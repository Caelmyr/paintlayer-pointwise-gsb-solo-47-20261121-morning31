"""Unit tests for the dual-exposure HDR merge.

Run in isolated mode (the host environment only exposes dependencies through an
external site-packages directory)::

    python3 -I tests/test_dualexposure.py -v
"""

import sys
from pathlib import Path

sys.path.insert(0, '/opt/pointwise-env/site-packages')
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import unittest

import numpy as np

from paintlayer.calibration import Correction, Frame, correct_plate
from paintlayer.dualexposure import (
    EXPOSURE_CONFLICT,
    LONG_ONLY,
    SHORT_ONLY,
    MergeSettings,
    merge_exposures,
)

FULL_SCALE = 65535


class MergeHarness(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.rng = rng
        self.height, self.width = 64, 96
        self.dark = np.full((self.height, self.width), 900, dtype=np.uint16)
        self.flat = np.full((self.height, self.width), 52000, dtype=np.uint16)
        truth = np.linspace(0.05, 2.6, self.width, dtype=np.float32)[None, :].repeat(self.height, axis=0)
        truth += rng.normal(0, 0.02, truth.shape).astype(np.float32)
        self.truth = truth
        self.transmission = np.exp(-truth)
        self.support = np.ones((self.height, self.width), dtype=bool)
        self.correction = Correction(0.995, 0.04, 0.008, 'attenuation')

    def frame(self, pixels):
        return Frame(pixels, self.support, FULL_SCALE)

    def expose(self, scale, noise, *, saturate=False, bury=False):
        data = (
            self.dark.astype(np.float32)
            + (self.flat - self.dark) * self.transmission * scale
            + self.rng.normal(0, noise, self.transmission.shape)
        )
        if saturate:
            data[self.transmission > 0.55] = FULL_SCALE
        if bury:
            # Densest readings collapse onto the dark floor and cannot be separated from it.
            floor = self.dark.astype(np.float32) - self.rng.uniform(0, 90, data.shape)
            data = np.where(data < self.dark + 4200, floor, data)
        return np.clip(np.rint(data), 0, FULL_SCALE).astype(np.uint16)

    def calibrate(self, pixels):
        return correct_plate(self.frame(pixels), self.frame(self.dark), self.frame(self.flat), None, self.correction)


class DualExposureMergeTests(MergeHarness):
    def test_recovers_both_extremes_and_aligns_scale(self):
        long_corrected = self.calibrate(self.expose(3.2, 90, saturate=True))
        short_corrected = self.calibrate(self.expose(1.0, 60, bury=True))
        merged = merge_exposures(long_corrected, short_corrected, MergeSettings.parse({}))

        ratio = merged.summary['alignment']['exposure_ratio_long_over_short']
        self.assertGreater(ratio, 2.5)
        self.assertLess(ratio, 4.0)

        long_saturated = ~long_corrected.surface.valid
        short_buried = ~short_corrected.surface.valid
        # The short exposure rescues the long exposure's clipped bright readings.
        self.assertGreater(int(np.count_nonzero(long_saturated & merged.surface.valid)), 100)
        # The long exposure rescues the short exposure's buried dark readings.
        self.assertGreater(int(np.count_nonzero(short_buried & merged.surface.valid)), 100)
        self.assertGreaterEqual(merged.surface.valid.sum(), long_corrected.surface.valid.sum())
        self.assertGreaterEqual(merged.surface.valid.sum(), short_corrected.surface.valid.sum())
        self.assertTrue(np.all(merged.owner[merged.surface.valid] > 0))

    def test_shared_region_tracks_true_attenuation(self):
        long_corrected = self.calibrate(self.expose(3.2, 70))
        short_corrected = self.calibrate(self.expose(1.0, 55))
        merged = merge_exposures(long_corrected, short_corrected, MergeSettings.parse({}))
        both = long_corrected.surface.valid & short_corrected.surface.valid
        error = np.abs(merged.surface.pixels[both] - self.truth[both])
        self.assertLess(float(np.median(error)), 0.05)
        self.assertEqual(merged.summary['conflict_pixels'], 0)
        self.assertEqual(int(np.count_nonzero(merged.reasons & EXPOSURE_CONFLICT)), 0)

    def test_flags_single_exposure_origin(self):
        long_corrected = self.calibrate(self.expose(3.2, 90, saturate=True))
        short_corrected = self.calibrate(self.expose(1.0, 60, bury=True))
        merged = merge_exposures(long_corrected, short_corrected, MergeSettings.parse({}))
        long_only = ~short_corrected.surface.valid & long_corrected.surface.valid
        short_only = ~long_corrected.surface.valid & short_corrected.surface.valid
        self.assertTrue(np.all((merged.reasons[long_only] & LONG_ONLY) != 0))
        self.assertTrue(np.all((merged.reasons[short_only] & SHORT_ONLY) != 0))
        self.assertTrue(np.all(merged.owner[long_only] == 1))
        self.assertTrue(np.all(merged.owner[short_only] == 2))

    def test_conflict_is_marked_and_downweighted(self):
        long_corrected = self.calibrate(self.expose(3.2, 70))
        short_corrected = self.calibrate(self.expose(1.0, 55))
        # Inject a localized physical disagreement on the short exposure only.
        region = slice(20, 30), slice(30, 45)
        short_pixels = short_corrected.raw.copy()
        short_pixels[region] = np.clip(short_pixels[region].astype(np.int32) + 9000, 0, FULL_SCALE).astype(np.uint16)
        short_conflicted = self.calibrate(short_pixels)
        merged = merge_exposures(long_corrected, short_conflicted, MergeSettings.parse({'conflict_tolerance': 0.05}))
        self.assertGreater(merged.summary['conflict_pixels'], 20)
        flagged = (merged.reasons & EXPOSURE_CONFLICT) != 0
        self.assertGreaterEqual(merged.summary['conflict_components']['count'], 1)
        # Conflicted readings stay usable but lose confidence versus the clean blend.
        clean = merge_exposures(long_corrected, short_corrected, MergeSettings.parse({'conflict_tolerance': 0.05}))
        self.assertTrue(np.all(merged.weight[flagged] <= clean.weight[flagged]))
        self.assertTrue(np.all(merged.surface.valid[flagged]))

    def test_rejects_dimension_mismatch_and_thin_overlap(self):
        long_corrected = self.calibrate(self.expose(3.2, 70))
        short_corrected = self.calibrate(self.expose(1.0, 55))
        cropped = type(short_corrected)(
            surface=type(long_corrected.surface)(
                short_corrected.surface.pixels[:-4], short_corrected.surface.valid[:-4],
            ),
            weight=short_corrected.weight[:-4],
            reasons=short_corrected.reasons[:-4],
            raw=short_corrected.raw[:-4],
            summary=short_corrected.summary,
        )
        with self.assertRaises(Exception):
            merge_exposures(long_corrected, cropped, MergeSettings.parse({}))

    def test_rejects_when_shared_support_is_too_small(self):
        long_corrected = self.calibrate(self.expose(3.2, 70, saturate=True))
        short_corrected = self.calibrate(self.expose(1.0, 55, bury=True))
        settings = MergeSettings.parse({'minimum_overlap': 100_000})
        with self.assertRaises(Exception):
            merge_exposures(long_corrected, short_corrected, settings)

    def test_transmission_model_merges_on_same_baseline(self):
        transmission_correction = Correction(0.995, 0.04, 0.008, 'transmission')
        long_result = correct_plate(
            self.frame(self.expose(2.5, 70)), self.frame(self.dark), self.frame(self.flat), None,
            transmission_correction,
        )
        short_result = correct_plate(
            self.frame(self.expose(1.0, 55)), self.frame(self.dark), self.frame(self.flat), None,
            transmission_correction,
        )
        merged = merge_exposures(long_result, short_result, MergeSettings.parse({}))
        self.assertEqual(merged.summary['unit'], 'relative transmission')
        shared = long_result.surface.valid & short_result.surface.valid
        error = np.abs(merged.surface.pixels[shared] - self.transmission[shared])
        self.assertLess(float(np.median(error)), 0.02)

    def test_conflict_detection_works_in_transmission_model(self):
        transmission_correction = Correction(0.995, 0.04, 0.008, 'transmission')
        long_result = correct_plate(
            self.frame(self.expose(2.5, 70)), self.frame(self.dark), self.frame(self.flat), None,
            transmission_correction,
        )
        short_pixels = self.expose(1.0, 55)
        region = slice(20, 30), slice(30, 45)
        short_pixels[region] = np.clip(
            short_pixels[region].astype(np.int32) + 9000, 0, FULL_SCALE,
        ).astype(np.uint16)
        short_result = correct_plate(
            self.frame(short_pixels), self.frame(self.dark), self.frame(self.flat), None,
            transmission_correction,
        )
        merged = merge_exposures(
            long_result, short_result, MergeSettings.parse({'conflict_tolerance': 0.05}),
        )
        self.assertGreater(merged.summary['conflict_pixels'], 20)
        self.assertTrue(np.all(np.isfinite(merged.residual)))
        self.assertEqual(int(np.count_nonzero(merged.reasons & EXPOSURE_CONFLICT)),
                         merged.summary['conflict_pixels'])


if __name__ == '__main__':
    unittest.main()
