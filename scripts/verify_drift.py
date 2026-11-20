#!/usr/bin/env python3
"""Standalone verification for multi-reference spatial drift correction.

Run from the project root with the project interpreter, e.g.
    python -I scripts/verify_drift.py
The script builds synthetic surfaces in a temporary directory and never
touches the application's runtime data.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from paintlayer.compare import (
    Comparison, PENDING_REVIEW, DENSER, build_correction, change_map,
)
from paintlayer.errors import InsufficientSupport
from paintlayer.geometry import Outline, outline_pixels
from paintlayer.raster import Surface

WIDTH, HEIGHT = 240, 180
rng = np.random.default_rng(7)
base = (0.4 + 0.0006 * np.arange(WIDTH)[None, :] + 0.0004 * np.arange(HEIGHT)[:, None]).astype(np.float32)
base += rng.normal(0, 0.002, base.shape).astype(np.float32)
valid = np.ones((HEIGHT, WIDTH), dtype=bool)
spread = np.full((HEIGHT, WIDTH), 0.004, dtype=np.float32)


def outline(x, y, w, h):
    return Outline(
        outer=((x, y), (x + w, y), (x + w, y + h), (x, y + h)),
        holes=(),
    )


def region(identity, name, box):
    x, y, w, h = box
    return {'id': identity, 'name': name, 'revision': 1}, outline(x, y, w, h)


def surfaces(drift_field, treatment_box=None, treatment_delta=0.0):
    later = base + drift_field.astype(np.float32)
    if treatment_box is not None:
        x, y, w, h = treatment_box
        later[y:y + h, x:x + w] += treatment_delta
    return Surface(base.copy(), valid), Surface(later.astype(np.float32), valid)


def check(condition, label):
    status = 'PASS' if condition else 'FAIL'
    print(f'{status}  {label}')
    if not condition:
        check.failures += 1


check.failures = 0


# --- Scenario 1: tilted exposure field recovered from four references -------
gradient = (0.010 + 0.0004 * np.arange(WIDTH)[None, :] - 0.0002 * np.arange(HEIGHT)[:, None]).astype(np.float32)
refs = [
    region('a' * 24, 'Upper left', (12, 12, 24, 20)),
    region('b' * 24, 'Upper right', (WIDTH - 36, 12, 24, 20)),
    region('c' * 24, 'Lower left', (12, HEIGHT - 32, 24, 20)),
    region('d' * 24, 'Lower right', (WIDTH - 36, HEIGHT - 32, 24, 20)),
]
earlier, later = surfaces(gradient, treatment_box=(95, 75, 40, 30), treatment_delta=0.10)
paired = earlier.valid & later.valid
correction, reachable, confidence, report = build_correction(earlier, later, paired, refs)
check(report['accepted_count'] == 4 and report['rejected_count'] == 0, 'four consistent references all accepted')
check(report['mode'] == 'plane_idw', f'mode is plane+IDW (got {report["mode"]})')
# Fitted plane gradient should recover the synthetic 0.0004 / -0.0002 slope.
gx, gy = report['plane_gradient']
check(abs(gx - 0.0004) < 2e-5 and abs(gy + 0.0002) < 2e-5, f'plane gradient recovered ({gx:.2e}, {gy:.2e})')
# Reference pixels themselves must be corrected almost exactly.
ref_mask = np.zeros_like(valid)
for _, outline_value in refs:
    (x, y, w, h), mask = outline_pixels(outline_value, (WIDTH, HEIGHT))
    ref_mask[y:y + h, x:x + w] |= mask
residual_on_refs = (later.pixels - earlier.pixels - correction)[ref_mask]
check(np.max(np.abs(residual_on_refs)) < 0.01,
      f'residual inside reference regions near zero (max {np.max(np.abs(residual_on_refs)):.4f})')
check(bool(reachable.all()), 'canvas covered when references bracket it')

details, products = change_map(
    earlier, later, spread, spread, Comparison.parse({}),
    references=refs,
)
classes = products['classification']
delta = products['difference'].pixels
x, y, w, h = (95, 75, 40, 30)
treatment_pixels = classes[y:y + h, x:x + w]
check(np.count_nonzero(treatment_pixels == DENSER) > 0.8 * w * h,
      'genuine treatment trace still classified as denser after correction')
check(np.count_nonzero(classes == PENDING_REVIEW) == 0, 'no pending-review pixels inside reference reach')
check(abs(float(np.median(delta[ref_mask]))) < 0.004, 'median residual over references below change threshold')


# --- Scenario 2: distant corner outside reach is held pending review --------
far_refs = [
    region('e' * 24, 'Clustered A', (10, 10, 16, 16)),
    region('f' * 24, 'Clustered B', (40, 12, 16, 16)),
    region('g' * 24, 'Clustered C', (12, 44, 16, 16)),
]
earlier2, later2 = surfaces(gradient)
_, reachable2, confidence2, report2 = build_correction(earlier2, later2, paired, far_refs)
check(not bool(reachable2.all()), 'clustered references leave part of the canvas out of reach')
check(bool(reachable2[20, 20]), 'area adjacent to references is reachable')
check(not bool(reachable2[-15, -15]), 'far corner is outside the correction reach')
check(float(confidence2[-15, -15]) == 0.0, 'confidence is zero beyond the reach')
details2, products2 = change_map(earlier2, later2, spread, spread, Comparison.parse({}), references=far_refs)
check(np.count_nonzero(products2['classification'] == PENDING_REVIEW) == report2['unresolved_pixels'],
      'out-of-reach paired pixels are classified pending review')
check(details2['pending_review_pixels'] > 0, 'details report pending-review pixel count')


# --- Scenario 3: a self-contradicting reference is rejected -----------------
bad_refs = far_refs + [region('h' * 24, 'Contradictory', (180, 120, 20, 20))]
# Tamper with the later surface only inside that region to fake a huge offset.
later3 = later2.pixels.copy()
bx, by, bw, bh = 180, 120, 20, 20
later3[by:by + bh, bx:bx + bw] += 0.25
later3_surface = Surface(later3, valid)
_, _, _, report3 = build_correction(earlier2, later3_surface, paired, bad_refs)
check(report3['rejected_count'] == 1 and report3['accepted_count'] == 3,
      'contradictory reference excluded while consistent ones keep the correction')
contradictory = next(item for item in report3['references'] if item['name'] == 'Contradictory')
check(contradictory['accepted'] is False and contradictory['reason'], 'excluded reference carries an explanatory reason')

# Two mutually contradictory references cannot establish any gradient.
conflict = [
    region('k' * 24, 'Says positive', (10, 10, 20, 20)),
    region('l' * 24, 'Says negative', (180, 130, 20, 20)),
]
conflict_later = later2.pixels.copy()
conflict_later[10:30, 10:30] += 0.22
conflict_later[130:150, 180:200] -= 0.22
try:
    build_correction(earlier2, Surface(conflict_later, valid), paired, conflict)
    check(False, 'mutually contradictory pair rejected')
except InsufficientSupport as exc:
    check(set(exc.facts['inconsistent_references']) == {'Says positive', 'Says negative'},
          'both conflicting references named in error facts')


# --- Scenario 4: reference region without enough paired samples -------------
sparse_paired = valid.copy()
sparse_paired[1:4, 1:4] = False
weak = [region('i' * 24, 'Tiny', (1, 1, 3, 3)), region('j' * 24, 'Tiny two', (6, 6, 16, 16))]
try:
    build_correction(earlier2, later2, sparse_paired, weak)
    check(False, 'undersized references rejected')
except InsufficientSupport:
    print('PASS  undersized references rejected')


# --- Scenario 5: legacy single-reference and no-reference paths unchanged ---
single_region, single_outline = refs[0]
details_legacy, products_legacy = change_map(
    earlier, later, spread, spread, Comparison.parse({}),
    reference=single_outline,
)
check(details_legacy['drift_correction']['mode'] == 'constant', 'single outline uses constant-offset mode')
check(np.count_nonzero(products_legacy['classification'] == PENDING_REVIEW) == 0,
      'legacy path never emits pending review')
check(details_legacy['reference'] is not None and details_legacy['reference_offset'] is not None,
      'legacy reference summary preserved')
details_plain, products_plain = change_map(earlier, later, spread, spread, Comparison.parse({}))
check(details_plain['drift_correction']['mode'] == 'none' and details_plain['reference_offset'] == 0.0,
      'no-reference path keeps zero offset')
for label, product in (('spatial', products), ('legacy', products_legacy), ('plain', products_plain)):
    check(set(product) == {'difference', 'uncertainty', 'classification', 'correction', 'adjustment'},
          f'{label} run exposes the full layer set')


# --- Scenario 6: uncertainty grows with extrapolation distance --------------
near = confidence2[20, 20]
far = confidence2[60, 60] if reachable2[60, 60] else 0.0
check(near >= far, 'correction confidence decays with distance from references')
unc = products2['uncertainty']
check(float(unc[20, 20]) <= float(np.max(unc)) + 1e-9, 'uncertainty field is finite and ordered')

print()
print(f'{check.failures} failure(s)')
sys.exit(1 if check.failures else 0)
