"""Merge a long and a short detector exposure of the same position into one plate.

The two exposures are calibrated independently with the same dark/flat references
(:mod:`paintlayer.calibration`). Their relative transmission surfaces share a
linear physical scale, so the overlap where both exposures are measurable fixes
a single exposure ratio and aligns the long exposure onto the short exposure's
numerical baseline. Where only one exposure reads, its value is recovered. Where
both read but disagree beyond the conflict tolerance, the disagreement is kept
visible and the pixel is marked so downstream measurements can treat it with
care. The output keeps the same numerical model and surface contract as a normal
calibrated plate and can be placed in an assembly unchanged.
"""

from dataclasses import dataclass

import numpy as np

from . import fields
from .errors import ExamError, InsufficientSupport
from .measure import describe_values
from .quality import components
from .raster import Surface, display_range, histogram, support_summary

# New acquisition states beyond the single-plate calibration flags (bits 0-4).
LONG_ONLY = 32
SHORT_ONLY = 64
EXPOSURE_CONFLICT = 128
MERGE_FLAGS = {
    'long_exposure_only': LONG_ONLY,
    'short_exposure_only': SHORT_ONLY,
    'exposure_conflict': EXPOSURE_CONFLICT,
}
# Ratios outside this span are treated as detector failures rather than physics.
RATIO_MIN = 0.01
RATIO_MAX = 100.0
LOG_RATIO_LIMIT = 8.0


@dataclass(frozen=True)
class MergeSettings:
    conflict_tolerance: float
    alignment_sigma: float
    minimum_overlap: int
    minimum_component_area: int

    @classmethod
    def parse(cls, raw):
        raw = fields.request_object(raw, field='merge')
        return cls(
            conflict_tolerance=fields.scalar(
                raw.get('conflict_tolerance', 0.09), field='conflict_tolerance', minimum=0.005, maximum=2,
            ),
            alignment_sigma=fields.scalar(
                raw.get('alignment_sigma', 3.0), field='alignment_sigma', minimum=1.5, maximum=8,
            ),
            minimum_overlap=fields.integer(
                raw.get('minimum_overlap', 40), field='minimum_overlap', minimum=10, maximum=100_000,
            ),
            minimum_component_area=fields.integer(
                raw.get('minimum_component_area', 6), field='minimum_component_area', minimum=1, maximum=50_000,
            ),
        )

    def as_json(self):
        return {
            'conflict_tolerance': self.conflict_tolerance,
            'alignment_sigma': self.alignment_sigma,
            'minimum_overlap': self.minimum_overlap,
            'minimum_component_area': self.minimum_component_area,
        }


@dataclass(frozen=True)
class CalibratedExposure:
    """Transmission (linear, before the optional log model) and its bookkeeping."""

    transmission: np.ndarray
    surface: Surface
    weight: np.ndarray
    reasons: np.ndarray
    raw: np.ndarray


@dataclass(frozen=True)
class MergedPlate:
    surface: Surface
    weight: np.ndarray
    reasons: np.ndarray
    raw: np.ndarray
    owner: np.ndarray
    conflict: np.ndarray
    residual: np.ndarray
    summary: dict


def calibrated_exposure(corrected):
    """Adapt a :class:`paintlayer.calibration.Corrected` result to merge inputs."""
    valid = corrected.surface.valid
    if corrected.summary['unit'].startswith('relative optical'):
        transmission = np.zeros(valid.shape, dtype=np.float32)
        transmission[valid] = np.exp(-corrected.surface.pixels[valid], dtype=np.float32)
    else:
        transmission = corrected.surface.pixels
    return CalibratedExposure(
        transmission=transmission.astype(np.float32),
        surface=corrected.surface,
        weight=corrected.weight.astype(np.float32),
        reasons=corrected.reasons.astype(np.uint8),
        raw=corrected.raw,
    )


def weighted_median(values, weights):
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights)
    cutoff = 0.5 * cumulative[-1]
    return float(sorted_values[np.searchsorted(cumulative, cutoff)])


def estimate_exposure_ratio(long_exposure, short_exposure, settings):
    """Robust median of long/short linear transmission over the shared support."""
    shared = long_exposure.surface.valid & short_exposure.surface.valid
    if int(np.count_nonzero(shared)) < settings.minimum_overlap:
        raise InsufficientSupport(
            'The two exposures share too few measurable pixels to align their scale.',
            facts={'shared_pixels': int(np.count_nonzero(shared)), 'required': settings.minimum_overlap},
        )
    long_t = long_exposure.transmission[shared]
    short_t = short_exposure.transmission[shared]
    physical = (long_t > 0) & (short_t > 0)
    ratios = long_t[physical] / short_t[physical]
    plausible = (ratios >= RATIO_MIN) & (ratios <= RATIO_MAX)
    if not np.any(plausible):
        raise InsufficientSupport('No physically plausible exposure ratio exists in the shared measurable region.')
    ratios = ratios[plausible].astype(np.float64)
    shared_weights = (
        long_exposure.weight[shared][physical][plausible].astype(np.float64)
        * short_exposure.weight[shared][physical][plausible].astype(np.float64)
    )
    log_ratios = np.log(ratios)
    centre = weighted_median(log_ratios, np.maximum(shared_weights, 1e-6))
    mad = float(np.median(np.abs(log_ratios - centre))) * 1.4826
    spread = max(mad, 1e-5)
    accepted = np.abs(log_ratios - centre) <= settings.alignment_sigma * spread
    if int(np.count_nonzero(accepted)) < max(8, settings.minimum_overlap // 4):
        accepted = np.ones(log_ratios.shape, dtype=bool)
    aligned = weighted_median(log_ratios[accepted], np.maximum(shared_weights[accepted], 1e-6))
    aligned = float(np.clip(aligned, -LOG_RATIO_LIMIT, LOG_RATIO_LIMIT))
    ratio = float(np.exp(aligned))
    report = {
        'exposure_ratio_long_over_short': ratio,
        'log_ratio_center': aligned,
        'robust_log_ratio_sigma': spread,
        'alignment_pixels': int(np.count_nonzero(accepted)),
        'candidate_pixels': int(ratios.size),
        'ratio_percentiles': {
            str(tile): float(value)
            for tile, value in zip((5, 50, 95), np.percentile(ratios, (5, 50, 95)), strict=True)
        },
    }
    return ratio, report


def merge_exposures(long_corrected, short_corrected, settings):
    if long_corrected.surface.pixels.shape != short_corrected.surface.pixels.shape:
        raise ExamError(
            'The long and short exposures must use the same detector dimensions.',
            field='exposures',
            facts={
                'long': list(long_corrected.surface.extent),
                'short': list(short_corrected.surface.extent),
            },
        )
    long_exposure = calibrated_exposure(long_corrected)
    short_exposure = calibrated_exposure(short_corrected)
    ratio, alignment = estimate_exposure_ratio(long_exposure, short_exposure, settings)

    long_valid = long_exposure.surface.valid
    short_valid = short_exposure.surface.valid
    both = long_valid & short_valid
    long_only = long_valid & ~short_valid
    short_only = short_valid & ~long_valid

    # Align the long exposure's linear transmission onto the short baseline.
    long_aligned = np.zeros_like(short_exposure.transmission)
    long_aligned[long_valid] = long_exposure.transmission[long_valid] / ratio
    merged_transmission = np.zeros(both.shape, dtype=np.float32)
    weight = np.zeros(both.shape, dtype=np.float32)
    owner = np.zeros(both.shape, dtype=np.uint8)
    conflict = np.zeros(both.shape, dtype=np.uint8)
    reasons = np.zeros(both.shape, dtype=np.uint8)

    # Preserve every per-exposure calibration flag on the pixels it describes.
    reasons |= long_exposure.reasons
    reasons |= short_exposure.reasons

    # Single-exposure recovery: the only readable exposure carries the pixel.
    merged_transmission[long_only] = long_aligned[long_only]
    weight[long_only] = long_exposure.weight[long_only]
    owner[long_only] = 1
    reasons[long_only] |= LONG_ONLY
    merged_transmission[short_only] = short_exposure.transmission[short_only]
    weight[short_only] = short_exposure.weight[short_only]
    owner[short_only] = 2
    reasons[short_only] |= SHORT_ONLY

    # Shared support: confidence-weighted blend on the common numerical baseline.
    long_weight = long_exposure.weight
    short_weight = short_exposure.weight
    total_weight = long_weight + short_weight
    merged_transmission[both] = (
        long_aligned[both] * long_weight[both]
        + short_exposure.transmission[both] * short_weight[both]
    ) / total_weight[both]

    residual = np.zeros(both.shape, dtype=np.float32)
    # Optical-depth separation on the aligned baseline, stable for either output model.
    safe_long = np.maximum(long_aligned[both], 1e-12)
    safe_short = np.maximum(short_exposure.transmission[both], 1e-12)
    residual[both] = np.abs(-np.log(safe_long, dtype=np.float32) + np.log(safe_short, dtype=np.float32))
    conflicting = both & (residual > settings.conflict_tolerance)
    conflict[conflicting] = 1
    reasons[conflicting] |= EXPOSURE_CONFLICT
    # A conflict means the two readings do not reconcile: down-rank confidence but
    # keep the weighted mean as the best available reading.
    weight[both] = np.minimum(long_weight[both], short_weight[both])
    weight[conflicting] *= 0.25
    owner[both] = np.where(long_weight[both] >= short_weight[both], 1, 2)

    supported = long_valid | short_valid
    if int(np.count_nonzero(supported)) < 12:
        raise InsufficientSupport(
            'The merged exposures leave fewer than twelve measurable pixels.',
            facts={'supported': int(np.count_nonzero(supported))},
        )

    model_is_attenuation = long_corrected.summary['unit'].startswith('relative optical')
    signal = np.zeros(supported.shape, dtype=np.float32)
    if model_is_attenuation:
        positive = supported & (merged_transmission > 0)
        signal[positive] = -np.log(merged_transmission[positive], dtype=np.float32)
        unit = 'relative optical depth'
    else:
        signal[supported] = merged_transmission[supported]
        unit = 'relative transmission'

    surface = Surface(signal, supported)
    limits = display_range(signal, supported)
    conflict_components = components(
        conflicting.astype(bool), settings.minimum_component_area, residual, 'exposure_conflict',
    )
    summary = {
        'extent': [int(supported.shape[1]), int(supported.shape[0])],
        'unit': unit,
        'display': list(limits),
        'support': support_summary(supported),
        'distribution': histogram(signal, supported, limits),
        'mean_weight': float(np.mean(weight[supported], dtype=np.float64)),
        'alignment': alignment,
        'settings': settings.as_json(),
        'exposure_pixels': {
            'shared': int(np.count_nonzero(both)),
            'long_only': int(np.count_nonzero(long_only)),
            'short_only': int(np.count_nonzero(short_only)),
            'neither': int(np.count_nonzero(~supported)),
        },
        'conflict_pixels': int(np.count_nonzero(conflicting)),
        'conflict_fraction': float(np.count_nonzero(conflicting) / max(1, np.count_nonzero(both))),
        'conflict_residual': describe_values(residual[conflicting]),
        'conflict_components': conflict_components,
        'reasons': merge_reason_counts(reasons),
    }
    return MergedPlate(surface, weight, reasons, long_corrected.raw, owner, conflict, residual, summary)


def merge_reason_counts(reasons):
    flags = {
        'acquisition_mask': 1,
        'saturated_detector': 2,
        'weak_flat_response': 4,
        'below_dark_signal': 8,
        'clipped_flat': 16,
        **MERGE_FLAGS,
    }
    result = {name: int(np.count_nonzero(reasons & bit)) for name, bit in flags.items()}
    result['flagged_pixels'] = int(np.count_nonzero(reasons))
    return result
