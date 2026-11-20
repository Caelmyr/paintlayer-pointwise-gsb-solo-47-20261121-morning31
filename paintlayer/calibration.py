"""Dark and flat field correction for immutable detector plate surfaces."""

from dataclasses import dataclass

import numpy as np

from . import fields
from .errors import ImageFormatError, InsufficientSupport
from .raster import Surface, display_range, histogram, support_summary

MASKED = 1
SATURATED = 2
DEAD_RESPONSE = 4
BELOW_DARK = 8
UNRELIABLE_FLAT = 16
CAL_FLAGS = {
    'acquisition_mask': MASKED,
    'saturated_detector': SATURATED,
    'weak_flat_response': DEAD_RESPONSE,
    'below_dark_signal': BELOW_DARK,
    'clipped_flat': UNRELIABLE_FLAT,
}


@dataclass(frozen=True)
class Correction:
    saturation: float
    min_span: float
    min_transmission: float
    model: str

    @classmethod
    def from_payload(cls, payload):
        payload = fields.request_object(payload, field='correction')
        return cls(
            saturation=fields.scalar(payload.get('saturation', 0.995), field='saturation', minimum=0.6, maximum=1),
            min_span=fields.scalar(payload.get('min_span', 0.04), field='min_span', minimum=0.0001, maximum=0.45),
            min_transmission=fields.scalar(
                payload.get('min_transmission', 0.008),
                field='min_transmission',
                minimum=0.0001,
                maximum=0.25,
            ),
            model=fields.choice(payload.get('model', 'attenuation'), field='model', options={'attenuation', 'transmission'}),
        )

    def as_json(self):
        return {
            'saturation': self.saturation,
            'min_span': self.min_span,
            'min_transmission': self.min_transmission,
            'model': self.model,
        }


@dataclass(frozen=True)
class Frame:
    pixels: np.ndarray
    support: np.ndarray
    full_scale: int

    def normalized(self):
        return self.pixels.astype(np.float32) / self.full_scale


@dataclass(frozen=True)
class Corrected:
    surface: Surface
    weight: np.ndarray
    reasons: np.ndarray
    raw: np.ndarray
    summary: dict


def check_references(detector, dark, flat, mask):
    extent = detector.support.shape
    for frame, name in ((dark, 'dark'), (flat, 'flat')):
        if frame is None:
            continue
        if frame.support.shape != extent:
            raise ImageFormatError(
                'Calibration references must use the same detector dimensions.',
                field=name,
                facts={'detector': list(extent[::-1]), 'reference': list(frame.support.shape[::-1])},
            )
        if frame.full_scale != detector.full_scale:
            raise ImageFormatError('Calibration references must use the same detector bit depth.', field=name)
    if mask is not None and mask.shape != extent:
        raise ImageFormatError('The acquisition mask must use the same detector dimensions.', field='mask')


def baseline_refs(detector, dark, flat):
    if dark is None:
        dark_data = np.zeros(detector.support.shape, dtype=np.float32)
        dark_support = detector.support
    else:
        dark_data = dark.normalized()
        dark_support = dark.support
    if flat is None:
        flat_data = np.ones(detector.support.shape, dtype=np.float32)
        flat_support = detector.support
    else:
        flat_data = flat.normalized()
        flat_support = flat.support
    return dark_data, flat_data, dark_support & flat_support


def annotate_bits(reasons, mask, flag):
    reasons[mask] |= flag


def reason_counts(reasons):
    result = {name: int(np.count_nonzero(reasons & bit)) for name, bit in CAL_FLAGS.items()}
    result['flagged_pixels'] = int(np.count_nonzero(reasons))
    return result


def confidence(span, numerator, signal, valid, correction):
    weight = np.zeros(span.shape, dtype=np.float32)
    span_grade = np.clip((span[valid] - correction.min_span) / (0.40 - min(0.35, correction.min_span)), 0.05, 1)
    shadow_grade = np.clip(numerator[valid] / (span[valid] * correction.min_transmission * 6), 0.08, 1)
    clipping_grade = np.clip((correction.saturation - signal[valid]) / 0.08, 0.08, 1)
    weight[valid] = span_grade * shadow_grade * clipping_grade
    return weight


def correct_plate(detector, dark, flat, mask, correction):
    check_references(detector, dark, flat, mask)
    dark_image, flat_image, ref_support = baseline_refs(detector, dark, flat)
    detector_data = detector.normalized()
    hard_support = detector.support & ref_support
    if mask is not None:
        hard_support &= mask.astype(bool)
    reasons = np.zeros(detector.support.shape, dtype=np.uint8)
    annotate_bits(reasons, ~hard_support, MASKED)
    annotate_bits(reasons, detector_data >= correction.saturation, SATURATED)
    if flat is not None:
        annotate_bits(reasons, flat_image >= correction.saturation, UNRELIABLE_FLAT)
    span = flat_image - dark_image
    numerator = detector_data - dark_image
    annotate_bits(reasons, span < correction.min_span, DEAD_RESPONSE)
    confidence_floor = span * correction.min_transmission
    annotate_bits(reasons, numerator < confidence_floor, BELOW_DARK)
    valid = reasons == 0
    num_valid = int(np.count_nonzero(valid))
    if num_valid < 12:
        raise InsufficientSupport(
            'Calibration leaves fewer than twelve measurable pixels. Check the references or acquisition mask.',
            facts={'reasons': reason_counts(reasons), 'valid': num_valid},
        )
    output = np.zeros(valid.shape, dtype=np.float32)
    output[valid] = numerator[valid] / span[valid]
    if correction.model == 'attenuation':
        output[valid] = -np.log(output[valid])
    weight = confidence(span, numerator, detector_data, valid, correction)
    numeric_limits = display_range(output, valid)
    between = numerator[valid] > span[valid]
    summary = {
        'extent': [int(valid.shape[1]), int(valid.shape[0])],
        'unit': 'relative optical depth' if correction.model == 'attenuation' else 'relative transmission',
        'display': list(numeric_limits),
        'reasons': reason_counts(reasons),
        'support': support_summary(valid),
        'distribution': histogram(output, valid, numeric_limits),
        'above_reference_pixels': int(np.count_nonzero(between)),
        'mean_weight': float(np.mean(weight[valid], dtype=np.float64)),
        'references': {'dark_used': dark is not None, 'flat_used': flat is not None, 'mask_used': mask is not None},
    }
    return Corrected(Surface(output, valid), weight, reasons, detector.pixels, summary)


def interpolate_reference_probe(frame, position, radius):
    x, y = position
    start_x = max(0, x - radius)
    end_x = min(frame.support.shape[1], x + radius + 1)
    start_y = max(0, y - radius)
    end_y = min(frame.support.shape[0], y + radius + 1)
    local = frame.pixels[start_y:end_y, start_x:end_x]
    support = frame.support[start_y:end_y, start_x:end_x]
    xs = np.arange(start_x, end_x, dtype=np.int32) - x
    ys = np.arange(start_y, end_y, dtype=np.int32) - y
    inside = xs[None, :] ** 2 + ys[:, None] ** 2 <= radius * radius
    values = local[support & inside]
    if values.size == 0:
        return {'position': [x, y], 'radius': radius, 'samples': 0, 'mean': None, 'median': None}
    return {
        'position': [x, y],
        'radius': radius,
        'samples': int(values.size),
        'mean': float(np.mean(values, dtype=np.float64)),
        'median': float(np.median(values)),
        'minimum': int(values.min()),
        'maximum': int(values.max()),
        'standard_deviation': float(np.std(values, dtype=np.float64)),
        'full_scale': frame.full_scale,
    }
