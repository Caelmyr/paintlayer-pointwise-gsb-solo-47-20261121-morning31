"""Evaluate calibration reference quality independently of an artwork acquisition."""

from dataclasses import dataclass
import math

import numpy as np
from scipy import ndimage

from . import fields
from .calibration import check_references
from .errors import InsufficientSupport
from .measure import describe_values
from .quality import components
from .raster import display_range, histogram, png_bytes, support_summary
from .views import COLORS, flag_palette, gradient_rgba

DARK_HOT = 1
FLAT_CLIPPED = 2
WEAK_GAIN = 4
GAIN_OUTLIER = 8
ROW_ANOMALY = 16
COLUMN_ANOMALY = 32
FLAG_NAMES = {
    'hot_dark': DARK_HOT,
    'clipped_flat': FLAT_CLIPPED,
    'weak_gain': WEAK_GAIN,
    'gain_outlier': GAIN_OUTLIER,
    'row_deviation': ROW_ANOMALY,
    'column_deviation': COLUMN_ANOMALY,
}
DIAGNOSTIC_COLORS = tuple(COLORS[label] for label in ('rust', 'gold', 'violet', 'royal', 'aqua', 'pink'))


@dataclass(frozen=True)
class DetectorCheck:
    outlier_sigma: float
    dark_limit: float
    minimum_gain: float
    clip_limit: float
    axis_deviation: float
    neighbourhood: int
    component_area: int

    @classmethod
    def parse(cls, raw):
        raw = fields.request_object(raw, field='detector')
        return cls(
            outlier_sigma=fields.scalar(raw.get('outlier_sigma', 5), field='outlier_sigma', minimum=2, maximum=15),
            dark_limit=fields.scalar(raw.get('dark_limit', 0.12), field='dark_limit', minimum=0.0001, maximum=0.8),
            minimum_gain=fields.scalar(raw.get('minimum_gain', 0.06), field='minimum_gain', minimum=0.001, maximum=0.9),
            clip_limit=fields.scalar(raw.get('clip_limit', 0.995), field='clip_limit', minimum=0.5, maximum=1),
            axis_deviation=fields.scalar(raw.get('axis_deviation', 0.06), field='axis_deviation', minimum=0.001, maximum=0.5),
            neighbourhood=fields.integer(raw.get('neighbourhood', 17), field='neighbourhood', minimum=5, maximum=85),
            component_area=fields.integer(raw.get('component_area', 4), field='component_area', minimum=1, maximum=128),
        )

    def as_json(self):
        return {
            'outlier_sigma': self.outlier_sigma,
            'dark_limit': self.dark_limit,
            'minimum_gain': self.minimum_gain,
            'clip_limit': self.clip_limit,
            'axis_deviation': self.axis_deviation,
            'neighbourhood': self.neighbourhood,
            'component_area': self.component_area,
        }


def robust_noise(values):
    centre = float(np.median(values))
    dev = float(np.median(np.abs(values - centre))) * 1.4826
    return centre, max(dev, 1e-6)


def proportional_background(data, eligible, radius):
    available = eligible.astype(np.float32)
    intensity = np.where(eligible, data, 0).astype(np.float32)
    numerator = ndimage.uniform_filter(intensity, size=radius, mode='reflect')
    denominator = ndimage.uniform_filter(available, size=radius, mode='reflect')
    fit_valid = denominator >= 0.5
    average = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=fit_valid)
    return average, fit_valid


def axis_trace(gain, open_samples, axis):
    rows = gain if axis == 1 else gain.T
    usable = open_samples if axis == 1 else open_samples.T
    minimum = math.ceil(rows.shape[1] * 0.45)
    estimates = np.full(rows.shape[0], np.nan, dtype=np.float64)
    counts = np.count_nonzero(usable, axis=1)
    known = counts >= minimum
    for index in np.flatnonzero(known):
        estimates[index] = np.median(rows[index, usable[index]])
    return estimates, known, counts


def axis_background(trace, known, radius):
    support = known.astype(np.float64)
    samples = np.where(known, trace, 0)
    mean = ndimage.uniform_filter1d(samples, size=radius, mode='reflect')
    population = ndimage.uniform_filter1d(support, size=radius, mode='reflect')
    eligibility = known & (population > 0.55)
    result = np.zeros_like(trace)
    np.divide(mean, population, out=result, where=eligibility)
    return result, eligibility


def spans(marker):
    padded = np.pad(marker.astype(np.int8), 1)
    starts = np.flatnonzero(np.diff(padded) == 1)
    stops = np.flatnonzero(np.diff(padded) == -1)
    return [(int(a), int(b - 1)) for a, b in zip(starts, stops, strict=True)]


def axis_report_before_aggregate(gain, eligible, axis, settings):
    trace, measured, counts = axis_trace(gain, eligible, axis)
    estimate, reliable = axis_background(trace, measured, settings.neighbourhood)
    magnitude = np.zeros_like(trace)
    accepted = reliable & (estimate > settings.minimum_gain / 2)
    magnitude[accepted] = (trace[accepted] - estimate[accepted]) / estimate[accepted]
    suspicious = accepted & (np.abs(magnitude) >= settings.axis_deviation)
    grouped = [{
        'start': left,
        'end': right,
        'length': right - left + 1,
        'max_deviation': float(np.max(np.abs(magnitude[left:right + 1]))),
    } for left, right in spans(suspicious)]
    profile = [{
        'index': index,
        'median_gain': float(trace[index]) if measured[index] else None,
        'normalized_deviation': float(magnitude[index]) if accepted[index] else None,
        'sample_count': int(counts[index]),
    } for index in range(len(trace))]
    return suspicious, {
        'measurable': int(np.count_nonzero(measured)),
        'flagged': int(np.count_nonzero(suspicious)),
        'groups': grouped,
        'profile': profile,
    }


def diagnostic_categories(flags, gain, component_floor):
    findings = []
    for name, tag in FLAG_NAMES.items():
        hit = (flags & tag) != 0
        current = components(hit, component_floor, gain if name in {'weak_gain', 'gain_outlier'} else None, name)
        findings.append({
            'kind': name,
            'flag': tag,
            'affected': int(np.count_nonzero(hit)),
        } | current)
    return findings


def flat_diagnostics(dark, flat, mask, settings):
    check_references(dark, None, flat, mask)
    raw_dark = dark.normalized()
    raw_flat = flat.normalized()
    domain = dark.support & flat.support
    if mask is not None:
        domain &= mask != 0
    if np.count_nonzero(domain) < 25:
        raise InsufficientSupport('The references share fewer than 25 usable detector samples.')
    gain = raw_flat - raw_dark
    dark_hot = domain & (raw_dark > settings.dark_limit)
    flat_clip = domain & (raw_flat >= settings.clip_limit)
    weak = domain & (gain <= settings.minimum_gain)
    eligible = domain & ~(dark_hot | flat_clip | weak)
    if np.count_nonzero(eligible) < 25:
        raise InsufficientSupport(
            'The flat and dark references cannot establish a usable detector response.',
            facts={'usable': int(np.count_nonzero(eligible))},
        )
    fit, fit_valid = proportional_background(gain, eligible, settings.neighbourhood)
    accepted = eligible & fit_valid & (fit > 0)
    local_anomalies = np.zeros_like(gain)
    local_anomalies[accepted] = (gain[accepted] - fit[accepted]) / fit[accepted]
    if np.any(accepted):
        center, sigma = robust_noise(local_anomalies[accepted])
        outlier = accepted & (np.abs(local_anomalies - center) > settings.outlier_sigma * sigma)
    else:
        center = sigma = None
        outlier = np.zeros_like(accepted)
    striped_rows, row = axis_report_before_aggregate(gain, eligible, 1, settings)
    striped_columns, column = axis_report_before_aggregate(gain, eligible, 0, settings)
    tags = np.zeros_like(raw_dark, dtype=np.uint8)
    for affected, flag in (
        (dark_hot, DARK_HOT), (flat_clip, FLAT_CLIPPED), (weak, WEAK_GAIN), (outlier, GAIN_OUTLIER),
        (domain & striped_rows[:, None], ROW_ANOMALY), (domain & striped_columns[None, :], COLUMN_ANOMALY),
    ):
        tags[affected] |= flag
    reference = display_range(gain, eligible)
    details = {
        'settings': settings.as_json(),
        'extent': [gain.shape[1], gain.shape[0]],
        'support': support_summary(domain),
        'usable': support_summary(eligible),
        'gain': describe_values(gain[eligible]),
        'dark': describe_values(raw_dark[domain]),
        'flat': describe_values(raw_flat[domain]),
        'gain_histogram': histogram(gain, eligible, reference),
        'gain_limits': list(reference),
        'relative_residual_center': center,
        'robust_residual_sigma': sigma,
        'affected_pixels': int(np.count_nonzero(tags)),
        'rows': row,
        'columns': column,
    }
    details['categories'] = diagnostic_categories(tags, gain, settings.component_area)
    visual = gradient_rgba(gain, domain, reference, (25, 30, 45, 255), (238, 217, 150, 255))
    suspicious = flag_palette(tags, DIAGNOSTIC_COLORS)
    return details, png_bytes(visual), png_bytes(suspicious)
