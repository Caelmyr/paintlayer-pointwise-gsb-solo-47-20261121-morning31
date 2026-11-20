"""Reproducible detector measurements over outlined areas and canvas transects."""

from dataclasses import dataclass
import math

import numpy as np

from . import fields
from .errors import InsufficientSupport, InvalidGeometry
from .geometry import bilinear_values, outline_pixels
from .raster import histogram, support_summary
from .settings import MAX_PROFILE_SAMPLES

PERCENTILES = (5, 25, 50, 75, 95)


def describe_values(values):
    if values.size == 0:
        return None
    percentiles = np.percentile(values, PERCENTILES)
    mean = float(np.mean(values, dtype=np.float64))
    median = float(percentiles[2])
    deviations = np.abs(values - median)
    return {
        'count': int(values.size),
        'minimum': float(np.min(values)),
        'maximum': float(np.max(values)),
        'mean': mean,
        'median': median,
        'std': float(np.std(values, dtype=np.float64)),
        'median_absolute_deviation': float(np.median(deviations)),
        'percentiles': {str(quantile): float(value) for quantile, value in zip(PERCENTILES, percentiles, strict=True)},
    }


def four_part_summary(values, geometry, valid):
    height, width = geometry.shape
    parts = []
    for vertical, horizontal, label in (
        (slice(None, height // 2), slice(None, width // 2), 'northwest'),
        (slice(None, height // 2), slice(width // 2, None), 'northeast'),
        (slice(height // 2, None), slice(None, width // 2), 'southwest'),
        (slice(height // 2, None), slice(width // 2, None), 'southeast'),
    ):
        area = geometry[vertical, horizontal]
        good = valid[vertical, horizontal]
        measured = values[vertical, horizontal][good]
        inside = int(np.count_nonzero(area))
        parts.append({
            'section': label,
            'included': inside,
            'measured': int(measured.size),
            'median': float(np.median(measured)) if measured.size else None,
            'mean': float(np.mean(measured, dtype=np.float64)) if measured.size else None,
            'usable_fraction': measured.size / inside if inside else None,
        })
    return parts


def area_measurement(surface, outline, *, scale, unit, coverage=None, limit_pair=None):
    bounds, mask = outline_pixels(outline, surface.extent)
    cropped = surface.crop(bounds)
    valid = mask & cropped.valid & np.isfinite(cropped.pixels)
    included = int(np.count_nonzero(mask))
    measured = int(np.count_nonzero(valid))
    if not measured:
        raise InsufficientSupport('No measurable detector samples lie inside this region.')
    values = cropped.pixels[valid]
    report = {
        'bounds': list(bounds),
        'extent': list(surface.extent),
        'included_pixels': included,
        'measured_pixels': measured,
        'unmeasured_pixels': included - measured,
        'valid_fraction': measured / included,
        'outline_area_square_pixels': outline.geometric_area,
        'outline_area_square_mm': outline.geometric_area * scale ** 2 if scale else None,
        'measured_square_mm': measured * scale ** 2 if scale else None,
        'units': unit,
        'distribution': describe_values(values),
        'histogram': histogram(cropped.pixels, valid, limit_pair, bins=40),
        'sections': four_part_summary(cropped.pixels, mask, valid),
    }
    if coverage is not None:
        x, y, width, height = bounds
        local = coverage[y:y + height, x:x + width][valid]
        bins = np.bincount(local)
        report['coverage'] = {str(index): int(count) for index, count in enumerate(bins) if count}
    if limit_pair is not None:
        low, high = limit_pair
        below = int(np.count_nonzero(values < low))
        above = int(np.count_nonzero(values > high))
        report['thresholds'] = {
            'range': list(limit_pair),
            'below': below,
            'inside': measured - below - above,
            'above': above,
            'inside_fraction': (measured - below - above) / measured,
        }
    return report, mask, valid


@dataclass(frozen=True)
class Transect:
    start: tuple
    end: tuple
    half_width: int
    step: float
    min_fraction: float
    aggregate: str

    @classmethod
    def parse(cls, raw, extent):
        raw = fields.request_object(raw, field='transect')
        start = fields.point(raw.get('start'), field='start', extent=extent)
        end = fields.point(raw.get('end'), field='end', extent=extent)
        step = fields.scalar(raw.get('step', 1), field='step', minimum=0.4, maximum=8)
        definition = cls(
            start=start,
            end=end,
            half_width=fields.integer(raw.get('half_width', 0), field='half_width', minimum=0, maximum=16),
            step=step,
            min_fraction=fields.scalar(raw.get('min_fraction', 0.5), field='min_fraction', minimum=0.1, maximum=1),
            aggregate=fields.choice(raw.get('aggregate', 'median'), field='aggregate', options={'median', 'mean'}),
        )
        if definition.length < 2:
            raise InvalidGeometry('A transect must span at least two canvas pixels.')
        if definition.interval_count + 1 > MAX_PROFILE_SAMPLES:
            raise InvalidGeometry('Use a larger sample interval for this transect.')
        return definition

    @property
    def length(self):
        return math.dist(self.start, self.end)

    @property
    def interval_count(self):
        return max(1, math.ceil(self.length / self.step))

    def as_json(self):
        return {
            'start': list(self.start),
            'end': list(self.end),
            'half_width': self.half_width,
            'step': self.step,
            'aggregate': self.aggregate,
            'min_fraction': self.min_fraction,
        }

    def position_grid(self):
        origin = np.array(self.start, dtype=np.float64)
        delta = np.array(self.end, dtype=np.float64) - origin
        positions = np.linspace(0.0, 1.0, self.interval_count + 1)
        centres = origin + positions[:, None] * delta[None, :]
        normal = np.array([-delta[1], delta[0]]) / self.length
        offsets = np.arange(-self.half_width, self.half_width + 1, dtype=np.float64)
        across = centres[:, None, :] + offsets[None, :, None] * normal[None, None, :]
        return self.length * positions, centres, across


def aggregate_profile(values, usable, aggregator):
    if aggregator == 'mean':
        return float(np.mean(values[usable], dtype=np.float64))
    return float(np.median(values[usable]))


def break_intervals(positions, states):
    intervals = []
    start = 0
    for index, valid in enumerate(states):
        if index == 0 or valid == states[index - 1]:
            continue
        intervals.append((start, index - 1, states[index - 1]))
        start = index
    intervals.append((start, len(states) - 1, states[-1]))
    return [
        {
            'first_index': first,
            'last_index': last,
            'start': float(positions[first]),
            'finish': float(positions[last]),
            'supported': bool(ok),
        }
        for first, last, ok in intervals
    ]


def threshold_crossings(positions, values, valid, threshold):
    events = []
    index = 0
    while index < len(positions) - 1:
        if not valid[index]:
            index += 1
            continue
        if values[index] == threshold:
            start = index
            while index + 1 < len(positions) and valid[index + 1] and values[index + 1] == threshold:
                index += 1
            before = values[start - 1] if start > 0 and valid[start - 1] else None
            after = values[index + 1] if index + 1 < len(positions) and valid[index + 1] else None
            kind = 'plateau'
            if before is not None and after is not None and (before < threshold) != (after < threshold):
                kind = 'rise' if before < after else 'fall'
            events.append({
                'distance': float(positions[start]),
                'interval_end': float(positions[index]),
                'kind': kind,
            })
        elif valid[index + 1] and values[index + 1] != threshold:
            a, b = values[index], values[index + 1]
            if (a < threshold) != (b < threshold):
                fraction = (threshold - a) / (b - a)
                crossing = positions[index] + fraction * (positions[index + 1] - positions[index])
                events.append({'distance': float(crossing), 'interval_end': None, 'kind': 'rise' if b > a else 'fall'})
        index += 1
    if valid[-1] and values[-1] == threshold and (not valid[-2] or values[-2] != threshold):
        end = float(positions[-1])
        events.append({'distance': end, 'interval_end': end, 'kind': 'plateau'})
    return events


def transect_measurement(surface, definition, *, scale, unit, threshold=None):
    distances, centres, grid = definition.position_grid()
    values, valid, _ = bilinear_values(surface.pixels, surface.valid, grid[..., 0], grid[..., 1])
    common = np.count_nonzero(valid, axis=1)
    count = values.shape[1]
    accepted = common >= math.ceil(definition.min_fraction * count - 1e-8)
    signal = np.empty(len(distances), dtype=np.float64)
    signal.fill(np.nan)
    if not np.any(accepted):
        raise InsufficientSupport('The transect has no supported sample stations.')
    samples = []
    for index, (distance, centre, contributors, ok) in enumerate(zip(distances, centres, common, accepted, strict=True)):
        if ok:
            signal[index] = aggregate_profile(values[index], valid[index], definition.aggregate)
        samples.append({
            'x': float(centre[0]),
            'y': float(centre[1]),
            'distance': float(distance),
            'distance_mm': float(distance * scale) if scale else None,
            'value': float(signal[index]) if ok else None,
            'contributors': int(contributors),
            'possible': count,
        })
    crossings = threshold_crossings(distances, signal, accepted, threshold) if threshold is not None else []
    return {
        'definition': definition.as_json(),
        'length': float(distances[-1]),
        'length_mm': float(distances[-1] * scale) if scale else None,
        'units': unit,
        'summary': describe_values(signal[accepted]),
        'threshold': threshold,
        'crossings': crossings,
        'segments': break_intervals(distances, accepted),
        'samples': samples,
    }


def compare_outlines(surface, first, second, *, scale, unit):
    a, _, _ = area_measurement(surface, first, scale=scale, unit=unit)
    b, _, _ = area_measurement(surface, second, scale=scale, unit=unit)
    left, right = a['distribution'], b['distribution']
    return {
        'first': a,
        'second': b,
        'median_offset': right['median'] - left['median'],
        'mean_offset': right['mean'] - left['mean'],
        'relative_support': right['valid_fraction'] - left['valid_fraction'],
        'relative_spread': right['median_absolute_deviation'] - left['median_absolute_deviation'],
    }


def pixel_probe(surface, position, radius, *, unit):
    x, y = position
    x0, y0 = max(x - radius, 0), max(y - radius, 0)
    x1, y1 = min(x + radius + 1, surface.width), min(y + radius + 1, surface.height)
    local = surface.crop((x0, y0, x1 - x0, y1 - y0))
    dx = np.arange(x0, x1, dtype=np.float64) - x
    dy = np.arange(y0, y1, dtype=np.float64) - y
    within = dx[None, :] ** 2 + dy[:, None] ** 2 <= radius ** 2
    usable = within & local.valid
    return {
        'position': [x, y],
        'units': unit,
        'value': float(surface.pixels[y, x]) if surface.valid[y, x] else None,
        'sample_summary': describe_values(local.pixels[usable]),
        'support': support_summary(usable),
        'radius': radius,
    }
