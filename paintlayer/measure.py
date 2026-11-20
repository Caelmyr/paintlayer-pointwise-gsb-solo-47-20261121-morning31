"""Reproducible detector measurements over outlined areas and canvas transects."""

from dataclasses import dataclass
import math

import numpy as np

from . import fields
from .errors import ExamError, InsufficientSupport, InvalidGeometry
from .geometry import bilinear_values, outline_pixels
from .raster import histogram, support_summary
from .settings import MAX_PROFILE_SAMPLES

PERCENTILES = (5, 25, 50, 75, 95)
BAND_SIDES = {'above', 'below'}
BAND_IN, BAND_OUT, BAND_UNKNOWN = 1, 0, -1


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


@dataclass(frozen=True)
class BandRequest:
    side: str
    baseline_window: float
    min_width: float
    bridge_gap: float

    @classmethod
    def parse(cls, raw):
        window = raw.get('baseline_window')
        min_width = raw.get('min_width')
        bridge_gap = raw.get('bridge_gap')
        return cls(
            side=fields.choice(raw.get('band_side'), field='band_side', options=BAND_SIDES),
            baseline_window=fields.scalar(window if window is not None else 0, field='baseline_window', minimum=0, maximum=1000),
            min_width=fields.scalar(min_width if min_width is not None else 2, field='min_width', minimum=0, maximum=200),
            bridge_gap=fields.scalar(bridge_gap if bridge_gap is not None else 2, field='bridge_gap', minimum=0, maximum=100),
        )


def moving_baseline(signal, accepted, window_stations):
    count = signal.size
    half = window_stations // 2
    values = np.where(accepted, signal, np.nan)
    padded = np.pad(values, half, constant_values=np.nan)
    windows = np.lib.stride_tricks.sliding_window_view(padded, window_stations)
    minimum = max(3, window_stations // 3)
    baseline = np.full(count, np.nan)
    for start in range(0, count, 256):
        chunk = windows[start:start + 256]
        enough = np.count_nonzero(np.isfinite(chunk), axis=1) >= minimum
        medians = np.full(chunk.shape[0], np.nan)
        if np.any(enough):
            medians[enough] = np.nanmedian(chunk[enough], axis=1)
        baseline[start:start + chunk.shape[0]] = medians
    # Stations closer than half a window to either end use the largest centred window instead.
    for station in list(range(min(half, count))) + list(range(max(count - half, 0), count)):
        shrink = min(half, station, count - 1 - station)
        size = 2 * shrink + 1
        if size < 5:
            baseline[station] = np.nan
            continue
        local = values[station - shrink:station + shrink + 1]
        local = local[np.isfinite(local)]
        baseline[station] = float(np.median(local)) if local.size >= max(3, size // 3) else np.nan
    return baseline


def classify_bands(corrected, classifiable, threshold, side):
    states = np.full(corrected.shape, BAND_UNKNOWN, dtype=np.int8)
    inside = corrected > threshold if side == 'above' else corrected < threshold
    states[classifiable] = BAND_OUT
    states[classifiable & inside] = BAND_IN
    return states


def maximal_runs(states, target):
    runs = []
    index = 0
    while index < len(states):
        if states[index] == target:
            first = index
            while index + 1 < len(states) and states[index + 1] == target:
                index += 1
            runs.append((first, index))
        index += 1
    return runs


def clean_band_states(states, positions, spacing, bridge_gap, min_width):
    cleaned = states.copy()
    last_index = len(states) - 1
    if min_width > 0:
        for first, last in maximal_runs(cleaned, BAND_IN):
            if first == 0 or last == last_index:
                continue
            if cleaned[first - 1] == BAND_OUT and cleaned[last + 1] == BAND_OUT:
                if positions[last] - positions[first] + spacing < min_width - 1e-9:
                    cleaned[first:last + 1] = BAND_OUT
    if bridge_gap > 0:
        for first, last in maximal_runs(cleaned, BAND_OUT):
            if first == 0 or last == last_index:
                continue
            if cleaned[first - 1] == BAND_IN and cleaned[last + 1] == BAND_IN:
                if positions[last] - positions[first] + spacing <= bridge_gap + 1e-9:
                    cleaned[first:last + 1] = BAND_IN
    return cleaned


def interpolate_crossing(positions, values, threshold, outer, inner):
    a, b = float(values[outer]), float(values[inner])
    fraction = 0.5 if b == a else min(1.0, max(0.0, (threshold - a) / (b - a)))
    return float(positions[outer] + fraction * (positions[inner] - positions[outer]))


def locate_edge(states, positions, corrected, threshold, boundary, direction):
    neighbour = boundary + direction
    if 0 <= neighbour < len(states) and states[neighbour] == BAND_OUT:
        outer, inner = (neighbour, boundary) if direction < 0 else (boundary, neighbour)
        point = interpolate_crossing(positions, corrected, threshold, outer, inner)
        return 'resolved', point, point, False
    scan = neighbour
    while 0 <= scan < len(states) and states[scan] == BAND_UNKNOWN:
        scan += direction
    may_continue = 0 <= scan < len(states) and states[scan] == BAND_IN
    if direction < 0:
        if scan >= 0:
            return 'gap', float(positions[scan]), float(positions[boundary]), may_continue
        return 'open', None, float(positions[boundary]), False
    if scan < len(states):
        return 'gap', float(positions[boundary]), float(positions[scan]), may_continue
    return 'open', float(positions[boundary]), None, False


def edge_report(edge, scale):
    kind, lower, upper, may_continue = edge
    if kind == 'resolved':
        position = lower
    elif lower is not None and upper is not None:
        position = (lower + upper) / 2
    else:
        position = None
    return {
        'kind': kind,
        'position': position,
        'lower': lower,
        'upper': upper,
        'position_mm': position * scale if scale and position is not None else None,
        'lower_mm': lower * scale if scale and lower is not None else None,
        'upper_mm': upper * scale if scale and upper is not None else None,
        'may_continue': may_continue,
    }


def band_items(states, positions, corrected, threshold, scale):
    items = []
    for first, last in maximal_runs(states, BAND_IN):
        start = edge_report(locate_edge(states, positions, corrected, threshold, first, -1), scale)
        end = edge_report(locate_edge(states, positions, corrected, threshold, last, 1), scale)
        width_min = max(0.0, end['lower'] - start['upper'])
        if start['lower'] is not None and end['upper'] is not None:
            width_max = max(width_min, end['upper'] - start['lower'])
            width = (width_min + width_max) / 2
        else:
            width_max = None
            width = None
        items.append({
            'index': len(items) + 1,
            'stations': [int(first), int(last)],
            'span': [float(positions[first]), float(positions[last])],
            'span_mm': [float(positions[first] * scale), float(positions[last] * scale)] if scale else None,
            'start': start,
            'end': end,
            'width': width,
            'width_min': width_min,
            'width_max': width_max,
            'width_mm': width * scale if scale and width is not None else None,
            'width_min_mm': width_min * scale if scale else None,
            'width_max_mm': width_max * scale if scale and width_max is not None else None,
            'resolved': start['kind'] == 'resolved' and end['kind'] == 'resolved',
        })
    return items


def measure_bands(distances, signal, accepted, threshold, request, scale, unit):
    spacing = float(distances[1] - distances[0]) if len(distances) > 1 else 1.0
    effective = signal
    classifiable = accepted
    baseline = None
    if request.baseline_window > 0:
        window_stations = int(round(request.baseline_window / spacing))
        if window_stations < 5:
            raise ExamError(
                f'Use a background window of at least {5 * spacing:.3g} px for this sample interval.',
                field='baseline_window',
            )
        window_stations += 1 - window_stations % 2
        baseline = moving_baseline(signal, accepted, window_stations)
        residual = signal - baseline
        classifiable = accepted & np.isfinite(residual)
        states = classify_bands(residual, classifiable, threshold, request.side)
        excluded = np.zeros(signal.shape, dtype=bool)
        for first, last in maximal_runs(states, BAND_IN):
            if distances[last] - distances[first] + spacing >= request.min_width - 1e-9:
                excluded[first:last + 1] = True
        if np.any(excluded):
            baseline = moving_baseline(signal, accepted & ~excluded, window_stations)
        effective = signal - baseline
        classifiable = accepted & np.isfinite(baseline)
    states = classify_bands(effective, classifiable, threshold, request.side)
    states = clean_band_states(states, distances, spacing, request.bridge_gap, request.min_width)
    report = {
        'side': request.side,
        'threshold': threshold,
        'baseline_window': request.baseline_window or None,
        'min_width': request.min_width,
        'bridge_gap': request.bridge_gap,
        'units': unit,
        'items': band_items(states, distances, effective, threshold, scale),
    }
    return report, effective, classifiable, baseline


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


def transect_measurement(surface, definition, *, scale, unit, threshold=None, band_request=None):
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
    effective, classifiable, baseline, bands = signal, accepted, None, None
    if band_request is not None and threshold is not None:
        bands, effective, classifiable, baseline = measure_bands(
            distances, signal, accepted, threshold, band_request, scale, unit,
        )
    crossings = threshold_crossings(distances, effective, classifiable, threshold) if threshold is not None else []
    return {
        'definition': definition.as_json(),
        'length': float(distances[-1]),
        'length_mm': float(distances[-1] * scale) if scale else None,
        'units': unit,
        'summary': describe_values(signal[accepted]),
        'threshold': threshold,
        'crossings': crossings,
        'baseline': [float(value) if np.isfinite(value) else None for value in baseline] if baseline is not None else None,
        'bands': bands,
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
