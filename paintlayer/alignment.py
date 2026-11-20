"""Small translation refinement between calibrated plates that share an overlap.

Only the common canvas footprint of a plate pair is compared, on a band-pass
surface that suppresses the smooth detector trend and down-weights unreliable
samples.  Every competing peak is reported instead of one being chosen silently:
repeated weave, occlusions that leave a narrow strip and shifts that run into
the search limit stay visible as positions a person still has to confirm.
"""

from dataclasses import dataclass, replace
from io import BytesIO
from itertools import combinations
import math

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage, signal

from . import fields
from .errors import CapacityExceeded, InsufficientSupport
from .raster import Surface, display_range, radiograph_rgba
from .settings import MAX_ALIGNMENT_PAIRS, MAX_ALIGNMENT_WORK, MAX_STACK_TILES

BAND_INNER = 1.5
BAND_OUTER = 6.0
ERODE_MARGIN = 6
MIN_SAMPLES = 512
MIN_SPAN = 12
PEAK_FLOOR = 0.40
SECOND_RATIO = 0.94
MAX_RESIDUAL = 1.5
MAX_CANDIDATES = 4
SOLVE_ROUNDS = 6
MAX_PAIR_PIXELS = 4_000_000
PREVIEW_EDGE = 1100


def inline_png(payload):
    import base64
    return 'data:image/png;base64,' + base64.b64encode(payload).decode('ascii')

CONFIDENT = 'confident'
PENDING = ('ambiguous', 'narrow_overlap', 'weak_support', 'inconclusive', 'beyond_search')
UNAVAILABLE = ('separated', 'too_large', 'over_budget')
PAIR_NOTES = {
    'confident': 'One shift stands clearly above every alternative.',
    'ambiguous': 'Several shifts score almost equally; a person must choose.',
    'narrow_overlap': 'Too little common area remains to fix the shift.',
    'weak_support': 'Too few usable common samples for a reliable shift.',
    'inconclusive': 'No shift stands above the surrounding correlation.',
    'beyond_search': 'The best shift touches the search limit; widen the radius.',
    'inconsistent': 'This shift disagrees with the other confirmed pairs.',
    'separated': 'The plates share no canvas area to compare.',
    'too_large': 'The common area exceeds the allowance for one pair.',
    'over_budget': 'The alignment work allowance was reached before this pair.',
}
LEVEL_RESOLVED = 1
LEVEL_PENDING = 2
LEVEL_UNRESOLVED = 3
PREVIEW_TINTS = (
    (0, 0, 0, 0),
    (88, 166, 124, 44),
    (228, 178, 84, 72),
    (204, 96, 72, 82),
)
OUTLINE = {
    'current': (188, 194, 182, 110),
    'resolved': (144, 214, 174, 240),
    'unresolved': (226, 126, 100, 235),
    'alternative': (232, 186, 96, 150),
}


@dataclass(frozen=True)
class PlateFeature:
    plate_id: str
    label: str
    geometry: object
    feature: np.ndarray
    usable: np.ndarray
    pixels: np.ndarray
    usable_pixels: int


@dataclass(frozen=True)
class PairReading:
    first: int
    second: int
    status: str
    box: tuple | None
    candidates: tuple
    samples: int
    span: tuple | None
    peak: float | None
    runner_up: float | None
    shift: tuple | None
    solved: bool = False
    residual: float | None = None

    def as_json(self, features):
        first, second = features[self.first], features[self.second]
        return {
            'first': first.plate_id,
            'second': second.plate_id,
            'first_index': self.first + 1,
            'second_index': self.second + 1,
            'first_label': first.label,
            'second_label': second.label,
            'status': self.status,
            'note': PAIR_NOTES[self.status],
            'solved': self.solved,
            'shift': [round(value, 2) for value in self.shift] if self.shift else None,
            'score': self.peak,
            'runner_up': self.runner_up,
            'margin': round(self.peak - self.runner_up, 4) if None not in (self.peak, self.runner_up) else None,
            'samples': self.samples,
            'span': list(self.span) if self.span else None,
            'common': list(self.box) if self.box else None,
            'residual': self.residual,
            'candidates': [
                {'shift': list(shift), 'score': score, 'samples': samples}
                for shift, score, samples in self.candidates
            ],
        }


@dataclass(frozen=True)
class PlateReading:
    index: int
    status: str
    delta: tuple
    current: tuple
    proposed: tuple
    note: str

    def as_json(self):
        return {
            'index': self.index + 1,
            'current': list(self.current),
            'delta': list(self.delta),
            'proposed': list(self.proposed),
            'status': self.status,
            'note': self.note,
        }


@dataclass(frozen=True)
class Alignment:
    extent: tuple
    radius: int
    features: tuple
    pairs: tuple
    plates: tuple
    skipped: tuple
    status_map: np.ndarray
    base: np.ndarray
    marks: tuple

    def report(self):
        return {
            'extent': list(self.extent),
            'radius': self.radius,
            'thresholds': {
                'peak_score': PEAK_FLOOR,
                'second_ratio': SECOND_RATIO,
                'minimum_samples': MIN_SAMPLES,
                'minimum_span_pixels': MIN_SPAN,
                'maximum_residual_pixels': MAX_RESIDUAL,
            },
            'summary': {
                'pairs': len(self.pairs),
                'confident_pairs': sum(pair.status == CONFIDENT for pair in self.pairs),
                'pending_pairs': sum(pair.status in PENDING for pair in self.pairs),
                'unavailable_pairs': sum(pair.status in UNAVAILABLE for pair in self.pairs),
                'plates_moved': sum(row.status == CONFIDENT and row.delta != (0, 0) for row in self.plates),
                'plates_held': sum(row.status not in {CONFIDENT, 'anchored'} for row in self.plates),
                'plates_anchored': sum(row.status == 'anchored' for row in self.plates),
                'largest_shift': max((max(abs(value) for value in row.delta) for row in self.plates), default=0),
                'skipped_plates': list(self.skipped),
            },
            'plates': [
                row.as_json() | {'plate': feature.plate_id, 'label': feature.label}
                for row, feature in zip(self.plates, self.features, strict=True)
            ],
            'pairs': [pair.as_json(self.features) for pair in self.pairs],
            'legend': [
                {'status': 'anchored', 'note': 'Kept where you placed it; the other plates are measured against it.'},
                {'status': 'confident', 'note': 'Every consistent pair agrees on one shift; safe to apply.'},
                {'status': 'uncertain', 'note': 'Competing or insufficient shifts involve this plate; confirm by hand.'},
                {'status': 'unresolved', 'note': 'No confirmed pair connects this plate to a fixed position.'},
                {'status': 'ambiguous', 'note': 'Several shifts fit the common area almost equally.'},
                {'status': 'narrow_overlap', 'note': 'The common area left after masks and erosion is too small.'},
            ],
        }

    def proposals(self):
        return {
            feature.plate_id: {
                'delta': list(reading.delta),
                'proposed': list(reading.proposed),
                'status': reading.status,
            }
            for reading, feature in zip(self.plates, self.features, strict=True)
        }

    def alternatives(self):
        """Candidate translations for a plate whose position is not yet settled.

        A pending pair is only actionable when its counterpart is already fixed,
        because a shift between two loose plates can be split in many ways.
        """
        placed = {CONFIDENT, 'anchored'}
        options = {feature.plate_id: [] for feature in self.features}
        for pair in self.pairs:
            if pair.status not in PENDING or not pair.candidates:
                continue
            first, second = self.features[pair.first], self.features[pair.second]
            first_set = self.plates[pair.first].status in placed
            second_set = self.plates[pair.second].status in placed
            if not (first_set or second_set):
                continue
            first_base = np.asarray(self.plates[pair.first].delta, dtype=np.float64)
            second_base = np.asarray(self.plates[pair.second].delta, dtype=np.float64)
            for shift, score, samples in pair.candidates:
                offset = np.asarray(shift, dtype=np.float64)
                shared = {
                    'score': score,
                    'samples': samples,
                    'pair': [pair.first + 1, pair.second + 1],
                    'pair_labels': [first.label, second.label],
                    'pair_status': pair.status,
                }
                if second_set:
                    options[first.plate_id].append(shared | {
                        'delta': [int(value) for value in np.rint(second_base + offset - first_base)],
                        'basis': f'keeps {second.label} fixed',
                    })
                if first_set:
                    options[second.plate_id].append(shared | {
                        'delta': [int(value) for value in np.rint(first_base - offset - second_base)],
                        'basis': f'keeps {first.label} fixed',
                    })
        for values in options.values():
            values.sort(key=lambda row: row['score'], reverse=True)
            del values[MAX_CANDIDATES:]
        return options


def parse_radius(raw):
    return fields.integer(raw.get('radius', 8), field='radius', minimum=0, maximum=32)


def clipped_box(geometry, shape, extent):
    height, width = shape[:2]
    x0, y0 = max(0, geometry.x), max(0, geometry.y)
    x1, y1 = min(extent[0], geometry.x + width), min(extent[1], geometry.y + height)
    return None if x0 >= x1 or y0 >= y1 else (x0, y0, x1, y1)


def grown_box(box, distance, extent):
    return (
        max(0, box[0] - distance),
        max(0, box[1] - distance),
        min(extent[0], box[2] + distance),
        min(extent[1], box[3] + distance),
    )


def common_box(left, right):
    if left is None or right is None:
        return None
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[2], right[2]), min(left[3], right[3])
    return None if x0 >= x1 or y0 >= y1 else (x0, y0, x1, y1)


def canvas_window(array, geometry, box):
    x0, y0, x1, y1 = box
    height, width = array.shape[:2]
    output = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
    left, top = x0 - geometry.x, y0 - geometry.y
    sx0, sx1 = max(0, left), min(width, left + (x1 - x0))
    sy0, sy1 = max(0, top), min(height, top + (y1 - y0))
    if sx0 < sx1 and sy0 < sy1:
        output[sy0 - top:sy1 - top, sx0 - left:sx1 - left] = array[sy0:sy1, sx0:sx1]
    return output


def shifted(mask, dy, dx):
    output = np.zeros_like(mask)
    height, width = mask.shape
    ys, yd = (slice(0, height - dy), slice(dy, height)) if dy >= 0 else (slice(-dy, height), slice(0, height + dy))
    xs, xd = (slice(0, width - dx), slice(dx, width)) if dx >= 0 else (slice(-dx, width), slice(0, width + dx))
    output[yd, xd] = mask[ys, xs]
    return output


def feature_surface(plate):
    geometry = plate.placement.geometry
    raw = np.asarray(plate.surface.pixels)
    finite = np.isfinite(raw)
    pixels = np.nan_to_num(geometry.orient(raw.astype(np.float32)), nan=0.0, posinf=0.0, neginf=0.0)
    weights = geometry.orient(np.asarray(plate.weights, dtype=np.float32))
    usable = geometry.orient(plate.surface.valid) & finite & (weights > 0)
    eroded = ndimage.binary_erosion(
        usable, structure=np.ones((2 * ERODE_MARGIN + 1, 2 * ERODE_MARGIN + 1), dtype=bool), border_value=0,
    )
    usable = eroded if eroded.any() else usable
    if not usable.any():
        return None
    nearest = ndimage.distance_transform_edt(~usable, return_distances=False, return_indices=True)
    filled = pixels[tuple(nearest)]
    band = np.subtract(
        ndimage.gaussian_filter(filled, BAND_INNER),
        ndimage.gaussian_filter(filled, BAND_OUTER),
    )
    scale = float(np.std(band[usable]))
    if not math.isfinite(scale) or scale <= 1e-7:
        return None
    feature = (band / scale * np.sqrt(np.clip(weights, 0, None))).astype(np.float32)
    feature[~usable] = 0.0
    return PlateFeature(
        plate_id=plate.placement.plate_id,
        label=plate.label,
        geometry=geometry,
        feature=feature,
        usable=usable,
        pixels=pixels,
        usable_pixels=int(np.count_nonzero(usable)),
    )


def correlation_surface(first, second, radius):
    a, am = first
    b, bm = second
    height, width = a.shape
    if min(height, width) <= 2 * radius:
        return None
    flip = lambda values: values[::-1, ::-1]
    surface = signal.fftconvolve(a, flip(b), mode='full')
    counts = signal.fftconvolve(am, flip(bm), mode='full')
    left = signal.fftconvolve(a, flip(bm), mode='full')
    right = signal.fftconvolve(am, flip(b), mode='full')
    left_power = signal.fftconvolve(a * a, flip(bm), mode='full')
    right_power = signal.fftconvolve(am, flip(b * b), mode='full')
    rows = slice(height - 1 - radius, height + radius)
    columns = slice(width - 1 - radius, width + radius)
    surface, counts, left, right, left_power, right_power = (
        value[rows, columns] for value in (surface, counts, left, right, left_power, right_power)
    )
    enough = counts >= max(MIN_SAMPLES, 0.01 * float(counts.max()))
    safe = np.maximum(counts, 1.0)
    centered = surface - left * right / safe
    spread_left = np.maximum(left_power - left * left / safe, 0)
    spread_right = np.maximum(right_power - right * right / safe, 0)
    valid = enough & (spread_left > 1e-6) & (spread_right > 1e-6)
    scores = np.zeros(counts.shape, dtype=np.float32)
    scores[valid] = (centered[valid] / np.sqrt(spread_left[valid] * spread_right[valid])).astype(np.float32)
    return scores, counts, valid


def local_peaks(scores, counts, valid, radius):
    if not valid.any():
        return []
    masked = np.where(valid, scores, -np.inf)
    peaks = masked == ndimage.maximum_filter(masked, size=3)
    rows, columns = np.nonzero(peaks & valid)
    chosen = []
    for position in np.argsort(masked[rows, columns])[::-1]:
        row, column = int(rows[position]), int(columns[position])
        score = float(masked[row, column])
        if not math.isfinite(score) or score <= 0:
            continue
        if any(max(abs(column - other[1]), abs(row - other[0])) <= 3 for other in chosen):
            continue
        chosen.append((row, column, score, int(counts[row, column])))
        if len(chosen) >= MAX_CANDIDATES:
            break
    return chosen


def subpixel(values, index):
    if index <= 0 or index >= len(values) - 1:
        return 0.0
    left, center, right = (float(value) for value in values[index - 1:index + 2])
    curvature = left - 2 * center + right
    if not math.isfinite(curvature) or curvature >= -1e-9:
        return 0.0
    return float(np.clip(0.5 * (left - right) / curvature, -0.5, 0.5))


def footprint_span(left, right, shift):
    common = (left > 0) & shifted(right > 0, shift[1], shift[0])
    if not common.any():
        return None, 0
    rows, columns = np.nonzero(common)
    return (int(columns.max() - columns.min() + 1), int(rows.max() - rows.min() + 1)), int(common.sum())


def classify(candidates, radius, span, samples):
    shift, score, _ = candidates[0][:3]
    if score < PEAK_FLOOR:
        return 'inconclusive', None
    if max(abs(shift[0]), abs(shift[1])) >= radius > 0:
        return 'beyond_search', None
    if samples < MIN_SAMPLES or span is None or min(span) < MIN_SPAN:
        return 'narrow_overlap', None
    runner = competitors(candidates, radius)
    if runner is not None and runner >= score * SECOND_RATIO:
        return 'ambiguous', runner
    return CONFIDENT, runner


def competitors(candidates, radius):
    """Best score clearly away from the winning peak (at least five samples apart)."""
    (shift, score, _), *rest = candidates
    far = [
        found for other, found, _ in rest
        if max(abs(other[0] - shift[0]), abs(other[1] - shift[1])) > 5
    ]
    if far:
        return max(far)
    rim = [
        found for other, found, _ in rest
        if max(abs(other[0]), abs(other[1])) >= radius
    ]
    return max(rim) if rim else None


def compare_pair(first, second, extent, radius):
    """Compare two plates and report every competing shift inside the search radius."""
    shared = {'first': first[0], 'second': second[0]}
    first, second = first[1], second[1]
    empty = dict(candidates=(), samples=0, span=None, peak=None, runner_up=None, shift=None)
    boxes = [clipped_box(item.geometry, item.feature.shape, extent) for item in (first, second)]
    box = common_box(*boxes)
    if box is None:
        return PairReading(status='separated', box=None, **empty, **shared)
    window = common_box(grown_box(boxes[0], radius, extent), grown_box(boxes[1], radius, extent))
    if (window[2] - window[0]) * (window[3] - window[1]) > MAX_PAIR_PIXELS:
        return PairReading(status='too_large', box=box, **empty, **shared)
    windows = [
        (canvas_window(item.feature, item.geometry, window), canvas_window(item.usable.astype(np.float32), item.geometry, window))
        for item in (first, second)
    ]
    scored = correlation_surface(windows[0], windows[1], radius)
    if scored is None:
        return PairReading(status='narrow_overlap', box=box, **empty, **shared)
    scores, counts, valid = scored
    peaks = local_peaks(scores, counts, valid, radius)
    if not peaks:
        return PairReading(status='weak_support', box=box, **empty, **shared)
    row, column, score, _ = peaks[0]
    precise = (
        peaks[0][1] - radius + subpixel(scores[row, :], column),
        peaks[0][0] - radius + subpixel(scores[:, column], row),
    )
    rounded = (int(round(precise[0])), int(round(precise[1])))
    span, samples = footprint_span(windows[0][1], windows[1][1], rounded)
    keyed = tuple(
        ((item[1] - radius, item[0] - radius), float(item[2]), int(item[3])) for item in peaks
    )
    status, runner = classify(keyed, radius, span, samples)
    return PairReading(
        **shared,
        status=status,
        box=box,
        candidates=keyed,
        samples=samples,
        span=span,
        peak=round(score, 5),
        runner_up=round(runner, 5) if runner is not None else None,
        shift=precise,
    )


def solve(features, pairs):
    """Fit one translation per plate, rejecting pairs that disagree with the rest."""
    accepted = [pair for pair in pairs if pair.status == CONFIDENT]
    diverged = set()
    translations, solved, anchored = solve_once(features, accepted)
    for _ in range(SOLVE_ROUNDS):
        worst = None
        for pair in accepted:
            if pair in diverged or not (solved[pair.first] and solved[pair.second]):
                continue
            error = math.dist(translations[pair.second], translations[pair.first] + np.asarray(pair.shift))
            if error > MAX_RESIDUAL and (worst is None or error > worst[0]):
                worst = (error, pair)
        if worst is None:
            break
        diverged.add(worst[1])
        translations, solved, anchored = solve_once(features, [pair for pair in accepted if pair not in diverged])
    final = [pair for pair in accepted if pair not in diverged]
    residuals = {}
    for pair in final:
        if solved[pair.first] and solved[pair.second]:
            error = math.dist(translations[pair.second], translations[pair.first] + np.asarray(pair.shift))
            residuals[(pair.first, pair.second)] = round(error, 3)
    return translations, solved, anchored, finish(pairs, residuals)


def finish(pairs, residuals):
    readings = []
    for pair in pairs:
        if pair.status != CONFIDENT:
            readings.append(pair)
        elif (pair.first, pair.second) in residuals:
            readings.append(replace(pair, solved=True, residual=residuals[(pair.first, pair.second)]))
        else:
            readings.append(replace(pair, status='inconsistent', solved=False))
    return tuple(readings)


def solve_once(features, accepted):
    """Least-squares translation per plate with the group kept near its input positions.

    One reference plate fixes the scale of the solution, then the whole group is
    slid by the median translation so the arrangement that needs the fewest
    changes wins.  A group-wide slide cannot be measured from overlaps alone,
    so it is never invented.
    """
    count = len(features)
    translations = np.zeros((count, 2))
    solved = np.zeros(count, dtype=bool)
    anchored = np.zeros(count, dtype=bool)
    for group in components(count, accepted):
        if len(group) < 2:
            continue
        reference = max(group, key=lambda index: (features[index].usable_pixels, -index))
        unknown = [index for index in group if index != reference]
        slot = {index: position for position, index in enumerate(unknown)}
        edges = [pair for pair in accepted if pair.first in group and pair.second in group]
        design = np.zeros((len(edges) * 2, len(unknown) * 2))
        target = np.zeros(len(edges) * 2)
        weight = np.zeros(len(edges) * 2)
        for row, pair in enumerate(edges):
            weight[row * 2:row * 2 + 2] = math.sqrt(max(1, pair.samples) * max(0.05, pair.peak or 0))
            for axis in (0, 1):
                line = row * 2 + axis
                if pair.second != reference:
                    design[line, slot[pair.second] * 2 + axis] = 1
                if pair.first != reference:
                    design[line, slot[pair.first] * 2 + axis] = -1
                target[line] = pair.shift[axis]
        solution = np.linalg.lstsq(design * weight[:, None], target * weight, rcond=None)[0]
        values = np.zeros((len(group), 2))
        for position, index in enumerate(group):
            if index != reference:
                values[position] = solution[slot[index] * 2:slot[index] * 2 + 2]
        if len(group) > 2:
            # With three or more plates the most common correction is the one the
            # arrangement supports; the rest were the plates actually misplaced.
            values -= np.rint(np.median(values, axis=0))
        for position, index in enumerate(group):
            translations[index] = values[position]
            solved[index] = True
        still = [index for index in group if np.all(np.rint(translations[index]) == 0)]
        anchored[max(still or group, key=lambda index: (features[index].usable_pixels, -index))] = True
    return translations, solved, anchored


def components(count, edges):
    parent = list(range(count))

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for pair in edges:
        left, right = find(pair.first), find(pair.second)
        if left != right:
            parent[right] = left
    groups = {}
    for index in range(count):
        groups.setdefault(find(index), []).append(index)
    return list(groups.values())


def plate_readings(features, pairs, translations, solved, anchored, extent):
    readings = []
    for index, feature in enumerate(features):
        current = (feature.geometry.x, feature.geometry.y)
        if anchored[index]:
            readings.append(PlateReading(
                index, 'anchored', (0, 0), current, current,
                'Kept where you placed it; the other plates in its group are measured against it.',
            ))
        elif solved[index]:
            delta = tuple(int(round(value)) for value in translations[index])
            proposed = current[0] + delta[0], current[1] + delta[1]
            if not (-4096 <= proposed[0] <= extent[0] - 1 and -4096 <= proposed[1] <= extent[1] - 1):
                readings.append(PlateReading(
                    index, 'unresolved', (0, 0), current, current,
                    'The confirmed shift would move this plate outside the placement range.',
                ))
                continue
            note = (
                'The confirmed pairs place this plate where it already is.'
                if delta == (0, 0) else 'Every confirmed pair agrees on this translation.'
            )
            readings.append(PlateReading(index, CONFIDENT, delta, current, proposed, note))
        elif any(
            pair.status in PENDING for pair in pairs if index in (pair.first, pair.second)
        ):
            readings.append(PlateReading(
                index, 'uncertain', (0, 0), current, current,
                'Competing or insufficient shifts involve this plate; choose a position by hand.',
            ))
        else:
            readings.append(PlateReading(
                index, 'unresolved', (0, 0), current, current,
                'No confirmed pair connects this plate to a fixed position.',
            ))
    return tuple(readings)


def status_map(extent, features, readings, pairs):
    layer = np.zeros((extent[1], extent[0]), dtype=np.uint8)
    settled = {CONFIDENT, 'anchored'}
    for pair in pairs:
        if pair.status in UNAVAILABLE or pair.box is None:
            continue
        x0, y0, x1, y1 = pair.box
        unresolved = readings[pair.first].status not in settled or readings[pair.second].status not in settled
        level = LEVEL_UNRESOLVED if unresolved else (LEVEL_RESOLVED if pair.solved else LEVEL_PENDING)
        layer[y0:y1, x0:x1] = np.maximum(layer[y0:y1, x0:x1], level)
    for reading, feature in zip(readings, features, strict=True):
        if reading.status in settled:
            continue
        box = clipped_box(feature.geometry, feature.feature.shape, extent)
        if box is None:
            continue
        x0, y0, x1, y1 = box
        layer[y0:y1, x0:x1] = np.maximum(layer[y0:y1, x0:x1], LEVEL_UNRESOLVED)
    return layer


def placed_box(reading, feature, extent):
    """Footprint a plate would occupy after its proposed shift is applied."""
    return offset_box(feature.geometry, feature.feature.shape, placed_offset(reading), extent)


def placed_offset(reading):
    return (reading.proposed if reading.status in {CONFIDENT, 'anchored'} else reading.current)


def offset_box(geometry, shape, offset, extent):
    height, width = shape[:2]
    x0, y0 = max(0, offset[0]), max(0, offset[1])
    x1, y1 = min(extent[0], offset[0] + width), min(extent[1], offset[1] + height)
    return (x0, y0, x1, y1)


def preview_base(features, readings, extent):
    """Composite as it would look with every confirmed shift applied."""
    width, height = extent
    total = np.zeros((height, width), dtype=np.float32)
    weight = np.zeros((height, width), dtype=np.float32)
    for reading, feature in zip(readings, features, strict=True):
        x, y = placed_offset(reading)
        left, top = x - feature.geometry.x, y - feature.geometry.y
        shape = feature.pixels.shape
        sx0, sx1 = max(0, -left), min(shape[1], extent[0] - x)
        sy0, sy1 = max(0, -top), min(shape[0], extent[1] - y)
        if sx0 >= sx1 or sy0 >= sy1:
            continue
        canvas = (slice(y + sy0, y + sy1), slice(x + sx0, x + sx1))
        usable = feature.usable[sy0:sy1, sx0:sx1]
        total[canvas] += np.where(usable, feature.pixels[sy0:sy1, sx0:sx1], 0.0)
        weight[canvas] += usable
    covered = weight > 0
    signal = np.zeros((height, width), dtype=np.float32)
    signal[covered] = total[covered] / weight[covered]
    return radiograph_rgba(Surface(signal, covered), limits=display_range(signal, covered))


def preview_marks(features, readings, extent):
    marks = []
    for reading, feature in zip(readings, features, strict=True):
        shape = feature.feature.shape
        current = clipped_box(feature.geometry, shape, extent)
        if current is not None:
            marks.append({'box': current, 'color': OUTLINE['current'], 'width': 1, 'label': str(reading.index + 1)})
        target = placed_box(reading, feature, extent)
        if target[0] >= target[2] or target[1] >= target[3]:
            continue
        if reading.status in {CONFIDENT, 'anchored'}:
            marks.append({'box': target, 'color': OUTLINE['resolved'], 'width': 2})
        else:
            marks.append({'box': target, 'color': OUTLINE['unresolved'], 'width': 2})
    return tuple(marks)


def alternative_marks(features, readings, options, extent, limit=8):
    marks = []
    settled = {CONFIDENT, 'anchored'}
    for reading, feature in zip(readings, features, strict=True):
        if reading.status in settled or not options.get(feature.plate_id):
            continue
        for option in options[feature.plate_id]:
            offset = (feature.geometry.x + option['delta'][0], feature.geometry.y + option['delta'][1])
            box = offset_box(feature.geometry, feature.feature.shape, offset, extent)
            if box[0] >= box[2] or box[1] >= box[3]:
                continue
            marks.append({'box': box, 'color': OUTLINE['alternative'], 'width': 1})
            if len(marks) >= limit:
                return tuple(marks)
    return tuple(marks)


def render_preview(alignment):
    image = Image.fromarray(alignment.base)
    if max(alignment.extent) > PREVIEW_EDGE:
        scale = PREVIEW_EDGE / max(alignment.extent)
        image = image.resize(
            (max(1, round(alignment.extent[0] * scale)), max(1, round(alignment.extent[1] * scale))),
            Image.Resampling.LANCZOS,
        )
    else:
        scale = 1.0
    tint = Image.fromarray(np.asarray(PREVIEW_TINTS, dtype=np.uint8)[np.minimum(alignment.status_map, 3)])
    if tint.size != image.size:
        tint = tint.resize(image.size, Image.Resampling.NEAREST)
    image = Image.alpha_composite(image, tint)
    draw = ImageDraw.Draw(image, 'RGBA')
    for mark in alignment.marks:
        x, y, width, height = mark['box']
        draw.rectangle(
            (x * scale, y * scale, (x + width) * scale - 1, (y + height) * scale - 1),
            outline=mark['color'], width=mark['width'],
        )
        if mark.get('label'):
            draw.rectangle((x * scale + 2, y * scale + 2, x * scale + 17, y * scale + 17), fill=(20, 26, 22, 215))
            draw.text((x * scale + 6, y * scale + 4), mark['label'], fill=(242, 238, 224, 255))
    draw.rectangle((0, 0, image.size[0] - 1, image.size[1] - 1), outline=(82, 90, 80, 190), width=1)
    with BytesIO() as output:
        image.save(output, 'PNG', compress_level=4)
        return output.getvalue()


def measure(plates, extent, radius):
    if len(plates) < 2:
        raise InsufficientSupport('Choose at least two calibrated plates to compare.')
    if len(plates) > MAX_STACK_TILES:
        raise CapacityExceeded('Too many plates for one alignment.')
    if len(plates) * (len(plates) - 1) // 2 > MAX_ALIGNMENT_PAIRS:
        raise CapacityExceeded('Too many plate pairs for one alignment.')
    features = []
    skipped = []
    for plate in plates:
        feature = feature_surface(plate)
        if feature is None:
            skipped.append(plate.label)
        else:
            features.append(feature)
    if len(features) < 2:
        raise InsufficientSupport(
            'At least two calibrated plates need a usable interior to compare.',
            facts={'skipped': skipped},
        )
    pairs = []
    budget = 0
    indexed = list(enumerate(features))
    for (first, feature), (second, other) in combinations(indexed, 2):
        window = common_box(
            grown_box(clipped_box(feature.geometry, feature.feature.shape, extent), radius, extent),
            grown_box(clipped_box(other.geometry, other.feature.shape, extent), radius, extent),
        )
        estimate = (window[2] - window[0]) * (window[3] - window[1]) if window else 0
        if budget + estimate > MAX_ALIGNMENT_WORK:
            pairs.append(PairReading(first, second, 'over_budget', None, (), 0, None, None, None, None))
            continue
        budget += estimate
        pairs.append(compare_pair((first, feature), (second, other), extent, radius))
    translations, solved, anchored, graded = solve(features, pairs)
    readings = plate_readings(features, graded, translations, solved, anchored, extent)
    result = Alignment(
        extent=extent,
        radius=radius,
        features=tuple(features),
        pairs=graded,
        plates=readings,
        skipped=tuple(skipped),
        status_map=status_map(extent, features, readings, graded),
        base=preview_base(features, readings, extent),
        marks=(),
    )
    alternatives = alternative_marks(features, readings, result.alternatives(), extent)
    marks = preview_marks(features, readings, extent) + alternatives
    return replace(result, marks=marks), skipped
