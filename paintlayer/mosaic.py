"""Confidence-weighted plate assembly with observable acquisition coverage."""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from . import fields
from .errors import CapacityExceeded, InsufficientSupport, InvalidGeometry
from .geometry import TileGeometry
from .raster import Surface, display_range, histogram, support_summary
from .settings import MAX_PLATE_INPUT_PIXELS, MAX_STACK_TILES


@dataclass(frozen=True)
class Placement:
    plate_id: str
    geometry: TileGeometry
    emphasis: float

    @classmethod
    def parse_many(cls, payload, extent):
        raw = fields.sequence(payload, field='placements', minimum=1, maximum=MAX_STACK_TILES)
        parsed = []
        for index, entry in enumerate(raw):
            field = f'placements[{index}]'
            entry = fields.request_object(entry, field=field)
            parsed.append(cls(
                plate_id=fields.item_id(entry.get('plate'), field=f'{field}.plate'),
                geometry=TileGeometry.from_payload(entry, field=field, extent=extent),
                emphasis=fields.scalar(entry.get('emphasis', 1), field=f'{field}.emphasis', minimum=0.05, maximum=4),
            ))
        fields.unique([entry.plate_id for entry in parsed], field='placements')
        return parsed

    def as_json(self):
        return {'plate': self.plate_id, 'emphasis': self.emphasis} | self.geometry.to_json()


@dataclass(frozen=True)
class PlateData:
    placement: Placement
    surface: Surface
    weights: np.ndarray
    reason_flags: np.ndarray
    label: str


@dataclass(frozen=True)
class Assembly:
    surface: Surface
    coverage: np.ndarray
    owner: np.ndarray
    spread: np.ndarray
    quality: np.ndarray
    summary: dict


def distance_weights(valid, feather):
    if feather == 0:
        return valid.astype(np.float32)
    inset = np.pad(valid, 1, constant_values=False)
    distance = ndimage.distance_transform_edt(inset)[1:-1, 1:-1]
    weights = (distance / feather).astype(np.float32)
    np.clip(weights, 0, 1, out=weights)
    return weights


def coverage_facts(coverage):
    count = np.bincount(coverage.ravel())
    return {
        'uncovered': int(count[0]),
        'single_tile': int(count[1]) if len(count) > 1 else 0,
        'multi_tile': int(count[2:].sum()),
        'maximum': int(len(count) - 1),
        'histogram': {str(index): int(population) for index, population in enumerate(count)},
    }


def candidate_slices(plate, extent):
    geometry = plate.placement.geometry
    oriented = geometry.orient(plate.surface.valid)
    windows = geometry.overlap(oriented, extent)
    if windows is None:
        raise InvalidGeometry(
            f'Plate {plate.label} is wholly outside the examination canvas.',
            field='placements',
            facts={'plate': plate.placement.plate_id, 'extent': list(extent)},
        )
    return oriented, windows


class Accumulator:
    def __init__(self, extent):
        self.extent = extent
        shape = extent[1], extent[0]
        self.total = np.zeros(shape, dtype=np.float64)
        self.total_weight = np.zeros(shape, dtype=np.float64)
        self.low = np.full(shape, np.inf, dtype=np.float32)
        self.high = np.full(shape, -np.inf, dtype=np.float32)
        self.best = np.zeros(shape, dtype=np.float32)
        self.quality = np.zeros(shape, dtype=np.uint8)
        self.coverage = np.zeros(shape, dtype=np.uint8)
        self.owner = np.zeros(shape, dtype=np.uint8)
        self.tiles = []

    def add(self, plate, index, feather):
        oriented, (source, target) = candidate_slices(plate, self.extent)
        distance = distance_weights(oriented, feather)[source]
        valid = oriented[source]
        geometry = plate.placement.geometry
        weights = geometry.orient(plate.weights)[source] * distance * plate.placement.emphasis
        signal = geometry.orient(plate.surface.pixels)[source]
        reasons = geometry.orient(plate.reason_flags)[source]
        contributors = valid & (weights > 0) & np.isfinite(signal)
        usable = int(np.count_nonzero(contributors))
        if not usable:
            raise InsufficientSupport(f'No usable pixels from {plate.label} overlap the examination canvas.')
        self.total[target] += np.where(contributors, signal, 0.0) * weights
        self.total_weight[target] += np.where(contributors, weights, 0.0)
        self.coverage[target] += contributors
        self.quality[target] |= reasons
        self.clip_spread(target, signal, contributors)
        better = contributors & (weights > self.best[target])
        self.best[target][better] = weights[better]
        self.owner[target][better] = index
        self.tiles.append({
            'plate': plate.placement.plate_id,
            'label': plate.label,
            'index': index,
            'contributing_pixels': usable,
            'outside_canvas_pixels': int(np.count_nonzero(oriented)) - int(np.count_nonzero(valid)),
            'position': plate.placement.as_json(),
            'support_inside': support_summary(contributors),
        })

    def clip_spread(self, target, signal, usable):
        current_min = self.low[target]
        current_max = self.high[target]
        np.minimum(current_min, signal, out=current_min, where=usable)
        np.maximum(current_max, signal, out=current_max, where=usable)

    def finalize(self, unit, feather):
        supported = self.total_weight > 0
        signal = np.zeros(supported.shape, dtype=np.float32)
        signal[supported] = (self.total[supported] / self.total_weight[supported]).astype(np.float32)
        overlap = self.coverage > 1
        spread = np.zeros(supported.shape, dtype=np.float32)
        spread[overlap] = self.high[overlap] - self.low[overlap]
        limits = display_range(signal, supported)
        owner_pop = np.bincount(self.owner.ravel(), minlength=len(self.tiles) + 1)
        for layer in self.tiles:
            layer['dominant_pixels'] = int(owner_pop[layer['index']])
        summary = {
            'unit': unit,
            'display': list(limits),
            'extent': list(self.extent),
            'blend_radius': feather,
            'tiles': self.tiles,
            'support': support_summary(supported),
            'coverage': coverage_facts(self.coverage),
            'distribution': histogram(signal, supported, limits),
            'median_overlap_spread': float(np.median(spread[overlap])) if np.any(overlap) else None,
            'peak_overlap_spread': float(np.max(spread[overlap])) if np.any(overlap) else None,
        }
        return Assembly(Surface(signal, supported), self.coverage, self.owner, spread, self.quality, summary)


def assemble(plates, extent, radius, unit):
    loaded_pixels = sum(plate.surface.valid.size for plate in plates)
    if loaded_pixels > MAX_PLATE_INPUT_PIXELS:
        raise CapacityExceeded('These plates exceed the total input pixel allowance for one assembly.')
    accumulator = Accumulator(extent)
    for index, plate in enumerate(plates, start=1):
        accumulator.add(plate, index, radius)
    return accumulator.finalize(unit, radius)


def canvas_tile_probe(assembly, position, tile_labels):
    x, y = position
    has_data = bool(assembly.surface.valid[y, x])
    index = int(assembly.owner[y, x])
    return {
        'position': [x, y],
        'valid': has_data,
        'value': float(assembly.surface.pixels[y, x]) if has_data else None,
        'coverage': int(assembly.coverage[y, x]),
        'owner': tile_labels.get(index),
        'spread': float(assembly.spread[y, x]) if assembly.coverage[y, x] > 1 else None,
        'reason_bits': int(assembly.quality[y, x]),
    }
