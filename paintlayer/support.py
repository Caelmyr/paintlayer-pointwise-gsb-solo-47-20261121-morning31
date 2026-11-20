"""Inspect possible long stretcher and crossbar bands on the calibrated surface."""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage, signal

from . import fields
from .errors import InsufficientSupport
from .raster import Surface, support_summary
from .geometry import outline_pixels


@dataclass(frozen=True)
class BandSearch:
    axis: str
    width: float
    continuity: float
    minimum_strength: float
    max_bands: int

    @classmethod
    def parse(cls, payload):
        payload = fields.request_object(payload, field='support')
        return cls(
            axis=fields.choice(payload.get('axis', 'both'), field='axis', options={'horizontal', 'vertical', 'both'}),
            width=fields.scalar(payload.get('width', 12), field='width', minimum=3, maximum=90),
            continuity=fields.scalar(payload.get('continuity', 0.45), field='continuity', minimum=0.05, maximum=0.95),
            minimum_strength=fields.scalar(payload.get('minimum_strength', 0.045), field='minimum_strength', minimum=0.0001, maximum=2),
            max_bands=fields.integer(payload.get('max_bands', 8), field='max_bands', minimum=1, maximum=25),
        )

    def as_json(self):
        return {
            'axis': self.axis,
            'width': self.width,
            'continuity': self.continuity,
            'minimum_strength': self.minimum_strength,
            'max_bands': self.max_bands,
        }


def average_along(samples, support, width, axis):
    filled = np.where(support, samples, 0).astype(np.float32)
    count = support.astype(np.float32)
    pixels = max(3, int(np.rint(width)))
    values = ndimage.uniform_filter1d(filled, size=pixels, axis=axis, mode='constant')
    weights = ndimage.uniform_filter1d(count, size=pixels, axis=axis, mode='constant')
    valid = weights > 0.67
    response = np.zeros_like(values)
    np.divide(values, weights, out=response, where=valid)
    return response, valid


def offset_values(means, usable, offset, axis):
    shift = [0, 0]
    shift[axis] = offset
    shifted = ndimage.shift(means, shift=shift, order=0, mode='constant', cval=0)
    valid = ndimage.shift(usable, shift=shift, order=0, mode='constant', cval=False)
    return shifted, valid


def response_field(surface, domain, width, axis):
    usable = surface.valid & domain
    interior, interior_valid = average_along(surface.pixels, usable, width * 0.62, axis)
    shoulder, shoulder_valid = average_along(surface.pixels, usable, width * 0.45, axis)
    delta = max(2, int(np.rint(width * 0.68)))
    near, near_valid = offset_values(shoulder, shoulder_valid, delta, axis)
    far, far_valid = offset_values(shoulder, shoulder_valid, -delta, axis)
    valid = usable & interior_valid & near_valid & far_valid
    response = np.zeros(usable.shape, dtype=np.float32)
    response[valid] = interior[valid] - 0.5 * (near[valid] + far[valid])
    return response, valid


def inspect_projection(response, valid, *, axis, settings):
    longitudinal_axis = 1 - axis
    supported = np.sum(valid, axis=longitudinal_axis)
    minimum_span = max(4, int(response.shape[longitudinal_axis] * settings.continuity))
    ok = supported >= minimum_span
    positive = valid & (response >= settings.minimum_strength)
    negative = valid & (response <= -settings.minimum_strength)
    hits_bright = np.sum(positive, axis=longitudinal_axis)
    hits_dark = np.sum(negative, axis=longitudinal_axis)
    strength_bright = np.divide(hits_bright, supported, out=np.zeros_like(supported, dtype=float), where=ok)
    strength_dark = np.divide(hits_dark, supported, out=np.zeros_like(supported, dtype=float), where=ok)
    distance = max(2, int(settings.width * 0.75))
    hypotheses = []
    for polarity, fractions, hits, mask in (
        ('denser', strength_bright, hits_bright, positive),
        ('less_dense', strength_dark, hits_dark, negative),
    ):
        peaks, _ = signal.find_peaks(np.pad(fractions, 1), height=settings.continuity, distance=distance)
        for peak in peaks - 1:
            if not 0 <= peak < len(fractions) or not ok[peak]:
                continue
            if axis == 0:
                hit_mask = mask[peak]
                observations = response[peak, hit_mask]
            else:
                hit_mask = mask[:, peak]
                observations = response[hit_mask, peak]
            extent = np.flatnonzero(hit_mask)
            hypotheses.append({
                'axis': 'horizontal' if axis == 0 else 'vertical',
                'pixel_coordinate': int(peak),
                'polarity': polarity,
                'continuity': float(fractions[peak]),
                'supported_pixels': int(supported[peak]),
                'matching_pixels': int(hits[peak]),
                'median_response': float(np.median(observations)),
                'range_along_band': [int(extent[0]), int(extent[-1])],
            })
    hypotheses.sort(key=lambda band: (-band['continuity'], -abs(band['median_response']), band['pixel_coordinate']))
    return hypotheses[:settings.max_bands]


def draw_support_map(shape, candidates, valid_maps, width):
    layers = np.zeros(shape, dtype=np.uint8)
    half = max(1, int(width / 3))
    for finding in candidates:
        middle = finding['pixel_coordinate']
        from_axis, to_axis = finding['range_along_band']
        if finding['axis'] == 'horizontal':
            slc = slice(max(0, middle - half), min(shape[0], middle + half + 1)), slice(from_axis, to_axis + 1)
            bit = 1 if finding['polarity'] == 'denser' else 2
        else:
            slc = slice(from_axis, to_axis + 1), slice(max(0, middle - half), min(shape[1], middle + half + 1))
            bit = 4 if finding['polarity'] == 'denser' else 8
        layers[slc] |= np.where(valid_maps[finding['axis']][slc], bit, 0).astype(np.uint8)
    return layers


def analyze_bands(surface, settings, outline=None):
    domain = np.ones(surface.valid.shape, dtype=bool)
    if outline is not None:
        domain.fill(False)
        bounds, mask = outline_pixels(outline, surface.extent)
        x, y, width, height = bounds
        domain[y:y + height, x:x + width] = mask
    directions = {'horizontal': 0, 'vertical': 1} if settings.axis == 'both' else {
        settings.axis: 0 if settings.axis == 'horizontal' else 1,
    }
    usable = np.zeros(surface.valid.shape, dtype=bool)
    responses = np.zeros(surface.valid.shape, dtype=np.float32)
    candidates = []
    maps = {}
    for label, axis in directions.items():
        response, valid = response_field(surface, domain, settings.width, axis)
        above = valid & (np.abs(response) > np.abs(responses))
        responses[above] = response[above]
        usable |= valid
        maps[label] = valid
        candidates.extend(inspect_projection(response, valid, axis=axis, settings=settings))
    if not np.any(usable):
        raise InsufficientSupport('The chosen band width and inspection area have no usable neighbourhoods.')
    candidates.sort(key=lambda band: (-band['continuity'], -abs(band['median_response']), band['axis']))
    map_data = draw_support_map(surface.valid.shape, candidates, maps, settings.width)
    summary = {
        'extent': list(surface.extent),
        'settings': settings.as_json(),
        'support': support_summary(usable),
        'hypotheses': candidates,
        'band_pixels': int(np.count_nonzero(map_data)),
        'method': 'Local strip-to-shoulder contrast and continuous line evidence',
    }
    return summary, responses, usable, map_data
