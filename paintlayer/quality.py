"""Review acquisition quality by coverage, plate disagreement and local impulses."""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

from . import fields
from .geometry import outline_pixels
from .measure import describe_values
from .raster import Surface, support_summary
from .errors import InsufficientSupport

UNMEASURED = 1
SPARSE = 2
DISAGREEMENT = 4
IMPULSE = 8
CLIPPED_SOURCE = 16
QUALITY_BITS = {
    'uncovered': UNMEASURED,
    'single_support': SPARSE,
    'overlap_disagreement': DISAGREEMENT,
    'impulse': IMPULSE,
    'clipped_source': CLIPPED_SOURCE,
}
DETAIL_LIMIT = 120


@dataclass(frozen=True)
class QualityGate:
    min_overlap: int
    max_spread: float
    spike_threshold: float
    minimum_area: int

    @classmethod
    def parse(cls, payload):
        payload = fields.request_object(payload, field='quality')
        return cls(
            min_overlap=fields.integer(payload.get('min_overlap', 2), field='min_overlap', minimum=1, maximum=16),
            max_spread=fields.scalar(payload.get('max_spread', 0.075), field='max_spread', minimum=0, maximum=10),
            spike_threshold=fields.scalar(payload.get('spike_threshold', 0.18), field='spike_threshold', minimum=0.0001, maximum=10),
            minimum_area=fields.integer(payload.get('minimum_area', 8), field='minimum_area', minimum=1, maximum=50_000),
        )

    def as_json(self):
        return {
            'min_overlap': self.min_overlap,
            'max_spread': self.max_spread,
            'spike_threshold': self.spike_threshold,
            'minimum_area': self.minimum_area,
        }


def quality_domain(extent, outline):
    width, height = extent
    if outline is None:
        return np.ones((height, width), dtype=bool)
    bounds, mask = outline_pixels(outline, extent)
    x, y, width, height = bounds
    domain = np.zeros((extent[1], extent[0]), dtype=bool)
    domain[y:y + height, x:x + width] = mask
    return domain


def local_spikes(surface):
    signal = np.where(surface.valid, surface.pixels, 0.0).astype(np.float32)
    valid = surface.valid.astype(np.float32)
    local_sum = ndimage.uniform_filter(signal, size=3, mode='constant') * 9 - signal
    population = ndimage.uniform_filter(valid, size=3, mode='constant') * 9 - valid
    eligible = surface.valid & (population >= 5.99)
    metric = np.zeros_like(signal)
    metric[eligible] = np.abs(signal[eligible] - local_sum[eligible] / population[eligible])
    return metric, eligible


def add_binary_flags(output, mask, flag_bit):
    output[mask] |= flag_bit


def components(mask, minimum, response, category):
    labeled, count = ndimage.label(mask, structure=ndimage.generate_binary_structure(2, 2))
    if not count:
        return {'count': 0, 'small': 0, 'details': [], 'omitted': 0}
    sizes = np.bincount(labeled.ravel(), minlength=count + 1)
    eligible = np.flatnonzero(sizes[1:] >= minimum) + 1
    ordering = sorted(eligible.tolist(), key=lambda index: (-sizes[index], index))
    boxes = ndimage.find_objects(labeled, max_label=count)
    details = []
    for index in ordering[:DETAIL_LIMIT]:
        y_slice, x_slice = boxes[index - 1]
        local_mask = labeled[y_slice, x_slice] == index
        local_rows, local_columns = np.where(local_mask)
        size = int(sizes[index])
        score = None
        if response is not None:
            score = float(np.max(response[y_slice, x_slice][local_mask]))
        details.append({
            'category': category,
            'area': size,
            'bounds': [
                int(x_slice.start), int(y_slice.start),
                int(x_slice.stop - x_slice.start), int(y_slice.stop - y_slice.start),
            ],
            'centroid': [
                float(x_slice.start + local_columns.mean()),
                float(y_slice.start + local_rows.mean()),
            ],
            'peak_score': score,
        })
    return {
        'count': int(len(ordering)),
        'small': int(count - len(ordering)),
        'omitted': int(max(0, len(ordering) - DETAIL_LIMIT)),
        'details': details,
    }


def boundary_contacts(mask):
    return {
        'left': bool(np.any(mask[:, 0])),
        'right': bool(np.any(mask[:, -1])),
        'top': bool(np.any(mask[0, :])),
        'bottom': bool(np.any(mask[-1, :])),
    }


def summarize_classes(tags, domain, minimum, metrics):
    results = []
    for label, bit in QUALITY_BITS.items():
        classified = (tags & bit) != 0
        count = int(np.count_nonzero(classified))
        found = components(classified, minimum, metrics.get(label), label)
        area = int(np.count_nonzero(domain))
        results.append({
            'key': label,
            'flag': bit,
            'pixels': count,
            'fraction': count / area,
            'boundary_contacts': boundary_contacts(classified),
        } | found)
    return results


def quality_audit(surface, coverage, spread, flags, gate, outline=None):
    domain = quality_domain(surface.extent, outline)
    if not domain.any():
        raise InsufficientSupport('Select an inspection area with raster support.')
    valid = domain & surface.valid
    spike, can_test = local_spikes(surface)
    tags = np.zeros(surface.valid.shape, dtype=np.uint8)
    add_binary_flags(tags, domain & ~surface.valid, UNMEASURED)
    add_binary_flags(tags, valid & (coverage < gate.min_overlap), SPARSE)
    add_binary_flags(tags, valid & (coverage >= 2) & (spread > gate.max_spread), DISAGREEMENT)
    add_binary_flags(tags, valid & can_test & (spike >= gate.spike_threshold), IMPULSE)
    add_binary_flags(tags, domain & ((flags & (2 | 16)) > 0), CLIPPED_SOURCE)
    response = np.maximum(
        np.where(valid & (coverage > 1), spread / max(0.0001, gate.max_spread), 0.0),
        np.where(valid & can_test, spike / gate.spike_threshold, 0.0),
    ).astype(np.float32)
    metrics = {'overlap_disagreement': spread, 'impulse': spike}
    classified = summarize_classes(tags, domain, gate.minimum_area, metrics)
    found = int(np.count_nonzero(tags))
    report = {
        'gate': gate.as_json(),
        'extent': list(surface.extent),
        'support': support_summary(valid),
        'domain_pixels': int(np.count_nonzero(domain)),
        'flagged_pixels': found,
        'clean_pixels': int(np.count_nonzero(domain)) - found,
        'classes': classified,
        'multiple_reasons': int(np.count_nonzero((tags & (tags - np.uint8(1))) != 0)),
        'numerical_disagreement': describe_values(spread[valid & (coverage >= 2)]),
    }
    return report, tags, response, domain


def quality_scores(tags, response, domain, box):
    x, y, width, height = box
    region = (slice(y, y + height), slice(x, x + width))
    local_tags = tags[region]
    local_domain = domain[region]
    sources = {}
    for name, bit in QUALITY_BITS.items():
        sources[name] = int(np.count_nonzero(local_domain & ((local_tags & bit) != 0)))
    active = local_domain & (local_tags != 0)
    source_count = int(np.count_nonzero(local_domain))
    return {
        'counts': sources,
        'domain_pixels': source_count,
        'flagged_pixels': int(np.count_nonzero(active)),
        'mean_response': float(np.mean(response[region][active], dtype=np.float64)) if np.any(active) else None,
        'worst_response': float(np.max(response[region][active])) if np.any(active) else None,
    }
