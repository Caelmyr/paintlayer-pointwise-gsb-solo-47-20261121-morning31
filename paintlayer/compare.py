"""Compare two date-specific numerical surfaces in their shared canvas coordinates."""

from dataclasses import dataclass

import numpy as np

from . import fields
from .errors import InsufficientSupport, InvalidGeometry
from .geometry import outline_pixels
from .measure import describe_values
from .raster import Surface, histogram, support_summary

EARLIER_ONLY = 1
LATER_ONLY = 2
STABLE = 3
DENSER = 4
LESS_DENSE = 5
UNRESOLVED = 6
CHANGE_CLASSES = {
    'earlier_only': EARLIER_ONLY,
    'later_only': LATER_ONLY,
    'stable': STABLE,
    'positive': DENSER,
    'negative': LESS_DENSE,
    'unresolved': UNRESOLVED,
}


@dataclass(frozen=True)
class Comparison:
    threshold: float
    spread_weight: float

    @classmethod
    def parse(cls, payload):
        payload = fields.request_object(payload, field='comparison')
        return cls(
            threshold=fields.scalar(payload.get('threshold', 0.065), field='threshold', minimum=0.0001, maximum=10),
            spread_weight=fields.scalar(payload.get('spread_weight', 0.5), field='spread_weight', minimum=0, maximum=3),
        )

    def as_json(self):
        return {'threshold': self.threshold, 'spread_weight': self.spread_weight}


def comparison_domain(extent, domain_outline):
    if domain_outline is None:
        return np.ones((extent[1], extent[0]), dtype=bool)
    bounds, mask = outline_pixels(domain_outline, extent)
    x, y, width, height = bounds
    output = np.zeros((extent[1], extent[0]), dtype=bool)
    output[y:y + height, x:x + width] = mask
    return output


def stable_reference(earlier, later, paired, outline):
    if outline is None:
        return 0.0, None
    bounds, mask = outline_pixels(outline, earlier.extent)
    x, y, width, height = bounds
    y_slice, x_slice = slice(y, y + height), slice(x, x + width)
    common = paired[y_slice, x_slice] & mask
    if np.count_nonzero(common) < 8:
        raise InsufficientSupport('The reference region needs at least eight samples measured at both visits.')
    before = earlier.pixels[y_slice, x_slice][common]
    after = later.pixels[y_slice, x_slice][common]
    offsets = after - before
    offset = float(np.median(offsets))
    summary = {
        'area': int(np.count_nonzero(mask)),
        'paired': int(offsets.size),
        'raw_offsets': describe_values(offsets),
        'bounds': list(bounds),
    }
    return offset, summary


def classify_change(earlier, later, paired, delta, uncertainty, threshold):
    classes = np.zeros(earlier.shape, dtype=np.uint8)
    classes[earlier & ~later] = EARLIER_ONLY
    classes[later & ~earlier] = LATER_ONLY
    magnitude = np.abs(delta)
    clear_stable = paired & (magnitude + uncertainty < threshold)
    clear_change = paired & (magnitude - uncertainty >= threshold - 1e-7)
    classes[paired] = UNRESOLVED
    classes[clear_stable] = STABLE
    classes[clear_change & (delta > 0)] = DENSER
    classes[clear_change & (delta < 0)] = LESS_DENSE
    return classes


def comparison_facts(classes, paired, domain, delta, uncertainty, definition):
    counts = {kind: int(np.count_nonzero(classes == code)) for kind, code in CHANGE_CLASSES.items()}
    peak = max(float(np.percentile(np.abs(delta[paired]), 98)), definition.threshold)
    return {
        'class_counts': counts,
        'paired_support': support_summary(paired),
        'domain_pixels': int(np.count_nonzero(domain)),
        'no_data_in_domain': int(np.count_nonzero(domain & (classes == 0))),
        'delta': describe_values(delta[paired]),
        'uncertainty': describe_values(uncertainty[paired]),
        'display': [-peak, peak],
        'distribution': histogram(delta, paired, (-peak, peak), bins=36),
    }


def change_map(earlier, later, earlier_spread, later_spread, definition, outline=None, reference=None):
    if earlier.extent != later.extent:
        raise InvalidGeometry('The two surfaces use different canvas dimensions.')
    domain = comparison_domain(earlier.extent, outline)
    earlier_valid = domain & earlier.valid
    later_valid = domain & later.valid
    paired = earlier_valid & later_valid
    if np.count_nonzero(paired) < 8:
        raise InsufficientSupport('The selected visits share fewer than eight measurable samples.')
    offset, reference_summary = stable_reference(earlier, later, paired, reference)
    delta = np.zeros(earlier.valid.shape, dtype=np.float32)
    delta[paired] = later.pixels[paired] - earlier.pixels[paired] - offset
    uncertainty = np.zeros_like(delta)
    uncertainty[paired] = definition.spread_weight * np.hypot(earlier_spread[paired], later_spread[paired])
    classes = classify_change(earlier_valid, later_valid, paired, delta, uncertainty, definition.threshold)
    details = {
        'extent': list(earlier.extent),
        'settings': definition.as_json(),
        'reference_offset': offset,
        'reference': reference_summary,
    } | comparison_facts(classes, paired, domain, delta, uncertainty, definition)
    return details, Surface(delta, paired), uncertainty, classes
