"""Compare two date-specific numerical surfaces in their shared canvas coordinates.

Exposure drift between visits is corrected either from one stable region
(legacy constant offset) or from several distributed stable regions.  The
spatial correction fits a weighted least-squares plane through the per-region
median offsets to restore a linear gradient, blends it with inverse-distance
interpolation between the regions, and only trusts the result within a bounded
reach of the nearest reference.  Out-of-reach pixels are held as pending review
rather than being called changes, and the measured rasters themselves are never
modified, so genuine treatment traces survive in the residual delta.
"""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage

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
PENDING_REVIEW = 7
CHANGE_CLASSES = {
    'earlier_only': EARLIER_ONLY,
    'later_only': LATER_ONLY,
    'stable': STABLE,
    'positive': DENSER,
    'negative': LESS_DENSE,
    'unresolved': UNRESOLVED,
    'pending_review': PENDING_REVIEW,
}
MIN_PAIRED_SAMPLES = 8
DISAGREEMENT_SIGMAS = 3.5
REACH_FACTOR = 1.35
IDW_POWER = 2.0


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


@dataclass
class ReferenceProbe:
    region_id: str
    name: str
    bounds: tuple
    samples: int
    median_offset: float
    scatter: float
    accepted: bool
    reason: str | None
    mask: np.ndarray | None = None
    centre: tuple | None = None

    def as_json(self):
        return {
            'region_id': self.region_id, 'name': self.name, 'bounds': list(self.bounds),
            'paired_samples': self.samples, 'median_offset': self.median_offset,
            'robust_scatter': self.scatter, 'accepted': self.accepted, 'reason': self.reason,
        }


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
    offsets = later.pixels[y_slice, x_slice][common] - earlier.pixels[y_slice, x_slice][common]
    return float(np.median(offsets)), {
        'area': int(np.count_nonzero(mask)), 'paired': int(offsets.size),
        'raw_offsets': describe_values(offsets), 'bounds': list(bounds),
    }


def _probe_region(earlier, later, paired, region, outline):
    bounds, mask = outline_pixels(outline, earlier.extent)
    x, y, width, height = bounds
    y_slice, x_slice = slice(y, y + height), slice(x, x + width)
    common = paired[y_slice, x_slice] & mask
    if int(np.count_nonzero(common)) < MIN_PAIRED_SAMPLES:
        return ReferenceProbe(
            region['id'], region['name'], bounds, int(np.count_nonzero(common)), 0.0, 0.0, False,
            f'Fewer than {MIN_PAIRED_SAMPLES} paired samples were measured in both visits.',
        )
    offsets = later.pixels[y_slice, x_slice][common] - earlier.pixels[y_slice, x_slice][common]
    median = float(np.median(offsets))
    scatter = float(1.4826 * np.median(np.abs(offsets - median)))
    rows, columns = np.nonzero(common)
    centre = (x + float(np.mean(columns)), y + float(np.mean(rows)))
    return ReferenceProbe(
        region['id'], region['name'], bounds, int(common.size), median, scatter, True, None,
        mask=common, centre=centre,
    )


def _consensus(probes):
    """Flag reference offsets that do not belong to one mutually consistent story."""
    usable = [probe for probe in probes if probe.accepted]
    if len(usable) == 1:
        return
    if len(usable) == 2:
        first, second = usable
        # With two estimates the MAD cannot expose a conflict; use the regions'
        # own within-region scatter as the tolerance instead.
        if abs(first.median_offset - second.median_offset) > DISAGREEMENT_SIGMAS * (first.scatter + second.scatter) + 1e-6:
            for probe in usable:
                probe.accepted = False
                probe.reason = (
                    'Its offset cannot be reconciled with the other reference region; '
                    'two conflicting regions cannot establish a drift gradient.'
                )
        return
    offsets = np.array([probe.median_offset for probe in usable], dtype=np.float64)
    centre = float(np.median(offsets))
    robust_sigma = max(1.4826 * float(np.median(np.abs(offsets - centre))), 1e-6)
    for probe in usable:
        if abs(probe.median_offset - centre) > DISAGREEMENT_SIGMAS * robust_sigma:
            probe.accepted = False
            probe.reason = 'Its offset disagrees with the other reference regions; excluded from the correction.'


def _fit_plane(points):
    data = np.asarray(points, dtype=np.float64)
    weights = data[:, 4] / np.maximum(data[:, 3], 1e-6) ** 2
    weights = weights / np.mean(weights)
    design = np.column_stack((np.ones(len(data)), data[:, :2]))
    try:
        beta, *_ = np.linalg.lstsq(design * np.sqrt(weights)[:, None], data[:, 2] * np.sqrt(weights), rcond=None)
    except np.linalg.LinAlgError:
        return None
    return (float(beta[0]), float(beta[1]), float(beta[2])) if np.all(np.isfinite(beta)) else None


def _idw_field(shape, points):
    height, width = shape
    ys, xs = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing='ij')
    field = np.zeros(shape, dtype=np.float32)
    weights = np.zeros(shape, dtype=np.float32)
    for x, y, offset, _scatter, _n in points:
        influence = 1.0 / np.maximum((xs - x) ** 2 + (ys - y) ** 2, 0.25) ** (IDW_POWER / 2.0)
        field += influence * offset
        weights += influence
    return field / np.maximum(weights, 1e-12)


def build_correction(earlier, later, paired, references):
    """Per-pixel drift offset, reach mask, confidence and an audit report.

    An empty reference list reproduces the legacy zero-offset behaviour.
    """
    shape = earlier.valid.shape
    zero = np.zeros(shape, dtype=np.float32)
    if not references:
        return zero, np.ones(shape, dtype=bool), zero, {
            'mode': 'none', 'reference_count': 0, 'accepted_count': 0, 'rejected_count': 0,
            'consensus_offset': 0.0, 'plane_gradient': [0.0, 0.0], 'reach_radius_px': None,
            'unresolved_pixels': int(np.count_nonzero(paired)), 'references': [],
        }
    probes = [_probe_region(earlier, later, paired, region, outline) for region, outline in references]
    if not any(probe.accepted for probe in probes):
        raise InsufficientSupport(
            'Every stable reference region needs at least eight samples measured at both visits.',
            facts={'references': [probe.as_json() for probe in probes]},
        )
    _consensus(probes)
    accepted = [probe for probe in probes if probe.accepted]
    if len(accepted) < 2:
        raise InsufficientSupport(
            'The stable reference regions do not agree on an exposure drift. '
            'At least two mutually consistent regions are needed for a spatial correction.',
            facts={
                'inconsistent_references': [probe.name for probe in probes if not probe.accepted],
                'references': [probe.as_json() for probe in probes],
            },
        )
    points = [
        (probe.centre[0], probe.centre[1], probe.median_offset, max(probe.scatter, 1e-6), probe.samples)
        for probe in accepted
    ]
    centres = np.asarray([probe.centre for probe in accepted], dtype=np.float64)
    gaps = []
    for index, centre in enumerate(centres):
        distances = np.linalg.norm(centres - centre, axis=1)
        distances[index] = np.inf
        gaps.append(float(np.min(distances)))
    reach_radius = REACH_FACTOR * max(gaps)

    reference_pixels = np.zeros(shape, dtype=bool)
    for probe in accepted:
        x, y, width, height = probe.bounds
        reference_pixels[y:y + height, x:x + width] |= probe.mask
    nearest = ndimage.distance_transform_edt(~reference_pixels).astype(np.float32)
    reachable = nearest <= reach_radius

    plane = _fit_plane(points)
    ys, xs = np.meshgrid(np.arange(shape[0], dtype=np.float32), np.arange(shape[1], dtype=np.float32), indexing='ij')
    if plane is None:
        correction = np.full(shape, float(np.average([p[2] for p in points], weights=[p[4] for p in points])),
                             dtype=np.float32)
        mode = 'constant'
    else:
        b0, bx, by = plane
        # Local IDW dominates on and between references; the fitted plane takes
        # over smoothly as the nearest reference fades toward the reach edge.
        local_weight = np.exp(-((nearest / reach_radius) ** 2)).astype(np.float32)
        correction = _idw_field(shape, points) * local_weight + (b0 + bx * xs + by * ys) * (1.0 - local_weight)
        mode = 'plane' if len(accepted) == 2 else 'plane_idw'
    correction[~reachable] = 0.0
    confidence = np.clip(1.0 - nearest / reach_radius, 0.0, 1.0).astype(np.float32)
    confidence[~reachable] = 0.0
    return correction, reachable, confidence, {
        'mode': mode, 'reference_count': len(references), 'accepted_count': len(accepted),
        'rejected_count': sum(1 for probe in probes if not probe.accepted),
        'consensus_offset': float(np.average([p[2] for p in points], weights=[p[4] for p in points])),
        'plane_gradient': [plane[1], plane[2]] if plane is not None else [0.0, 0.0],
        'reach_radius_px': float(reach_radius),
        'unresolved_pixels': int(np.count_nonzero(paired & ~reachable)),
        'references': [probe.as_json() for probe in probes],
    }


def correction_uncertainty(reference_scatter, confidence, paired, spread_uncertainty):
    """Add the drift estimate's own uncertainty, relaxed with extrapolation distance."""
    floor = float(np.median(reference_scatter)) if reference_scatter else 0.0
    model = np.zeros_like(confidence)
    model[paired] = floor * (1.35 - 0.85 * confidence)[paired]
    return np.hypot(spread_uncertainty, model), model


def classify_change(earlier, later, paired, reachable, delta, uncertainty, threshold):
    classes = np.zeros(earlier.shape, dtype=np.uint8)
    classes[earlier & ~later] = EARLIER_ONLY
    classes[later & ~earlier] = LATER_ONLY
    classes[paired & ~reachable] = PENDING_REVIEW
    judged = paired & reachable
    magnitude = np.abs(delta)
    clear_stable = judged & (magnitude + uncertainty < threshold)
    clear_change = judged & (magnitude - uncertainty >= threshold - 1e-7)
    classes[judged] = UNRESOLVED
    classes[clear_stable] = STABLE
    classes[clear_change & (delta > 0)] = DENSER
    classes[clear_change & (delta < 0)] = LESS_DENSE
    return classes


def comparison_facts(classes, paired, reachable, domain, delta, uncertainty, definition):
    judged = paired & reachable
    counts = {kind: int(np.count_nonzero(classes == code)) for kind, code in CHANGE_CLASSES.items()}
    peak = max(float(np.percentile(np.abs(delta[judged]), 98)) if judged.any() else 0.0, definition.threshold)
    return {
        'class_counts': counts,
        'paired_support': support_summary(paired),
        'domain_pixels': int(np.count_nonzero(domain)),
        'no_data_in_domain': int(np.count_nonzero(domain & (classes == 0))),
        'delta': describe_values(delta[judged]),
        'uncertainty': describe_values(uncertainty[judged]),
        'display': [-peak, peak],
        'distribution': histogram(delta, judged, (-peak, peak), bins=36),
        'pending_review_pixels': int(np.count_nonzero(classes == PENDING_REVIEW)),
        'corrected_support_fraction': float(np.count_nonzero(judged) / max(1, np.count_nonzero(paired))),
    }


def change_map(earlier, later, earlier_spread, later_spread, definition, outline=None, reference=None,
               references=None):
    if earlier.extent != later.extent:
        raise InvalidGeometry('The two surfaces use different canvas dimensions.')
    domain = comparison_domain(earlier.extent, outline)
    paired = domain & earlier.valid & later.valid
    if np.count_nonzero(paired) < 8:
        raise InsufficientSupport('The selected visits share fewer than eight measurable samples.')

    if references:
        drift_surface, reachable, confidence, drift_report = build_correction(
            earlier, later, paired, references,
        )
        reference_scatter = [
            max(item['robust_scatter'], 1e-6) for item in drift_report['references'] if item['accepted']
        ]
        legacy_summary = None
    else:
        drift_surface = np.zeros(earlier.valid.shape, dtype=np.float32)
        reachable = np.ones_like(paired)
        confidence = np.ones_like(drift_surface)
        legacy_offset, legacy_summary = stable_reference(earlier, later, paired, reference)
        drift_surface[:] = legacy_offset
        reference_scatter = [legacy_summary['raw_offsets']['median_absolute_deviation']] if legacy_summary else []
        drift_report = {
            'mode': 'constant' if reference is not None else 'none',
            'reference_count': 1 if reference is not None else 0,
            'accepted_count': 1 if reference is not None else 0, 'rejected_count': 0,
            'consensus_offset': legacy_offset, 'plane_gradient': [0.0, 0.0],
            'reach_radius_px': None, 'unresolved_pixels': 0,
            'references': [{
                'paired_samples': legacy_summary['paired'] if legacy_summary else 0,
                'median_offset': legacy_offset,
                'robust_scatter': reference_scatter[0] if reference_scatter else 0.0,
                'accepted': reference is not None, 'reason': None,
                'bounds': legacy_summary['bounds'] if legacy_summary else None,
            }] if reference is not None else [],
        }

    delta = np.zeros(earlier.valid.shape, dtype=np.float32)
    delta[paired] = later.pixels[paired] - earlier.pixels[paired] - drift_surface[paired]
    spread_uncertainty = np.zeros_like(delta)
    spread_uncertainty[paired] = definition.spread_weight * np.hypot(
        earlier_spread[paired], later_spread[paired],
    )
    uncertainty, _model = correction_uncertainty(reference_scatter, confidence, paired, spread_uncertainty)
    classes = classify_change(earlier.valid, later.valid, paired, reachable, delta, uncertainty, definition.threshold)
    details = {
        'extent': list(earlier.extent), 'settings': definition.as_json(),
        'reference_offset': drift_report['consensus_offset'], 'reference': legacy_summary,
        'drift_correction': drift_report,
    } | comparison_facts(classes, paired, reachable, domain, delta, uncertainty, definition)
    return details, {
        'difference': Surface(delta, paired), 'uncertainty': uncertainty,
        'classification': classes, 'correction': confidence, 'adjustment': drift_surface,
    }
