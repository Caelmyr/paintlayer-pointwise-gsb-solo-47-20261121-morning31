"""Render numerical and registered-photo layers without mutating stored values."""

from dataclasses import dataclass
import math

import numpy as np

from . import fields
from .errors import ExamError
from .library import get_array
from .raster import (
    Surface, color_indexed, constrained_preview, display_range, photo_rgba,
    png_bytes, radiograph_rgba,
)

NIGHT = (0, 0, 0, 0)
COLORS = {
    'rust': (213, 99, 73, 220),
    'gold': (241, 183, 91, 215),
    'aqua': (95, 189, 173, 210),
    'teal': (76, 144, 142, 190),
    'violet': (159, 134, 210, 220),
    'royal': (97, 135, 217, 198),
    'gray': (154, 173, 182, 148),
    'pink': (220, 115, 147, 215),
}
COVERAGE = (
    NIGHT, (67, 112, 120, 208), (98, 165, 158, 232), (153, 200, 162, 240),
    (191, 206, 142, 242), (237, 212, 133, 246), (245, 176, 106, 250),
)
CHANGE_PALETTE = (
    NIGHT, COLORS['gold'], COLORS['royal'], COLORS['gray'], COLORS['rust'],
    COLORS['aqua'], COLORS['violet'],
)
OWNER_PALETTE = (
    NIGHT, COLORS['aqua'], COLORS['gold'], COLORS['violet'], COLORS['rust'],
    COLORS['royal'], COLORS['pink'], COLORS['gray'], COLORS['teal'],
)
REASON_COLORS = tuple(COLORS[key] for key in ('gray', 'rust', 'gold', 'royal', 'violet'))
QUALITY_COLORS = tuple(COLORS[key] for key in ('royal', 'gray', 'gold', 'rust', 'violet'))
SUPPORT_COLORS = tuple(COLORS[key] for key in ('gold', 'pink', 'aqua', 'royal'))

DISPLAY_LAYERS = {
    'plate': {'signal', 'support', 'weight', 'reasons', 'raw'},
    'mosaic': {'signal', 'support', 'coverage', 'spread', 'owner', 'quality'},
    'registration': {'photo', 'support', 'grid'},
    'compare': {'difference', 'support', 'classification', 'uncertainty'},
    'quality': {'signal', 'support', 'classification', 'response'},
    'support': {'signal', 'support', 'classification', 'response'},
}
SCALED_LAYERS = {'spread', 'uncertainty'}
VIEW_MODES = {'blend', 'split', 'checker'}
IMAGE_DEFAULTS = {'registration': 'photo', 'compare': 'difference'}


@dataclass(frozen=True)
class Presentation:
    limits: tuple | None
    gamma: float
    invert: bool
    palette: str

    @classmethod
    def parse(cls, raw):
        raw = fields.request_object(raw, field='display')
        bounds = raw.get('limits')
        lower_upper = fields.limits_pair(bounds, field='limits', minimum=-100, maximum=100_000) if bounds is not None else None
        scaling = fields.scalar(raw.get('gamma', 1), field='gamma', minimum=0.25, maximum=4)
        invert = fields.flag(raw.get('invert', False), field='invert')
        palette = fields.choice(raw.get('palette', 'grayscale'), field='palette', options={'grayscale', 'warm', 'cool'})
        return cls(lower_upper, scaling, invert, palette)

    def as_json(self):
        return {
            'limits': list(self.limits) if self.limits else None,
            'gamma': self.gamma,
            'invert': self.invert,
            'palette': self.palette,
        }


@dataclass(frozen=True)
class Overlay:
    mode: str
    opacity: float
    division: float
    block: int

    @classmethod
    def parse(cls, raw):
        raw = fields.request_object(raw, field='overlay')
        return cls(
            mode=fields.choice(raw.get('mode', 'blend'), field='mode', options=VIEW_MODES),
            opacity=fields.scalar(raw.get('opacity', 0.48), field='opacity', minimum=0, maximum=1),
            division=fields.scalar(raw.get('division', 0.5), field='division', minimum=0, maximum=1),
            block=fields.integer(raw.get('block', 72), field='block', minimum=8, maximum=300),
        )

    def as_json(self):
        return {'mode': self.mode, 'opacity': self.opacity, 'division': self.division, 'block': self.block}


def display_lut(rgba, palette):
    if palette == 'grayscale':
        return rgba
    gray = rgba[..., 0].astype(np.float32) / 255
    selected = rgba[..., 3] > 0
    if palette == 'warm':
        curve = np.stack((gray ** 0.82, gray * 0.90, gray ** 1.38 * 0.78), axis=-1)
    else:
        curve = np.stack((gray ** 1.35 * 0.90, gray ** 0.94 * 0.96, gray ** 0.77), axis=-1)
    rgba[..., :3][selected] = np.rint(curve[selected] * 255).astype(np.uint8)
    return rgba


def display_numerical(surface, presentation, limits):
    rgba = radiograph_rgba(
        surface,
        limits=presentation.limits or limits,
        gamma=presentation.gamma,
        negative=presentation.invert,
    )
    return display_lut(rgba, presentation.palette)


def flag_palette(flags, colors):
    bit_count = len(colors)
    table = np.zeros((1 << bit_count, 4), dtype=np.uint8)
    for flag in range(1, 1 << bit_count):
        rows = [color for bit, color in enumerate(colors) if (1 << bit) & flag]
        table[flag] = np.asarray(rows, dtype=np.uint16).mean(axis=0).astype(np.uint8)
        if len(rows) > 1:
            table[flag, 3] = max(int(table[flag, 3]), 233)
    return color_indexed(flags.astype(np.uint8) & ((1 << bit_count) - 1), table)


def gradient_rgba(values, support, limits, low, high):
    lower, upper = limits
    strength = np.clip((values - lower) / max(upper - lower, 1e-7), 0, 1)
    variation = np.asarray(high, dtype=np.float32) - np.asarray(low, dtype=np.float32)
    rgba = np.rint(np.array(low, dtype=np.float32) + strength[..., None] * variation).astype(np.uint8)
    rgba[~support] = 0
    return rgba


def diverging_rgba(values, support, limits):
    negative = values < 0
    magnitude = np.where(
        negative,
        np.abs(values) / max(abs(limits[0]), 1e-7),
        np.abs(values) / max(limits[1], 1e-7),
    )
    strength = np.minimum(1, magnitude).astype(np.float32)
    neutral = np.array([205, 209, 201], dtype=np.float32)
    positive = np.array([216, 73, 45], dtype=np.float32)
    blue = np.array([21, 120, 152], dtype=np.float32)
    tint = np.where(negative[..., None], blue, positive)
    colors = neutral + (tint - neutral) * strength[..., None]
    rgba = np.empty((*values.shape, 4), dtype=np.uint8)
    rgba[..., :3] = np.rint(colors).astype(np.uint8)
    rgba[..., 3] = support.astype(np.uint8) * 255
    rgba[~support] = 0
    return rgba


def as_surface(array, alpha=None):
    if alpha is None:
        alpha = np.ones(array.shape[:2], dtype=bool)
    return Surface(array, alpha)


def clipped_array(array, box):
    if box is None:
        return array
    x, y, width, height = box
    return array[y:y + height, x:x + width]


def overlay_selector(spec, extent, shape, origin):
    width, _ = extent
    columns = np.arange(shape[1], dtype=np.int32)[None, :] + origin[0]
    if spec.mode == 'split':
        return np.broadcast_to(columns >= width * spec.division, shape)
    rows = np.arange(shape[0], dtype=np.int32)[:, None] + origin[1]
    return ((rows // spec.block + columns // spec.block) % 2) == 1


def fuse_layers(detector, photo, spec, extent, origin):
    if detector.shape != photo.shape:
        raise ExamError('The radiograph and photograph use different canvas dimensions.')
    signal_support = detector[..., 3] > 0
    image_support = photo[..., 3] > 0
    overlap = signal_support & image_support
    merged = detector.copy()
    photo_only = image_support & ~signal_support
    merged[photo_only] = photo[photo_only]
    if spec.mode == 'blend':
        mixed = detector[overlap].astype(np.float32) * (1 - spec.opacity) + photo[overlap].astype(np.float32) * spec.opacity
        merged[overlap] = np.rint(mixed).astype(np.uint8)
    else:
        choose_photo = overlay_selector(spec, extent, overlap.shape, origin) & overlap
        merged[choose_photo] = photo[choose_photo]
    return merged


class Views:
    def __init__(self, store, library, processing):
        self.store = store
        self.library = library
        self.processing = processing

    def source_image(self, case_id, source_id, request):
        presentation = Presentation.parse(request.get('display', {}))
        edge = fields.integer(request.get('edge', 1600), field='edge', minimum=120, maximum=4096)
        with self.store.snapshot() as queries:
            source = queries.record(case_id, source_id, 'source')
            surface = self.library.source_surface(queries, case_id, source)
        role = source['details']['role']
        if role == 'visible':
            rgba = photo_rgba(surface)
        elif role == 'mask':
            rgba = color_indexed(surface.pixels, (NIGHT, COLORS['aqua']))
        else:
            rgba = display_numerical(surface, presentation, tuple(source['details']['display']))
        return constrained_preview(rgba, edge)

    def run_image(self, case_id, run_id, request):
        presentation = Presentation.parse(request.get('display', {}))
        edge = fields.integer(request.get('edge', 1800), field='edge', minimum=120, maximum=4096)
        with self.store.snapshot() as queries:
            run = queries.record(case_id, run_id, 'run')
            kind = run['details']['kind']
            layer = fields.choice(request.get('layer', IMAGE_DEFAULTS.get(kind, 'signal')), field='layer', options=DISPLAY_LAYERS[kind])
            extent = tuple(run['details']['extent'])
            crop = fields.box(request.get('box'), field='box', extent=extent) if request.get('box') else None
            rgba = self.render(queries, case_id, run, layer, presentation, crop)
        return constrained_preview(rgba, edge)

    def render(self, queries, case_id, run, layer, presentation, crop=None):
        numeric = get_array(queries, case_id, run['id'], layer)
        values = clipped_array(numeric, crop)
        kind = run['details']['kind']
        if layer == 'support':
            return color_indexed(values.astype(np.uint8), (NIGHT, COLORS['aqua']))
        if layer == 'photo':
            support = clipped_array(get_array(queries, case_id, run['id'], 'support'), crop)
            return photo_rgba(as_surface(values, support))
        if layer == 'grid':
            return color_indexed(values, (NIGHT, COLORS['gold'], COLORS['aqua'], (245, 231, 176, 233)))
        if layer == 'coverage':
            return color_indexed(np.minimum(values, len(COVERAGE) - 1), COVERAGE)
        if layer == 'owner':
            choices = np.where(values > 0, 1 + (values - 1) % (len(OWNER_PALETTE) - 1), 0)
            return color_indexed(choices, OWNER_PALETTE)
        if layer in {'quality', 'reasons'}:
            return flag_palette(values, REASON_COLORS)
        if layer == 'classification':
            return self.classification(kind, values)
        if layer == 'weight':
            return gradient_rgba(values, values > 0, (0, 1), (35, 67, 67, 204), COLORS['aqua'])
        if layer in SCALED_LAYERS:
            return self.magnitude(values, layer, run['details'])
        if layer == 'response':
            measured = clipped_array(get_array(queries, case_id, run['id'], 'support'), crop)
            if kind == 'quality':
                return gradient_rgba(values, measured, (0, 2), (26, 50, 62, 78), COLORS['rust'])
            limits = display_range(values, measured, (2, 98))
            return diverging_rgba(values, measured, limits)
        return self.signal_layer(queries, case_id, run, layer, values, presentation, crop)

    def classification(self, kind, values):
        if kind == 'compare':
            return color_indexed(values, CHANGE_PALETTE)
        palette = QUALITY_COLORS if kind == 'quality' else SUPPORT_COLORS
        return flag_palette(values, palette)

    def magnitude(self, values, layer, details):
        legend = details.get('peak_overlap_spread')
        if legend is None or layer == 'uncertainty':
            legend = details.get('settings', {}).get('threshold', 0.1)
        positive = values > 0
        if np.any(positive):
            high = float(np.percentile(values[positive], 97))
        else:
            high = 0
        maximum = max(0.001, high, float(legend or 0))
        return gradient_rgba(values, positive, (0, maximum), (61, 60, 90, 140), COLORS['gold'])

    def signal_layer(self, queries, case_id, run, layer, values, presentation, crop):
        actual = run['details']
        if layer == 'raw':
            valid = np.ones(values.shape, dtype=bool)
            limits = display_range(values, valid)
        else:
            valid = clipped_array(get_array(queries, case_id, run['id'], 'support'), crop)
            limits = actual.get('display') or display_range(values, valid)
        if layer == 'difference':
            return diverging_rgba(values, valid, presentation.limits or limits)
        return display_numerical(Surface(values, valid), presentation, limits)

    def combined(self, case_id, raw):
        raw = fields.request_object(raw)
        presentation = Presentation.parse(raw.get('display', {}))
        overlay = Overlay.parse(raw.get('overlay', {}))
        edge = fields.integer(raw.get('edge', 1800), field='edge', minimum=120, maximum=4096)
        with self.store.snapshot() as queries:
            mosaic = self.processing.run(queries, case_id, raw.get('mosaic'), {'mosaic'})
            photo = self.processing.run(queries, case_id, raw.get('registration'), {'registration'})
            extent = tuple(mosaic['details']['extent'])
            box = fields.box(raw.get('box'), field='box', extent=extent) if raw.get('box') else None
            rgba = self.compose(queries, case_id, mosaic, photo, presentation, overlay, box)
        return constrained_preview(rgba, edge)

    def compose(self, queries, case_id, mosaic, photo, presentation, overlay, box):
        radiograph = self.render(queries, case_id, mosaic, 'signal', presentation, box)
        visible = self.render(queries, case_id, photo, 'photo', presentation, box)
        extent = tuple(mosaic['details']['extent'])
        origin = tuple(box[:2]) if box else (0, 0)
        return fuse_layers(radiograph, visible, overlay, extent, origin)
