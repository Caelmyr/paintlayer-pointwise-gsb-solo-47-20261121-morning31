"""Canvas and detector geometry in zero-based pixel-centre coordinates."""

from dataclasses import dataclass
import math

import numpy as np

from . import fields
from .errors import InsufficientSupport, InvalidGeometry

EPSILON = 1e-8
ROWS_PER_CHUNK = 96


@dataclass(frozen=True)
class Outline:
    outer: tuple
    holes: tuple

    def bounds(self, extent):
        xs, ys = zip(*self.outer)
        x0 = max(0, math.floor(min(xs)))
        y0 = max(0, math.floor(min(ys)))
        x1 = min(extent[0], math.ceil(max(xs)) + 1)
        y1 = min(extent[1], math.ceil(max(ys)) + 1)
        return x0, y0, x1 - x0, y1 - y0

    def as_json(self):
        return {'outer': [list(point) for point in self.outer], 'holes': [list(map(list, ring)) for ring in self.holes]}

    @property
    def geometric_area(self):
        return abs(signed_area(self.outer)) - sum(abs(signed_area(hole)) for hole in self.holes)


@dataclass(frozen=True)
class TileGeometry:
    x: int
    y: int
    turns: int
    mirror: bool

    def orient(self, array):
        if self.mirror:
            array = np.flip(array, axis=1)
        return np.rot90(array, k=-self.turns, axes=(0, 1))

    def overlap(self, array, extent):
        height, width = array.shape[:2]
        x0 = max(0, self.x)
        y0 = max(0, self.y)
        x1 = min(extent[0], self.x + width)
        y1 = min(extent[1], self.y + height)
        if x0 >= x1 or y0 >= y1:
            return None
        source = (
            slice(y0 - self.y, y1 - self.y),
            slice(x0 - self.x, x1 - self.x),
        )
        target = (slice(y0, y1), slice(x0, x1))
        return source, target

    @classmethod
    def from_payload(cls, payload, *, field, extent):
        payload = fields.request_object(payload, field=field)
        width, height = extent
        return cls(
            x=fields.integer(payload.get('x'), field=f'{field}.x', minimum=-4096, maximum=width - 1),
            y=fields.integer(payload.get('y'), field=f'{field}.y', minimum=-4096, maximum=height - 1),
            turns=fields.integer(payload.get('turns', 0), field=f'{field}.turns', minimum=0, maximum=3),
            mirror=fields.flag(payload.get('mirror', False), field=f'{field}.mirror'),
        )

    def to_json(self):
        return {'x': self.x, 'y': self.y, 'turns': self.turns, 'mirror': self.mirror}


def validate_canvas(width, height, max_pixels):
    if width * height > max_pixels:
        raise InvalidGeometry('The shared canvas exceeds the maximum number of pixels.')
    return width, height


def edges(vertices):
    return zip(vertices, (*vertices[1:], vertices[0]))


def signed_area(vertices):
    return sum(a[0] * b[1] - b[0] * a[1] for a, b in edges(vertices)) / 2


def cross(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def lies_on(a, b, point):
    if abs(cross(a, b, point)) > EPSILON:
        return False
    return (
        min(a[0], b[0]) - EPSILON <= point[0] <= max(a[0], b[0]) + EPSILON
        and min(a[1], b[1]) - EPSILON <= point[1] <= max(a[1], b[1]) + EPSILON
    )


def segments_meet(a, b, c, d):
    cc, dd = cross(a, b, c), cross(a, b, d)
    aa, bb = cross(c, d, a), cross(c, d, b)
    if min(cc, dd) < -EPSILON < EPSILON < max(cc, dd) and min(aa, bb) < -EPSILON < EPSILON < max(aa, bb):
        return True
    return lies_on(a, b, c) or lies_on(a, b, d) or lies_on(c, d, a) or lies_on(c, d, b)


def check_ring(vertices, field):
    if abs(signed_area(vertices)) < 0.5:
        raise InvalidGeometry('The outline encloses less than half a pixel.', field=field)
    segments = list(edges(vertices))
    for index, (a, b) in enumerate(segments):
        c = segments[(index + 1) % len(segments)][1]
        if lies_on(a, b, c) or lies_on(b, c, a):
            raise InvalidGeometry('An outline edge doubles back over its neighbour.', field=field)
        for second in range(index + 2, len(segments)):
            if index == 0 and second == len(segments) - 1:
                continue
            c, d = segments[second]
            if segments_meet(a, b, c, d):
                raise InvalidGeometry('Outline edges may not cross or touch nonadjacent edges.', field=field)


def rings_meet(first, second):
    return any(segments_meet(a, b, c, d) for a, b in edges(first) for c, d in edges(second))


def point_in_ring(point, vertices):
    x, y = point
    in_shape = False
    for a, b in edges(vertices):
        if lies_on(a, b, point):
            return True
        if (a[1] > y) != (b[1] > y):
            crossing = (b[0] - a[0]) * (y - a[1]) / (b[1] - a[1]) + a[0]
            if x < crossing:
                in_shape = not in_shape
    return in_shape


def parse_outline(payload, extent):
    payload = fields.request_object(payload, field='geometry')
    outer = fields.polygon(payload.get('outer'), field='outline', extent=extent)
    check_ring(outer, 'outline')
    raw_holes = fields.sequence(payload.get('holes', []), field='holes', maximum=6)
    holes = []
    for index, raw_hole in enumerate(raw_holes):
        field = f'holes[{index}]'
        hole = fields.polygon(raw_hole, field=field, extent=extent, max_vertices=36)
        check_ring(hole, field)
        if rings_meet(outer, hole) or not point_in_ring(hole[0], outer):
            raise InvalidGeometry('Each exclusion must lie strictly inside the outer outline.', field=field)
        for previous in holes:
            if rings_meet(previous, hole) or point_in_ring(hole[0], previous) or point_in_ring(previous[0], hole):
                raise InvalidGeometry('Exclusions cannot overlap, touch or contain one another.', field=field)
        holes.append(tuple(hole))
    return Outline(tuple(outer), tuple(holes))


def stored_outline(payload):
    return Outline(tuple(map(tuple, payload['outer'])), tuple(tuple(map(tuple, ring)) for ring in payload['holes']))


def rasterize_ring(vertices, bounds):
    x0, y0, width, height = bounds
    xs = np.arange(x0, x0 + width, dtype=np.float64)[None, :]
    ys = np.arange(y0, y0 + height, dtype=np.float64)[:, None]
    inside = np.zeros((height, width), dtype=bool)
    boundary = np.zeros_like(inside)
    for a, b in edges(vertices):
        dx = b[0] - a[0]
        dy = b[1] - a[1]
        x_relevant = (xs >= min(a[0], b[0]) - EPSILON) & (xs <= max(a[0], b[0]) + EPSILON)
        y_relevant = (ys >= min(a[1], b[1]) - EPSILON) & (ys <= max(a[1], b[1]) + EPSILON)
        boundary |= x_relevant & y_relevant & (np.abs(dx * (ys - a[1]) - dy * (xs - a[0])) <= EPSILON)
        if abs(dy) > EPSILON:
            crosses = (a[1] > ys) != (b[1] > ys)
            intercept = a[0] + dx * (ys - a[1]) / dy
            inside ^= crosses & (xs < intercept)
    return inside | boundary


def outline_pixels(outline, extent):
    x, y, w, h = outline.bounds(extent)
    pixels = np.empty((h, w), dtype=bool)
    for offset in range(0, h, ROWS_PER_CHUNK):
        rows = min(ROWS_PER_CHUNK, h - offset)
        bounds = x, y + offset, w, rows
        selected = rasterize_ring(outline.outer, bounds)
        for hole in outline.holes:
            selected &= ~rasterize_ring(hole, bounds)
        pixels[offset:offset + rows] = selected
    if not pixels.any():
        raise InsufficientSupport('The selected outline includes no pixel centres.')
    return (x, y, w, h), pixels


def bilinear_values(pixels, support, x, y, *, min_weight=1.0):
    height, width = support.shape
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x, y = np.broadcast_arrays(x, y)
    in_frame = (x >= -EPSILON) & (x <= width - 1 + EPSILON) & (y >= -EPSILON) & (y <= height - 1 + EPSILON)
    safe_x = np.clip(x, 0, width - 1)
    safe_y = np.clip(y, 0, height - 1)
    x0 = np.floor(safe_x).astype(np.intp)
    y0 = np.floor(safe_y).astype(np.intp)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    fx = safe_x - x0
    fy = safe_y - y0
    weights = (
        ((1 - fx) * (1 - fy), x0, y0),
        (fx * (1 - fy), x1, y0),
        ((1 - fx) * fy, x0, y1),
        (fx * fy, x1, y1),
    )
    weight = np.zeros(x.shape, dtype=np.float32)
    is_color = pixels.ndim == 3
    result_shape = (*x.shape, pixels.shape[2]) if is_color else x.shape
    result = np.zeros(result_shape, dtype=np.float32)
    for fraction, xs, ys in weights:
        usable = support[ys, xs]
        contribution = (fraction * usable).astype(np.float32)
        value = pixels[ys, xs]
        if is_color:
            result += contribution[..., None] * np.where(usable[..., None], value, 0.0)
        else:
            result += contribution * np.where(usable, value, 0.0)
        weight += contribution
    accepted = in_frame & (weight > 0) & (weight >= min_weight - 2e-7)
    denominator = np.where(accepted, weight, 1)
    if is_color:
        result /= denominator[..., None]
    else:
        result /= denominator
    result[~accepted] = 0
    return result, accepted, weight
