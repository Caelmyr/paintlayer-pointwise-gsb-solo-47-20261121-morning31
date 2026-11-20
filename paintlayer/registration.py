"""Tie-point registration and support-aware cross-modal sampling."""

from dataclasses import dataclass
import math

import numpy as np

from . import fields
from .errors import InsufficientSupport, InvalidGeometry
from .geometry import ROWS_PER_CHUNK, bilinear_values
from .raster import Surface, support_summary

ALIGNMENT_MODELS = {'similarity', 'affine'}
MAX_CONDITION = 150
MAX_LANDMARKS = 28
MIN_SPAN = 3


@dataclass(frozen=True)
class TiePoint:
    canvas: tuple
    photograph: tuple
    influence: float
    label: str

    @classmethod
    def parse(cls, row, field, canvas_extent, photo_extent):
        row = fields.request_object(row, field=field)
        return cls(
            canvas=fields.point(row.get('canvas'), field=f'{field}.canvas', extent=canvas_extent),
            photograph=fields.point(row.get('photograph'), field=f'{field}.photograph', extent=photo_extent),
            influence=fields.scalar(row.get('influence', 1), field=f'{field}.influence', minimum=0.1, maximum=5),
            label=fields.text(row.get('label'), field=f'{field}.label', limit=40, required=False),
        )

    def record(self):
        return {
            'canvas': list(self.canvas),
            'photograph': list(self.photograph),
            'influence': self.influence,
            'label': self.label,
        }


def parse_points(payload, mode, canvas_extent, photo_extent):
    minimum = 2 if mode == 'similarity' else 3
    rows = fields.sequence(payload, field='points', minimum=minimum, maximum=MAX_LANDMARKS)
    points = [
        TiePoint.parse(row, f'points[{index}]', canvas_extent, photo_extent)
        for index, row in enumerate(rows)
    ]
    fields.unique([point.canvas for point in points], field='canvas points')
    fields.unique([point.photograph for point in points], field='photograph points')
    return points


@dataclass(frozen=True)
class Fit:
    matrix: np.ndarray
    mode: str
    condition: float
    points: list

    def canvas_to_photo(self, coordinates):
        return coordinates @ self.matrix[:, :2].T + self.matrix[:, 2]

    def photo_to_canvas(self, coordinates):
        inverse = np.linalg.inv(self.matrix[:, :2])
        return (coordinates - self.matrix[:, 2]) @ inverse.T

    def facts(self):
        transform = self.matrix[:, :2]
        determinant = float(np.linalg.det(transform))
        scales = np.linalg.svd(transform, compute_uv=False)
        return {
            'canvas_to_photo': self.matrix.tolist(),
            'angle_degrees': math.degrees(math.atan2(transform[1, 0], transform[0, 0])),
            'axis_scales': scales.tolist(),
            'reflected': determinant < 0,
            'area_scale': abs(determinant),
            'condition': self.condition,
        }


def landmark_coordinates(points):
    canvas = np.array([point.canvas for point in points], dtype=np.float64)
    photograph = np.array([point.photograph for point in points], dtype=np.float64)
    weights = np.array([point.influence for point in points], dtype=np.float64)
    return canvas, photograph, weights


def center_and_scale(coords, weights):
    midpoint = np.average(coords, axis=0, weights=weights)
    centered = coords - midpoint
    rms = float(np.sqrt(np.average(np.sum(centered ** 2, axis=1), weights=weights)))
    if rms < MIN_SPAN:
        raise InvalidGeometry('Spread corresponding landmarks farther apart.', field='points')
    return centered / rms, midpoint, rms


def solve_similarity(src, dst, weights):
    count = src.shape[0]
    design = np.empty((count * 2, 2), dtype=np.float64)
    design[::2, 0] = src[:, 0]
    design[::2, 1] = -src[:, 1]
    design[1::2, 0] = src[:, 1]
    design[1::2, 1] = src[:, 0]
    root_weights = np.sqrt(np.repeat(weights, 2))
    values, _, rank, singular = np.linalg.lstsq(design * root_weights[:, None], dst.ravel() * root_weights, rcond=None)
    if rank != 2:
        raise InvalidGeometry('Landmarks cannot determine a similarity transform.', field='points')
    real, imaginary = values
    linear = np.array([[real, -imaginary], [imaginary, real]], dtype=np.float64)
    return linear, float(singular[0] / singular[-1])


def solve_affine(src, dst, weights):
    factors = np.sqrt(weights)[:, None]
    coef, _, rank, values = np.linalg.lstsq(src * factors, dst * factors, rcond=None)
    if rank != 2 or values[-1] <= 0 or values[0] / values[-1] > MAX_CONDITION:
        raise InvalidGeometry('Affine landmarks must span two canvas directions.', field='points')
    return coef.T, float(values[0] / values[-1])


def fit_points(points, mode):
    canvas, photo, weights = landmark_coordinates(points)
    src, center_src, extent_src = center_and_scale(canvas, weights)
    dst, center_dst, extent_dst = center_and_scale(photo, weights)
    if mode == 'similarity':
        linear, condition = solve_similarity(src, dst, weights)
    else:
        linear, condition = solve_affine(src, dst, weights)
    linear *= extent_dst / extent_src
    scales = np.linalg.svd(linear, compute_uv=False)
    if scales[0] > 20 or scales[-1] < 0.05 or scales[0] / scales[-1] > 12:
        raise InvalidGeometry('Landmarks imply a collapsed or extreme transform.', field='points')
    translation = center_dst - linear @ center_src
    matrix = np.column_stack((linear, translation))
    return Fit(matrix, mode, condition, points)


def point_errors(fit):
    canvas, photo, weights = landmark_coordinates(fit.points)
    target = fit.canvas_to_photo(canvas)
    back = fit.photo_to_canvas(photo)
    forward_errors = np.linalg.norm(target - photo, axis=1)
    backward_errors = np.linalg.norm(back - canvas, axis=1)
    return [
        point.record() | {
            'estimated_photograph': target[index].tolist(),
            'photo_residual': float(forward_errors[index]),
            'canvas_residual': float(backward_errors[index]),
            'influence': float(weights[index]),
        }
        for index, point in enumerate(fit.points)
    ]


def holdout_errors(fit):
    minimum = 2 if fit.mode == 'similarity' else 3
    tested = []
    if len(fit.points) <= minimum:
        return tested
    for index, withheld in enumerate(fit.points):
        retained = fit.points[:index] + fit.points[index + 1:]
        try:
            subfit = fit_points(retained, fit.mode)
        except InvalidGeometry:
            tested.append({'point': index + 1, 'status': 'underdetermined', 'error': None})
            continue
        estimate = subfit.photo_to_canvas(np.array(withheld.photograph))
        distance = float(np.linalg.norm(estimate - np.array(withheld.canvas)))
        tested.append({'point': index + 1, 'status': 'measured', 'error': distance})
    return tested


def geometric_diagnostics(fit):
    residuals = point_errors(fit)
    canvas_errors = np.array([item['canvas_residual'] for item in residuals])
    weights = np.array([point.influence for point in fit.points])
    withheld = holdout_errors(fit)
    measurable = [row['error'] for row in withheld if row['error'] is not None]
    return {
        'mode': fit.mode,
        'transform': fit.facts(),
        'landmarks': residuals,
        'rms_canvas_pixels': float(np.sqrt(np.average(canvas_errors ** 2, weights=weights))),
        'maximum_canvas_pixels': float(np.max(canvas_errors)),
        'holdout': withheld,
        'holdout_worst': max(measurable, default=None),
    }


@dataclass(frozen=True)
class Registered:
    surface: Surface
    grid: np.ndarray
    summary: dict


def sample_photo(fit, photograph, extent):
    width, height = extent
    rows = np.arange(height, dtype=np.float64)
    columns = np.arange(width, dtype=np.float64)
    output = np.zeros((height, width, 3), dtype=np.uint8)
    support = np.zeros((height, width), dtype=bool)
    matrix = fit.matrix
    for start in range(0, height, ROWS_PER_CHUNK):
        stop = min(height, start + ROWS_PER_CHUNK)
        x = columns[None, :]
        y = rows[start:stop, None]
        mapped_x = matrix[0, 0] * x + matrix[0, 1] * y + matrix[0, 2]
        mapped_y = matrix[1, 0] * x + matrix[1, 1] * y + matrix[1, 2]
        samples, usable, _ = bilinear_values(photograph.pixels, photograph.valid, mapped_x, mapped_y)
        output[start:stop] = np.clip(np.rint(samples), 0, 255).astype(np.uint8)
        support[start:stop] = usable
    return Surface(output, support)


def grid_layer(fit, photo_extent, extent, valid):
    width, height = extent
    grid = np.zeros((height, width), dtype=np.uint8)
    photo_width, photo_height = photo_extent
    step = max(12, min(photo_width, photo_height) // 10)
    matrix = fit.matrix
    columns = np.arange(width, dtype=np.float64)[None, :]
    for start in range(0, height, ROWS_PER_CHUNK):
        stop = min(height, start + ROWS_PER_CHUNK)
        rows = np.arange(start, stop, dtype=np.float64)[:, None]
        photo_x = matrix[0, 0] * columns + matrix[0, 1] * rows + matrix[0, 2]
        photo_y = matrix[1, 0] * columns + matrix[1, 1] * rows + matrix[1, 2]
        x_mod = np.abs(np.remainder(photo_x + step / 2, step) - step / 2)
        y_mod = np.abs(np.remainder(photo_y + step / 2, step) - step / 2)
        lines_x = x_mod <= 0.65
        lines_y = y_mod <= 0.65
        grid[start:stop] = np.where(valid[start:stop], lines_x.astype(np.uint8) + lines_y.astype(np.uint8) * 2, 0)
    return grid, step


def register(mode, points, photograph, extent, target_support=None):
    fit = fit_points(points, mode)
    matched = sample_photo(fit, photograph, extent)
    if matched.count() < 16:
        raise InsufficientSupport('The photograph does not cover enough of the examination canvas.')
    lines, spacing = grid_layer(fit, photograph.extent, extent, matched.valid)
    summary = geometric_diagnostics(fit)
    summary['extent'] = list(extent)
    summary['photo_extent'] = list(photograph.extent)
    summary['support'] = support_summary(matched.valid)
    summary['grid_spacing_photo_pixels'] = spacing
    if target_support is not None:
        paired = target_support & matched.valid
        summary['paired_support'] = support_summary(paired)
        summary['photo_coverage_of_radiograph'] = int(np.count_nonzero(paired)) / max(1, int(np.count_nonzero(target_support)))
    return Registered(matched, lines, summary)
