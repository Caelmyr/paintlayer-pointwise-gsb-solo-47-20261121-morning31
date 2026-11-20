"""Estimate repeat spacing and dominant directions in selected radiograph patches."""

from dataclasses import dataclass
import math

import numpy as np
from scipy import ndimage

from . import fields
from .errors import InsufficientSupport
from .raster import png_bytes, support_summary

PATCH_MINIMUM = 32
PATCH_MAXIMUM = 640
DEFAULT_ANGLE_BINS = 36


@dataclass(frozen=True)
class TextureRead:
    bounds: tuple
    small_period: float
    large_period: float
    minimum_support: float
    subtract_plane: bool

    @classmethod
    def parse(cls, raw, extent):
        raw = fields.request_object(raw, field='texture')
        bounds = fields.box(raw.get('box'), field='box', extent=extent, limits=(PATCH_MAXIMUM, PATCH_MAXIMUM))
        if min(bounds[2:]) < PATCH_MINIMUM:
            raise InsufficientSupport(f'Choose a patch at least {PATCH_MINIMUM} pixels wide and high.')
        short, long = fields.limits_pair(raw.get('periods', [3, 18]), field='periods', minimum=2.2, maximum=min(bounds[2:]) * 0.75)
        return cls(
            bounds=bounds,
            small_period=short,
            large_period=long,
            minimum_support=fields.scalar(raw.get('min_support', 0.72), field='min_support', minimum=0.5, maximum=1),
            subtract_plane=fields.flag(raw.get('subtract_plane', True), field='subtract_plane'),
        )

    def as_json(self):
        return {
            'box': list(self.bounds),
            'periods': [self.small_period, self.large_period],
            'min_support': self.minimum_support,
            'subtract_plane': self.subtract_plane,
        }


def demean_and_window(values, support, remove_plane):
    height, width = values.shape
    y = np.linspace(-1, 1, height)[:, None]
    x = np.linspace(-1, 1, width)[None, :]
    weights = np.sqrt(np.hanning(height)[:, None] * np.hanning(width)[None, :])
    if not np.any(weights[support] > 0.1):
        raise InsufficientSupport('Usable samples are too close to the edge of the selected patch.')
    selected = values[support].astype(np.float64)
    scale = np.std(selected)
    if not math.isfinite(scale) or scale <= 1e-7:
        raise InsufficientSupport('The patch contains too little detector variation for a texture estimate.')
    weights[~support] = 0
    if remove_plane:
        rows, columns = np.nonzero(support)
        design = np.column_stack((np.ones(rows.size), x[0, columns], y[rows, 0]))
        corner_weights = weights[support][:, None]
        trend, _, rank, _ = np.linalg.lstsq(design * corner_weights, (selected * corner_weights[:, 0]), rcond=None)
        if rank < 3:
            raise InsufficientSupport('The selected patch cannot determine a two-dimensional background.')
        plane = trend[0] + trend[1] * x + trend[2] * y
    else:
        average = np.average(selected, weights=weights[support])
        plane = np.full(values.shape, average)
    high_pass = np.where(support, values - plane, 0)
    residual = float(np.sqrt(np.average((values[support] - plane[support]) ** 2, weights=weights[support])))
    return high_pass * weights, residual


@dataclass(frozen=True)
class FrequencyField:
    pxx: np.ndarray
    px: np.ndarray
    py: np.ndarray
    radial: np.ndarray
    band: np.ndarray


def frequency_field(windowed, short_period, long_period):
    height, width = windowed.shape
    spectrum = np.fft.fftshift(np.fft.fft2(windowed))
    power = np.abs(spectrum) ** 2
    px = np.fft.fftshift(np.fft.fftfreq(width))
    py = np.fft.fftshift(np.fft.fftfreq(height))
    radial = np.hypot(px[None, :], py[:, None])
    band = (radial >= 1 / long_period) & (radial <= 1 / short_period)
    if not np.any(band):
        raise InsufficientSupport('This patch cannot resolve the selected repeat spacing interval.')
    return FrequencyField(power, px, py, radial, band)


def peak_neighbours(power, row, column):
    y0, y1 = max(0, row - 1), min(power.shape[0], row + 2)
    x0, x1 = max(0, column - 1), min(power.shape[1], column + 2)
    cropped = power[y0:y1, x0:x1]
    median = float(np.median(cropped))
    excess = np.maximum(cropped - median, 0)
    if excess.sum() == 0:
        return float(row), float(column)
    ys, xs = np.meshgrid(np.arange(y0, y1), np.arange(x0, x1), indexing='ij')
    normalizer = excess.sum()
    return float(np.sum(ys * excess) / normalizer), float(np.sum(xs * excess) / normalizer)


def find_peaks(field, limit=7):
    power = field.pxx
    smooth = ndimage.gaussian_filter(power, sigma=0.42)
    ridges = smooth == ndimage.maximum_filter(smooth, size=5, mode='nearest')
    upper_half = (field.py[:, None] > 0) | ((field.py[:, None] == 0) & (field.px[None, :] > 0))
    rows, columns = np.nonzero(ridges & field.band & upper_half)
    median_noise = float(np.median(power[field.band]))
    rank = np.argsort(-smooth[rows, columns], kind='stable')
    found = []
    for index in rank[:limit]:
        row, col = rows[index], columns[index]
        sub_y, sub_x = peak_neighbours(power, row, col)
        dx = float(np.interp(sub_x, np.arange(len(field.px)), field.px))
        dy = float(np.interp(sub_y, np.arange(len(field.py)), field.py))
        frequency = math.hypot(dx, dy)
        if frequency <= 0:
            continue
        strength = float(power[row, col] / max(median_noise, 1e-12))
        wave_angle = math.degrees(math.atan2(dy, dx)) % 180
        found.append({
            'sample_frequency': [dx, dy],
            'period_pixels': 1 / frequency,
            'wavevector_degrees': wave_angle,
            'thread_axis_degrees': (wave_angle + 90) % 180,
            'contrast': strength,
            'grid_index': [int(col), int(row)],
        })
    return found


def directional_profile(field, bins=DEFAULT_ANGLE_BINS):
    fx = np.broadcast_to(field.px[None, :], field.pxx.shape)[field.band]
    fy = np.broadcast_to(field.py[:, None], field.pxx.shape)[field.band]
    theta = (np.degrees(np.arctan2(fy, fx)) + 90) % 180
    edges = np.linspace(0, 180, bins + 1)
    signal = np.log1p(field.pxx[field.band])
    totals, _ = np.histogram(theta, bins=edges, weights=signal)
    count, _ = np.histogram(theta, bins=edges)
    averages = np.divide(totals, count, out=np.zeros_like(totals), where=count > 0)
    n = max(float(averages.max()), 1e-12)
    return [
        {
            'direction_degrees': float((left + right) / 2),
            'count': int(population),
            'score': float(power / n) if population else None,
        }
        for left, right, population, power in zip(edges[:-1], edges[1:], count, averages, strict=True)
    ]


def spectrum_view(field, peaks):
    spectrum = np.log1p(field.pxx)
    vmax = float(np.percentile(spectrum[field.band], 99.6))
    band_map = (np.clip(spectrum / max(vmax, 1e-8), 0, 1) * 255).astype(np.uint8)
    rgba = np.empty((*spectrum.shape, 4), dtype=np.uint8)
    rgba[..., 0] = np.where(field.band, band_map, band_map * 0.25)
    rgba[..., 1] = np.where(field.band, band_map * 0.82, band_map * 0.25)
    rgba[..., 2] = np.where(field.band, band_map * 0.52, band_map * 0.28)
    rgba[..., 3] = 255
    for peak in peaks:
        column, row = peak['grid_index']
        for delta in (-4, -3, -2, 2, 3, 4):
            if 0 <= column + delta < spectrum.shape[1]:
                rgba[row, column + delta, :] = (125, 225, 214, 255)
            if 0 <= row + delta < spectrum.shape[0]:
                rgba[row + delta, column, :] = (125, 225, 214, 255)
    return rgba


def analyze_texture(surface, settings, case_scale):
    window = surface.crop(settings.bounds)
    support = support_summary(window.valid)
    if support['fraction'] < settings.minimum_support:
        raise InsufficientSupport(
            'The selected texture patch includes too much unmeasured area.',
            facts={'valid_fraction': support['fraction'], 'required': settings.minimum_support},
        )
    high_pass, residual = demean_and_window(window.pixels, window.valid, settings.subtract_plane)
    field = frequency_field(high_pass, settings.small_period, settings.large_period)
    peaks = find_peaks(field)
    for peak in peaks:
        peak['period_mm'] = peak['period_pixels'] * case_scale if case_scale else None
    report = {
        'settings': settings.as_json(),
        'support': support,
        'residual_rms': residual,
        'directional_profile': directional_profile(field),
        'peaks': peaks,
        'angle_reference': 'Canvas +x to +y on the display, modulo 180°',
    }
    return report, png_bytes(spectrum_view(field, peaks))
