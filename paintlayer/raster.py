"""Detector-safe image intake, storage, numerical encoding and view generation."""

from dataclasses import dataclass
import io
from pathlib import Path
import warnings

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from scipy import ndimage

from .errors import CapacityExceeded, ImageFormatError, InsufficientSupport
from .settings import MAX_SIDE, MAX_UPLOAD_BYTES, MAX_UPLOAD_PIXELS, MIN_IMAGE_EDGE

# Display-only local tone mapping for inspecting faint underdrawing. The transform is
# never written back into the stored rasters and uses a single wide local-average
# scale, so it cannot produce the two-sided ringing of an edge-sharpening kernel.
DETAIL_CAP = 0.30
DETAIL_BACKGROUND_LEVEL = 0.60
DETAIL_SIGMA_MIN = 8.0
DETAIL_SIGMA_MAX = 400.0
DETAIL_STRENGTH_MIN = 1.0
DETAIL_STRENGTH_MAX = 2.2

DETECTOR_MODES = {'L', 'I', 'I;16', 'I;16L', 'I;16B', 'I;16N'}
VISIBLE_MODES = {'RGB', 'RGBA', 'L', 'LA', 'P', 'CMYK'}
FORMAT_NAMES = {'TIFF', 'PNG', 'JPEG'}
TYPE_MAP = {'TIFF': 'image/tiff', 'PNG': 'image/png', 'JPEG': 'image/jpeg'}


@dataclass(frozen=True)
class Decoded:
    pixels: np.ndarray
    support: np.ndarray
    metadata: dict


@dataclass(frozen=True)
class Surface:
    pixels: np.ndarray
    valid: np.ndarray

    @property
    def height(self):
        return self.valid.shape[0]

    @property
    def width(self):
        return self.valid.shape[1]

    @property
    def extent(self):
        return self.width, self.height

    def count(self):
        return int(np.count_nonzero(self.valid))

    def crop(self, bounds):
        x, y, width, height = bounds
        return Surface(self.pixels[y:y + height, x:x + width], self.valid[y:y + height, x:x + width])


def png_bytes(array, *, compress=3):
    with io.BytesIO() as output:
        Image.fromarray(np.ascontiguousarray(array)).save(output, format='PNG', compress_level=compress)
        return output.getvalue()


def tiff_bytes(array):
    with io.BytesIO() as output:
        Image.fromarray(np.ascontiguousarray(array)).save(output, format='TIFF', compression='tiff_deflate')
        return output.getvalue()


def as_bytes(array):
    with io.BytesIO() as output:
        np.save(output, array, allow_pickle=False)
        return output.getvalue()


def from_blob(path):
    with Path(path).open('rb') as source:
        array = np.load(source, allow_pickle=False)
    return array


def image_from_bytes(payload):
    with Image.open(io.BytesIO(payload)) as image:
        image.load()
        return np.array(image)


def assert_extent(width, height):
    if min(width, height) < MIN_IMAGE_EDGE:
        raise ImageFormatError(f'Each image side must have at least {MIN_IMAGE_EDGE} samples.')
    if max(width, height) > MAX_SIDE or width * height > MAX_UPLOAD_PIXELS:
        raise CapacityExceeded('Image dimensions exceed the import allowance.')


def read_frame(payload, role, frame):
    if len(payload) == 0:
        raise ImageFormatError('This image file is empty.')
    if len(payload) > MAX_UPLOAD_BYTES:
        raise CapacityExceeded('This image exceeds the upload limit.')
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(payload)) as image:
                return decode_image(image, role, frame)
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise CapacityExceeded('Image dimensions exceed the safe decoding limit.') from exc
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise ImageFormatError('The image could not be decoded. Check the file format and contents.') from exc


def decode_image(image, role, frame):
    media_format = image.format
    if media_format not in FORMAT_NAMES:
        raise ImageFormatError('Use a TIFF, PNG or JPEG file.')
    if media_format == 'JPEG' and role != 'visible':
        raise ImageFormatError('Lossy JPEG is accepted for compatible visible photographs.')
    if media_format != 'TIFF' and getattr(image, 'is_animated', False):
        raise ImageFormatError('Import a still PNG or choose a page from a TIFF.')
    if frame >= 48:
        raise CapacityExceeded('Choose among the first 48 pages of a TIFF.')
    if frame:
        try:
            image.seek(frame)
        except EOFError as exc:
            raise ImageFormatError('The selected TIFF page does not exist.', field='page') from exc
    width, height = image.size
    assert_extent(width, height)
    if role == 'visible':
        decoded, support, mode_facts = decode_photo(image)
    elif role == 'mask':
        decoded, support, mode_facts = decode_mask(image)
    else:
        decoded, support, mode_facts = decode_detector(image, media_format)
    return Decoded(
        pixels=decoded,
        support=support,
        metadata={
            'format': media_format,
            'content_type': TYPE_MAP[media_format],
            'page': frame + 1,
            'width': int(decoded.shape[1]),
            'height': int(decoded.shape[0]),
            'source_width': width,
            'source_height': height,
            'mode': image.mode,
        } | mode_facts,
    )


def decode_photo(image):
    if image.mode not in VISIBLE_MODES:
        raise ImageFormatError('Use an RGB or 8-bit grayscale photograph.')
    orientation = image.getexif().get(274, 1)
    oriented = ImageOps.exif_transpose(image)
    rgba = np.asarray(oriented.convert('RGBA'), dtype=np.uint8)
    support = rgba[..., 3] > 0
    rgb = rgba[..., :3].copy()
    rgb[~support] = 0
    return rgb, support.copy(), {
        'bit_depth': 8,
        'orientation_applied': int(orientation) if isinstance(orientation, int) else 1,
        'alpha_validity': bool(np.any(~support)),
    }


def decode_mask(image):
    if image.mode == '1':
        data = np.asarray(image, dtype=bool)
    elif image.mode in DETECTOR_MODES:
        data = np.asarray(image) > 0
    else:
        raise ImageFormatError('Acquisition masks must be grayscale. Nonzero samples mark usable pixels.')
    if not np.any(data):
        raise ImageFormatError('The acquisition mask marks no usable pixels.')
    return data.astype(np.uint8), np.ones(data.shape, dtype=bool), {
        'bit_depth': 1,
        'accepted_pixels': int(np.count_nonzero(data)),
    }


def decode_detector(image, media_format):
    if image.mode not in DETECTOR_MODES:
        raise ImageFormatError('Detector, dark and flat images require a single unsigned 8-bit or 16-bit channel.')
    samples = np.asarray(image)
    inferred_depth = 8 if image.mode == 'L' else 16
    if media_format == 'TIFF':
        bits = image.tag_v2.get(258, inferred_depth)
        sample_format = image.tag_v2.get(339, 1)
        photometric = image.tag_v2.get(262, 1)
        if isinstance(bits, tuple):
            bits = bits[0] if len(bits) == 1 else None
        if isinstance(sample_format, tuple):
            sample_format = sample_format[0] if len(sample_format) == 1 else None
        if bits not in (8, 16) or sample_format != 1 or photometric != 1:
            raise ImageFormatError('Use unsigned 8-bit or 16-bit TIFF with MinIsBlack grayscale interpretation.')
        inferred_depth = bits
    ceiling = (1 << inferred_depth) - 1
    if samples.ndim != 2 or samples.min() < 0 or samples.max() > ceiling:
        raise ImageFormatError('Detector values fall outside the unsigned range of the declared image depth.')
    pixels = np.array(samples, dtype=np.uint8 if inferred_depth == 8 else np.uint16, copy=True)
    return pixels, np.ones(pixels.shape, dtype=bool), {
        'bit_depth': inferred_depth,
        'full_scale': ceiling,
        'raw_min': int(pixels.min()),
        'raw_max': int(pixels.max()),
    }


def support_summary(valid):
    count = int(np.count_nonzero(valid))
    size = valid.size
    if count:
        rows, columns = np.where(valid)
        x0, y0 = int(columns.min()), int(rows.min())
        width, height = int(columns.max()) - x0 + 1, int(rows.max()) - y0 + 1
        bbox = [x0, y0, width, height]
    else:
        bbox = None
    return {'valid': count, 'total': int(size), 'fraction': count / size, 'bounds': bbox}


def display_range(samples, support, percentiles=(1, 99)):
    values = samples[support & np.isfinite(samples)]
    if values.size == 0:
        return 0.0, 1.0
    low, high = (float(value) for value in np.percentile(values, percentiles))
    if low >= high:
        offset = max(0.5, abs(low) * 0.01)
        return low - offset, high + offset
    return low, high


def histogram(samples, support, bounds=None, bins=32):
    values = samples[support & np.isfinite(samples)]
    if values.size == 0:
        return {'counts': [], 'edges': [], 'under': 0, 'over': 0, 'valid': 0}
    low, high = bounds or display_range(samples, support, (0, 100))
    result, edges = np.histogram(values, bins=bins, range=(low, high))
    return {
        'counts': result.tolist(),
        'edges': edges.tolist(),
        'under': int(np.count_nonzero(values < low)),
        'over': int(np.count_nonzero(values > high)),
        'valid': int(values.size),
    }


def sampled_values(values, valid, *, minimum=1):
    finite = valid & np.isfinite(values)
    selected = values[finite]
    if selected.size < minimum:
        raise InsufficientSupport(
            'Too few usable samples remain inside this selection.',
            facts={'valid': int(selected.size), 'required': minimum},
        )
    return selected


def radiograph_rgba(surface, *, limits, gamma=1.0, negative=False):
    low, high = limits
    signal = np.zeros(surface.valid.shape, dtype=np.float32)
    accepted = surface.valid & np.isfinite(surface.pixels)
    signal[accepted] = surface.pixels[accepted]
    signal -= low
    signal /= high - low
    np.clip(signal, 0, 1, out=signal)
    np.power(signal, 1 / gamma, out=signal)
    if negative:
        signal = 1 - signal
    gray = np.asarray(np.rint(signal * 255), dtype=np.uint8)
    result = np.empty((*gray.shape, 4), dtype=np.uint8)
    result[..., :3] = gray[..., None]
    result[..., 3] = accepted.astype(np.uint8) * 255
    result[~accepted] = 0
    return result


def local_detail_surface(surface, *, limits, sigma, strength):
    """Return a display-only Surface with gentle local contrast redistribution.

    The working rasters are not modified: the returned pixels are re-expressed in the
    same measurement units as ``limits`` so the regular window and gamma pipeline
    renders them. A single wide local average balances slow background changes, and
    the detail layer passes through a soft saturating curve. Faint drawing therefore
    keeps nearly linear gain while strong boundaries are capped to a one-sided lobe,
    which avoids the halo of an outline drawn around an edge. Enhancement fades to
    nothing beside unsupported samples so masked boundaries are never embellished.
    """
    low, high = limits
    span = max(high - low, 1e-12)
    accepted = surface.valid & np.isfinite(surface.pixels)
    if not np.any(accepted):
        return surface
    windowed = np.zeros(surface.valid.shape, dtype=np.float32)
    windowed[accepted] = (surface.pixels[accepted].astype(np.float32) - low) / span
    mask = accepted.astype(np.float32)
    # A normalized box average gives the same slow background estimate as a Gaussian
    # but runs in linear time; the soft-capped detail layer makes the kernel shape
    # indistinguishable at visible edges while preserving the single-lobe response.
    diameter = max(3, int(round(4.0 * sigma)) | 1)
    weights = ndimage.uniform_filter(mask, diameter, mode='constant', cval=0.0)
    weights = np.maximum(weights, 1e-6)
    local_mean = ndimage.uniform_filter(
        windowed * mask, diameter, mode='constant', cval=0.0,
    ) / weights
    anchor = float(windowed[accepted].mean())
    detail = windowed - local_mean
    shaped_detail = DETAIL_CAP * np.tanh(strength * detail / DETAIL_CAP)
    softened_background = anchor + DETAIL_BACKGROUND_LEVEL * (local_mean - anchor)
    remapped = softened_background + shaped_detail
    if accepted.all():
        feather = np.ones(accepted.shape, dtype=np.float32)
    else:
        distance = ndimage.distance_transform_edt(accepted)
        feather = np.clip(distance / sigma, 0.0, 1.0) ** 2
    blended = anchor + (1.0 - feather) * (windowed - anchor) + feather * (remapped - anchor)
    blended = np.clip(blended, 0.0, 1.0)
    restored = np.zeros(surface.valid.shape, dtype=np.float32)
    restored[accepted] = low + blended[accepted] * span
    return Surface(restored, surface.valid)


def photo_rgba(surface):
    result = np.empty((*surface.valid.shape, 4), dtype=np.uint8)
    result[..., :3] = surface.pixels
    result[..., 3] = surface.valid.astype(np.uint8) * 255
    result[~surface.valid] = 0
    return result


def color_indexed(indices, palette, *, valid=None):
    result = np.asarray(palette, dtype=np.uint8)[indices]
    if valid is not None:
        result[~valid] = 0
    return result


def constrained_preview(rgba, edge):
    height, width = rgba.shape[:2]
    if max(width, height) <= edge:
        return png_bytes(rgba)
    image = Image.fromarray(rgba)
    image.thumbnail((edge, edge), resample=Image.Resampling.LANCZOS)
    with io.BytesIO() as output:
        image.save(output, 'PNG', compress_level=3)
        return output.getvalue()
