"""Inspect a local image container before committing a selected frame to the case."""

import base64
import io
import warnings

import numpy as np
from PIL import Image, ImageDraw, ImageFont, UnidentifiedImageError

from . import fields
from .errors import CapacityExceeded, ExamError, ImageFormatError
from .raster import (
    FORMAT_NAMES, Surface, decode_image, display_range, histogram, photo_rgba, png_bytes,
    radiograph_rgba, support_summary,
)
from .settings import MAX_UPLOAD_BYTES, SOURCE_ROLES

INSPECT_LIMIT = 48
INSPECT_PAGE_WINDOW = 8
PREVIEW_SIDE = 190
PREVIEW_HEIGHT = 162


def upload_bytes(upload):
    if upload is None:
        raise ExamError('Choose a local image file.', field='image')
    if upload.content_length is not None and upload.content_length > MAX_UPLOAD_BYTES:
        raise CapacityExceeded('This image exceeds the upload allowance.')
    data = upload.stream.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise ImageFormatError('The selected file is empty.')
    if len(data) > MAX_UPLOAD_BYTES:
        raise CapacityExceeded('This image exceeds the upload allowance.')
    return data


def request_options(form):
    role = fields.choice(form.get('role'), field='role', options=SOURCE_ROLES)
    try:
        first = int(form.get('page', 1))
    except (TypeError, ValueError) as exc:
        raise ExamError('Enter a valid starting page.', field='page') from exc
    start = fields.integer(first, field='page', minimum=1, maximum=INSPECT_LIMIT)
    return role, start


def visual_metadata(decoded, role):
    support = support_summary(decoded.support)
    shape = decoded.metadata
    metadata = {
        'dimensions': [shape['width'], shape['height']],
        'raw_dimensions': [shape['source_width'], shape['source_height']],
        'depth': shape['bit_depth'],
        'usable_fraction': support['fraction'],
    }
    if role == 'visible':
        metadata['orientation'] = shape['orientation_applied']
    elif role == 'mask':
        metadata['marked_pixels'] = shape['accepted_pixels']
    else:
        limits = display_range(decoded.pixels, decoded.support, (2, 98))
        metadata['preview_limits'] = list(limits)
        metadata['distribution'] = histogram(decoded.pixels, decoded.support, limits, bins=20)
        metadata['raw_range'] = [shape['raw_min'], shape['raw_max']]
        ceiling = shape['full_scale']
        metadata['ceiling_pixels'] = int(np.count_nonzero(decoded.pixels == ceiling))
    return metadata


def preview_frame(decoded, role, metadata):
    valid = decoded.support
    if role == 'visible':
        image = Image.fromarray(photo_rgba(Surface(decoded.pixels, valid)))
    elif role == 'mask':
        tint = np.zeros((*decoded.pixels.shape, 4), dtype=np.uint8)
        tint[decoded.pixels > 0] = (83, 192, 175, 255)
        image = Image.fromarray(tint)
    else:
        limits = metadata['preview_limits']
        image = Image.fromarray(radiograph_rgba(Surface(decoded.pixels, valid), limits=limits))
    image.thumbnail((PREVIEW_SIDE - 12, PREVIEW_HEIGHT - 36), Image.Resampling.LANCZOS)
    return image


def error_row(index, source_mode, source_size, error):
    return {
        'page': index,
        'mode': source_mode,
        'dimensions': list(source_size),
        'usable': False,
        'reason': error.message,
    }


def frame_row(container, role, page):
    original_mode = container.mode
    original_size = container.size
    try:
        decoded = decode_image(container, role, page - 1)
    except (CapacityExceeded, ImageFormatError) as error:
        return error_row(page, original_mode, original_size, error), None
    except (OSError, ValueError, SyntaxError):
        error = ImageFormatError('The frame is readable as metadata but its pixels could not be decoded.')
        return error_row(page, original_mode, original_size, error), None
    metadata = visual_metadata(decoded, role)
    return {
        'page': page,
        'mode': original_mode,
        'usable': True,
    } | metadata, preview_frame(decoded, role, metadata)


def sheet_grid(rows, images):
    columns = min(len(rows), 4)
    height = ((len(rows) + columns - 1) // columns) * PREVIEW_HEIGHT
    width = columns * PREVIEW_SIDE
    canvas = Image.new('RGB', (width, height), (32, 38, 38))
    painter = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=11)
    for index, (record, image) in enumerate(zip(rows, images, strict=True)):
        x = (index % columns) * PREVIEW_SIDE
        y = (index // columns) * PREVIEW_HEIGHT
        painter.rectangle((x + 3, y + 3, x + PREVIEW_SIDE - 3, y + PREVIEW_HEIGHT - 3), fill=(22, 28, 27))
        if image is not None:
            paste_x = x + (PREVIEW_SIDE - image.width) // 2
            paste_y = y + (PREVIEW_HEIGHT - 27 - image.height) // 2
            canvas.paste(image, (paste_x, paste_y), mask=image.getchannel('A'))
        else:
            painter.text((x + 14, y + 62), 'Frame unavailable for this role', font=font, fill=(235, 166, 125))
        w, h = record['dimensions']
        caption = f'Page {record["page"]} · {w} × {h}'
        painter.text((x + 11, y + PREVIEW_HEIGHT - 22), caption, font=font, fill=(219, 232, 219))
    return canvas


def inspect_pages(data, role, start):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error', Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as container:
                return read_page_window(container, role, start)
    except (Image.DecompressionBombWarning, Image.DecompressionBombError) as exc:
        raise CapacityExceeded('Image dimensions exceed the safe inspection limit.') from exc
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
        raise ImageFormatError('This file could not be opened for page inspection.') from exc


def read_page_window(container, role, start):
    file_format = container.format
    if file_format not in FORMAT_NAMES:
        raise ImageFormatError('Use a TIFF, PNG or JPEG container.')
    if file_format != 'TIFF' and start != 1:
        raise ImageFormatError('Choose page one for a still PNG or JPEG.', field='page')
    upper = min(start + INSPECT_PAGE_WINDOW - 1, INSPECT_LIMIT) if file_format == 'TIFF' else start
    rows, previews = [], []
    for page in range(start, upper + 1):
        try:
            container.seek(page - 1)
        except EOFError:
            break
        row, preview = frame_row(container, role, page)
        rows.append(row)
        previews.append(preview)
    if not rows:
        raise ImageFormatError('The selected starting page is not present.', field='page')
    more = False
    if upper < INSPECT_LIMIT and file_format == 'TIFF' and len(rows) == INSPECT_PAGE_WINDOW:
        try:
            container.seek(upper)
        except EOFError:
            pass
        else:
            more = True
    contact = sheet_grid(rows, previews)
    sheet = png_bytes(np.asarray(contact))
    return {
        'format': file_format,
        'role': role,
        'first_page': start,
        'next_page': upper + 1 if more else None,
        'inspection_limit': INSPECT_LIMIT,
        'pages': rows,
        'contact_sheet': 'data:image/png;base64,' + base64.b64encode(sheet).decode('ascii'),
    }
