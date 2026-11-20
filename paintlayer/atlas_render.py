"""Produce a print-readable raster contact sheet and PDF from measured page layout."""

from dataclasses import dataclass
import io
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .evidence import scale_mark_value

INK = (34, 43, 42)
PALE = (246, 242, 233)
FRAME = (192, 182, 165)
SAND = (222, 215, 202)
ACCENT = (115, 69, 43)


@dataclass(frozen=True)
class PlateBlock:
    image: object
    title: str
    note: str
    label: str
    canvas_bounds: tuple
    scale: float | None
    is_synthetic: bool

    @property
    def extent(self):
        return self.canvas_bounds[2:]


def measure_line(text, font, limit):
    if font.getlength(text) <= limit:
        return text
    shortened = text
    while shortened and font.getlength(shortened + '…') > limit:
        shortened = shortened[:-1]
    return shortened + '…' if shortened else ''


def line_chunks(text, font, max_width, max_rows):
    result = []
    active = ''
    for word in text.split():
        combined = f'{active} {word}' if active else word
        if font.getlength(combined) <= max_width:
            active = combined
            continue
        if active:
            result.append(active)
            if len(result) >= max_rows:
                break
        active = measure_line(word, font, max_width)
    else:
        if active:
            result.append(active)
    if len(result) == max_rows and ' '.join(result).removesuffix('…') != ' '.join(text.split()):
        result[-1] = measure_line(result[-1] + '…', font, max_width)
    return result[:max_rows]


def format_size(allocation, case_scale):
    x_mm, y_mm = allocation.printed_extent_mm
    if case_scale is None:
        return f'Printed image {x_mm:.1f} × {y_mm:.1f} mm'
    return f'Reproduced {x_mm:.1f} × {y_mm:.1f} mm'


def print_check(draw, x, y, plan, fonts):
    length = plan.format.dim(20)
    width = max(1, plan.format.dim(0.25))
    tick = plan.format.dim(1.2)
    for px in (x, x + length):
        draw.line((px, y - tick, px, y + tick), fill=INK, width=width)
    draw.line((x, y, x + length, y), fill=INK, width=width)
    label = '20 mm print check'
    draw.text((x + length + plan.format.dim(2), y - plan.format.dim(1.8)), label, font=fonts['small'], fill=INK)


def page_chrome(image, case, name, plan, page, fonts):
    painter = ImageDraw.Draw(image)
    left = plan.inner.x
    right = plan.inner.right
    top = plan.inner.y
    if case['synthetic']:
        kicker = 'PAINTLAYER / SYNTHETIC TRAINING EXAMINATION'
    else:
        kicker = 'PAINTLAYER / CONSERVATION EXAMINATION'
    painter.text((left, top), kicker, fill=ACCENT, font=fonts['kicker'])
    title = measure_line(name, fonts['title'], plan.inner.width)
    painter.text((left, top + plan.format.dim(5)), title, fill=INK, font=fonts['title'])
    metadata = ' • '.join(value for value in (case['title'], case['accession'], case['collection']) if value)
    painter.text(
        (left, top + plan.format.dim(17)),
        measure_line(metadata, fonts['label'], plan.inner.width),
        fill=INK,
        font=fonts['label'],
    )
    y = plan.body.y - plan.format.dim(3)
    painter.line((left, y, right, y), fill=FRAME, width=max(1, plan.format.dim(0.2)))
    footer_y = plan.body.bottom + plan.format.dim(5)
    print_check(painter, left, footer_y, plan, fonts)
    page_number = f'{page.number} / {len(plan.pages)}'
    painter.text((right, footer_y), page_number, font=fonts['label'], fill=INK, anchor='rm')
    if plan.print_scale:
        text = f'Artwork ratio {plan.print_scale:.3g}×'
    else:
        text = 'Relative comparison scale'
    painter.text((right, footer_y + plan.format.dim(4)), text, font=fonts['small'], fill=INK, anchor='rm')
    return painter


def missing_background(rect):
    col = np.arange(rect.width)[None, :] // 9
    row = np.arange(rect.height)[:, None] // 9
    gray = ((row + col) % 2) * 17 + 221
    monochrome = np.broadcast_to(gray[..., None], (rect.height, rect.width, 3))
    return Image.fromarray(monochrome.astype(np.uint8))


def paste_plate(image, block, rect, paper):
    size = rect.width, rect.height
    resample = Image.Resampling.BICUBIC if rect.width > block.image.width else Image.Resampling.LANCZOS
    plate = block.image.resize(size, resample)
    paper_tile = missing_background(rect)
    paper_tile.paste(plate, mask=plate.getchannel('A'))
    image.paste(paper_tile, (rect.x, rect.y))
    painter = ImageDraw.Draw(image)
    painter.rectangle((rect.x, rect.y, rect.right - 1, rect.bottom - 1), outline=FRAME, width=1)
    if block.scale is not None and min(size) > paper.dim(30):
        paint_image_scale(painter, rect, block, paper)


def paint_image_scale(painter, rect, block, paper):
    canvas_width = block.canvas_bounds[2]
    nominal = scale_mark_value(canvas_width, block.scale)
    fraction = nominal / (block.scale * canvas_width)
    if fraction <= 0 or fraction > 0.23:
        return
    mark = max(1, int(round(rect.width * fraction)))
    left = rect.x + paper.dim(3)
    bottom = rect.bottom - paper.dim(3)
    font = ImageFont.load_default(size=paper.dim(2.4))
    label = f'{nominal:g} mm'
    margin = paper.dim(1.5)
    right = left + max(mark, math.ceil(font.getlength(label)))
    painter.rectangle(
        (left - margin, bottom - paper.dim(4.6), right + margin, bottom + margin),
        fill=(39, 44, 43),
    )
    painter.line((left, bottom, left + mark, bottom), fill=PALE, width=max(1, paper.dim(0.25)))
    painter.text((left, bottom - paper.dim(3.5)), label, font=font, fill=PALE)


def block_caption(draw, allocate, block, plan, fonts):
    x = allocate.slot.x + plan.format.dim(1)
    caption_y = allocate.raster.bottom + plan.format.dim(2)
    width = allocate.slot.width - plan.format.dim(2)
    title = f'{allocate.order:02d}  {block.title}'
    draw.text((x, caption_y), measure_line(title, fonts['label'], width), font=fonts['label'], fill=INK)
    cx, cy, cw, ch = block.canvas_bounds
    context = f'{block.label} • {cw}×{ch} px from ({cx}, {cy})'
    y = caption_y + plan.format.dim(4.2)
    draw.text((x, y), measure_line(context, fonts['small'], width), font=fonts['small'], fill=INK)
    y += plan.format.dim(3.4)
    draw.text((x, y), measure_line(format_size(allocate, block.scale), fonts['small'], width), font=fonts['small'], fill=ACCENT)
    if block.note:
        excerpt = line_chunks(block.note, fonts['small'], width, 2)
        y += plan.format.dim(3.5)
        step = plan.format.dim(3.2)
        for offset, line in enumerate(excerpt):
            draw.text((x, y + step * offset), line, font=fonts['small'], fill=INK)


def contact_sheet_pages(case, title, plan, images):
    format = plan.format
    fonts = {
        'kicker': ImageFont.load_default(size=format.dim(2.7)),
        'title': ImageFont.load_default(size=format.dim(6)),
        'label': ImageFont.load_default(size=format.dim(3.1)),
        'small': ImageFont.load_default(size=format.dim(2.45)),
    }
    for page in plan.pages:
        output = Image.new('RGB', (plan.full.width, plan.full.height), PALE)
        painter = page_chrome(output, case, title, plan, page, fonts)
        for allocation in page.entries:
            block = images[allocation.key]
            paste_plate(output, block, allocation.raster, format)
            block_caption(painter, allocation, block, plan, fonts)
        yield output


def encode_pdf(images, dpi):
    first, *remaining = images
    with io.BytesIO() as output:
        first.save(output, 'PDF', save_all=True, append_images=remaining, resolution=float(dpi), quality=93)
        return output.getvalue()


def encode_preview(image):
    thumb = image.copy()
    thumb.thumbnail((960, 960), Image.Resampling.LANCZOS)
    with io.BytesIO() as output:
        thumb.save(output, 'PNG', compress_level=4)
        return output.getvalue()
