"""Render actual-size tiled print pages with rulers, overlap bands and match marks."""

from dataclasses import dataclass
import io
import math

from PIL import Image, ImageDraw, ImageFont

from .atlas_layout import Rectangle
from .atlas_render import ACCENT, FRAME, INK, PALE, SAND, missing_background, print_check

HATCH_LINE = (150, 96, 52)
BAND_TINT = (246, 222, 188, 26)
CUT = (86, 60, 40)
MARK = (43, 64, 78)
PDF_POINTS_PER_MM = 72 / 25.4


@dataclass(frozen=True)
class TileSource:
    image: object
    title: str
    note: str
    label: str
    canvas_bounds: tuple
    scale: float
    is_synthetic: bool


def font_set(paper):
    return {
        'kicker': ImageFont.load_default(size=paper.dim(2.6)),
        'title': ImageFont.load_default(size=paper.dim(4.6)),
        'label': ImageFont.load_default(size=paper.dim(3.0)),
        'small': ImageFont.load_default(size=paper.dim(2.4)),
        'tiny': ImageFont.load_default(size=paper.dim(2.0)),
    }


def clipped(text, font, width):
    if font.getlength(text) <= width:
        return text
    shortened = text
    while shortened and font.getlength(shortened + '...') > width:
        shortened = shortened[:-1]
    return shortened + '...' if shortened else ''


def wrap(text, font, width, rows):
    result = []
    active = ''
    for word in text.split():
        combined = f'{active} {word}' if active else word
        if font.getlength(combined) <= width:
            active = combined
        else:
            if active:
                result.append(active)
            active = word
            if len(result) == rows:
                break
    else:
        if active:
            result.append(active)
    return result[:rows]


def ruler_steps(pixels_per_canvas, minimum_mm):
    needed = minimum_mm / max(1e-9, pixels_per_canvas)
    for step in (10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000):
        if step >= needed:
            return step
    return 5000


def draw_rulers(painter, plan, window, fonts):
    """Canvas-coordinate rulers on the two inner edges of the plate."""
    paper = plan.format
    body = plan.body
    x0, y0, width, height = window
    k = plan.pixels_per_canvas_pixel
    step = ruler_steps(plan.scale, 13.0)
    gap = max(1, paper.dim(0.25))
    start = int(math.ceil(x0 / step) * step)
    while start <= x0 + width:
        x = body.x + int(round((start - x0) * k))
        if x > body.right:
            break
        painter.line((x, body.y - gap * 3, x, body.y), fill=INK, width=gap)
        painter.text((x, body.y - paper.dim(4.2)), str(start), font=fonts['tiny'], fill=INK, anchor='ms')
        start += step
    start = int(math.ceil(y0 / step) * step)
    while start <= y0 + height:
        y = body.y + int(round((start - y0) * k))
        if y > body.bottom:
            break
        painter.line((body.x - gap * 3, y, body.x, y), fill=INK, width=gap)
        painter.text((body.x - gap * 4, y), str(start), font=fonts['tiny'], fill=INK, anchor='rm')
        start += step
    painter.text((body.x, body.y - paper.dim(8.2)), 'canvas x / canvas y, in pixels', font=fonts['tiny'], fill=ACCENT)


def band_overlay(rect, paper):
    overlay = Image.new('RGBA', (rect.width, rect.height), BAND_TINT)
    draw = ImageDraw.Draw(overlay)
    step = max(3, paper.dim(4.0))
    width = max(1, paper.dim(0.22))
    for offset in range(-rect.height, rect.width + rect.height, step):
        draw.line((offset, 0, offset + rect.height, rect.height), fill=HATCH_LINE + (46,), width=width)
    return overlay


def paint_band(image, painter, plan, band, page_rect, fonts):
    """Draw a shared band, its dashed join line and matching crosses."""
    paper = plan.format
    overlay = band_overlay(page_rect, paper)
    image.paste(overlay, (page_rect.x, page_rect.y), overlay)
    painter.rectangle((page_rect.x, page_rect.y, page_rect.right, page_rect.bottom), outline=HATCH_LINE, width=max(1, paper.dim(0.3)))
    cut = band['cut']
    k = plan.pixels_per_canvas_pixel
    dash = max(2, paper.dim(2.6))
    if band['axis'] == 'x':
        cut_x = page_rect.x + int(round((cut - band['canvas'][0]) * k))
        cut_x = min(max(cut_x, page_rect.x), page_rect.right)
        for y in range(page_rect.y, page_rect.bottom, dash):
            painter.line((cut_x, y, cut_x, min(y + paper.dim(1.4), page_rect.bottom)), fill=CUT, width=max(1, paper.dim(0.35)))
        painter.text((cut_x + paper.dim(1.0), page_rect.bottom - paper.dim(4.2)), 'join', font=fonts['tiny'], fill=CUT)
    else:
        cut_y = page_rect.y + int(round((cut - band['canvas'][1]) * k))
        cut_y = min(max(cut_y, page_rect.y), page_rect.bottom)
        for x in range(page_rect.x, page_rect.right, dash):
            painter.line((x, cut_y, min(x + paper.dim(1.4), page_rect.right), cut_y), fill=CUT, width=max(1, paper.dim(0.35)))
        painter.text((page_rect.x + paper.dim(1.0), cut_y - paper.dim(3.2)), 'join', font=fonts['tiny'], fill=CUT)
    size = paper.dim(2.4)
    for point in plan.mark_canvas_points(band):
        if band['axis'] == 'x':
            x = page_rect.x + page_rect.width // 2
            y = page_rect.y + int(round((point[1] - band['canvas'][1]) * k))
        else:
            x = page_rect.x + int(round((point[0] - band['canvas'][0]) * k))
            y = page_rect.y + page_rect.height // 2
        painter.line((x - size, y, x + size, y), fill=MARK, width=max(1, size // 6))
        painter.line((x, y - size, x, y + size), fill=MARK, width=max(1, size // 6))
        painter.ellipse((x - size, y - size, x + size, y + size), outline=MARK, width=max(1, size // 8))
        painter.text((x + size * 1.5, y - size * 1.6), f'{point[0]},{point[1]}', font=fonts['tiny'], fill=MARK)
    share = band['canvas'][2] if band['axis'] == 'x' else band['canvas'][3]
    return round(share * plan.scale)


def neighbour_note(painter, plan, notes, fonts):
    if not notes:
        return
    paper = plan.format
    y = plan.body.bottom + paper.dim(8.0)
    for index, line in enumerate(notes[:4]):
        offset = index % 2
        painter.text(
            (plan.body.x + offset * (plan.body.width // 2), y + (index // 2) * paper.dim(3.2)),
            clipped(line, fonts['tiny'], plan.body.width // 2 - paper.dim(4)),
            font=fonts['tiny'],
            fill=ACCENT,
        )


def assembly_map(painter, plan, row, column, fonts):
    paper = plan.format
    cell = min(paper.dim(3.6), max(paper.dim(2.2), plan.inner.width // 24))
    gap = max(1, paper.dim(0.6))
    width = plan.columns * cell + (plan.columns - 1) * gap
    height = plan.rows * cell + (plan.rows - 1) * gap
    right = plan.inner.right
    top = plan.inner.y + paper.dim(4.2)
    left = right - width
    if left < plan.inner.x + plan.inner.width * 0.55 or top + height > plan.body.y - paper.dim(2):
        return
    for r in range(plan.rows):
        for c in range(plan.columns):
            x = left + c * (cell + gap)
            y = top + r * (cell + gap)
            selected = (r, c) == (row, column)
            painter.rectangle((x, y, x + cell, y + cell), fill=MARK if selected else SAND, outline=FRAME, width=1)
            painter.text(
                (x + cell / 2, y + cell / 2),
                str(plan.page_number(r, c)),
                font=fonts['tiny'],
                fill=PALE if selected else INK,
                anchor='mm',
            )
    painter.text((left, top - paper.dim(3.2)), 'assembly map', font=fonts['tiny'], fill=ACCENT)


def page_header(painter, plan, source, row, column, fonts):
    paper = plan.format
    left = plan.inner.x
    kicker = 'PAINTLAYER / SYNTHETIC TRAINING EXAMINATION' if source.is_synthetic else 'PAINTLAYER / CONSERVATION EXAMINATION'
    painter.text((left, plan.inner.y), kicker, fill=ACCENT, font=fonts['kicker'])
    painter.text((left, plan.inner.y + paper.dim(4.4)), clipped(source.title, fonts['title'], plan.inner.width), fill=INK, font=fonts['title'])
    subtitle = (
        f'Actual-size tiled print - page {plan.page_number(row, column)}/{plan.pages} '
        f'(row {row + 1}/{plan.rows}, column {column + 1}/{plan.columns}) - '
        f'assembled {plan.content_mm[0]:.0f} x {plan.content_mm[1]:.0f} mm'
    )
    painter.text((left, plan.inner.y + paper.dim(10.2)), subtitle, fill=INK, font=fonts['label'])
    x0, y0, width, height = plan.window(row, column)
    detail = f'{source.label} - canvas window ({x0}, {y0}) {width}x{height} px - {plan.scale:g} mm per canvas px'
    painter.text((left, plan.inner.y + paper.dim(14.2)), clipped(detail, fonts['small'], plan.inner.width), fill=INK, font=fonts['small'])
    assembly_map(painter, plan, row, column, fonts)


def draw_focus(painter, plan, row, column, fonts):
    """Dashed outline of the protected area wherever it falls on this page."""
    if plan.focus_bounds is None:
        return None
    x, y, width, height = plan.focus_bounds
    x0, y0, page_width, page_height = plan.window(row, column)
    left = max(x, x0)
    top = max(y, y0)
    right = min(x + width, x0 + page_width)
    bottom = min(y + height, y0 + page_height)
    if right - left < 1 or bottom - top < 1:
        return None
    paper = plan.format
    k = plan.pixels_per_canvas_pixel
    rect = (
        plan.body.x + int(round((left - x0) * k)),
        plan.body.y + int(round((top - y0) * k)),
        plan.body.x + int(round((right - x0) * k)),
        plan.body.y + int(round((bottom - y0) * k)),
    )
    dash = max(2, paper.dim(2.8))
    width_px = max(1, paper.dim(0.4))
    for x_start in range(rect[0], rect[2], dash):
        painter.line((x_start, rect[1], min(x_start + paper.dim(1.6), rect[2]), rect[1]), fill=ACCENT, width=width_px)
        painter.line((x_start, rect[3], min(x_start + paper.dim(1.6), rect[2]), rect[3]), fill=ACCENT, width=width_px)
    for y_start in range(rect[1], rect[3], dash):
        painter.line((rect[0], y_start, rect[0], min(y_start + paper.dim(1.6), rect[3])), fill=ACCENT, width=width_px)
        painter.line((rect[2], y_start, rect[2], min(y_start + paper.dim(1.6), rect[3])), fill=ACCENT, width=width_px)
    painter.text(
        (plan.body.x + paper.dim(1.0), plan.body.bottom + paper.dim(4.0)),
        f'protected area ({x}, {y}) {width}x{height} px, outlined in orange',
        font=fonts['tiny'],
        fill=ACCENT,
    )
    return rect


def tile_page(plan, source, row, column):
    paper = plan.format
    fonts = font_set(paper)
    output = Image.new('RGB', (plan.full.width, plan.full.height), PALE)
    painter = ImageDraw.Draw(output)
    page_header(painter, plan, source, row, column, fonts)

    x0, y0, width, height = plan.window(row, column)
    bx, by = source.canvas_bounds[0], source.canvas_bounds[1]
    rect_width = max(1, int(round(width * plan.pixels_per_canvas_pixel)))
    rect_height = max(1, int(round(height * plan.pixels_per_canvas_pixel)))
    rect = Rectangle(plan.body.x, plan.body.y, rect_width, rect_height)
    crop = source.image.crop((x0 - bx, y0 - by, x0 - bx + width, y0 - by + height))
    resample = Image.Resampling.BICUBIC if rect_width >= crop.width else Image.Resampling.LANCZOS
    plate = crop.resize((rect_width, rect_height), resample)
    tile = missing_background(rect)
    tile.paste(plate, mask=plate.getchannel('A'))
    output.paste(tile, (rect.x, rect.y))
    painter.rectangle((rect.x, rect.y, rect.right - 1, rect.bottom - 1), outline=FRAME, width=1)
    draw_rulers(painter, plan, (x0, y0, width, height), fonts)

    notes = []
    for band in plan.bands_for_page(row, column):
        page_rect = Rectangle(
            plan.body.x + int(round((band['canvas'][0] - x0) * plan.pixels_per_canvas_pixel)),
            plan.body.y + int(round((band['canvas'][1] - y0) * plan.pixels_per_canvas_pixel)),
            max(1, int(round(band['canvas'][2] * plan.pixels_per_canvas_pixel))),
            max(1, int(round(band['canvas'][3] * plan.pixels_per_canvas_pixel))),
        )
        shared = paint_band(output, painter, plan, band, page_rect, fonts)
        join = f"join {band['side']} - page {band['neighbour_page']} - shared {shared} mm - match crosses"
        notes.append(join)
    draw_focus(painter, plan, row, column, fonts)
    neighbour_note(painter, plan, notes, fonts)

    footer_y = plan.full.height - paper.dim(paper.margin_mm) - paper.dim(5.4)
    painter.line((plan.inner.x, footer_y - paper.dim(1.6), plan.inner.right, footer_y - paper.dim(1.6)), fill=FRAME, width=1)
    print_check(painter, plan.inner.x, footer_y - paper.dim(7.2), plan, fonts)
    note = source.note or 'Join with the pages named above. Print at 100%, never fit to page.'
    note_width = plan.inner.width - paper.dim(58)
    for index, line in enumerate(wrap(note, fonts['tiny'], note_width, 2)):
        painter.text((plan.inner.x, footer_y + index * paper.dim(2.9)), line, font=fonts['tiny'], fill=INK)
    painter.text(
        (plan.inner.right, footer_y + paper.dim(0.2)),
        f'{plan.content_mm[0]:.0f} x {plan.content_mm[1]:.0f} mm assembled',
        font=fonts['small'],
        fill=ACCENT,
        anchor='ra',
    )
    painter.text(
        (plan.inner.right, footer_y + paper.dim(4.6)),
        f'Canvas {source.canvas_bounds[2]}x{source.canvas_bounds[3]} px from ({bx}, {by}) - 1:1 artwork ratio',
        font=fonts['tiny'],
        fill=ACCENT,
        anchor='ra',
    )
    return output


def tile_pages(plan, source):
    for row in range(plan.rows):
        for column in range(plan.columns):
            yield tile_page(plan, source, row, column)


def page_marks(plan):
    return [
        {'page': plan.page_number(row, column), 'marks': plan.page_marks(row, column)}
        for row in range(plan.rows)
        for column in range(plan.columns)
    ]


def encode_preview(image):
    thumb = image.copy()
    thumb.thumbnail((1100, 1100), Image.Resampling.LANCZOS)
    with io.BytesIO() as output:
        thumb.save(output, 'PNG', compress_level=4)
        return output.getvalue()


def encode_pdf(pages, paper):
    """Write one JPEG image per page at its exact physical size.

    Pages are encoded one at a time, so a tiled print does not hold every
    raster and its PDF bytes in memory at once.
    """
    width_mm, height_mm = paper.dimensions_mm
    box = (width_mm * PDF_POINTS_PER_MM, height_mm * PDF_POINTS_PER_MM)
    output = io.BytesIO()
    output.write(b'%PDF-1.4\n%\xe2\xe3\xcf\xd3\n')
    offsets = {}
    order = []

    def write_object(body):
        number = len(order) + 1
        order.append(number)
        offsets[number] = output.tell()
        output.write(f'{number} 0 obj\n'.encode('ascii'))
        output.write(body)
        output.write(b'\nendobj\n')
        return number

    # The catalog and page tree are known only after the pages exist, so reserve
    # fixed-size slots and fill them in at the end.
    catalog = write_object(b'<< /Type /Catalog /Pages 2 0 R >>' + b' ' * 512)
    tree = write_object(b'<< /Type /Pages /Kids [] /Count 0 >>' + b' ' * 4096)
    page_ids = []
    for image in pages:
        with io.BytesIO() as jpeg:
            image.save(jpeg, 'JPEG', quality=92, subsampling=0)
            data = jpeg.getvalue()
        image_id = write_object(
            (
                f'<< /Type /XObject /Subtype /Image /Width {image.width} /Height {image.height} '
                f'/ColorSpace /DeviceRGB /BitsPerComponent 8 /Filter /DCTDecode /Length {len(data)} >>\nstream\n'
            ).encode('ascii')
            + data
            + b'\nendstream'
        )
        content = f'q {box[0]:.3f} 0 0 {box[1]:.3f} 0 0 cm /Im0 Do Q'.encode('ascii')
        content_id = write_object(f'<< /Length {len(content)} >>\nstream\n'.encode('ascii') + content + b'\nendstream')
        page_ids.append(
            write_object(
                (
                    f'<< /Type /Page /Parent {tree} 0 R /MediaBox [0 0 {box[0]:.3f} {box[1]:.3f}] '
                    f'/Resources << /XObject << /Im0 {image_id} 0 R >> >> /Contents {content_id} 0 R >>'
                ).encode('ascii')
            )
        )
    tree_body = f'<< /Type /Pages /Kids [{" ".join(f"{number} 0 R" for number in page_ids)}] /Count {len(page_ids)} >>'.encode('ascii')
    catalog_body = f'<< /Type /Catalog /Pages {tree} 0 R >>'.encode('ascii')
    patch_object(output, offsets[catalog], catalog, catalog_body)
    patch_object(output, offsets[tree], tree, tree_body)
    start = output.tell()
    output.write(f'xref\n0 {len(order) + 1}\n'.encode('ascii'))
    output.write(b'0000000000 65535 f \n')
    for number in order:
        output.write(f'{offsets[number]:010d} 00000 n \n'.encode('ascii'))
    output.write(f'trailer\n<< /Size {len(order) + 1} /Root {catalog} 0 R >>\nstartxref\n{start}\n%%EOF\n'.encode('ascii'))
    return output.getvalue()


def patch_object(output, offset, number, body):
    """Replace a reserved object in place; ``body`` must not outgrow the slot."""
    data = bytearray(output.getbuffer())
    start = data.index(b' 0 obj\n', offset) + len(b' 0 obj\n')
    end = data.index(b'\nendobj\n', start)
    if len(body) > end - start:
        raise ValueError('PDF object replacement grew unexpectedly')
    data[start:end] = body + b' ' * (end - start - len(body))
    output.seek(0)
    output.write(data)
    output.truncate()
