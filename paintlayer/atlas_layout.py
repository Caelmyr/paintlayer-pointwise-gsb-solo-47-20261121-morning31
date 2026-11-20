"""Lay out examination plates at a declared physical image scale."""

from dataclasses import dataclass
import math

from . import fields
from .errors import CapacityExceeded, InvalidGeometry

PAPERS = {
    'a4': (210, 297),
    'letter': (215.9, 279.4),
    'a3': (297, 420),
}
RESOLUTION = {144, 192, 240}
MAX_PAGES = 8
MAX_PAPER_PIXELS = 13_000_000
PX_PER_MM_AT_DPI_1 = 1 / 25.4


@dataclass(frozen=True)
class Rectangle:
    x: int
    y: int
    width: int
    height: int

    @property
    def right(self):
        return self.x + self.width

    @property
    def bottom(self):
        return self.y + self.height

    def inset(self, pixels):
        if min(self.width, self.height) <= 2 * pixels:
            raise InvalidGeometry('The print margin consumes the available area.')
        return Rectangle(self.x + pixels, self.y + pixels, self.width - 2 * pixels, self.height - 2 * pixels)

    def as_json(self):
        return [self.x, self.y, self.width, self.height]


@dataclass(frozen=True)
class PageFormat:
    paper: str
    orientation: str
    dpi: int
    margins_mm: int
    scale_factor: float | None
    columns: int

    @classmethod
    def parse(cls, raw):
        paper = fields.choice(raw.get('paper', 'a4'), field='paper', options=PAPERS)
        direction = fields.choice(raw.get('orientation', 'portrait'), field='orientation', options={'portrait', 'landscape'})
        resolution = fields.integer(raw.get('dpi', 144), field='dpi', minimum=144, maximum=240)
        if resolution not in RESOLUTION:
            raise InvalidGeometry('Choose one of the supported sheet resolutions.', field='dpi', facts={'resolutions': sorted(RESOLUTION)})
        margins_mm = fields.integer(raw.get('margin', 13), field='margin', minimum=7, maximum=25)
        columns = fields.integer(raw.get('columns', 2), field='columns', minimum=1, maximum=3)
        scale = raw.get('scale')
        scale = fields.scalar(scale, field='scale', minimum=0.05, maximum=4) if scale is not None else None
        return cls(paper, direction, resolution, margins_mm, scale, columns)

    @property
    def dimensions_mm(self):
        paper = PAPERS[self.paper]
        return paper if self.orientation == 'portrait' else paper[::-1]

    @property
    def px_per_mm(self):
        return self.dpi * PX_PER_MM_AT_DPI_1

    def dim(self, millimetres):
        return max(1, int(round(millimetres * self.px_per_mm)))

    def page(self):
        width, height = (self.dim(value) for value in self.dimensions_mm)
        if width * height > MAX_PAPER_PIXELS:
            raise CapacityExceeded('This page size and DPI exceed the contact sheet budget.')
        return Rectangle(0, 0, width, height)

    def as_json(self):
        return {
            'paper': self.paper,
            'orientation': self.orientation,
            'size_mm': list(self.dimensions_mm),
            'dpi': self.dpi,
            'margin_mm': self.margins_mm,
            'columns': self.columns,
            'physical_scale': self.scale_factor,
        }


@dataclass(frozen=True)
class PrintEntry:
    key: str
    extent: tuple
    label: str
    has_note: bool


@dataclass(frozen=True)
class Allocation:
    key: str
    slot: Rectangle
    raster: Rectangle
    order: int
    scale_pixels: float
    printed_extent_mm: tuple

    def as_json(self):
        return {
            'id': self.key,
            'slot': self.slot.as_json(),
            'raster': self.raster.as_json(),
            'order': self.order,
            'scale_pixels': self.scale_pixels,
            'printed_extent_mm': list(self.printed_extent_mm),
        }


@dataclass(frozen=True)
class LayoutPage:
    number: int
    entries: list


@dataclass(frozen=True)
class SheetPlan:
    format: PageFormat
    full: Rectangle
    inner: Rectangle
    body: Rectangle
    note_height: int
    actual_pixels_per_source: float
    group_gap: int
    column_gap: int
    print_scale: float | None
    pages: list

    def manifest(self):
        return {
            'paper': self.format.as_json(),
            'page_pixels': [self.full.width, self.full.height],
            'body_box': self.body.as_json(),
            'pixels_per_canvas_pixel': self.actual_pixels_per_source,
            'actual_ratio_to_artwork': self.print_scale,
            'pages': [
                {'number': page.number, 'entries': [item.as_json() for item in page.entries]}
                for page in self.pages
            ],
        }


def canvas_interiors(paper):
    full = paper.page()
    inner = full.inset(paper.dim(paper.margins_mm))
    head = paper.dim(29)
    foot = paper.dim(12)
    body = Rectangle(inner.x, inner.y + head, inner.width, inner.height - head - foot)
    if min(body.width, body.height) < paper.dim(38):
        raise InvalidGeometry('Page margins leave too little room to place plates.')
    return full, inner, body


def choose_scale(entries, body, paper, art_scale, column_gap, label_height):
    available_column = (body.width - column_gap * (paper.columns - 1)) / paper.columns
    max_width = max(item.extent[0] for item in entries)
    max_height = max(item.extent[1] for item in entries)
    must_fit = min(available_column / max_width, (body.height - label_height) / max_height)
    if paper.scale_factor is None:
        # A single contact sheet uses one raster-to-paper ratio so matched areas stay comparable.
        return must_fit, int(math.floor(available_column))
    if art_scale is None:
        raise InvalidGeometry('Set the examination scale before selecting a physical printing ratio.', field='scale')
    actual = paper.scale_factor * art_scale * paper.px_per_mm
    if actual > must_fit + 1e-8:
        raise InvalidGeometry(
            'At this scale a crop exceeds its printable slot. Use fewer columns or a smaller ratio.',
            field='scale',
            facts={'maximum_ratio_for_selection': must_fit / art_scale / paper.px_per_mm},
        )
    return actual, int(math.floor(available_column))


def arrange(entries, paper, art_scale):
    full, inner, body = canvas_interiors(paper)
    column_gap = paper.dim(5)
    group_gap = paper.dim(6)
    label_height = paper.dim(21 if any(item.has_note for item in entries) else 15)
    actual, col_width = choose_scale(entries, body, paper, art_scale, column_gap, label_height)
    pages = [[]]
    y = body.y
    for start in range(0, len(entries), paper.columns):
        row = entries[start:start + paper.columns]
        row_height = max(int(math.floor(item.extent[1] * actual)) for item in row) + label_height
        if y + row_height > body.bottom:
            if len(pages) >= MAX_PAGES:
                raise CapacityExceeded('The selected plates would produce too many contact sheet pages.')
            pages.append([])
            y = body.y
        for ordinal, item in enumerate(row):
            x = body.x + ordinal * (col_width + column_gap)
            width = max(1, int(math.floor(item.extent[0] * actual)))
            height = max(1, int(math.floor(item.extent[1] * actual)))
            raster = Rectangle(x + (col_width - width) // 2, y, width, height)
            allocation = Allocation(
                key=item.key,
                slot=Rectangle(x, y, col_width, row_height),
                raster=raster,
                order=start + ordinal + 1,
                scale_pixels=actual,
                printed_extent_mm=(width / paper.px_per_mm, height / paper.px_per_mm),
            )
            pages[-1].append(allocation)
        y += row_height + group_gap
    effective_ratio = actual / paper.px_per_mm / art_scale if art_scale else None
    return SheetPlan(
        format=paper, full=full, inner=inner, body=body, note_height=label_height,
        actual_pixels_per_source=actual, group_gap=group_gap, column_gap=column_gap,
        print_scale=effective_ratio,
        pages=[LayoutPage(index, page) for index, page in enumerate(pages, start=1)],
    )
