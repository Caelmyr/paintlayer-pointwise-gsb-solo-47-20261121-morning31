"""Plan an actual-size print of one canvas region across a grid of paper pages.

Every page is printed at the examination's own physical canvas scale, so the
joined pages reproduce the region at real size.  Page windows advance in whole
canvas pixels at one common scale, so a shared band shows the same canvas
content on both pages and a missing-data gap stays registered to its canvas
coordinates across the join.

A focus area (an examination region or box) is kept whole on one page whenever
the paper allows, with the join line pushed to the free edge of its band.  When
that is impossible the shared band at the join is widened so both neighbouring
pages carry the focus for checking.  Only if even that is beyond the paper or
page budget does a join cross the focus, and the layout reports it.
"""

from dataclasses import dataclass
import math

from . import fields
from .atlas_layout import PAPERS, PX_PER_MM_AT_DPI_1, RESOLUTION, Rectangle
from .errors import CapacityExceeded, InvalidGeometry

MAX_TILE_PAGES = 24
MAX_TILE_PIXELS = 60_000_000
MIN_OVERLAP_MM = 4.0
MAX_OVERLAP_MM = 40.0

HEADER_MM = 22
FOOTER_MM = 23
RULER_MM = 5


@dataclass(frozen=True)
class TileFormat:
    paper: str
    orientation: str
    dpi: int
    margin_mm: int
    overlap_mm: float

    @classmethod
    def parse(cls, raw):
        paper = fields.choice(raw.get('paper', 'a4'), field='paper', options=PAPERS)
        direction = fields.choice(raw.get('orientation', 'portrait'), field='orientation', options={'portrait', 'landscape'})
        resolution = fields.integer(raw.get('dpi', 144), field='dpi', minimum=144, maximum=240)
        if resolution not in RESOLUTION:
            raise InvalidGeometry('Choose one of the supported print resolutions.', field='dpi', facts={'resolutions': sorted(RESOLUTION)})
        margins_mm = fields.integer(raw.get('margin', 10), field='margin', minimum=7, maximum=25)
        overlap_mm = fields.scalar(raw.get('overlap', 12), field='overlap', minimum=MIN_OVERLAP_MM, maximum=MAX_OVERLAP_MM)
        return cls(paper, direction, resolution, margins_mm, overlap_mm)

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
        if width * height > 13_000_000:
            raise CapacityExceeded('This page size and resolution exceed the print budget.')
        return Rectangle(0, 0, width, height)

    def as_json(self):
        return {
            'paper': self.paper,
            'orientation': self.orientation,
            'size_mm': list(self.dimensions_mm),
            'dpi': self.dpi,
            'margin_mm': self.margin_mm,
            'overlap_mm': self.overlap_mm,
        }


@dataclass(frozen=True)
class AxisPlan:
    """Window starts and shared bands along one canvas axis, in canvas pixels.

    Window ``i`` shows canvas pixels ``[starts[i], starts[i] + capacity)``.  The
    band between windows ``i`` and ``i + 1`` is printed on both pages.
    """

    starts: tuple
    capacity: int

    def bands(self):
        return tuple(
            (self.starts[index + 1], self.starts[index] + self.capacity)
            for index in range(len(self.starts) - 1)
        )

    def clear_cut(self, index, low, high, clearance):
        """Join position in this band that stays clear of the focus, if any."""
        band_low, band_high = self.bands()[index]
        center = (band_low + band_high) / 2
        if high <= band_low or low >= band_high:
            return center, True
        if band_low <= low:
            position = min(center, low - clearance)
            return max(float(band_low), position), True
        if band_high >= high:
            position = max(center, high + clearance)
            return min(float(band_high), position), True
        return center, False

    def focus_pages(self, low, high):
        return [index for index, start in enumerate(self.starts) if start <= low and high <= start + self.capacity]

    def duplicated(self, low, high):
        """Fraction of the focus span lying inside some shared band."""
        span = high - low
        if span <= 0:
            return 1.0
        covered = []
        for band_low, band_high in self.bands():
            start = max(band_low, low)
            end = min(band_high, high)
            if end > start:
                covered.append((start, end))
        covered.sort()
        total = 0.0
        cursor = low
        for start, end in covered:
            start = max(start, cursor)
            if start < end:
                total += end - start
                cursor = end
        return min(1.0, total / span)


def chain(start, end, count):
    """``count + 1`` integer positions from ``start`` to ``end`` inclusive."""
    if count <= 0:
        return [int(round(start))]
    return [int(round(start + (end - start) * index / count)) for index in range(count + 1)]


def sampled(low, high, limit):
    if high < low:
        return []
    if high - low <= limit:
        return list(range(int(low), int(high) + 1))
    step = (high - low) / limit
    values = {int(round(low + step * index)) for index in range(limit + 1)}
    values.update({int(round(low)), int(round(high))})
    return sorted(value for value in values if low <= value <= high)


def valid(starts, extent, capacity, minimum):
    if not starts or starts[0] != 0 or starts[-1] != extent - capacity:
        return False
    ordered = sorted(set(starts))
    if len(ordered) != len(starts):
        return False
    return all(
        minimum <= ordered[index + 1] - ordered[index] <= capacity - minimum
        for index in range(len(ordered) - 1)
    )


def uniform_starts(extent, capacity, count):
    return chain(0, extent - capacity, count - 1)


def anchored_starts(extent, capacity, count, anchor):
    """Evenly spaced pages with one page start fixed over the focus."""
    last = extent - capacity
    left_count = int(round((count - 1) * anchor / last)) if last else 0
    left_count = min(max(left_count, 0), count - 1)
    left = chain(0, anchor, left_count) if left_count else [0]
    right = chain(anchor, last, count - 1 - left_count) if count - 1 - left_count else [last]
    return sorted(set(left + right))


def band_starts(extent, capacity, count, band_low, band_width):
    """Pages arranged around one shared band [band_low, band_low + width]."""
    last = extent - capacity
    left_start = band_low + band_width - capacity
    right_start = band_low
    if left_start < 0 or right_start > last or right_start <= left_start:
        return None
    return chained_pair(extent, capacity, count, left_start, right_start)


def chained_pair(extent, capacity, count, left_start, right_start):
    """Chain pages from both outer edges to a chosen pair of neighbouring starts."""
    last = extent - capacity
    room = count - 2
    if room < 0:
        return None
    span_left = left_start
    span_right = last - right_start
    total = span_left + span_right
    left_count = 0 if total <= 0 else min(room, max(0, int(round(room * span_left / total))))
    right_count = room - left_count
    left = chain(0, left_start, left_count) if left_count else [0]
    right = chain(right_start, last, right_count) if right_count else [last]
    return sorted(set(left + right))


def seam_starts(extent, capacity, count, seam, band_width):
    """One wide band centred on ``seam`` when the focus is wider than a page."""
    left_start = int(round(seam + band_width / 2 - capacity))
    right_start = int(round(seam - band_width / 2))
    if left_start < 0 or right_start > extent - capacity or right_start <= left_start:
        return None
    return chained_pair(extent, capacity, count, left_start, right_start)


def candidate_layouts(extent, capacity, minimum, low, high, budget):
    last = extent - capacity
    widest = capacity - minimum
    candidates = []
    for count in range(2, budget + 1):
        starts = uniform_starts(extent, capacity, count)
        if valid(starts, extent, capacity, minimum):
            candidates.append(AxisPlan(tuple(starts), capacity))
        if low is None:
            continue
        anchor_low = max(0, math.ceil(high - capacity))
        anchor_high = min(last, math.floor(low))
        for anchor in sampled(anchor_low, anchor_high, 3):
            starts = anchored_starts(extent, capacity, count, anchor)
            if valid(starts, extent, capacity, minimum):
                candidates.append(AxisPlan(tuple(starts), capacity))
        if count >= 3:
            widths = sorted({
                widest,
                max(minimum, int(widest * 0.75)),
                max(minimum, int(widest * 0.5)),
                max(minimum, int(widest * 0.3)),
            })
            band_low_low = max(0, math.ceil(high - capacity))
            band_low_high = min(last, math.floor(low))
            for width in widths:
                if width < minimum or width > widest:
                    continue
                for band_low in sampled(band_low_low, band_low_high, 4):
                    starts = band_starts(extent, capacity, count, band_low, width)
                    if starts and valid(starts, extent, capacity, minimum):
                        candidates.append(AxisPlan(tuple(starts), capacity))
            # Focus wider than a page: still give the join a wide shared band.
            for width in widths:
                if width < minimum or width > widest:
                    continue
                for seam in sampled(low, high, 3):
                    starts = seam_starts(extent, capacity, count, seam, width)
                    if starts and valid(starts, extent, capacity, minimum):
                        candidates.append(AxisPlan(tuple(starts), capacity))
    return candidates


def focus_rank(plan, low, high):
    """0 contained, 1 fully shared, 2 partly shared, 3 crosses without sharing."""
    if low is None:
        return 0
    if plan.focus_pages(low, high):
        return 0
    coverage = plan.duplicated(low, high)
    if coverage >= 1 - 1e-9:
        return 1
    return 2 if coverage > 0 else 3


def axis_rank(plan, low, high):
    rank = focus_rank(plan, low, high)
    coverage = plan.duplicated(low, high) if low is not None else 0.0
    gaps = [plan.starts[index + 1] - plan.starts[index] for index in range(len(plan.starts) - 1)]
    spread = max(gaps) - min(gaps) if gaps else 0
    # 0: focus contained, 1: shared band over it, 2: partly shared, 3: cut
    # straight through.  Within the extra-page allowance, prefer the better
    # treatment of the focus, then the most of it shared (in 5% steps so small
    # gains do not multiply sheets), then the fewest evenly spaced pages.
    if rank <= 1:
        return (rank, 0, len(plan.starts), spread)
    return (rank, -round(coverage * 20), len(plan.starts), spread)


def fewest_pages(extent, capacity, minimum):
    return max(2, math.ceil((extent - capacity) / (capacity - minimum) - 1e-9) + 1)


def plan_axis(extent, capacity, overlap, focus, budget):
    extent = int(extent)
    capacity = max(8, min(int(capacity), extent))
    minimum = max(1, min(int(overlap), capacity // 4))
    if extent <= capacity:
        return AxisPlan((0,), extent)
    span = clip_span(focus, extent)
    fewest = fewest_pages(extent, capacity, minimum)
    uniform = AxisPlan(tuple(uniform_starts(extent, capacity, fewest)), capacity)
    if span is None:
        return uniform
    budget = max(fewest, min(budget, MAX_TILE_PAGES))
    # Extra sheets are allowed only while they keep a focus that fits on one
    # page intact.  An oversized focus is worth one extra sheet at most.
    wanted = fewest + 2 if span[1] - span[0] <= capacity else fewest + 1
    allowance = min(budget, wanted)
    candidates = [
        plan
        for plan in candidate_layouts(extent, capacity, minimum, span[0], span[1], budget)
        if len(plan.starts) <= allowance
    ]
    if not candidates:
        return uniform
    return min(candidates, key=lambda plan: axis_rank(plan, span[0], span[1]))


def axis_budget(extent, capacity, overlap, other_fewest, limit):
    """Pages one axis may use so the two axes stay within the page limit."""
    minimum = max(1, min(int(overlap), capacity // 4))
    fewest = fewest_pages(extent, capacity, minimum)
    cap = max(fewest, limit // max(1, other_fewest))
    return min(MAX_TILE_PAGES, fewest + 2, cap)


def clip_span(span, extent):
    if span is None:
        return None
    low = max(0.0, float(span[0]))
    high = min(float(extent), float(span[1]))
    return (low, high) if high - low > 0.5 else None


def clip_focus(focus, bounds):
    if focus is None:
        return None
    x, y, width, height = focus
    bx, by, bw, bh = bounds
    left = max(x, bx)
    top = max(y, by)
    right = min(x + width, bx + bw)
    bottom = min(y + height, by + bh)
    if right - left < 1 or bottom - top < 1:
        return None
    return left, top, right - left, bottom - top


@dataclass(frozen=True)
class TilePlan:
    format: TileFormat
    full: Rectangle
    inner: Rectangle
    body: Rectangle
    scale: float
    source_bounds: tuple
    horizontal: AxisPlan
    vertical: AxisPlan
    focus_bounds: tuple | None
    focus_requested: tuple | None = None

    @property
    def columns(self):
        return len(self.horizontal.starts)

    @property
    def rows(self):
        return len(self.vertical.starts)

    @property
    def pages(self):
        return self.rows * self.columns

    @property
    def pixels_per_canvas_pixel(self):
        return self.scale * self.format.px_per_mm

    @property
    def content_mm(self):
        return self.source_bounds[2] * self.scale, self.source_bounds[3] * self.scale

    @property
    def clearance(self):
        return max(1, self.horizontal.capacity // 60)

    def cut_positions(self):
        """Chosen join line for each band, in local canvas pixels."""
        if self.focus_bounds is None:
            return {
                'x': [(low + high) / 2 for low, high in self.horizontal.bands()],
                'y': [(low + high) / 2 for low, high in self.vertical.bands()],
            }
        x, y, width, height = self.focus_bounds
        low_x, high_x = x - self.source_bounds[0], x - self.source_bounds[0] + width
        low_y, high_y = y - self.source_bounds[1], y - self.source_bounds[1] + height
        return {
            'x': [
                self.horizontal.clear_cut(index, low_x, high_x, self.clearance)[0]
                for index in range(self.columns - 1)
            ],
            'y': [
                self.vertical.clear_cut(index, low_y, high_y, self.clearance)[0]
                for index in range(self.rows - 1)
            ],
        }

    def bands_for_page(self, row, column):
        """Canvas-space shared bands on each side of a page.

        A band's canvas coordinates come from the axis window starts directly,
        so the same band yields identical numbers on both neighbouring pages.
        """
        x0, y0, width, height = self.window(row, column)
        ox, oy = self.source_bounds[0], self.source_bounds[1]
        result = []
        if column > 0:
            low, high = self.horizontal.bands()[column - 1]
            result.append(self._band('left', 'x', column - 1, ox + low, y0, high - low, height, (row, column - 1)))
        if column < self.columns - 1:
            low, high = self.horizontal.bands()[column]
            result.append(self._band('right', 'x', column, ox + low, y0, high - low, height, (row, column + 1)))
        if row > 0:
            low, high = self.vertical.bands()[row - 1]
            result.append(self._band('top', 'y', row - 1, x0, oy + low, width, high - low, (row - 1, column)))
        if row < self.rows - 1:
            low, high = self.vertical.bands()[row]
            result.append(self._band('bottom', 'y', row, x0, oy + low, width, high - low, (row + 1, column)))
        return result

    def _band(self, side, axis, index, x, y, width, height, neighbour):
        return {
            'side': side,
            'axis': axis,
            'index': index,
            'canvas': (x, y, width, height),
            'neighbour': neighbour,
            'neighbour_page': self.page_number(*neighbour),
            'cut': self.cut_positions()[axis][index],
        }

    def mark_canvas_points(self, band):
        """Three shared canvas coordinates that both pages print identically."""
        x, y, width, height = band['canvas']
        points = []
        for fraction in (0.2, 0.5, 0.8):
            if band['axis'] == 'x':
                points.append((x + width // 2, int(round(y + height * fraction))))
            else:
                points.append((int(round(x + width * fraction)), y + height // 2))
        return points

    def page_marks(self, row, column):
        found = []
        for band in self.bands_for_page(row, column):
            for point in self.mark_canvas_points(band):
                found.append(
                    {
                        'canvas': list(point),
                        'side': band['side'],
                        'with_page': band['neighbour_page'],
                    }
                )
        return found

    def axis_focus_score(self, axis, low, high):
        """1.0 when the focus is safe on that axis (contained or fully shared)."""
        if axis.focus_pages(low, high):
            return 1.0
        return axis.duplicated(low, high)

    def focus_report(self):
        if self.focus_bounds is None:
            return {'bounds': None, 'status': 'none', 'coverage': 1.0, 'bands': []}
        x, y, width, height = self.focus_bounds
        low_x, high_x = x - self.source_bounds[0], x - self.source_bounds[0] + width
        low_y, high_y = y - self.source_bounds[1], y - self.source_bounds[1] + height
        contained = bool(self.horizontal.focus_pages(low_x, high_x)) and bool(self.vertical.focus_pages(low_y, high_y))
        coverage = min(
            self.axis_focus_score(self.horizontal, low_x, high_x),
            self.axis_focus_score(self.vertical, low_y, high_y),
        )
        bands = []
        for name, axis, low, high, origin in (
            ('x', self.horizontal, low_x, high_x, self.source_bounds[0]),
            ('y', self.vertical, low_y, high_y, self.source_bounds[1]),
        ):
            for index, (band_low, band_high) in enumerate(axis.bands()):
                if band_high <= low or band_low >= high:
                    continue
                _, clear = axis.clear_cut(index, low, high, self.clearance)
                band_low_abs, band_high_abs = band_low + origin, band_high + origin
                bands.append(
                    {
                        'axis': name,
                        'boundary': index + 1,
                        'band_px': [band_low_abs, band_high_abs],
                        'band_mm': (band_high - band_low) * self.scale,
                        'shared_mm': max(0.0, min(high, band_high) - max(low, band_low)) * self.scale,
                        'covers_focus': band_low <= low and high <= band_high,
                        'cut_clear': clear,
                        'pages': [index + 1, index + 2],
                    }
                )
        if contained:
            status = 'single_page'
        elif coverage >= 1 - 1e-9:
            status = 'overlap_band'
        elif coverage > 0:
            status = 'partly_shared'
        else:
            status = 'crosses_focus'
        return {
            'bounds': list(self.focus_bounds),
            'requested_bounds': list(self.focus_requested) if self.focus_requested else list(self.focus_bounds),
            'clipped_to_crop': self.focus_requested is not None and tuple(self.focus_requested) != tuple(self.focus_bounds),
            'status': status,
            'coverage': coverage,
            'bands': bands,
        }

    def page_number(self, row, column):
        return row * self.columns + column + 1

    def origin(self, row, column):
        return self.source_bounds[0] + self.horizontal.starts[column], self.source_bounds[1] + self.vertical.starts[row]

    def window(self, row, column):
        x, y = self.origin(row, column)
        return (
            x,
            y,
            min(self.horizontal.capacity, self.source_bounds[0] + self.source_bounds[2] - x),
            min(self.vertical.capacity, self.source_bounds[1] + self.source_bounds[3] - y),
        )

    def neighbours(self, row, column):
        result = {}
        if column > 0:
            result['left'] = (row, column - 1)
        if column < self.columns - 1:
            result['right'] = (row, column + 1)
        if row > 0:
            result['top'] = (row - 1, column)
        if row < self.rows - 1:
            result['bottom'] = (row + 1, column)
        return result

    def manifest(self):
        cuts = self.cut_positions()
        return {
            'mode': 'actual_size_tiles',
            'paper': self.format.as_json(),
            'page_pixels': [self.full.width, self.full.height],
            'body_box': self.body.as_json(),
            'canvas_bounds': list(self.source_bounds),
            'mm_per_canvas_pixel': self.scale,
            'pixels_per_canvas_pixel': self.pixels_per_canvas_pixel,
            'printed_size_mm': list(self.content_mm),
            'grid': {'rows': self.rows, 'columns': self.columns, 'pages': self.pages},
            'overlap_mm': self.format.overlap_mm,
            'column_windows_px': [list(band) for band in self.horizontal.bands()],
            'row_windows_px': [list(band) for band in self.vertical.bands()],
            'cut_lines_px': {
                axis: [round(position + self.source_bounds[0 if axis == 'x' else 1], 2) for position in positions]
                for axis, positions in cuts.items()
            },
            'focus': self.focus_report(),
            'pages': [
                {
                    'number': self.page_number(row, column),
                    'row': row + 1,
                    'column': column + 1,
                    'canvas_window_px': list(self.window(row, column)),
                    'neighbours': {
                        side: {'row': other[0] + 1, 'column': other[1] + 1, 'number': self.page_number(*other)}
                        for side, other in self.neighbours(row, column).items()
                    },
                }
                for row in range(self.rows)
                for column in range(self.columns)
            ],
            'assembly': (
                'Print every page at 100% / actual size with fit-to-page and auto-rotate disabled. '
                'Lay neighbouring pages overlapping inside the hatched band, align the crosses that carry the same '
                'canvas coordinate, then cut along the marked join line or tape from the reverse.'
            ),
        }


def page_metrics(paper):
    full = paper.page()
    inner = full.inset(paper.dim(paper.margin_mm))
    body = Rectangle(
        inner.x + paper.dim(RULER_MM),
        inner.y + paper.dim(HEADER_MM) + paper.dim(RULER_MM),
        inner.width - paper.dim(RULER_MM),
        inner.height - paper.dim(HEADER_MM) - paper.dim(FOOTER_MM) - paper.dim(RULER_MM),
    )
    if min(body.width, body.height) < paper.dim(38):
        raise InvalidGeometry('Page margins leave too little room to print tiles.')
    return full, inner, body


def plan_tiles(bounds, scale, focus, paper):
    if scale is None:
        raise InvalidGeometry('Set the examination scale before printing at actual size.', field='scale')
    full, inner, body = page_metrics(paper)
    k = scale * paper.px_per_mm
    capacity_x = max(8, int(body.width / k))
    capacity_y = max(8, int(body.height / k))
    overlap_px = max(1, int(round(paper.overlap_mm / scale)))
    focus_bounds = clip_focus(focus, bounds)
    if focus is not None and focus_bounds is None:
        raise InvalidGeometry('The protected area lies outside the printed crop.', field='focus')
    local_x = (focus_bounds[0] - bounds[0], focus_bounds[0] - bounds[0] + focus_bounds[2]) if focus_bounds else None
    local_y = (focus_bounds[1] - bounds[1], focus_bounds[1] - bounds[1] + focus_bounds[3]) if focus_bounds else None
    minimum_x = max(1, min(overlap_px, capacity_x // 4))
    minimum_y = max(1, min(overlap_px, capacity_y // 4))
    base_x = fewest_pages(bounds[2], capacity_x, minimum_x)
    base_y = fewest_pages(bounds[3], capacity_y, minimum_y)
    horizontal = plan_axis(bounds[2], capacity_x, overlap_px, local_x, MAX_TILE_PAGES)
    vertical = plan_axis(bounds[3], capacity_y, overlap_px, local_y, MAX_TILE_PAGES)
    if len(horizontal.starts) * len(vertical.starts) > MAX_TILE_PAGES:
        # Retry with a per-axis page budget so the product fits.
        budget_x = axis_budget(bounds[2], capacity_x, overlap_px, base_y, MAX_TILE_PAGES)
        budget_y = axis_budget(bounds[3], capacity_y, overlap_px, base_x, MAX_TILE_PAGES)
        horizontal = plan_axis(bounds[2], capacity_x, overlap_px, local_x, budget_x)
        vertical = plan_axis(bounds[3], capacity_y, overlap_px, local_y, budget_y)
    if len(horizontal.starts) * len(vertical.starts) > MAX_TILE_PAGES and focus_bounds is not None:
        # Keeping the protected area whole would cost more paper than the
        # print allowance.  Fall back to the plain grid and let the focus
        # report say plainly which joins cross the area.
        horizontal = plan_axis(bounds[2], capacity_x, overlap_px, None, MAX_TILE_PAGES)
        vertical = plan_axis(bounds[3], capacity_y, overlap_px, None, MAX_TILE_PAGES)
    plan = TilePlan(
        format=paper,
        full=full,
        inner=inner,
        body=body,
        scale=scale,
        source_bounds=tuple(bounds),
        horizontal=horizontal,
        vertical=vertical,
        focus_bounds=focus_bounds,
        focus_requested=tuple(focus) if focus is not None else None,
    )
    if plan.pages > MAX_TILE_PAGES:
        raise CapacityExceeded(
            'This area needs more pages than one tiled print allows. Print a smaller region or choose a lower resolution.',
            field='region',
            facts={'pages': plan.pages, 'allowed': MAX_TILE_PAGES},
        )
    if plan.pages * body.width * body.height > MAX_TILE_PIXELS:
        raise CapacityExceeded(
            'This tiled print exceeds the total page size allowance. Choose a lower resolution or a smaller region.',
            field='dpi',
        )
    return plan
