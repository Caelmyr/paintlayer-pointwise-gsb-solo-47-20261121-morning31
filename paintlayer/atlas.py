"""Prepare and preserve physical contact sheets and actual-size tiled prints."""

from dataclasses import dataclass
from itertools import chain
import hashlib

from PIL import Image

from . import fields
from .atlas_layout import PageFormat, PrintEntry, arrange
from .atlas_render import PlateBlock, contact_sheet_pages, encode_preview
from .atlas_render import encode_pdf as encode_sheet_pdf
from .atlas_tile_render import TileSource, encode_pdf as encode_tile_pdf, page_marks, tile_pages
from .atlas_tiles import TileFormat, plan_tiles
from .errors import CapacityExceeded, ExamError, InsufficientSupport
from .library import case_extent, public_record
from .store import json_text
from .views import DISPLAY_LAYERS, IMAGE_DEFAULTS, Overlay, Presentation

MAX_SHEET_ITEMS = 12
MAX_SHEET_RENDER_PIXELS = 35_000_000
MAX_CONTACT_PDF_BYTES = 44 * 1024 * 1024
MAX_TILE_PDF_BYTES = 60 * 1024 * 1024


@dataclass(frozen=True)
class AtlasSource:
    id: str
    run: dict
    photograph: dict | None
    region: dict | None
    bounds: tuple
    layer: str
    name: str
    note: str
    mode: str
    display: Presentation
    overlay: Overlay | None

    def as_json(self):
        return {
            'item': self.id,
            'run': self.run['id'],
            'registration': self.photograph['id'] if self.photograph else None,
            'region': self.region['id'] if self.region else None,
            'region_revision': self.region['revision'] if self.region else None,
            'bounds': list(self.bounds),
            'layer': self.layer,
            'title': self.name,
            'note': self.note,
            'composition': self.mode,
            'display': self.display.as_json(),
            'overlay': self.overlay.as_json() if self.overlay else None,
        }

    def layout_entry(self):
        return PrintEntry(self.id, self.bounds[2:], self.name, bool(self.note))

    def dependencies(self):
        refs = [(self.run, 'printed_surface')]
        if self.photograph:
            refs.append((self.photograph, 'printed_reference'))
        if self.region:
            refs.append((self.region, 'printed_region'))
        return refs


class Atlas:
    def __init__(self, store, processing, views):
        self.store = store
        self.processing = processing
        self.views = views

    def source(self, queries, case, raw, ordinal):
        field = f'items[{ordinal}]'
        raw = fields.request_object(raw, field=field)
        mode = fields.choice(raw.get('mode', 'layer'), field=f'{field}.mode', options={'layer', 'overlay'})
        accepted = {'mosaic'} if mode == 'overlay' else {'mosaic', 'compare', 'quality', 'support', 'registration'}
        run = self.processing.run(queries, case['id'], raw.get('run'), accepted)
        kind = run['details']['kind']
        selected = IMAGE_DEFAULTS.get(kind, 'signal')
        layer = fields.choice(raw.get('layer', selected), field=f'{field}.layer', options=DISPLAY_LAYERS[kind])
        region, outline = self.processing.region_input(queries, case['id'], raw, 'region')
        if outline is not None:
            box = outline.bounds(case_extent(case))
        elif raw.get('box') is not None:
            box = fields.box(raw.get('box'), field=f'{field}.box', extent=case_extent(case))
        else:
            box = (0, 0, *case_extent(case))
        if min(box[2:]) < 4:
            raise InsufficientSupport('Contact sheet crops require at least four pixels per side.', field=f'{field}.box')
        name = fields.text(raw.get('title'), field=f'{field}.title', required=False, limit=100) or run['name']
        note = fields.text(raw.get('note'), field=f'{field}.note', required=False, multiline=True, limit=180)
        photo = self.processing.run(queries, case['id'], raw.get('registration'), {'registration'}) if mode == 'overlay' else None
        return AtlasSource(
            id=str(ordinal + 1), run=run, photograph=photo, region=region,
            bounds=tuple(box), layer=layer, name=name, note=note, mode=mode,
            display=Presentation.parse(raw.get('display', {})),
            overlay=Overlay.parse(raw.get('overlay', {})) if photo else None,
        )

    def apply(self, queries, case, source):
        case_id = case['id']
        if source.photograph is not None:
            rgba = self.views.compose(
                queries, case_id, source.run, source.photograph, source.display, source.overlay, source.bounds,
            )
        else:
            rgba = self.views.render(queries, case_id, source.run, source.layer, source.display, source.bounds)
        if not rgba[..., 3].any():
            raise InsufficientSupport(f'No visible pixels remain in the crop for {source.name}.')
        signature = 'Registered overlay' if source.photograph else source.layer.replace('_', ' ').capitalize()
        return PlateBlock(
            image=Image.fromarray(rgba),
            title=source.name,
            note=source.note,
            label=signature,
            canvas_bounds=source.bounds,
            scale=case['scale'],
            is_synthetic=case['synthetic'],
        )

    def plan(self, queries, case, raw):
        title = fields.text(raw.get('name'), field='name', limit=100)
        format = PageFormat.parse(fields.request_object(raw.get('format', {}), field='format'))
        choices = fields.sequence(raw.get('items'), field='items', minimum=1, maximum=MAX_SHEET_ITEMS)
        images = [self.source(queries, case, row, index) for index, row in enumerate(choices)]
        estimated_pixels = sum(image.bounds[2] * image.bounds[3] for image in images)
        if estimated_pixels > 4 * MAX_SHEET_RENDER_PIXELS // 3:
            raise CapacityExceeded('Select smaller or fewer canvases to build this contact sheet.')
        plan = arrange([image.layout_entry() for image in images], format, case['scale'])
        total_pixels = len(plan.pages) * plan.full.width * plan.full.height
        if total_pixels > MAX_SHEET_RENDER_PIXELS:
            raise CapacityExceeded('This sheet exceeds the total page size allowance.')
        return title, format, images, plan

    def preview(self, case_id, form):
        raw = fields.request_object(form)
        with self.store.snapshot() as queries:
            case = queries.case(case_id)
            title, _, images, plan = self.plan(queries, case, raw)
            needed = {item.key for item in plan.pages[0].entries}
            blocks = {entry.id: self.apply(queries, case, entry) for entry in images if entry.id in needed}
            first = next(contact_sheet_pages(case, title, plan, blocks))
        return encode_preview(first)

    def save(self, case_id, form):
        raw = fields.request_object(form)
        with self.store.snapshot() as queries:
            case = queries.case(case_id)
            title, format, images, plan = self.plan(queries, case, raw)
            blocks = {entry.id: self.apply(queries, case, entry) for entry in images}
            pages = list(contact_sheet_pages(case, title, plan, blocks))
        preview = encode_preview(pages[0])
        payload = encode_sheet_pdf(pages, format.dpi)
        if len(payload) > MAX_CONTACT_PDF_BYTES:
            raise CapacityExceeded('The rendered PDF exceeds the contact sheet file allowance.')
        report = plan.manifest() | {
            'note': 'Print at actual size and verify the marked 20 mm reference.',
            'pdf_bytes': len(payload),
            'pdf_sha256': hashlib.sha256(payload).hexdigest(),
            'items': [entry.as_json() for entry in images],
        }
        deps = [pair for entry in images for pair in entry.dependencies()]
        with self.store.transaction() as queries:
            record = queries.add(case_id, 'sheet', title, report, blob_data=(
                ('preview', preview), ('pdf', payload), ('layout', json_text(report).encode('utf-8')),
            ), dependencies=deps)
            return public_record(queries, case_id, record)

    def tile_inputs(self, queries, case, raw):
        raw = fields.request_object(raw)
        title = fields.text(raw.get('name'), field='name', limit=100)
        format = TileFormat.parse(fields.request_object(raw.get('format', {}), field='format'))
        run = self.processing.run(queries, case['id'], raw.get('run'), {'mosaic', 'compare', 'quality', 'support', 'registration'})
        kind = run['details']['kind']
        layer = fields.choice(
            raw.get('layer', IMAGE_DEFAULTS.get(kind, 'signal')),
            field='layer',
            options=DISPLAY_LAYERS[kind],
        )
        region, outline = self.processing.region_input(queries, case['id'], raw, 'region')
        focus_region, focus_outline = self.processing.region_input(queries, case['id'], raw, 'focus')
        if outline is not None:
            bounds = outline.bounds(case_extent(case))
        elif raw.get('box') is not None:
            bounds = fields.box(raw.get('box'), field='box', extent=case_extent(case))
        else:
            bounds = (0, 0, *case_extent(case))
        if focus_outline is not None:
            focus = focus_outline.bounds(case_extent(case))
        elif raw.get('focus_box') is not None:
            focus = fields.box(raw.get('focus_box'), field='focus_box', extent=case_extent(case))
        else:
            focus = None
        if min(bounds[2:]) < 8:
            raise InsufficientSupport('Tiled prints require at least eight canvas pixels per side.', field='box')
        if case['scale'] is None:
            raise ExamError('Set a canvas scale in the examination details before printing at actual size.', field='scale')
        note = fields.text(raw.get('note'), field='note', required=False, multiline=True, limit=180)
        plan = plan_tiles(bounds, case['scale'], focus, format)
        source = AtlasSource(
            id='1', run=run, photograph=None, region=region, bounds=tuple(bounds), layer=layer,
            name=run['name'], note=note, mode='layer',
            display=Presentation.parse(raw.get('display', {})), overlay=None,
        )
        return title, source, region, focus_region, plan

    def tile_source(self, queries, case, source):
        rgba = self.views.render(queries, case['id'], source.run, source.layer, source.display, source.bounds)
        if not rgba[..., 3].any():
            raise InsufficientSupport(f'No visible pixels remain in the crop for {source.name}.')
        return TileSource(
            image=Image.fromarray(rgba),
            title=source.name,
            note=source.note,
            label=source.layer.replace('_', ' ').capitalize(),
            canvas_bounds=source.bounds,
            scale=case['scale'],
            is_synthetic=case['synthetic'],
        )

    def tile_preview(self, case_id, form):
        raw = fields.request_object(form)
        with self.store.snapshot() as queries:
            case = queries.case(case_id)
            _, source, _, _, plan = self.tile_inputs(queries, case, raw)
            image = self.tile_source(queries, case, source)
        first = next(tile_pages(plan, image))
        return encode_preview(first)

    def save_tiles(self, case_id, form):
        raw = fields.request_object(form)
        with self.store.snapshot() as queries:
            case = queries.case(case_id)
            title, source, region, focus_region, plan = self.tile_inputs(queries, case, raw)
            image = self.tile_source(queries, case, source)
            pages = tile_pages(plan, image)
            first = next(pages)
            preview = encode_preview(first)
            payload = encode_tile_pdf(chain([first], pages), plan.format)
            marks = page_marks(plan)
            expected = queries.blob_info(case_id, source.run['id'])
        if len(payload) > MAX_TILE_PDF_BYTES:
            raise CapacityExceeded('The rendered tiled PDF exceeds the print file allowance.')
        report = plan.manifest() | {
            'pdf_bytes': len(payload),
            'pdf_sha256': hashlib.sha256(payload).hexdigest(),
            'run': {'id': source.run['id'], 'name': source.run['name']},
            'layer': source.layer,
            'region': {'id': region['id'], 'revision': region['revision']} if region else None,
            'focus_region': {'id': focus_region['id'], 'revision': focus_region['revision']} if focus_region else None,
            'display': source.display.as_json(),
            'source_blobs': expected,
            'page_marks': marks,
            'note': (
                'Print every page at 100% / actual size. Verify the marked 20 mm check, then join neighbours '
                'inside the shared bands using the crosses that carry the same canvas coordinate.'
            ),
        }
        dependencies = [(source.run, 'printed_surface')]
        if region:
            dependencies.append((region, 'printed_region'))
        if focus_region:
            dependencies.append((focus_region, 'protected_print_area'))
        with self.store.transaction() as queries:
            record = queries.add(case_id, 'sheet', title, report, blob_data=(
                ('preview', preview), ('pdf', payload), ('layout', json_text(report).encode('utf-8')),
            ), dependencies=dependencies)
            return public_record(queries, case_id, record)
