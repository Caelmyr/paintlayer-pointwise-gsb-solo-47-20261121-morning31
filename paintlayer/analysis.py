"""Save measurable inspection results with the numerical and geometric inputs used."""

from . import fields
from .detector import DetectorCheck, flat_diagnostics
from .errors import ExamError
from .evidence import area_csv, csv_values, transect_csv
from .library import public_record
from .texture import TextureRead, analyze_texture

ANALYSIS_TYPES = {'texture', 'detector', 'area', 'transect'}


def peak_csv(details):
    return csv_values(
        details['peaks'],
        ('period_pixels', 'period_mm', 'wavevector_degrees', 'thread_axis_degrees', 'contrast'),
    )


def detector_csv(details):
    rows = []
    for axis in ('rows', 'columns'):
        for entry in details[axis]['profile']:
            rows.append({'axis': axis} | entry)
    return csv_values(rows, ('axis', 'index', 'median_gain', 'normalized_deviation', 'sample_count'))


class Analysis:
    def __init__(self, store, library, processing):
        self.store = store
        self.library = library
        self.processing = processing

    def save(self, case_id, kind, form, *, persist):
        raw = fields.request_object(form)
        kind = fields.choice(kind, field='analysis', options=ANALYSIS_TYPES)
        handlers = {
            'detector': self.detector,
            'texture': self.texture,
            'area': self.area,
            'transect': self.transect,
        }
        with self.store.snapshot() as queries:
            case = queries.case(case_id)
            details, assets, required = handlers[kind](queries, case, raw)
        if not persist:
            return details
        name = fields.text(raw.get('name'), field='name', limit=100)
        with self.store.transaction() as queries:
            record = queries.add(
                case_id, 'analysis', name, details | {'analysis_type': kind},
                blob_data=assets, dependencies=required,
            )
            return public_record(queries, case_id, record)

    def detector(self, queries, case, form):
        case_id = case['id']
        references = {
            'dark': self.processing.read_source(queries, case_id, form.get('dark'), 'dark'),
            'flat': self.processing.read_source(queries, case_id, form.get('flat'), 'flat'),
            'mask': self.processing.read_source(queries, case_id, form.get('mask'), 'mask', required=False),
        }
        dark = self.library.detector_frame(queries, case_id, references['dark'], 'dark')
        flat = self.library.detector_frame(queries, case_id, references['flat'], 'flat')
        mask_record = references.get('mask')
        mask = self.library.source_surface(queries, case_id, mask_record, role='mask').pixels if mask_record else None
        details, preview, warnings = flat_diagnostics(dark, flat, mask, DetectorCheck.parse(form))
        dependencies = [(record, key) for key, record in references.items() if record]
        if not form.get('save'):
            details['preview_inline'] = wrap_inline_png(preview)
            details['flags_inline'] = wrap_inline_png(warnings)
        blobs = [('preview', preview), ('flags', warnings), ('csv', detector_csv(details))]
        return details, blobs, dependencies

    def texture(self, queries, case, form):
        case_id = case['id']
        run = self.processing.run(queries, case_id, form.get('run'), {'mosaic', 'plate'})
        surface = self.processing.surface(queries, case_id, run)
        definition = TextureRead.parse(form, surface.extent)
        report, image = analyze_texture(surface, definition, case['scale'] if run['details']['kind'] == 'mosaic' else None)
        report['run'] = run['id']
        if not form.get('save'):
            report['spectrum_inline'] = wrap_inline_png(image)
        return report, [('spectrum', image), ('csv', peak_csv(report))], [(run, 'numerical_source')]

    def area(self, queries, case, form):
        case_id = case['id']
        run = self.processing.run(queries, case_id, form.get('run'), {'mosaic'})
        region, outline = self.processing.region_input(queries, case_id, form, 'region')
        if outline is None:
            raise ExamError('Choose an area to measure.', field='region')
        result = self.processing.area(case_id, form)
        if result['region_revision'] != region['revision']:
            raise ExamError('The region was revised during measurement. Reopen its latest outline.')
        result['outline'] = outline.as_json()
        deps = [(run, 'numerical_source'), (region, 'measured_region')]
        return result, [('csv', area_csv(result))], deps

    def transect(self, queries, case, form):
        case_id = case['id']
        run = self.processing.run(queries, case_id, form.get('run'), {'mosaic'})
        report = self.processing.transect(case_id, form)
        return report, [('csv', transect_csv(report))], [(run, 'numerical_source')]


def wrap_inline_png(payload):
    import base64
    return 'data:image/png;base64,' + base64.b64encode(payload).decode('ascii')
