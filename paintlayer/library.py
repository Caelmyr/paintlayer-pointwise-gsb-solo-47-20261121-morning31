"""Create and maintain the case notebook, raw image sources and Canvas regions."""

import datetime as dt
import re
from pathlib import PurePath

import numpy as np

from . import fields
from .calibration import Frame, interpolate_reference_probe
from .errors import ExamError, ImageFormatError, InsufficientSupport
from .geometry import outline_pixels, parse_outline, validate_canvas
from .intake import upload_bytes
from .raster import as_bytes, display_range, from_blob, histogram, read_frame, support_summary, Surface
from .settings import MAX_SIDE, MAX_SURFACE_PIXELS, SOURCE_ROLES

REGION_TAGS = {
    'treatment': 'Treatment area',
    'reference': 'Stable reference',
    'material': 'Material variation',
    'interest': 'Area of interest',
    'artifact': 'Acquisition artifact',
    'support': 'Support feature',
}


def case_fields(payload):
    return {
        'title': fields.text(payload.get('title'), field='title', limit=96),
        'accession': fields.text(payload.get('accession'), field='accession', limit=45, required=False),
        'collection': fields.text(payload.get('collection'), field='collection', limit=90, required=False),
        'notes': fields.text(payload.get('notes'), field='notes', limit=4000, required=False, multiline=True),
    }


def case_extent(case):
    return case['width'], case['height']


def expected_revision(payload):
    return fields.integer(payload.get('revision'), field='revision', minimum=1)


def get_array(queries, case_id, item_id, name):
    path, _ = queries.blob_path(case_id, item_id, name)
    return from_blob(path)


def new_date(value, field='date'):
    if not isinstance(value, str) or not fields.DATE_PATTERN.fullmatch(value):
        raise ExamError('Enter an acquisition date as YYYY-MM-DD.', field=field)
    try:
        return dt.date.fromisoformat(value).isoformat()
    except ValueError as exc:
        raise ExamError('Use a valid calendar date.', field=field) from exc


def public_record(queries, case_id, record):
    return record | {
        'inputs': queries.inputs(case_id, record['id']),
        'files': queries.blob_info(case_id, record['id']),
    }


def normalize_filename(name):
    name = PurePath(str(name or 'imported-image').replace('\\', '/')).name
    name = re.sub(r'[\x00-\x1f\x7f]', '', name).strip()
    return name[-120:] or 'imported-image'


def source_display(decoded, role):
    if role == 'visible':
        return {}
    support = decoded.support
    pixels = decoded.pixels
    limits = display_range(pixels, support)
    return {'display': list(limits), 'histogram': histogram(pixels, support, limits)}


class Library:
    def __init__(self, store):
        self.store = store

    def create(self, form, *, synthetic=False):
        payload = fields.request_object(form)
        width = fields.integer(payload.get('width'), field='width', minimum=8, maximum=MAX_SIDE)
        height = fields.integer(payload.get('height'), field='height', minimum=8, maximum=MAX_SIDE)
        validate_canvas(width, height, MAX_SURFACE_PIXELS)
        raw_scale = payload.get('scale')
        mm_per_pixel = fields.scalar(raw_scale, field='scale', minimum=0.0001, maximum=25) if raw_scale is not None else None
        details = case_fields(payload) | {
            'width': width,
            'height': height,
            'scale': mm_per_pixel,
            'synthetic': synthetic,
        }
        with self.store.transaction() as queries:
            return queries.create_case(details)

    def revise_case(self, case_id, form):
        payload = fields.request_object(form)
        with self.store.transaction() as queries:
            return queries.update_case(case_id, case_fields(payload), expected_revision(payload))

    def contents(self, case_id):
        with self.store.snapshot() as queries:
            case = queries.case(case_id)
            records = {
                collection: queries.records(case_id, kind)
                for collection, kind in (
                    ('sessions', 'session'),
                    ('sources', 'source'),
                    ('profiles', 'profile'),
                    ('runs', 'run'),
                    ('regions', 'region'),
                    ('findings', 'finding'),
                    ('analyses', 'analysis'),
                    ('archives', 'archive'),
                    ('sheets', 'sheet'),
                )
            }
            return {'case': case, 'events': queries.events(case_id)} | records

    def session_fields(self, raw):
        return {
            'date': new_date(raw.get('date')),
            'purpose': fields.text(raw.get('purpose'), field='purpose', required=False, limit=120),
            'notes': fields.text(raw.get('notes'), field='notes', required=False, multiline=True, limit=2000),
        }

    def create_session(self, case_id, form):
        payload = fields.request_object(form)
        name = fields.text(payload.get('name'), field='name', limit=90)
        details = self.session_fields(payload)
        with self.store.transaction() as queries:
            return queries.add(case_id, 'session', name, details)

    def revise_session(self, case_id, session_id, form):
        payload = fields.request_object(form)
        name = fields.text(payload.get('name'), field='name', limit=90)
        details = self.session_fields(payload)
        with self.store.transaction() as queries:
            return queries.revise(case_id, session_id, 'session', expected_revision(payload), name, details)

    def receive(self, case_id, upload, form):
        payload = upload_bytes(upload)
        options = {
            'name': form.get('name') or normalize_filename(upload.filename),
            'role': form.get('role'),
            'session': form.get('session') or None,
            'notes': form.get('notes'),
        }
        try:
            options['page'] = int(form.get('page', 1))
        except (ValueError, TypeError) as exc:
            raise ExamError('Enter a TIFF page number.', field='page') from exc
        return self.add_source(case_id, payload, upload.filename, options)

    def add_source(self, case_id, payload, file_name, form):
        options = fields.request_object(form)
        title = fields.text(options.get('name'), field='name', limit=120)
        role = fields.choice(options.get('role'), field='role', options=SOURCE_ROLES)
        page = fields.integer(options.get('page', 1), field='page', minimum=1, maximum=48)
        notes = fields.text(options.get('notes'), field='notes', limit=2000, required=False, multiline=True)
        session_id = options.get('session')
        session_id = fields.item_id(session_id, field='session') if session_id else None
        with self.store.snapshot() as queries:
            queries.case(case_id)
            session = queries.record(case_id, session_id, 'session') if session_id else None
        decoded = read_frame(payload, role, page - 1)
        name = normalize_filename(file_name)
        details = decoded.metadata | source_display(decoded, role) | {
            'role': role,
            'filename': name,
            'notes': notes,
            'support': support_summary(decoded.support),
            'session': session_id,
            'session_date': session['details']['date'] if session else None,
            'session_name': session['name'] if session else None,
        }
        usable = details['support']['valid']
        if not usable:
            raise InsufficientSupport('The imported image has no usable samples.')
        with self.store.transaction() as queries:
            dependency = [(session, 'acquisition_session')] if session else []
            record = queries.add(
                case_id,
                'source',
                title,
                details,
                blob_data=(
                    ('original', payload),
                    ('pixels', as_bytes(decoded.pixels)),
                    ('valid', as_bytes(decoded.support)),
                ),
                dependencies=dependency,
            )
            return public_record(queries, case_id, record)

    def source_surface(self, queries, case_id, source, *, role=None):
        if source['kind'] != 'source':
            raise ImageFormatError('Choose an imported image source.')
        if role is not None and source['details']['role'] != role:
            raise ImageFormatError(f'Choose an image with role {SOURCE_ROLES[role]}.', field=role)
        item_id = source['id']
        pixels = get_array(queries, case_id, item_id, 'pixels')
        valid = get_array(queries, case_id, item_id, 'valid')
        return Surface(pixels, valid)

    def detector_frame(self, queries, case_id, source, role):
        surface = self.source_surface(queries, case_id, source, role=role)
        return Frame(surface.pixels, surface.valid, source['details']['full_scale'])

    def source_probe(self, case_id, source_id, payload):
        payload = fields.request_object(payload)
        with self.store.snapshot() as queries:
            record = queries.record(case_id, source_id, 'source')
            role = record['details']['role']
            if role in {'visible', 'mask'}:
                raise ImageFormatError('Read raw counts from a detector or calibration reference.')
            frame = self.detector_frame(queries, case_id, record, role)
            coords = fields.point(payload.get('point'), field='point', extent=frame.support.shape[::-1])
            x, y = (int(round(value)) for value in coords)
            radius = fields.integer(payload.get('radius', 2), field='radius', minimum=0, maximum=50)
            return interpolate_reference_probe(frame, (x, y), radius)

    def region_data(self, case, form):
        outline = parse_outline(form.get('geometry'), case_extent(case))
        bounds, mask = outline_pixels(outline, case_extent(case))
        included = int(np.count_nonzero(mask))
        color = fields.choice(form.get('category', 'interest'), field='category', options=REGION_TAGS)
        return {
            'geometry': outline.as_json(),
            'category': color,
            'bounds': list(bounds),
            'area_pixels': included,
            'square_mm': included * case['scale'] ** 2 if case['scale'] else None,
            'note': fields.text(form.get('note'), field='note', required=False, multiline=True, limit=1800),
        }

    def save_region(self, case_id, form, region_id=None):
        payload = fields.request_object(form)
        name = fields.text(payload.get('name'), field='name', limit=90)
        with self.store.snapshot() as queries:
            case = queries.case(case_id)
        details = self.region_data(case, payload)
        with self.store.transaction() as queries:
            if region_id is None:
                return queries.add(case_id, 'region', name, details)
            return queries.revise(case_id, region_id, 'region', expected_revision(payload), name, details)

    def lookup(self, case_id, record_id):
        with self.store.snapshot() as queries:
            return public_record(queries, case_id, queries.record(case_id, record_id))

    def remove(self, case_id, record_id, kind):
        kind = fields.choice(kind, field='kind', options={'region', 'session', 'source', 'profile', 'run', 'finding', 'archive', 'analysis', 'sheet'})
        with self.store.transaction() as queries:
            queries.remove(case_id, record_id, kind)
        return {'removed': record_id, 'kind': kind}
