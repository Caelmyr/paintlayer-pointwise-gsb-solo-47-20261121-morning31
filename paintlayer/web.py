"""Responsive local application routes and HTTP representations."""

from functools import lru_cache, wraps
from pathlib import Path
from threading import BoundedSemaphore

from flask import Flask, Response, jsonify, request, send_file
from werkzeug.exceptions import BadRequest, HTTPException, RequestEntityTooLarge, UnsupportedMediaType

from . import __version__, fields
from .analysis import Analysis
from .atlas import Atlas
from .demo import load_demo, training_artwork
from .errors import CapacityExceeded, ExamError, MissingRecord
from .evidence import Evidence, download_name, verify_package
from .intake import inspect_pages, request_options, upload_bytes
from .library import Library
from .processing import Processing
from .settings import MAX_REQUEST_BYTES, MAX_SIDE, MAX_SURFACE_PIXELS, MAX_UPLOAD_BYTES, SOURCE_ROLES
from .store import Store
from .views import Views

STATIC_ROOT = Path(__file__).resolve().parents[1] / 'static'


def body():
    if not request.is_json:
        raise ExamError('Send an application/json request body.')
    try:
        payload = request.get_json()
    except (BadRequest, UnsupportedMediaType) as exc:
        raise ExamError('The request body is not valid JSON.') from exc
    return fields.request_object(payload)


def image_field(raw, format_name):
    if format_name == 'list':
        try:
            return [float(value) for value in raw.split(',')]
        except ValueError as exc:
            raise ExamError('Use a comma-separated list of numbers.') from exc
    if format_name == 'int':
        try:
            return int(raw)
        except ValueError as exc:
            raise ExamError('Use a valid integer.') from exc
    if format_name == 'float':
        try:
            return float(raw)
        except ValueError as exc:
            raise ExamError('Use a finite number.') from exc
    if format_name == 'bool':
        if raw not in ('1', '0', 'true', 'false'):
            raise ExamError('Use a valid flag.')
        return raw in ('1', 'true')
    return raw


def img_query():
    display_fields = {'gamma': 'float', 'limits': 'list', 'invert': 'bool', 'palette': 'text'}
    view_fields = {'mode': 'text', 'opacity': 'float', 'division': 'float', 'block': 'int'}
    display = {
        name: image_field(request.args[name], kind)
        for name, kind in display_fields.items()
        if name in request.args
    }
    overlay = {
        name: image_field(request.args[name], kind)
        for name, kind in view_fields.items()
        if name in request.args
    }
    result = {'display': display, 'overlay': overlay}
    for name in ('layer', 'mosaic', 'registration'):
        if name in request.args:
            result[name] = request.args[name]
    if request.args.get('edge'):
        result['edge'] = image_field(request.args['edge'], 'int')
    if request.args.get('box'):
        raw_box = request.args['box'].split(',')
        result['box'] = [image_field(value, 'int') for value in raw_box]
    return result


def image_response(data):
    return Response(data, mimetype='image/png', headers={'Cache-Control': 'private, max-age=900'})


def quality_payload(error):
    return jsonify({'error': error.as_dict()}), error.status


def policy_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['Referrer-Policy'] = 'no-referrer'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['Content-Security-Policy'] = (
        "default-src 'self'; img-src 'self' data: blob:; font-src 'self'; "
        "style-src 'self' 'unsafe-inline'; frame-ancestors 'none'; object-src 'none'; form-action 'self'"
    )
    if request.path.startswith('/api/') and 'Cache-Control' not in response.headers:
        response.headers['Cache-Control'] = 'private, no-store'
    return response


def create_app(runtime):
    app = Flask('paintlayer', static_folder=str(STATIC_ROOT), static_url_path='/assets')
    app.config.update(
        MAX_CONTENT_LENGTH=MAX_REQUEST_BYTES,
        MAX_FORM_MEMORY_SIZE=512 * 1024,
        MAX_FORM_PARTS=25,
        JSON_SORT_KEYS=False,
    )
    app.json.sort_keys = False
    store = Store(runtime.data_root)
    library = Library(store)
    processing = Processing(store, library)
    views = Views(store, library, processing)
    evidence = Evidence(store, library, processing, views)
    analysis = Analysis(store, library, processing)
    atlas = Atlas(store, processing, views)
    app.extensions['paintlayer'] = {
        'store': store,
        'library': library,
        'processing': processing,
        'views': views,
        'evidence': evidence,
        'analysis': analysis,
        'atlas': atlas,
    }
    active_heavy = BoundedSemaphore(1)

    def image_work(operation):
        @wraps(operation)
        def do_work(*args, **kwargs):
            if not active_heavy.acquire(timeout=18):
                err = ExamError('The previous image operation is still running. Retry in a moment.')
                err.status = 503
                err.code = 'image_worker_busy'
                raise err
            try:
                return operation(*args, **kwargs)
            finally:
                active_heavy.release()
        return do_work

    app.after_request(policy_headers)

    @app.errorhandler(ExamError)
    def invalid_operation(error):
        return quality_payload(error)

    @app.errorhandler(RequestEntityTooLarge)
    def oversized_request(error):
        return quality_payload(CapacityExceeded('The uploaded request exceeds the maximum size.'))

    @app.errorhandler(HTTPException)
    def http_error(error):
        if request.path.startswith('/api/'):
            return jsonify({'error': {'code': 'http_error', 'message': error.description}}), error.code
        return error

    @app.get('/')
    def landing():
        return app.send_static_file('index.html')

    cached_training = lru_cache(maxsize=1)(training_artwork)

    @app.get('/media/training-artwork.png')
    @image_work
    def training_artwork_view():
        return image_response(cached_training())

    @app.get('/api/health')
    def health():
        with store.snapshot() as queries:
            queries.db.execute('SELECT 1 FROM cases LIMIT 1').fetchone()
        return {'status': 'ok', 'version': __version__}

    @app.get('/api/about')
    def about():
        return {
            'name': 'PaintLayer Desk',
            'version': __version__,
            'source_roles': SOURCE_ROLES,
            'limits': {'side': MAX_SIDE, 'canvas_pixels': MAX_SURFACE_PIXELS, 'upload_bytes': MAX_UPLOAD_BYTES},
        }

    @app.get('/api/cases')
    def case_catalog():
        with store.snapshot() as queries:
            return {'cases': queries.catalog()}

    @app.post('/api/cases')
    def create_case():
        return library.create(body()), 201

    @app.post('/api/demo')
    @image_work
    def training():
        return load_demo(library, processing, store), 201

    @app.get('/api/cases/<case_id>')
    def case_contents(case_id):
        return library.contents(case_id)

    @app.put('/api/cases/<case_id>')
    def update_case(case_id):
        return library.revise_case(case_id, body())

    @app.post('/api/cases/<case_id>/sessions')
    def create_session(case_id):
        return library.create_session(case_id, body()), 201

    @app.put('/api/cases/<case_id>/sessions/<session_id>')
    def update_session(case_id, session_id):
        return library.revise_session(case_id, session_id, body())

    @app.put('/api/cases/<case_id>/working')
    def update_workspace(case_id):
        return processing.choose(case_id, body())

    @app.post('/api/cases/<case_id>/sources')
    @image_work
    def intake(case_id):
        return library.receive(case_id, request.files.get('image'), request.form), 201

    @app.post('/api/cases/<case_id>/sources/inspect')
    @image_work
    def inspect_source_frames(case_id):
        with store.snapshot() as queries:
            queries.case(case_id)
        role, start = request_options(request.form)
        payload = upload_bytes(request.files.get('image'))
        return inspect_pages(payload, role, start)

    @app.get('/api/cases/<case_id>/sources/<source_id>/view.png')
    @image_work
    def view_source(case_id, source_id):
        return image_response(views.source_image(case_id, source_id, img_query()))

    @app.post('/api/cases/<case_id>/sources/<source_id>/probe')
    @image_work
    def measure_source(case_id, source_id):
        return library.source_probe(case_id, source_id, body())

    @app.post('/api/cases/<case_id>/profiles')
    def create_profile(case_id):
        return processing.create_profile(case_id, body()), 201

    @app.post('/api/cases/<case_id>/regions')
    @image_work
    def create_region(case_id):
        return library.save_region(case_id, body()), 201

    @app.put('/api/cases/<case_id>/regions/<region_id>')
    @image_work
    def update_region(case_id, region_id):
        return library.save_region(case_id, body(), region_id=region_id)

    @app.post('/api/cases/<case_id>/runs/<operation>')
    @image_work
    def create_run(case_id, operation):
        handler = {
            'calibrate': processing.calibrate,
            'assemble': processing.assemble,
            'register': processing.register,
            'quality': processing.audit,
            'support': processing.support,
            'compare': processing.compare,
        }.get(operation)
        if handler is None:
            raise MissingRecord('This imaging operation is unavailable.')
        return handler(case_id, body()), 201

    @app.get('/api/cases/<case_id>/runs/<run_id>/view.png')
    @image_work
    def view_run(case_id, run_id):
        return image_response(views.run_image(case_id, run_id, img_query()))

    @app.get('/api/cases/<case_id>/overlay.png')
    @image_work
    def view_overlay(case_id):
        return image_response(views.combined(case_id, img_query()))

    @app.post('/api/cases/<case_id>/runs/<run_id>/probe')
    @image_work
    def probe_run(case_id, run_id):
        return processing.run_probe(case_id, run_id, body())

    @app.post('/api/cases/<case_id>/runs/<run_id>/audit-box')
    @image_work
    def inspect_quality_box(case_id, run_id):
        return processing.audit_box(case_id, run_id, body())

    @app.post('/api/cases/<case_id>/regions/compare')
    @image_work
    def compare_regions(case_id):
        return processing.area_comparison(case_id, body())

    @app.post('/api/cases/<case_id>/analysis/<kind>')
    @image_work
    def analyze_case(case_id, kind):
        payload = body()
        persist = fields.flag(payload.get('save', False), field='save')
        result = analysis.save(case_id, kind, payload, persist=persist)
        return result, 201 if persist else 200

    @app.post('/api/cases/<case_id>/findings')
    @image_work
    def capture_finding(case_id):
        return evidence.capture(case_id, body()), 201

    @app.post('/api/cases/<case_id>/archives')
    @image_work
    def archive_evidence(case_id):
        return evidence.create_archive(case_id, body()), 201

    @app.get('/api/cases/<case_id>/archives/<archive_id>/verify')
    @image_work
    def inspect_bundle(case_id, archive_id):
        with store.snapshot() as queries:
            record = queries.record(case_id, archive_id, 'archive')
            path, _ = queries.blob_path(case_id, archive_id, 'bundle')
            return verify_package(path, record['details']['sha256'])

    @app.post('/api/cases/<case_id>/sheets/preview')
    @image_work
    def preview_sheet(case_id):
        return image_response(atlas.preview(case_id, body()))

    @app.post('/api/cases/<case_id>/sheets')
    @image_work
    def save_sheet(case_id):
        return atlas.save(case_id, body()), 201

    @app.get('/api/cases/<case_id>/records/<record_id>')
    def record_metadata(case_id, record_id):
        return library.lookup(case_id, record_id)

    @app.delete('/api/cases/<case_id>/records/<kind>/<record_id>')
    @image_work
    def drop_record(case_id, kind, record_id):
        return library.remove(case_id, record_id, kind)

    @app.get('/api/cases/<case_id>/records/<record_id>/files/<label>')
    def download(case_id, record_id, label):
        with store.snapshot() as queries:
            record = queries.record(case_id, record_id)
            path, blob = queries.blob_path(case_id, record_id, label)
            kind = record['kind']
            media, extension, inline = file_representation(record, label)
            opened = path.open('rb')
        filename = download_name(record['name'], extension)
        response = send_file(opened, mimetype=media, as_attachment=not inline, download_name=filename, etag=blob['sha256'])
        response.headers['Content-Length'] = blob['bytes']
        return response

    return app


def file_representation(record, label):
    details = record['details']
    kind = record['kind']
    if kind == 'sheet':
        if label == 'preview':
            return 'image/png', '.png', True
        if label == 'pdf':
            return 'application/pdf', '.pdf', False
        if label == 'layout':
            return 'application/json', '-layout.json', False
    if kind == 'source' and label == 'original':
        ext = {'PNG': '.png', 'JPEG': '.jpg', 'TIFF': '.tiff'}[details['format']]
        return details['content_type'], ext, False
    if kind == 'finding' and label == 'snapshot':
        return 'image/png', '.png', True
    if kind == 'archive' and label == 'bundle':
        return 'application/zip', '.zip', False
    if kind == 'analysis' and label == 'csv':
        return 'text/csv', '.csv', False
    if kind == 'analysis' and label in {'preview', 'flags', 'spectrum'}:
        return 'image/png', f'-{label}.png', True
    if kind in {'source', 'run'}:
        return 'application/octet-stream', f'-{label}.npy', False
    raise MissingRecord('This file representation cannot be downloaded.')
