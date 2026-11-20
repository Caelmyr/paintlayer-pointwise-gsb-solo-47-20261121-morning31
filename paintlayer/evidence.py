"""Capture immutable visual findings and assemble self-describing local exports."""

import csv
import hashlib
import io
from pathlib import PurePosixPath
import re
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from PIL import Image, ImageDraw

from . import fields
from .errors import CapacityExceeded, ExamError, InsufficientSupport
from .geometry import outline_pixels, stored_outline
from .library import case_extent, get_array, public_record
from .measure import area_measurement, Transect, transect_measurement
from .raster import png_bytes, tiff_bytes
from .settings import MAX_EXPORT_BYTES
from .store import json_text, now
from .views import DISPLAY_LAYERS, Overlay, Presentation

FINDING_STATUS = {'observation', 'review', 'confirmed'}
ARCHIVE_ITEMS = 12
EVIDENCE_IMAGE_SIDE = 1500


def download_name(title, extension):
    stem = re.sub(r'[^\w.-]+', '-', title, flags=re.UNICODE).strip('.-_')[:72]
    return f'{stem or "paintlayer"}{extension}'


def captioned_preview(array, bounds, canvas_case, region):
    rendered = Image.fromarray(array.copy())
    draw = ImageDraw.Draw(rendered, 'RGBA')
    x, y, width, height = bounds
    if region:
        outline = stored_outline(region['details']['geometry'])
        for ring in (outline.outer, *outline.holes):
            points = [(column - x, row - y) for column, row in ring]
            draw.line(points + points[:1], fill=(240, 181, 105, 255), width=max(1, width // 380))
    if canvas_case['scale']:
        add_scale_mark(draw, width, height, canvas_case['scale'])
    return rendered


def scale_mark_value(width, mm_per_pixel):
    max_mark = width * mm_per_pixel / 5
    choices = [step * 10.0 ** exponent for exponent in range(-3, 6) for step in (1, 2, 5)]
    feasible = [number for number in choices if number <= max_mark]
    return max(feasible, default=max_mark)


def add_scale_mark(draw, width, height, mm_per_pixel):
    if min(width, height) < 65:
        return
    physical = scale_mark_value(width, mm_per_pixel)
    length = max(1, int(round(physical / mm_per_pixel)))
    left = max(8, width // 28)
    baseline = height - max(15, height // 20)
    right = left + length
    draw.rectangle((left - 4, baseline - 16, right + 30, baseline + 7), fill=(27, 30, 30, 214))
    draw.line((left, baseline, right, baseline), fill=(252, 240, 205, 255), width=2)
    draw.line((left, baseline - 3, left, baseline + 4), fill=(252, 240, 205, 255), width=1)
    draw.line((right, baseline - 3, right, baseline + 4), fill=(252, 240, 205, 255), width=1)
    draw.text((left, baseline - 14), f'{physical:g} mm', fill=(252, 240, 205, 255))


def encode_image(image):
    image.thumbnail((EVIDENCE_IMAGE_SIDE, EVIDENCE_IMAGE_SIDE), Image.Resampling.LANCZOS)
    with io.BytesIO() as output:
        image.save(output, format='PNG', compress_level=5)
        return output.getvalue()


def csv_values(rows, keys):
    text = io.StringIO(newline='')
    writer = csv.DictWriter(text, fieldnames=keys, extrasaction='ignore')
    writer.writeheader()
    for row in rows:
        clean = {}
        for key in keys:
            value = row.get(key)
            if isinstance(value, str) and value.startswith(('=', '+', '-', '@', '\t', '\r')):
                value = "'" + value
            clean[key] = value
        writer.writerow(clean)
    return text.getvalue().encode('utf-8')


def finding_csv(findings):
    keys = (
        'id', 'name', 'status', 'captured', 'run_id', 'run_name', 'x', 'y',
        'width', 'height', 'region_id', 'region_revision', 'notes',
    )
    rows = []
    for item in findings:
        data = item['details']
        x, y, width, height = data['bounds']
        region = data.get('region') or {}
        rows.append({
            'id': item['id'],
            'name': item['name'],
            'status': data['status'],
            'captured': data['captured'],
            'run_id': data['run']['id'],
            'run_name': data['run']['name'],
            'x': x,
            'y': y,
            'width': width,
            'height': height,
            'region_id': region.get('id'),
            'region_revision': region.get('revision'),
            'notes': data['note'],
        })
    return csv_values(rows, keys)


def area_csv(result):
    stats = result['distribution']
    return csv_values(
        [
            {'metric': 'included_pixels', 'value': result['included_pixels'], 'units': 'pixel'},
            {'metric': 'measured_pixels', 'value': result['measured_pixels'], 'units': 'pixel'},
            {'metric': 'valid_fraction', 'value': result['valid_fraction'], 'units': 'proportion'},
            {'metric': 'measured_area', 'value': result['measured_square_mm'], 'units': 'mm²'},
            {'metric': 'mean', 'value': stats['mean'], 'units': result['units']},
            {'metric': 'median', 'value': stats['median'], 'units': result['units']},
            {'metric': 'minimum', 'value': stats['minimum'], 'units': result['units']},
            {'metric': 'maximum', 'value': stats['maximum'], 'units': result['units']},
            {'metric': 'standard_deviation', 'value': stats['std'], 'units': result['units']},
        ],
        ('metric', 'value', 'units'),
    )


def transect_csv(result):
    return csv_values(result['samples'], ('x', 'y', 'distance', 'distance_mm', 'value', 'contributors', 'possible'))


def quality_csv(record):
    rows = []
    keys = ('class', 'area_pixels', 'left', 'top', 'width', 'height', 'peak_score')
    for group in record['details']['classes']:
        for component in group['details']:
            x, y, width, height = component['bounds']
            rows.append({
                'class': group['key'],
                'area_pixels': component['area'],
                'left': x,
                'top': y,
                'width': width,
                'height': height,
                'peak_score': component['peak_score'],
            })
    return csv_values(rows, keys)


class Bundle:
    def __init__(self, case):
        self.case = case
        self._files = {}
        self._entries = {}
        self._total = 0
        self.unresolved_inputs = set()

    def add(self, path, data, kind):
        path = str(PurePosixPath(path))
        if path in self._files or path.startswith('/') or '..' in PurePosixPath(path).parts:
            raise ValueError('Invalid evidence package path')
        if self._total + len(data) > MAX_EXPORT_BYTES:
            raise CapacityExceeded('Select fewer or smaller images for one evidence bundle.')
        self._total += len(data)
        self._files[path] = data
        self._entries[path] = {
            'kind': kind,
            'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest(),
        }

    def add_json(self, path, data, label):
        self.add(path, json_text(data).encode('utf-8'), label)

    def manifest(self, *, runs, findings):
        return {
            'format': 'PaintLayer examination evidence',
            'version': 2,
            'created': now(),
            'synthetic': self.case['synthetic'],
            'examination': self.case,
            'runs': [run['id'] for run in runs],
            'findings': [finding['id'] for finding in findings],
            'coordinates': 'Zero-based pixel centre coordinates in the shared examination canvas',
            'validity': 'Use TIFF validity alongside numerical rasters, including where a supported value is zero.',
            'files': self._entries.copy(),
            'unresolved_inputs': [
                {'record': record, 'revision': revision}
                for record, revision in sorted(self.unresolved_inputs)
            ],
        }

    def finish(self, manifest):
        self.add_json('manifest.json', manifest, 'manifest')
        with io.BytesIO() as output:
            with ZipFile(output, mode='w', compression=ZIP_DEFLATED, compresslevel=4) as archive:
                for name, payload in self._files.items():
                    archive.writestr(name, payload)
            payload = output.getvalue()
        return payload, hashlib.sha256(payload).hexdigest()


def region_snapshot(region):
    if region is None:
        return None
    return {
        'id': region['id'],
        'name': region['name'],
        'revision': region['revision'],
        'geometry': region['details']['geometry'],
    }


def run_snapshot(record):
    return {'id': record['id'], 'name': record['name'], 'kind': record['details']['kind']}


class Evidence:
    def __init__(self, store, library, processing, views):
        self.store = store
        self.library = library
        self.processing = processing
        self.views = views

    def capture(self, case_id, form):
        raw = fields.request_object(form)
        name = fields.text(raw.get('name'), field='name', limit=100)
        note = fields.text(raw.get('note'), field='note', limit=3500, required=False, multiline=True)
        status = fields.choice(raw.get('status', 'observation'), field='status', options=FINDING_STATUS)
        presentation = Presentation.parse(raw.get('display', {}))
        with self.store.snapshot() as queries:
            case = queries.case(case_id)
            run_key = fields.item_id(raw.get('run'), field='run')
            run = queries.record(case_id, run_key, 'run')
            base_kind = run['details']['kind']
            extent = tuple(run['details']['extent'])
            if extent != case_extent(case) or base_kind == 'plate':
                raise ExamError('Capture a result located in the shared examination canvas.')
            region, outline = self.processing.region_input(queries, case_id, raw, 'region')
            bounds = self.finding_bounds(raw, extent, outline)
            photo, overlay = self.finding_photo(queries, case_id, raw, run)
            layer = fields.choice(
                raw.get('layer', 'photo' if base_kind == 'registration' else 'difference' if base_kind == 'compare' else 'signal'),
                field='layer',
                options=DISPLAY_LAYERS[base_kind],
            )
            rgba = self.finding_view(queries, case_id, run, photo, layer, presentation, overlay, bounds)
            if not np.any(rgba[..., 3]):
                raise InsufficientSupport('The selected crop contains no visible information in this view.')
            measurements = self.finding_metrics(queries, case, run, outline, raw)
            expected_input = queries.blob_info(case_id, run_key)
        labelled = captioned_preview(rgba, bounds, case, region)
        stamps = {
            'status': status,
            'notes_unit': 'captured scene',
            'note': note,
            'captured': now(),
            'bounds': list(bounds),
            'run': run_snapshot(run),
            'photograph': run_snapshot(photo) if photo else None,
            'region': region_snapshot(region),
            'display': presentation.as_json(),
            'overlay': overlay.as_json() if overlay else None,
            'layer': layer,
            'measurement': measurements,
            'source_blobs': expected_input,
        }
        dependencies = [(run, 'captured_result')]
        if photo:
            dependencies.append((photo, 'registered_reference'))
        if region:
            dependencies.append((region, 'captured_region'))
        image = encode_image(labelled)
        with self.store.transaction() as queries:
            record = queries.add(case_id, 'finding', name, stamps, blob_data=[('snapshot', image)], dependencies=dependencies)
            return public_record(queries, case_id, record)

    def finding_bounds(self, raw, extent, outline):
        if raw.get('box'):
            return fields.box(raw.get('box'), field='box', extent=extent)
        if outline is None:
            return 0, 0, extent[0], extent[1]
        x, y, width, height = outline.bounds(extent)
        margin = max(6, int(max(width, height) * 0.1))
        x0, y0 = max(0, x - margin), max(0, y - margin)
        x1, y1 = min(extent[0], x + width + margin), min(extent[1], y + height + margin)
        return x0, y0, x1 - x0, y1 - y0

    def finding_photo(self, queries, case_id, raw, run):
        key = raw.get('registration')
        if not key:
            return None, None
        if run['details']['kind'] != 'mosaic':
            raise ExamError('A blended finding must capture an assembled radiograph.')
        role = self.processing.run(queries, case_id, key, {'registration'})
        return role, Overlay.parse(raw.get('overlay', {}))

    def finding_view(self, queries, case_id, run, photo, layer, presentation, overlay, bounds):
        if photo is not None:
            return self.views.compose(queries, case_id, run, photo, presentation, overlay, bounds)
        return self.views.render(queries, case_id, run, layer, presentation, bounds)

    def finding_metrics(self, queries, case, run, outline, raw):
        if run['details']['kind'] != 'mosaic':
            return {}
        if outline is None and not raw.get('transect'):
            return {}
        case_id = case['id']
        surface = self.processing.surface(queries, case_id, run)
        unit = run['details']['unit']
        result = {}
        if outline is not None:
            measurement, _, _ = area_measurement(surface, outline, scale=case['scale'], unit=unit)
            result['area'] = measurement
        if raw.get('transect'):
            definition = Transect.parse(raw['transect'], surface.extent)
            result['transect'] = transect_measurement(surface, definition, scale=case['scale'], unit=unit)
        return result

    def create_archive(self, case_id, form):
        raw = fields.request_object(form)
        name = fields.text(raw.get('name'), field='name', limit=100)
        runs = self.archive_choices(raw, 'runs')
        findings = self.archive_choices(raw, 'findings')
        include_numeric = fields.flag(raw.get('include_numeric', True), field='include_numeric')
        if not runs and not findings:
            raise ExamError('Select at least one run or finding for the evidence bundle.')
        with self.store.snapshot() as queries:
            case = queries.case(case_id)
            bundle = Bundle(case)
            run_data = [queries.record(case_id, key, 'run') for key in runs]
            finding_data = [queries.record(case_id, key, 'finding') for key in findings]
            for index, item in enumerate(run_data, start=1):
                self.add_run(queries, case_id, bundle, item, index, include_numeric)
            for index, item in enumerate(finding_data, start=1):
                self.add_finding(queries, case_id, bundle, item, index)
            if finding_data:
                bundle.add('findings/index.csv', finding_csv(finding_data), 'finding_index')
            manifest = bundle.manifest(runs=run_data, findings=finding_data)
        packed, digest = bundle.finish(manifest)
        summary = {
            'sha256': digest,
            'bytes': len(packed),
            'members': len(manifest['files']) + 1,
            'selected_runs': runs,
            'selected_findings': findings,
            'numeric_tiffs': include_numeric,
            'unresolved_input_revisions': len(bundle.unresolved_inputs),
        }
        dependencies = [(item, 'included_result') for item in run_data] + [
            (item, 'included_finding') for item in finding_data
        ]
        with self.store.transaction() as queries:
            record = queries.add(case_id, 'archive', name, summary, blob_data=[('bundle', packed)], dependencies=dependencies)
            return public_record(queries, case_id, record)

    def archive_choices(self, raw, field_name):
        options = fields.sequence(raw.get(field_name, []), field=field_name, maximum=ARCHIVE_ITEMS)
        keys = [fields.item_id(value, field=field_name) for value in options]
        return fields.unique(keys, field=field_name)

    def add_run(self, queries, case_id, bundle, run, index, with_numeric):
        labels = {'registration': 'photo', 'compare': 'difference'}
        kind = run['details']['kind']
        layer = labels.get(kind, 'signal')
        root = f'runs/{index:02d}'
        default = Presentation(None, 1, False, 'grayscale')
        preview = self.views.render(queries, case_id, run, layer, default)
        bundle.add(f'{root}/preview.png', png_bytes(preview), 'run_preview')
        bundle.add_json(f'{root}/record.json', run, 'processing_record')
        lineage = queries.lineage(case_id, run['id'])
        bundle.unresolved_inputs.update(
            (item['id'], item['revision']) for item in lineage
            if item['revision_status'] == 'unresolved_revision'
        )
        bundle.add_json(f'{root}/lineage.json', lineage, 'run_lineage')
        if kind == 'quality':
            bundle.add(f'{root}/components.csv', quality_csv(run), 'acquisition_components')
        if with_numeric:
            self.add_numeric(queries, case_id, bundle, run, root, layer)

    def add_numeric(self, queries, case_id, bundle, run, root, layer):
        record_id = run['id']
        data = get_array(queries, case_id, record_id, layer)
        usable = get_array(queries, case_id, record_id, 'support')
        suffix = 'photo' if data.ndim == 3 else 'values'
        bundle.add(f'{root}/{suffix}.tiff', tiff_bytes(data), 'numerical_raster')
        bundle.add(f'{root}/validity.tiff', tiff_bytes(usable.astype(np.uint8) * 255), 'usable_sample_mask')
        kind = run['details']['kind']
        outputs = {
            'mosaic': ('coverage', 'spread', 'owner'),
            'plate': ('reasons', 'raw', 'owner', 'conflict', 'residual'),
            'registration': ('grid',),
            'quality': ('classification', 'response'),
            'support': ('classification', 'response'),
            'compare': ('classification', 'uncertainty'),
        }[kind]
        for name in outputs:
            if name not in queries.blob_info(case_id, run['id']):
                continue
            raster = get_array(queries, case_id, record_id, name)
            bundle.add(f'{root}/{name}.tiff', tiff_bytes(raster), 'auxiliary_raster')

    def add_finding(self, queries, case_id, bundle, finding, index):
        root = f'findings/{index:02d}'
        snapshot, _ = queries.blob_path(case_id, finding['id'], 'snapshot')
        bundle.add(f'{root}/marked-crop.png', snapshot.read_bytes(), 'finding_image')
        bundle.add_json(f'{root}/record.json', finding, 'finding_record')
        metrics = finding['details']['measurement']
        if 'area' in metrics:
            bundle.add(f'{root}/area.csv', area_csv(metrics['area']), 'area_measurements')
        if 'transect' in metrics:
            bundle.add(f'{root}/transect.csv', transect_csv(metrics['transect']), 'transect_samples')


def verify_package(path, expected_digest):
    from json import JSONDecodeError, loads
    from zipfile import BadZipFile

    with path.open('rb') as stream:
        expected_file = hashlib.file_digest(stream, 'sha256').hexdigest()
    valid_container = expected_file == expected_digest
    if not valid_container:
        return {
            'verified': False,
            'reason': 'The stored bundle does not match the saved SHA-256.',
            'computed_sha256': expected_file,
            'member_count': None,
        }
    try:
        with ZipFile(path) as archive:
            entries = archive.infolist()
            files = archive.namelist()
            if len(files) != len(set(files)) or 'manifest.json' not in files:
                raise ValueError('Manifest is missing or a path occurs twice.')
            if sum(item.file_size for item in entries) > MAX_EXPORT_BYTES:
                raise ValueError('Bundle exceeds its declared size allowance.')
            manifest = loads(archive.read('manifest.json'))
            expected = manifest.get('files')
            if not isinstance(expected, dict):
                raise ValueError('The manifest does not declare its file list.')
            if set(files) - {'manifest.json'} != set(expected):
                raise ValueError('The package file list does not match its manifest.')
            errors = []
            for filename, information in expected.items():
                entry = archive.getinfo(filename)
                if entry.file_size != information.get('bytes'):
                    errors.append({'path': filename, 'problem': 'size'})
                    continue
                with archive.open(entry) as data:
                    digest = hashlib.file_digest(data, 'sha256').hexdigest()
                if digest != information.get('sha256'):
                    errors.append({'path': filename, 'problem': 'sha256'})
    except (BadZipFile, JSONDecodeError, KeyError, OSError, TypeError, ValueError) as exc:
        return {
            'verified': False,
            'reason': str(exc),
            'member_count': None,
            'computed_sha256': expected_file,
        }
    return {
        'verified': not errors,
        'computed_sha256': expected_file,
        'member_count': len(files),
        'errors': errors,
    }
