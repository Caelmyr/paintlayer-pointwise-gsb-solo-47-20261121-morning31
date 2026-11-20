"""Coordinate numerical operations, case roles and immutable processing runs."""

from dataclasses import dataclass

import numpy as np

from . import fields
from .calibration import Correction, correct_plate
from .compare import Comparison, change_map
from .errors import CapacityExceeded, ExamError, InsufficientSupport
from .geometry import stored_outline
from .library import case_extent, expected_revision, get_array, public_record
from .measure import BandRequest, Transect, area_measurement, compare_outlines, pixel_probe, transect_measurement
from .mosaic import Placement, PlateData, assemble
from .quality import QualityGate, quality_audit, quality_scores
from .raster import Surface, as_bytes, display_range
from .registration import ALIGNMENT_MODELS, parse_points, register
from .settings import CASE_VIEWS, MAX_PLATE_INPUT_PIXELS
from .support import BandSearch, analyze_bands

NUMERICAL_RUNS = {'plate', 'mosaic'}


def name_or(raw, default):
    name = fields.text(raw.get('name'), field='name', limit=100, required=False)
    return name or default


@dataclass(frozen=True)
class RunOutput:
    details: dict
    arrays: dict
    dependencies: list


class Processing:
    def __init__(self, store, library):
        self.store = store
        self.library = library

    def read_source(self, queries, case_id, raw_id, role, required=True):
        if not required and not raw_id:
            return None
        source_id = fields.item_id(raw_id, field=role)
        record = queries.record(case_id, source_id, 'source')
        if record['details']['role'] != role:
            raise ExamError(f'This image is not assigned the {role} role.', field=role)
        return record

    def run(self, queries, case_id, run_id, allowed):
        key = fields.item_id(run_id, field='run')
        record = queries.record(case_id, key, 'run')
        if record['details']['kind'] not in allowed:
            raise ExamError('This operation requires a compatible examination result.', field='run')
        return record

    def clear_selection(self, selection):
        if selection == 'working':
            return 'working_run', {'mosaic'}
        if selection == 'photograph':
            return 'registration_run', {'registration'}
        raise ExamError('Choose a working view to change.', field='selection')

    def choose(self, case_id, form):
        raw = fields.request_object(form)
        field, allowed = self.clear_selection(raw.get('selection'))
        proposed = raw.get('run')
        expected = expected_revision(raw)
        with self.store.transaction() as queries:
            run = self.run(queries, case_id, proposed, allowed) if proposed else None
            return queries.set_working(case_id, field, run['id'] if run else None, expected)

    def create_profile(self, case_id, form):
        raw = fields.request_object(form)
        name = name_or(raw, 'Detector correction')
        settings = Correction.from_payload(raw)
        with self.store.snapshot() as queries:
            queries.case(case_id)
            dark = self.read_source(queries, case_id, raw.get('dark'), 'dark', required=False)
            flat = self.read_source(queries, case_id, raw.get('flat'), 'flat', required=False)
        if dark and flat:
            self.validate_profile_sources(dark, flat)
        dependencies = [(item, role) for item, role in ((dark, 'dark'), (flat, 'flat')) if item]
        details = settings.as_json() | {
            'dark': dark['id'] if dark else None,
            'flat': flat['id'] if flat else None,
        }
        with self.store.transaction() as queries:
            return public_record(
                queries, case_id, queries.add(case_id, 'profile', name, details, dependencies=dependencies)
            )

    def validate_profile_sources(self, dark, flat):
        left = dark['details']
        right = flat['details']
        if (left['width'], left['height'], left['bit_depth']) != (right['width'], right['height'], right['bit_depth']):
            raise ExamError('Dark and flat reference images must share dimensions and detector bit depth.', field='profile')

    def calibrate(self, case_id, form):
        raw = fields.request_object(form)
        with self.store.snapshot() as queries:
            detector = self.read_source(queries, case_id, raw.get('detector'), 'radiograph')
            profile = None
            if raw.get('profile'):
                profile_id = fields.item_id(raw.get('profile'), field='profile')
                profile = queries.record(case_id, profile_id, 'profile')
            inputs = profile['details'] if profile else raw
            dark = self.read_source(queries, case_id, inputs.get('dark'), 'dark', required=False)
            flat = self.read_source(queries, case_id, inputs.get('flat'), 'flat', required=False)
            mask = self.read_source(queries, case_id, raw.get('mask'), 'mask', required=False)
            result = self.corrected_result(queries, case_id, detector, profile, dark, flat, mask, inputs)
        return self.persist(case_id, 'plate', name_or(raw, f"{detector['name']} correction"), result)

    def corrected_result(self, queries, case_id, detector, profile, dark, flat, mask, inputs):
        correction = Correction.from_payload(inputs)
        detector_frame = self.library.detector_frame(queries, case_id, detector, 'radiograph')
        dark_frame = self.library.detector_frame(queries, case_id, dark, 'dark') if dark else None
        flat_frame = self.library.detector_frame(queries, case_id, flat, 'flat') if flat else None
        mask_data = self.library.source_surface(queries, case_id, mask, role='mask').pixels if mask else None
        corrected = correct_plate(detector_frame, dark_frame, flat_frame, mask_data, correction)
        dependencies = [
            (record, label)
            for record, label in (
                (detector, 'detector'),
                (profile, 'correction_profile'),
                (dark, 'dark_reference'),
                (flat, 'flat_reference'),
                (mask, 'acquisition_mask'),
            )
            if record
        ]
        arrays = {
            'signal': corrected.surface.pixels,
            'support': corrected.surface.valid,
            'weight': corrected.weight,
            'reasons': corrected.reasons,
            'raw': corrected.raw,
        }
        details = corrected.summary | {
            'model': correction.model,
            'correction': correction.as_json(),
            'detector': detector['id'],
            'session': detector['details']['session'],
            'session_date': detector['details']['session_date'],
        }
        return RunOutput(details, arrays, dependencies)

    def surface(self, queries, case_id, record):
        kind = record['details']['kind']
        label = {'registration': 'photo', 'compare': 'difference'}.get(kind, 'signal')
        return Surface(
            get_array(queries, case_id, record['id'], label),
            get_array(queries, case_id, record['id'], 'support'),
        )

    def optional_array(self, queries, case_id, record, label):
        if label not in CASE_VIEWS[record['details']['kind']]:
            return None
        return get_array(queries, case_id, record['id'], label)

    def assemble(self, case_id, form):
        raw = fields.request_object(form)
        radius = fields.integer(raw.get('radius', 30), field='radius', minimum=0, maximum=350)
        with self.store.snapshot() as queries:
            case = queries.case(case_id)
            extent = case_extent(case)
            placements = Placement.parse_many(raw.get('placements'), extent)
            records = [self.run(queries, case_id, entry.plate_id, {'plate'}) for entry in placements]
            units = {record['details']['unit'] for record in records}
            if len(units) != 1:
                raise ExamError('Assemble only plates calibrated in the same unit.', field='placements')
            total_extent = sum(np.prod(entry['details']['extent'], dtype=np.int64) for entry in records)
            if total_extent > MAX_PLATE_INPUT_PIXELS:
                raise CapacityExceeded('This assembly would exceed its safe total input dimensions.')
            inputs = [
                PlateData(
                    placement=placement,
                    surface=self.surface(queries, case_id, record),
                    weights=get_array(queries, case_id, record['id'], 'weight'),
                    reason_flags=get_array(queries, case_id, record['id'], 'reasons'),
                    label=record['name'],
                )
                for placement, record in zip(placements, records, strict=True)
            ]
        product = assemble(inputs, extent, radius, units.pop())
        arrays = {
            'signal': product.surface.pixels,
            'support': product.surface.valid,
            'coverage': product.coverage,
            'spread': product.spread,
            'owner': product.owner,
            'quality': product.quality,
        }
        details = product.summary | {
            'session_dates': sorted({row['details']['session_date'] for row in records if row['details']['session_date']}),
            'placements': [item.as_json() for item in placements],
        }
        output = RunOutput(details, arrays, [(record, 'calibrated_tile') for record in records])
        return self.persist(case_id, 'mosaic', name_or(raw, 'Assembled radiograph'), output)

    def register(self, case_id, form):
        raw = fields.request_object(form)
        mode = fields.choice(raw.get('mode', 'affine'), field='mode', options=ALIGNMENT_MODELS)
        with self.store.snapshot() as queries:
            case = queries.case(case_id)
            source = self.read_source(queries, case_id, raw.get('photograph'), 'visible')
            photograph = self.library.source_surface(queries, case_id, source, role='visible')
            points = parse_points(raw.get('points'), mode, case_extent(case), photograph.extent)
            target = self.run(queries, case_id, raw['mosaic'], {'mosaic'}) if raw.get('mosaic') else None
            target_support = self.surface(queries, case_id, target).valid if target else None
        result = register(mode, points, photograph, case_extent(case), target_support)
        arrays = {'photo': result.surface.pixels, 'support': result.surface.valid, 'grid': result.grid}
        dependencies = [(source, 'visible_reference')]
        if target is not None:
            dependencies.append((target, 'paired_radiograph'))
        details = result.summary | {'points': [item.record() for item in points], 'photograph': source['id']}
        output = RunOutput(details, arrays, dependencies)
        return self.persist(case_id, 'registration', name_or(raw, 'Registered photograph'), output)

    def region_input(self, queries, case_id, raw, field_name):
        value = raw.get(field_name)
        if not value:
            return None, None
        key = fields.item_id(value, field=field_name)
        record = queries.record(case_id, key, 'region')
        return record, stored_outline(record['details']['geometry'])

    def audit(self, case_id, form):
        raw = fields.request_object(form)
        gate = QualityGate.parse(raw)
        with self.store.snapshot() as queries:
            mosaic = self.run(queries, case_id, raw.get('run'), {'mosaic'})
            region, outline = self.region_input(queries, case_id, raw, 'region')
            base = self.surface(queries, case_id, mosaic)
            coverage = get_array(queries, case_id, mosaic['id'], 'coverage')
            spread = get_array(queries, case_id, mosaic['id'], 'spread')
            flags = get_array(queries, case_id, mosaic['id'], 'quality')
        summary, classification, response, domain = quality_audit(base, coverage, spread, flags, gate, outline)
        arrays = {
            'signal': np.where(domain, base.pixels, 0),
            'support': domain & base.valid,
            'classification': classification,
            'response': response,
            'domain': domain,
        }
        details = summary | {'display': mosaic['details']['display'], 'unit': mosaic['details']['unit']}
        related = [(mosaic, 'audited_surface')]
        if region:
            related.append((region, 'inspection_region'))
        output = RunOutput(details, arrays, related)
        return self.persist(case_id, 'quality', name_or(raw, 'Acquisition audit'), output)

    def support(self, case_id, form):
        raw = fields.request_object(form)
        settings = BandSearch.parse(raw)
        with self.store.snapshot() as queries:
            mosaic = self.run(queries, case_id, raw.get('run'), {'mosaic'})
            source = self.surface(queries, case_id, mosaic)
            region, outline = self.region_input(queries, case_id, raw, 'region')
        summary, response, usable, classified = analyze_bands(source, settings, outline)
        related = [(mosaic, 'examined_surface')]
        if region:
            related.append((region, 'inspection_region'))
        details = summary | {'display': mosaic['details']['display'], 'unit': mosaic['details']['unit']}
        arrays = {
            'signal': np.where(usable, source.pixels, 0),
            'support': usable,
            'response': response,
            'classification': classified,
        }
        output = RunOutput(details, arrays, related)
        return self.persist(case_id, 'support', name_or(raw, 'Stretcher band inspection'), output)

    def compare(self, case_id, form):
        raw = fields.request_object(form)
        definition = Comparison.parse(raw)
        with self.store.snapshot() as queries:
            earlier = self.run(queries, case_id, raw.get('earlier'), {'mosaic'})
            later = self.run(queries, case_id, raw.get('later'), {'mosaic'})
            if earlier['id'] == later['id']:
                raise ExamError('Choose separate earlier and later results.', field='later')
            if earlier['details']['unit'] != later['details']['unit']:
                raise ExamError('Choose visits calibrated with the same numerical model.', field='later')
            domain_region, domain = self.region_input(queries, case_id, raw, 'region')
            reference_region, reference = self.region_input(queries, case_id, raw, 'reference')
            earlier_data = self.surface(queries, case_id, earlier)
            later_data = self.surface(queries, case_id, later)
            source_spread = get_array(queries, case_id, earlier['id'], 'spread')
            later_spread = get_array(queries, case_id, later['id'], 'spread')
        summary, difference, uncertainty, classes = change_map(
            earlier_data, later_data, source_spread, later_spread, definition, domain, reference,
        )
        details = summary | {
            'unit': earlier['details']['unit'],
            'earlier': earlier['id'],
            'later': later['id'],
            'earlier_dates': earlier['details']['session_dates'],
            'later_dates': later['details']['session_dates'],
        }
        deps = [(earlier, 'earlier_surface'), (later, 'later_surface')]
        if domain_region:
            deps.append((domain_region, 'inspection_region'))
        if reference_region:
            deps.append((reference_region, 'stable_reference'))
        arrays = {
            'difference': difference.pixels,
            'support': difference.valid,
            'uncertainty': uncertainty,
            'classification': classes,
        }
        return self.persist(case_id, 'compare', name_or(raw, 'Visit comparison'), RunOutput(details, arrays, deps))

    def persist(self, case_id, kind, name, output):
        details = output.details | {'kind': kind, 'method_version': 1}
        blobs = [(key, as_bytes(array)) for key, array in output.arrays.items()]
        with self.store.transaction() as queries:
            record = queries.add(case_id, 'run', name, details, blob_data=blobs, dependencies=output.dependencies)
            return public_record(queries, case_id, record)

    def area(self, case_id, form):
        raw = fields.request_object(form)
        with self.store.snapshot() as queries:
            case = queries.case(case_id)
            run = self.run(queries, case_id, raw.get('run'), {'mosaic'})
            region, outline = self.region_input(queries, case_id, raw, 'region')
            if outline is None:
                raise ExamError('Choose an examination region.', field='region')
            surface = self.surface(queries, case_id, run)
            coverage = self.optional_array(queries, case_id, run, 'coverage')
        limits = fields.limits_pair(raw['limits'], field='limits', minimum=-40, maximum=66000) if raw.get('limits') is not None else None
        result, _, _ = area_measurement(
            surface, outline, scale=case['scale'], unit=run['details']['unit'], coverage=coverage, limit_pair=limits,
        )
        return result | {
            'region': region['id'],
            'region_revision': region['revision'],
            'run': run['id'],
        }

    def area_comparison(self, case_id, form):
        raw = fields.request_object(form)
        with self.store.snapshot() as queries:
            case = queries.case(case_id)
            run = self.run(queries, case_id, raw.get('run'), {'mosaic'})
            first, a = self.region_input(queries, case_id, raw, 'first')
            second, b = self.region_input(queries, case_id, raw, 'second')
            if a is None or b is None or first['id'] == second['id']:
                raise ExamError('Choose two different regions.', field='regions')
            surface = self.surface(queries, case_id, run)
        result = compare_outlines(surface, a, b, scale=case['scale'], unit=run['details']['unit'])
        return result | {
            'run': run['id'],
            'first_region': {'id': first['id'], 'revision': first['revision']},
            'second_region': {'id': second['id'], 'revision': second['revision']},
        }

    def transect(self, case_id, form):
        raw = fields.request_object(form)
        with self.store.snapshot() as queries:
            case = queries.case(case_id)
            run = self.run(queries, case_id, raw.get('run'), {'mosaic'})
            source = self.surface(queries, case_id, run)
        definition = Transect.parse(raw, source.extent)
        threshold = raw.get('threshold')
        threshold = fields.scalar(threshold, field='threshold', minimum=-40, maximum=66000) if threshold is not None else None
        band_request = None
        if raw.get('band_side'):
            if threshold is None:
                raise ExamError('Set a crossing threshold to measure band widths.', field='threshold')
            band_request = BandRequest.parse(raw)
        return transect_measurement(
            source, definition, scale=case['scale'], unit=run['details']['unit'],
            threshold=threshold, band_request=band_request,
        ) | {'run': run['id']}

    def run_probe(self, case_id, run_id, form):
        raw = fields.request_object(form)
        with self.store.snapshot() as queries:
            run = self.run(queries, case_id, run_id, NUMERICAL_RUNS)
            surface = self.surface(queries, case_id, run)
            coords = fields.point(raw.get('point'), field='point', extent=surface.extent)
            x, y = (int(round(point)) for point in coords)
            radius = fields.integer(raw.get('radius', 2), field='radius', minimum=0, maximum=50)
            result = pixel_probe(surface, (x, y), radius, unit=run['details']['unit'])
            if run['details']['kind'] == 'mosaic':
                result['coverage'] = int(get_array(queries, case_id, run_id, 'coverage')[y, x])
                result['overlap_spread'] = float(get_array(queries, case_id, run_id, 'spread')[y, x])
            return result

    def audit_box(self, case_id, run_id, form):
        raw = fields.request_object(form)
        with self.store.snapshot() as queries:
            record = self.run(queries, case_id, run_id, {'quality'})
            extent = tuple(record['details']['extent'])
            box = fields.box(raw.get('box'), field='box', extent=extent)
            tags = get_array(queries, case_id, run_id, 'classification')
            response = get_array(queries, case_id, run_id, 'response')
            domain = get_array(queries, case_id, run_id, 'domain')
        return quality_scores(tags, response, domain, box)
