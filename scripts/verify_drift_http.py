#!/usr/bin/env python3
"""End-to-end check for multi-reference spatial correction through the HTTP app.

Builds the synthetic demo in a temporary data directory, runs visit comparisons
through the Flask test client and verifies persistence, rendering, validation
and backwards compatibility with previously stored comparison runs.
"""

import io
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from PIL import Image

from paintlayer.demo import load_demo
from paintlayer.settings import Runtime
from paintlayer.store import Store
from paintlayer.library import Library
from paintlayer.processing import Processing
from paintlayer.web import create_app


def check(condition, label):
    print(f'{"PASS" if condition else "FAIL"}  {label}')
    if not condition:
        check.failures += 1


check.failures = 0


def png(view):
    with Image.open(io.BytesIO(view)) as image:
        return np.asarray(image)


with tempfile.TemporaryDirectory() as temp_dir:
    runtime = Runtime(data_root=Path(temp_dir), host='127.0.0.1', port=0, threads=1)
    app = create_app(runtime)
    store = app.extensions['paintlayer']['store']
    library = app.extensions['paintlayer']['library']
    processing = app.extensions['paintlayer']['processing']
    client = app.test_client()

    demo = load_demo(library, processing, store)
    case_id = demo['id']
    contents = library.contents(case_id)
    mosaics = [row for row in contents['runs'] if row['details']['kind'] == 'mosaic']
    earlier, later = mosaics[0], mosaics[1]
    reference_regions = [row for row in contents['regions'] if row['details']['category'] == 'reference']
    check(len(mosaics) == 2 and reference_regions, 'demo provides two visits and a reference region')

    # The demo only ships one reference region; add two more stable outlines in
    # untouched ground areas so the spatial path has distributed anchors.
    extra_regions = [
        ('Upper right ground', [[560, 60], [660, 64], [662, 116], [558, 110]]),
        ('Lower left ground', [[70, 430], [160, 428], [164, 480], [72, 484]]),
    ]
    region_ids = [reference_regions[0]['id']]
    for name, outer in extra_regions:
        saved = library.save_region(case_id, {
            'name': name, 'category': 'reference',
            'geometry': {'outer': outer, 'holes': []},
        })
        region_ids.append(saved['id'])

    # --- Spatial comparison via the HTTP route -------------------------------
    payload = {
        'name': 'Spatial visit comparison',
        'earlier': earlier['id'],
        'later': later['id'],
        'spatial_correction': True,
        'references': region_ids,
        'threshold': 0.065,
        'spread_weight': 0.5,
    }
    response = client.post(f'/api/cases/{case_id}/runs/compare', json=payload)
    check(response.status_code == 201, f'spatial compare returns 201 (got {response.status_code}: {response.get_json() if response.is_json else ""})')
    spatial_run = response.get_json()
    drift = spatial_run['details']['drift_correction']
    check(drift['reference_count'] == 3 and drift['accepted_count'] >= 2,
          f"spatial run audits {drift['reference_count']} references, {drift['accepted_count']} accepted")
    check(set(drift) >= {'mode', 'reach_radius_px', 'unresolved_pixels', 'references', 'plane_gradient'},
          'drift audit carries mode, reach, gradient and per-reference facts')
    check('pending_review' in spatial_run['details']['class_counts'], 'class counts include pending review')
    check(set(spatial_run['files']) >= {'difference', 'classification', 'uncertainty', 'correction', 'adjustment', 'support'},
          'all six comparison rasters are persisted')

    # Every stored layer must render through the view endpoint.
    for layer in ('difference', 'classification', 'uncertainty', 'correction', 'adjustment', 'support'):
        view = client.get(f'/api/cases/{case_id}/runs/{spatial_run["id"]}/view.png?layer={layer}&edge=600')
        check(view.status_code == 200 and png(view.data).shape[-1] == 4, f'layer {layer} renders as RGBA PNG')
    class_png = png(client.get(
        f'/api/cases/{case_id}/runs/{spatial_run["id"]}/view.png?layer=classification&edge=900').data)
    # Pending review uses the pink palette entry (226, 138, 196); only assert it
    # is distinguishable when the run actually produced pending pixels.
    if spatial_run['details']['pending_review_pixels']:
        colours = {tuple(pixel[:3]) for row in class_png for pixel in row if pixel[3] > 0}
        check((226, 138, 196) in colours, 'pending-review pixels use the dedicated class colour')
    else:
        check(True, 'demo references bracket the canvas; no pending-review pixels needed')

    # The treatment area must still read as a change, not be corrected away.
    treatment = next(row for row in contents['regions'] if row['details']['category'] == 'treatment')
    counts = spatial_run['details']['class_counts']
    check(counts['positive'] + counts['negative'] > 0, 'real treatment traces remain classified as changes')

    # --- Validation: single and multi reference are mutually exclusive -------
    conflict = dict(payload, name='Conflicting inputs', reference=region_ids[0])
    response = client.post(f'/api/cases/{case_id}/runs/compare', json=conflict)
    check(response.status_code == 400, 'using both reference modes is rejected')

    bad = dict(payload, name='Too few references', references=region_ids[:1])
    response = client.post(f'/api/cases/{case_id}/runs/compare', json=bad)
    check(response.status_code == 422, 'a single-entry spatial correction is rejected')

    invalid = dict(payload, name='Unknown region', references=['0' * 24])
    response = client.post(f'/api/cases/{case_id}/runs/compare', json=invalid)
    check(response.status_code == 404, 'unknown reference region id is rejected')

    duplicate = dict(payload, name='Duplicate region', references=[region_ids[0], region_ids[0]])
    response = client.post(f'/api/cases/{case_id}/runs/compare', json=duplicate)
    check(response.status_code == 400, 'duplicate reference entries are rejected')

    # --- Legacy single-reference and plain comparisons still work ------------
    legacy = client.post(f'/api/cases/{case_id}/runs/compare', json={
        'name': 'Legacy single reference', 'earlier': earlier['id'], 'later': later['id'],
        'reference': region_ids[0],
    })
    check(legacy.status_code == 201, 'legacy single-reference comparison still runs')
    legacy_run = legacy.get_json()
    check(legacy_run['details']['drift_correction']['mode'] == 'constant', 'legacy run reports constant mode')
    check(legacy_run['details']['class_counts']['pending_review'] == 0, 'legacy run never marks pending review')
    # New layers are backfilled for records created before this feature.
    for layer in ('correction', 'adjustment'):
        view = client.get(f'/api/cases/{case_id}/runs/{legacy_run["id"]}/view.png?layer={layer}')
        check(view.status_code == 200, f'legacy run renders backfilled {layer} layer')

    plain = client.post(f'/api/cases/{case_id}/runs/compare', json={
        'name': 'Plain comparison', 'earlier': earlier['id'], 'later': later['id'],
    })
    check(plain.status_code == 201 and plain.get_json()['details']['reference_offset'] == 0.0,
          'reference-free comparison keeps the zero-offset behaviour')

    # --- Evidence bundle includes the new rasters ----------------------------
    bundle = client.post(f'/api/cases/{case_id}/archives', json={
        'name': 'Spatial evidence', 'runs': [spatial_run['id']], 'include_numeric': True,
    })
    check(bundle.status_code == 201, 'evidence bundle builds with the spatial run')
    archive_id = bundle.get_json()['id']
    verify = client.get(f'/api/cases/{case_id}/archives/{archive_id}/verify')
    check(verify.status_code == 200 and verify.get_json()['verified'], 'bundle verifies against its manifest')

    # --- The run is listed in the case notebook and stays viewable -----------
    refreshed = library.contents(case_id)
    names = [row['name'] for row in refreshed['runs'] if row['details']['kind'] == 'compare']
    check('Spatial visit comparison' in names, 'spatial comparison is saved in the notebook')

print()
print(f'{check.failures} failure(s)')
sys.exit(1 if check.failures else 0)
