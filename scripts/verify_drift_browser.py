#!/usr/bin/env python3
"""Browser smoke test for the multi-reference comparison workbench UI."""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from playwright.sync_api import sync_playwright
from werkzeug.serving import make_server

from paintlayer.demo import load_demo
from paintlayer.library import Library
from paintlayer.processing import Processing
from paintlayer.settings import Runtime
from paintlayer.store import Store
from paintlayer.web import create_app

failures = []


def check(condition, label):
    print(f'{"PASS" if condition else "FAIL"}  {label}')
    if not condition:
        failures.append(label)


with tempfile.TemporaryDirectory() as temp_dir:
    runtime = Runtime(data_root=Path(temp_dir), host='127.0.0.1', port=18977, threads=1)
    app = create_app(runtime)
    store = app.extensions['paintlayer']['store']
    library = app.extensions['paintlayer']['library']
    processing = app.extensions['paintlayer']['processing']
    demo = load_demo(library, processing, store)
    case_id = demo['id']
    contents = library.contents(case_id)
    reference_regions = [row for row in contents['regions'] if row['details']['category'] == 'reference']
    for name, outer in (
        ('Upper right ground', [[560, 60], [660, 64], [662, 116], [558, 110]]),
        ('Lower left ground', [[70, 430], [160, 428], [164, 480], [72, 484]]),
    ):
        library.save_region(case_id, {
            'name': name, 'category': 'reference',
            'geometry': {'outer': outer, 'holes': []},
        })

    server = make_server('127.0.0.1', 18977, app, threaded=True)
    import threading
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as browser_tool:
            browser = browser_tool.chromium.launch(
                headless=True,
                executable_path='/opt/pointwise-env/browsers/chromium_headless_shell-1255/chrome-headless-shell-linux64/chrome-headless-shell',
                args=['--no-sandbox'],
            )
            page = browser.new_page(viewport={'width': 1500, 'height': 950})
            errors = []
            page.on('pageerror', lambda exc: errors.append(str(exc)))
            page.goto(f'http://127.0.0.1:18977/#case={case_id}')
            page.wait_for_selector('.run-row')

            # Open the compare dialog.
            page.locator('.left-panel').get_by_role('button', name='Compare visits', exact=True).click()
            dialog = page.get_by_role('dialog')
            dialog.wait_for(state='visible')
            check(dialog.get_by_text('Multi-region spatial correction').is_visible(),
                  'compare dialog exposes the spatial correction toggle')
            check(dialog.locator('[name=reference]').is_enabled(), 'single reference selector enabled by default')

            # The reference checklist is hidden until the toggle is on.
            checklist = dialog.locator('.pick-list')
            check(check_list_hidden := not checklist.first.is_visible(), 'reference checklist hidden before toggling')
            dialog.locator('[name=spatial_correction]').check()
            check(checklist.first.is_visible(), 'reference checklist appears after toggling spatial mode')
            check(dialog.locator('[name=reference]').is_disabled(),
                  'single reference selector disabled while spatial mode is active')
            boxes = checklist.locator('input[type=checkbox]')
            check(boxes.count() == 3, f'three stable reference regions offered (got {boxes.count()})')
            boxes.first.check()
            boxes.nth(1).check()
            boxes.nth(2).check()

            dialog.get_by_role('button', name='Calculate change').click()
            page.wait_for_selector('.toast', timeout=15000)
            page.wait_for_selector('.run-row.active', timeout=15000)

            # The new run opens on the density-change layer; switch through layers.
            check(not page.get_by_role('dialog').is_visible(), 'dialog closes after the comparison is saved')
            readout = page.locator('.right-panel')
            check(readout.get_by_text('Reference regions', exact=True).is_visible(),
                  'inspector shows reference region stats')
            check(readout.get_by_text('Pending review', exact=True).first.is_visible(),
                  'inspector shows pending review stat')
            check(readout.get_by_text('Median drift', exact=True).is_visible(), 'inspector shows median drift')
            check(readout.get_by_text('Upper left ground').first.is_visible(), 'per-reference audit table is shown')
            check(readout.locator('.legend').get_by_text('Pending review').is_visible(),
                  'legend names the pending-review class')

            for layer in ('Class map', 'Uncertainty', 'Correction reach', 'Exposure drift', 'Usable samples'):
                tab = page.get_by_role('button', name=layer)
                if tab.count():
                    tab.first.click()
                else:
                    page.locator('select[aria-label="More image layers"]').select_option(label=layer)
                page.wait_for_timeout(300)
                check('/view.png' in page.locator('.photo-stage img').get_attribute('src'),
                      f'{layer} layer loads an image')
            check(not errors, f'no uncaught browser errors ({errors})')

            # Reopen the dialog and confirm spatial mode resets to off.
            page.locator('.left-panel').get_by_role('button', name='Compare visits', exact=True).click()
            fresh = page.get_by_role('dialog')
            fresh.wait_for(state='visible')
            check(not fresh.locator('[name=spatial_correction]').is_checked(),
                  'spatial toggle defaults off in a fresh dialog')
            check(fresh.locator('[name=reference]').is_enabled(), 'single reference available again in fresh dialog')
            fresh.locator('[name=spatial_correction]').check()
            boxes = fresh.locator('.pick-list input[type=checkbox]')
            boxes.first.check()  # only one reference selected
            fresh.get_by_role('button', name='Calculate change').click()
            page.wait_for_selector('.form-error:not(:empty)', timeout=8000)
            check('at least two stable reference' in fresh.locator('.form-error').inner_text().lower(),
                  'client blocks spatial correction with fewer than two references')
            browser.close()
    finally:
        server.shutdown()

print()
print(f'{len(failures)} failure(s)')
sys.exit(1 if failures else 0)
