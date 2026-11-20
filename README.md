# PaintLayer Desk

PaintLayer Desk is a local radiographic examination workbench for painted canvases. It keeps raw detector counts, calibrated rasters, usable sample masks, registered visible photographs and examination evidence together in a case.

## Start

With Docker Compose:

```sh
docker compose up --build
```

Open http://127.0.0.1:19950. Stop with Ctrl+C or `docker compose stop`. The named `examinations` volume stores the notebook and binaries. A different instance can use its own Compose project name and `PAINTLAYER_PORT`.

With Python 3.11:

```sh
python3.11 -m venv .venv
. .venv/bin/activate
python -m pip install --only-binary=:all: -r requirements.txt
./start.sh
```

Open http://127.0.0.1:18950. The standalone service accepts `HOST`, `PORT`, `PAINTLAYER_THREADS` and `PAINTLAYER_DATA`. It binds loopback by default and stores local data under `runtime-data/`. Use a single application process per data directory. Back up the complete data directory while the service is stopped. Without authentication, the normal service is intended for one trusted local workstation.

## Workflows

Open the training examination for a fictional painting with an invented acquisition history and measurement values. The visible still life, four 16-bit detector plates, corresponding masks, dark and flat references and two sessions are generated locally. They are synthetic and marked as such in the app and reports.

| Where | Input and result |
| --- | --- |
| Sources → Import source | Inspect a TIFF page window and select a compatible frame. Accepted bytes, frame identity, decoded sample values and validity are stored separately. |
| Sources → Inspect | Examine metadata, sample a neighbourhood in native detector counts and download the original file. |
| Sources → New profile / Workbench → Calibrate plate | Select dark and flat references and an optional acquisition mask. Inspect the relative attenuation or transmission surface, confidence and flagged sample classes. |
| Workbench → Assemble radiograph | Place calibrated plates by integer position, quarter turns and optional mirror. Review the confidence-weighted assembly, dominant plate, coverage and overlap spread. |
| Workbench → Register photograph | Pick matching landmarks to fit similarity or affine geometry. Review the mapped image, held-out residuals and grid. |
| Workbench → Radiograph, Overlay or More layers | Adjust window, gamma, palette and radiograph/visible composition. The working data remains quantitative. |
| Workbench → Trace region / Regions | Outline a canonical canvas region and add exclusion rings in its editor. Each edit increments the region revision. Saved findings preserve the outline used when they were captured. |
| Measurements → Measure region / Compare regions | Inspect usable support, detector statistics, thresholds and distributions with canvas and optional physical area figures. |
| Workbench → Place transect | Pick endpoints and measure a strip along the canvas. Missing stations remain gaps in the plot and CSV; threshold events use only adjacent supported stations. |
| Workbench → Audit acquisition quality | Generate a reason map and connected components for overlap disagreement, missing data, sparse coverage, impulses and clipping. |
| Workbench → Inspect stretcher bands | Consider long horizontal and vertical contrast structures with usable fraction and supporting extent. The results are visual hypotheses. |
| Measurements → Inspect texture | Inspect spatial frequency, directional strength and possible repeat spacing in a bounded patch. |
| Measurements → Audit detector | Compare flat/dark gain and detector axes using gain and irregularity maps. |
| Workbench → Compare visits | Review earlier-only, later-only, stable, changed and unresolved pixels using shared usable samples and optional stable-reference correction. Tick several distributed stable regions for a spatial correction: a weighted plane plus local interpolation restores the exposure gradient between them, while pixels beyond the references' bounded reach are held as pending review instead of being called changes. |
| Workbench → Capture finding / Evidence → Export bundle | Preserve a marked crop, current display state, numerical measurement and source hashes. Verify and download a ZIP containing selected findings, TIFF values, validity masks and provenance. |
| Evidence → Build contact sheet | Align crop reproductions at the same print scale, check the first-page preview and download a PDF with an actual-size print reference and layout data. |

The working representation uses zero-based pixel centres, x from the left and y from the top. Bounds are x, y, width, height. Imported visible photographs use their orientation metadata. Detector images preserve stored pixel order and accept unsigned MinIsBlack 8- or 16-bit TIFF/PNG; visible JPEG is also accepted. Masks count nonzero grayscale pixels as usable. The page dialog inspects up to eight frames per window among the first 48 TIFF pages.

An imported image can be at most 28 MiB, 4096 pixels on a side and five million pixels. An examination canvas can contain up to six million pixels. One assembly can use at most 16 plates and 22 million aggregate plate pixels. ZIP, sheet and case allowances bound long-running work and storage. The demo uses 718 × 545 canvas pixels. The TIFF export accompanying each numerical layer carries a separate validity mask. A valid numerical zero is a measurement. Current runs retain the input region and session revisions they used. Evidence exports from older workspaces flag historical revisions whose original state cannot be recovered.

Values are relative and depend on the acquisition conditions. The calibrated blend is a confidence-weighted feathered mean. Difference classification carries overlap disagreement as a heuristic uncertainty, and stretcher/texture inspection summarizes patterns from known sample support. Spatial drift correction trusts its fitted gradient only within a bounded reach of the accepted stable references; out-of-reach samples and self-contradicting references are reported explicitly rather than corrected. These are not absolute material identification or instruments for physical radiation acquisition.

## Source structure

- `paintlayer/web.py` and `paintlayer/__main__.py` provide the HTTP and Waitress entry points.
- `paintlayer/library.py`, `paintlayer/store.py` and `paintlayer/intake.py` maintain the notebook, blob references and image ingestion.
- `paintlayer/processing.py` joins detector correction, placement, registration and quantitative outputs.
- `paintlayer/views.py`, `paintlayer/evidence.py` and the atlas modules generate presentation and portable reports.
- `static/` holds the browser entry point and examination controls. All resources are local.

Recalculate Python backend effective lines with `python scripts/count_backend.py --project-root . --source-root paintlayer`.
