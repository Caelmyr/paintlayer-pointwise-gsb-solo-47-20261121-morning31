"""Deterministic conservation training plates for an invented painted still life."""

from dataclasses import dataclass
import math
from threading import Lock

import numpy as np
from scipy import ndimage

from .raster import png_bytes, tiff_bytes

DEMO_SIZE = (718, 545)
TITLE = 'Still Life with Quinces'
DEMO_LOCK = Lock()


@dataclass(frozen=True)
class Scene:
    visible: np.ndarray
    first: np.ndarray
    second: np.ndarray
    weave: np.ndarray


def ellipse_grid(x, y, center, radii, angle=0):
    dx = x - center[0]
    dy = y - center[1]
    ca = math.cos(angle)
    sa = math.sin(angle)
    return ((dx * ca + dy * sa) / radii[0]) ** 2 + ((dy * ca - dx * sa) / radii[1]) ** 2


def blend_rgb(canvas, color, alpha):
    canvas *= 1 - alpha[..., None]
    canvas += alpha[..., None] * np.asarray(color, dtype=np.float32)


def veil(ellipse, edge=0.09):
    return np.clip((1 - ellipse) / edge + 0.5, 0, 1)


def segment_distance(x, y, first, last):
    a, b = np.array(first), np.array(last)
    dx, dy = b - a
    factor = np.clip(((x - a[0]) * dx + (y - a[1]) * dy) / (dx * dx + dy * dy), 0, 1)
    return np.sqrt((x - a[0] - factor * dx) ** 2 + (y - a[1] - factor * dy) ** 2)


def seed_scene(width, height):
    rng = np.random.default_rng(265117)
    x, y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    random = rng.normal(0, 1, (height, width)).astype(np.float32)
    plaster = ndimage.gaussian_filter(random, sigma=8)
    brushes = ndimage.gaussian_filter(random, sigma=(1.1, 4.5))
    grain = ndimage.gaussian_filter(random, sigma=(0.45, 0.3))
    return x, y, plaster, brushes, grain


def drawing_room(x, y, plaster, brushes, grain, weave):
    height, width = y.shape
    rgb = np.empty((height, width, 3), dtype=np.float32)
    light = np.exp(-ellipse_grid(x, y, (248, 150), (340, 330))) * 35
    wall = light + plaster * 180 + brushes * 8 + grain * 2
    rgb[...] = (33, 39, 31)
    rgb += wall[..., None] * np.array([1.23, 1.00, 0.54])
    edge = 384 + 0.028 * (x - 350) + 0.003 * (x - 350) ** 2 / 50
    table = np.clip((y - edge) / 2, 0, 1)
    timber = np.sin(x / 14 + np.sin(y / 38)) * 7 + np.cos(y / 2.2 + x * 0.016) * 3
    table_light = veil(ellipse_grid(x, y, (297, 417), (380, 130)), 0.9) * 27
    tone = 25 + timber + table_light + 155 * plaster
    color = np.stack((tone * 2.0, tone * 1.13, tone * 0.69), axis=-1)
    blend_rgb(rgb, color, table)
    cast = veil(ellipse_grid(x, y, (408, 433), (260, 33), 0.025), 0.95)
    blend_rgb(rgb, (22, 25, 19), cast * 0.38)
    detector = 0.26 + 0.055 * x / width + 0.038 * y / height + 0.012 * grain + weave
    detector += 0.085 * table
    return rgb, detector


def pigment_stroke(rgb, plate, x, y, grain, first, last, width, color, density):
    d = segment_distance(x, y, first, last)
    alpha = np.clip((width - d + grain * 0.6) / max(1, width * 0.45), 0, 1)
    blend_rgb(rgb, color, alpha * 0.9)
    plate += alpha * density


def fruit(rgb, plate, x, y, position, radii, angle, tone, plaster, grain):
    shape = ellipse_grid(x, y, position, radii, angle) + 0.019 * np.sin(x / 5 + y / 9) + 0.026 * grain
    alpha = veil(shape, 0.08)
    shade_map = ellipse_grid(x, y, (position[0] - radii[0] * 0.27, position[1] - radii[1] * 0.45), radii, angle)
    shading = np.exp(-shade_map * 1.7)
    color = np.array(tone, dtype=np.float32) * (0.46 + shading[..., None] * 0.63)
    mottle = 15 * grain + 110 * plaster + 8 * np.sin((x + y) / 15)
    color += mottle[..., None] * np.array([0.81, 0.64, 0.41])
    blend_rgb(rgb, color, alpha)
    plate += alpha * (0.16 + shading * 0.21 + 0.017 * grain)
    stem = (position[0] + radii[0] * 0.39, position[1] - radii[1] * 0.86)
    pigment_stroke(rgb, plate, x, y, grain, stem, (stem[0] + 9, stem[1] - 14), 3.4, (76, 55, 30), 0.09)


def vine(rgb, plate, x, y, grain, plaster):
    branches = (
        ((375, 406), (335, 150), 6.3, (99, 88, 49)),
        ((342, 220), (225, 117), 5.4, (102, 85, 41)),
        ((351, 250), (454, 106), 5.5, (106, 87, 48)),
        ((360, 286), (532, 192), 4.5, (95, 87, 50)),
        ((326, 194), (275, 72), 3.2, (114, 94, 52)),
        ((292, 178), (172, 161), 2.6, (104, 87, 43)),
        ((401, 174), (477, 155), 3.0, (100, 81, 43)),
        ((436, 238), (489, 282), 3.9, (114, 93, 58)),
    )
    for first, last, thickness, shade in branches:
        pigment_stroke(rgb, plate, x, y, grain, first, last, thickness, shade, 0.094)
    leaves = (
        ((249, 115), (55, 13), -2.3), ((194, 109), (44, 12), 0.28), ((270, 78), (40, 13), 0.83),
        ((307, 112), (49, 13), -2.0), ((206, 169), (43, 16), 0.12), ((153, 147), (37, 13), 0.44),
        ((375, 144), (42, 12), -1.3), ((419, 120), (43, 13), 0.31), ((478, 91), (42, 13), -0.63),
        ((468, 140), (50, 12), 0.06), ((501, 158), (49, 12), -0.26), ((458, 212), (49, 15), -0.46),
        ((527, 191), (53, 12), -0.34), ((548, 222), (42, 12), 0.52), ((481, 269), (54, 15), 0.78),
        ((279, 204), (50, 13), 0.19), ((368, 296), (53, 13), -0.83), ((403, 269), (36, 12), -0.5),
    )
    for index, (center, radii, angle) in enumerate(leaves):
        texture = np.sin(x / 4 + y / 7 + index) * 0.029 + grain * 0.024
        guide = ellipse_grid(x, y, center, radii, angle)
        leaf_alpha = veil(guide + texture, 0.1)
        brightness = np.exp(-ellipse_grid(x, y, (center[0] - 11, center[1] - 3), radii, angle) * 1.5)
        light = 0.50 + brightness * 0.70 + plaster * 2.5
        hue = np.asarray((109 + index % 3 * 11, 113 + index % 4 * 6, 64), dtype=np.float32)
        blend_rgb(rgb, light[..., None] * hue, leaf_alpha * 0.96)
        plate += leaf_alpha * (0.038 + brightness * 0.082 + grain * 0.009)
        run = (math.cos(angle) * radii[0] * 0.8, math.sin(angle) * radii[0] * 0.8)
        first = (center[0] - run[0], center[1] - run[1])
        last = (center[0] + run[0], center[1] + run[1])
        pigment_stroke(rgb, plate, x, y, grain, first, last, 0.85, (129, 128, 77), 0.022)


def porcelain(rgb, plate, x, y, plaster, brushes, grain):
    bowl = ellipse_grid(x, y, (346, 398), (186, 79), 0.018)
    topcut = np.clip((y - 361) / 3, 0, 1)
    bowl_mask = veil(bowl, 0.09) * topcut
    shade = 0.68 + 0.34 * np.exp(-ellipse_grid(x, y, (290, 400), (134, 65), 0))
    shade += plaster * 4 + brushes * 0.14 + grain * 0.018
    pottery = shade[..., None] * np.array([156, 173, 165], dtype=np.float32)
    blend_rgb(rgb, pottery, bowl_mask)
    plate += bowl_mask * (0.14 + 0.09 * shade)
    rim_outer = ellipse_grid(x, y, (346, 367), (186, 40), 0.016)
    rim_inner = ellipse_grid(x, y, (346, 366), (175, 33), 0.016)
    whole_rim = veil(rim_outer, 0.09)
    rim = whole_rim * (1 - veil(rim_inner, 0.12))
    blend_rgb(rgb, (65, 72, 54), whole_rim)
    blend_rgb(rgb, (192, 191, 165), rim)
    plate += rim * 0.22
    band = np.maximum(0, 1 - np.abs(y - (432 + 0.00087 * (x - 346) ** 2)) / 2.8)
    band *= bowl_mask * np.clip((158 - np.abs(x - 346)) / 26, 0, 0.75)
    blend_rgb(rgb, (50, 88, 88), band)
    plate += band * 0.13


def make_scene():
    width, height = DEMO_SIZE
    x, y, plaster, brushes, grain = seed_scene(width, height)
    wave = 0.012 * np.sin(2 * np.pi * (x + y * 0.026) / 7.2) + 0.011 * np.sin(2 * np.pi * (y - x * 0.017) / 6.3)
    rgb, plate = drawing_room(x, y, plaster, brushes, grain, wave)
    vine(rgb, plate, x, y, grain, plaster)
    porcelain(rgb, plate, x, y, plaster, brushes, grain)
    for position, radii, angle, color in (
        ((350, 301), (64, 56), -0.30, (189, 143, 57)),
        ((432, 335), (60, 48), 0.12, (208, 159, 66)),
        ((260, 346), (60, 52), -0.26, (178, 139, 51)),
        ((329, 359), (62, 49), 0.20, (218, 160, 66)),
        ((500, 433), (49, 34), -0.32, (183, 131, 57)),
    ):
        fruit(rgb, plate, x, y, position, radii, angle, color, plaster, grain)
    pin_holes = np.zeros_like(plate)
    for position in (24, 93, 169, 258, 355, 446, 533, 624, 692):
        up = ellipse_grid(x, y, (position, 26), (4, 3))
        down = ellipse_grid(x, y, (position, height - 27), (4, 3))
        pin_holes += veil(up, 0.45) + veil(down, 0.45)
    horizontal = np.exp(-((y - 50) / 10) ** 8) + np.exp(-((y - height + 49) / 11) ** 8)
    vertical = np.exp(-((x - 33) / 8) ** 8) + np.exp(-((x - width + 34) / 9) ** 8)
    midbeam = np.exp(-((x - 363 - 0.019 * y) / 16) ** 8)
    trace = np.abs(ellipse_grid(x, y, (440, 286), (86, 115), -0.22) - 1)
    deleted_drawing = np.exp(-trace ** 2 / 0.0025) * (0.045 + 0.04 * np.sin(y / 65))
    initial = plate + horizontal * 0.13 + vertical * 0.16 + midbeam * 0.17 + pin_holes * 0.85 + deleted_drawing
    repaired = veil(ellipse_grid(x, y, (460, 382), (59, 31), 0.21), 0.48) * 0.12
    other = veil(ellipse_grid(x, y, (294, 196), (26, 19), -0.19), 0.65) * 0.135
    later = initial + repaired - other + 0.026
    clay = np.clip(np.rint(rgb + wave[..., None] * 70), 0, 255).astype(np.uint8)
    return Scene(clay, initial, later, wave)


def detector_references(width=445, height=545):
    xs = np.arange(width, dtype=np.float32)[None, :]
    ys = np.arange(height, dtype=np.float32)[:, None]
    dark = np.broadcast_to(845 + 30 * np.sin(xs / 73) + 22 * np.cos(ys / 84), (height, width)).copy()
    field = 53000 + 2850 * np.cos(xs / 172) - 3000 * np.sin(ys / 165) + 400 * np.cos((xs + ys) / 211)
    return np.rint(dark).astype(np.uint16), np.rint(field).astype(np.uint16)


def detector_tile(attenuation, x_offset, *, visit, tile, dark, flat):
    height, width = dark.shape
    detector = attenuation[:, x_offset:x_offset + width]
    rng = np.random.default_rng(48008 + visit * 991 + tile * 79)
    jitter = rng.normal(0, 87, detector.shape)
    level_offset = 0.012 if tile else -0.006
    transmission = np.exp(-detector + level_offset)
    data = dark.astype(np.float32) + (flat.astype(np.float32) - dark) * transmission + jitter
    data[:, :3] = 65535
    data[-3:, :] = 100
    data[99 + visit * 10:104 + visit * 10, 274:281] = 65535
    return np.clip(np.rint(data), 0, 65535).astype(np.uint16)


def detector_mask(take, tile, width=445, height=545):
    mask = np.full((height, width), 255, dtype=np.uint8)
    mask[:5, :] = 0
    mask[:, -6:] = 0
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    if tile:
        hole = ellipse_grid(x, y, (356, 437 if take else 410), (17, 27), 0.21)
    else:
        hole = ellipse_grid(x, y, (47, 134 if take else 121), (14, 26), -0.22)
    mask[hole < 1] = 0
    return mask


def image_case(library, name, role, pixels, case, session=None):
    as_tiff = role in {'dark', 'flat', 'radiograph'}
    extension = 'tiff' if as_tiff else 'png'
    encoder = tiff_bytes if as_tiff else png_bytes
    return library.add_source(case, encoder(pixels), f'{name.lower().replace(" ", "-")}.{extension}', {
        'name': name,
        'role': role,
        'session': session,
        'notes': 'Synthetic training sample generated for this demonstration.',
    })


def load_demo(library, processing, store):
    with DEMO_LOCK:
        with store.snapshot() as queries:
            existing = next((row for row in queries.catalog() if row['synthetic'] and row['working_run']), None)
        if existing is not None:
            return existing
        return create_exam(library, processing)


def create_exam(library, processing):
    width, height = DEMO_SIZE
    scene = make_scene()
    case = library.create({
        'title': TITLE,
        'accession': 'PL-TRAINING-07',
        'collection': 'Studio conservation exercises',
        'width': width,
        'height': height,
        'scale': 0.39,
        'notes': 'Simulated visible and radiographic acquisitions of a fictional painting. All reported optical depths are relative.',
    }, synthetic=True)
    case_id = case['id']
    photo = image_case(library, 'Visible reference', 'visible', scene.visible, case_id)
    dark, flat = detector_references()
    dark_source = image_case(library, 'Detector dark', 'dark', dark, case_id)
    flat_source = image_case(library, 'Detector flat', 'flat', flat, case_id)
    profile = processing.create_profile(case_id, {
        'name': 'Studio detector reference',
        'dark': dark_source['id'],
        'flat': flat_source['id'],
    })
    mosaics = []
    for take, (date, name, attenuation) in enumerate((
        ('2025-03-04', 'Before treatment', scene.first),
        ('2026-09-18', 'Follow-up visit', scene.second),
    )):
        session = library.create_session(case_id, {'name': name, 'date': date, 'purpose': 'Synthetic comparison study'})
        placements = []
        for index, left in enumerate((0, width - 445)):
            letter = 'A' if index == 0 else 'B'
            pixels = detector_tile(attenuation, left, visit=take, tile=index, dark=dark, flat=flat)
            source = image_case(library, f'{name} · Plate {letter}', 'radiograph', pixels, case_id, session['id'])
            mask_pixels = detector_mask(take, index)
            mask = image_case(library, f'{name} · Mask {letter}', 'mask', mask_pixels, case_id, session['id'])
            plate = processing.calibrate(case_id, {
                'name': f'{name} · Calibrated {letter}',
                'detector': source['id'],
                'mask': mask['id'],
                'profile': profile['id'],
            })
            placements.append({'plate': plate['id'], 'x': left, 'y': 0})
        mosaics.append(processing.assemble(case_id, {'name': name, 'placements': placements, 'radius': 38}))
    points = [(50, 67), (620, 88), (612, 481), (80, 464)]
    photo_reg = processing.register(case_id, {
        'name': 'Registered visible reference',
        'mode': 'affine',
        'mosaic': mosaics[-1]['id'],
        'photograph': photo['id'],
        'points': [{'canvas': list(point), 'photograph': list(point)} for point in points],
    })
    for title, category, outer, holes in (
        ('Porcelain rim', 'material', [[194, 351], [482, 347], [517, 391], [201, 410]], [[[258, 365], [310, 360], [303, 379], [259, 389]]]),
        ('Upper left ground', 'reference', [[89, 58], [179, 64], [179, 112], [89, 106]], []),
        ('Treatment review', 'treatment', [[401, 353], [521, 353], [550, 423], [432, 435]], []),
    ):
        library.save_region(case_id, {'name': title, 'category': category, 'geometry': {'outer': outer, 'holes': holes}})
    case = processing.choose(case_id, {'selection': 'photograph', 'run': photo_reg['id'], 'revision': case['revision']})
    return processing.choose(case_id, {'selection': 'working', 'run': mosaics[-1]['id'], 'revision': case['revision']})


def training_artwork():
    return png_bytes(make_scene().visible)
