"""Parse request fields before any array allocation or database activity."""

import math
import re
import unicodedata

from .errors import ExamError, InvalidGeometry

ID_PATTERN = re.compile(r'^[0-9a-f]{24}$')
DATE_PATTERN = re.compile(r'^\d{4}-\d{2}-\d{2}$')
CONTROL_CHARS = {'\x00', '\r', '\t'}


def request_object(value, *, field='request'):
    if not isinstance(value, dict):
        raise ExamError('Expected an object.', field=field)
    return value


def text(value, *, field, limit=160, required=True, multiline=False):
    if value is None and not required:
        return ''
    if not isinstance(value, str):
        raise ExamError('Expected text.', field=field)
    value = unicodedata.normalize('NFC', value).strip()
    if len(value) > limit:
        raise ExamError(f'Use at most {limit} characters.', field=field)
    if any(character in CONTROL_CHARS for character in value):
        raise ExamError('Text contains a control character.', field=field)
    if not multiline and '\n' in value:
        raise ExamError('Use a single line.', field=field)
    if required and not value:
        raise ExamError('Enter a value.', field=field)
    return value


def item_id(value, *, field):
    if not isinstance(value, str) or not ID_PATTERN.fullmatch(value):
        raise ExamError('Select an existing record.', field=field)
    return value


def choice(value, *, field, options):
    if not isinstance(value, str) or value not in options:
        raise ExamError('Select one of the available options.', field=field, facts={'options': sorted(options)})
    return value


def scalar(value, *, field, minimum=None, maximum=None):
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ExamError('Enter a number.', field=field)
    try:
        number = float(value)
    except OverflowError as exc:
        raise ExamError('Use a finite number.', field=field) from exc
    if not math.isfinite(number):
        raise ExamError('Use a finite number.', field=field)
    if minimum is not None and number < minimum:
        raise ExamError(f'Use a number at least {minimum}.', field=field)
    if maximum is not None and number > maximum:
        raise ExamError(f'Use a number at most {maximum}.', field=field)
    return number


def integer(value, *, field, minimum=None, maximum=None):
    if not isinstance(value, int) or isinstance(value, bool):
        raise ExamError('Enter a whole number.', field=field)
    if minimum is not None and value < minimum:
        raise ExamError(f'Use a whole number at least {minimum}.', field=field)
    if maximum is not None and value > maximum:
        raise ExamError(f'Use a whole number at most {maximum}.', field=field)
    return value


def flag(value, *, field):
    if not isinstance(value, bool):
        raise ExamError('Use a check box value.', field=field)
    return value


def sequence(value, *, field, minimum=0, maximum=200):
    if not isinstance(value, list):
        raise ExamError('Expected a list.', field=field)
    if not minimum <= len(value) <= maximum:
        raise ExamError(f'Provide {minimum} to {maximum} entries.', field=field)
    return value


def point(value, *, field, extent):
    values = sequence(value, field=field, minimum=2, maximum=2)
    limits = tuple(maximum - 1 for maximum in extent)
    return tuple(
        scalar(entry, field=f'{field}.{axis}', minimum=0, maximum=limit)
        for axis, entry, limit in zip('xy', values, limits, strict=True)
    )


def polygon(value, *, field, extent, max_vertices=64):
    vertices = sequence(value, field=field, minimum=3, maximum=max_vertices)
    points = [point(vertex, field=f'{field}[{index}]', extent=extent) for index, vertex in enumerate(vertices)]
    if len(set(points)) != len(points):
        raise InvalidGeometry('A vertex occurs more than once. Close the outline without repeating its first vertex.', field=field)
    return points


def box(value, *, field, extent, limits=None):
    values = sequence(value, field=field, minimum=4, maximum=4)
    width, height = extent
    x = integer(values[0], field=f'{field}.x', minimum=0, maximum=width - 1)
    y = integer(values[1], field=f'{field}.y', minimum=0, maximum=height - 1)
    w = integer(values[2], field=f'{field}.width', minimum=1, maximum=width - x)
    h = integer(values[3], field=f'{field}.height', minimum=1, maximum=height - y)
    if limits is not None and (w, h) != (min(w, limits[0]), min(h, limits[1])):
        raise InvalidGeometry(f'Inspect a box at most {limits[0]} × {limits[1]} pixels.', field=field)
    return x, y, w, h


def unique(records, *, field):
    seen = set()
    for record in records:
        if record in seen:
            raise ExamError('An entry was selected more than once.', field=field)
        seen.add(record)
    return records


def limits_pair(value, *, field, minimum, maximum):
    values = sequence(value, field=field, minimum=2, maximum=2)
    start = scalar(values[0], field=f'{field}.low', minimum=minimum, maximum=maximum)
    end = scalar(values[1], field=f'{field}.high', minimum=minimum, maximum=maximum)
    if start >= end:
        raise ExamError('The upper bound must exceed the lower bound.', field=field)
    return start, end
