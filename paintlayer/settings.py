"""Product and resource limits shared by the examination engine and UI."""

from dataclasses import dataclass
from os import environ
from pathlib import Path

MAX_SIDE = 4096
MAX_SURFACE_PIXELS = 6_000_000
MAX_UPLOAD_PIXELS = 5_000_000
MAX_UPLOAD_BYTES = 28 * 1024 * 1024
MAX_REQUEST_BYTES = MAX_UPLOAD_BYTES + 512 * 1024
MAX_CASES = 100
MAX_SOURCES = 100
MAX_RUNS = 130
MAX_REGIONS = 100
MAX_FINDINGS = 160
MAX_SESSIONS = 20
MAX_PROFILES = 24
MAX_STACK_TILES = 16
MAX_PLATE_INPUT_PIXELS = 22_000_000
MAX_EXPORT_BYTES = 85 * 1024 * 1024
MAX_PROFILE_SAMPLES = 4000
MIN_IMAGE_EDGE = 8
MAX_CASE_BYTES = 768 * 1024 * 1024
MAX_ARCHIVES = 25

SOURCE_ROLES = {
    'radiograph': 'Detector plate',
    'visible': 'Visible photograph',
    'dark': 'Dark exposure',
    'flat': 'Flat exposure',
    'mask': 'Acquisition mask',
}

RUN_KINDS = {
    'plate': 'Calibrated plate',
    'mosaic': 'Assembled radiograph',
    'registration': 'Registered photograph',
    'compare': 'Visit comparison',
    'quality': 'Acquisition quality map',
    'support': 'Support inspection',
}

CASE_VIEWS = {
    'plate': ('signal', 'support', 'weight', 'reasons', 'raw', 'owner', 'conflict', 'residual'),
    'mosaic': ('signal', 'support', 'coverage', 'spread', 'owner', 'quality'),
    'registration': ('photo', 'support', 'grid'),
    'compare': ('difference', 'support', 'classification', 'uncertainty'),
    'quality': ('signal', 'support', 'classification', 'response'),
    'support': ('signal', 'support', 'response', 'classification'),
}


@dataclass(frozen=True)
class Runtime:
    data_root: Path
    host: str
    port: int
    threads: int

    @classmethod
    def from_environment(cls):
        default = Path.cwd() / 'runtime-data'
        return cls(
            data_root=Path(environ.get('PAINTLAYER_DATA', default)).expanduser().resolve(),
            host=environ.get('HOST', '127.0.0.1'),
            port=int(environ.get('PORT', '18950')),
            threads=int(environ.get('PAINTLAYER_THREADS', '3')),
        )
