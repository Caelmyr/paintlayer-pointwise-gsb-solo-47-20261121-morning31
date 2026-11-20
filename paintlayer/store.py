"""Transactional examination records and bounded content-addressed blob storage."""

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import sqlite3
from tempfile import NamedTemporaryFile
from threading import RLock

from .errors import CapacityExceeded, ExamError, MissingRecord, RevisionConflict
from .settings import (
    MAX_ARCHIVES, MAX_CASE_BYTES, MAX_CASES, MAX_FINDINGS, MAX_PROFILES, MAX_REGIONS, MAX_RUNS,
    MAX_SESSIONS, MAX_SOURCES,
)


SCHEMA_VERSION = 2
SCHEMA = """
CREATE TABLE IF NOT EXISTS cases (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    accession TEXT NOT NULL,
    collection TEXT NOT NULL,
    notes TEXT NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    scale REAL,
    synthetic INTEGER NOT NULL,
    created TEXT NOT NULL,
    updated TEXT NOT NULL,
    revision INTEGER NOT NULL,
    working_run TEXT,
    registration_run TEXT
);
CREATE TABLE IF NOT EXISTS records (
    id TEXT PRIMARY KEY,
    case_id TEXT NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    name TEXT NOT NULL,
    revision INTEGER NOT NULL,
    created TEXT NOT NULL,
    updated TEXT NOT NULL,
    details TEXT NOT NULL,
    UNIQUE(case_id, id)
);
CREATE INDEX IF NOT EXISTS records_case_kind ON records(case_id, kind, updated);
CREATE TABLE IF NOT EXISTS blobs (
    hash TEXT PRIMARY KEY,
    size INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS record_blobs (
    record_id TEXT NOT NULL REFERENCES records(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    hash TEXT NOT NULL REFERENCES blobs(hash) ON DELETE RESTRICT,
    PRIMARY KEY(record_id, name)
);
CREATE INDEX IF NOT EXISTS record_blob_hash ON record_blobs(hash);
CREATE TABLE IF NOT EXISTS provenance (
    case_id TEXT NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    derived_id TEXT NOT NULL,
    source_id TEXT NOT NULL,
    relation TEXT NOT NULL,
    source_revision INTEGER NOT NULL,
    source_snapshot TEXT,
    PRIMARY KEY(derived_id, source_id, relation),
    FOREIGN KEY(case_id, derived_id) REFERENCES records(case_id, id) ON DELETE CASCADE,
    FOREIGN KEY(case_id, source_id) REFERENCES records(case_id, id) ON DELETE RESTRICT,
    CHECK (derived_id != source_id)
);
CREATE INDEX IF NOT EXISTS provenance_source ON provenance(source_id);
CREATE TABLE IF NOT EXISTS events (
    seq INTEGER PRIMARY KEY AUTOINCREMENT,
    case_id TEXT NOT NULL REFERENCES cases(id) ON DELETE CASCADE,
    action TEXT NOT NULL,
    record_id TEXT,
    occurred TEXT NOT NULL,
    details TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_case_sequence ON events(case_id, seq);
"""

COLLECTION_LIMITS = {
    'archive': MAX_ARCHIVES,
    'analysis': 100,
    'finding': MAX_FINDINGS,
    'profile': MAX_PROFILES,
    'region': MAX_REGIONS,
    'run': MAX_RUNS,
    'session': MAX_SESSIONS,
    'source': MAX_SOURCES,
    'sheet': 30,
}
BLOB_PATTERN = re.compile(r'^[0-9a-f]{64}$')
CASE_STATE_COLUMNS = {'working_run', 'registration_run'}


def now():
    return datetime.now(timezone.utc).isoformat(timespec='milliseconds')


def new_id():
    return secrets.token_hex(12)


def json_text(payload):
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)


def resolve_detail(row):
    record = dict(row)
    record['details'] = json.loads(record['details'])
    return record


@dataclass(frozen=True)
class IncomingBlob:
    name: str
    data: bytes

    @property
    def hash(self):
        return hashlib.sha256(self.data).hexdigest()


class BlobStore:
    def __init__(self, root):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def path(self, digest):
        if not BLOB_PATTERN.fullmatch(digest):
            raise ValueError('Invalid blob identity')
        return self.root / digest[:2] / digest[2:]

    def write(self, digest, payload):
        target = self.path(digest)
        if target.is_file():
            if target.stat().st_size != len(payload):
                raise RuntimeError('Existing image blob does not match its expected size')
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        with NamedTemporaryFile(dir=target.parent, prefix='.pending-', delete=False) as temporary:
            temporary_path = Path(temporary.name)
            try:
                temporary.write(payload)
                temporary.flush()
                os.fsync(temporary.fileno())
                os.replace(temporary_path, target)
            finally:
                temporary_path.unlink(missing_ok=True)

    def remove(self, digest):
        self.path(digest).unlink(missing_ok=True)


class Store:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.database = self.root / 'examinations.sqlite3'
        self.blobs = BlobStore(self.root / 'blobs')
        self.writer = RLock()
        with self.connection() as connection:
            connection.execute('PRAGMA journal_mode=WAL')
            version = connection.execute('PRAGMA user_version').fetchone()[0]
            if version > SCHEMA_VERSION:
                raise RuntimeError(f'Unsupported examination database schema {version}')
            connection.executescript(SCHEMA)
            connection.execute('BEGIN IMMEDIATE')
            try:
                columns = {row['name'] for row in connection.execute('PRAGMA table_info(provenance)')}
                if 'source_snapshot' not in columns:
                    connection.execute('ALTER TABLE provenance ADD COLUMN source_snapshot TEXT')
                connection.execute(f'PRAGMA user_version={SCHEMA_VERSION}')
                connection.commit()
            except BaseException:
                connection.rollback()
                raise

    @contextmanager
    def connection(self):
        connection = sqlite3.connect(self.database, isolation_level=None, timeout=8)
        connection.row_factory = sqlite3.Row
        connection.execute('PRAGMA foreign_keys=ON')
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def transaction(self):
        with self.writer, self.connection() as connection:
            connection.execute('BEGIN IMMEDIATE')
            queries = Queries(connection, self.blobs)
            try:
                yield queries
                connection.commit()
            except BaseException:
                connection.rollback()
                for digest in queries.new_blob_hashes:
                    exists = connection.execute('SELECT 1 FROM blobs WHERE hash = ?', (digest,)).fetchone()
                    if exists is None:
                        self.blobs.remove(digest)
                raise
            for digest in queries.retired_blob_hashes:
                self.blobs.remove(digest)

    @contextmanager
    def snapshot(self):
        with self.connection() as connection:
            connection.execute('BEGIN')
            try:
                yield Queries(connection, self.blobs)
            finally:
                connection.rollback()


class Queries:
    def __init__(self, db, blobs):
        self.db = db
        self.blobs = blobs
        self.new_blob_hashes = set()
        self.retired_blob_hashes = set()

    def catalog(self):
        rows = self.db.execute(
            """
            SELECT cases.*,
                   COUNT(CASE WHEN records.kind = 'source' THEN 1 END) AS source_count,
                   COUNT(CASE WHEN records.kind = 'finding' THEN 1 END) AS finding_count
            FROM cases LEFT JOIN records ON cases.id = records.case_id
            GROUP BY cases.id
            ORDER BY cases.updated DESC, cases.id
            """
        )
        return [dict(row) | {'synthetic': bool(row['synthetic'])} for row in rows]

    def case(self, case_id):
        row = self.db.execute('SELECT * FROM cases WHERE id = ?', (case_id,)).fetchone()
        if row is None:
            raise MissingRecord('This examination is no longer available.', field='case')
        return dict(row) | {'synthetic': bool(row['synthetic'])}

    def records(self, case_id, kind):
        rows = self.db.execute(
            """
            SELECT * FROM records WHERE case_id = ? AND kind = ?
            ORDER BY created, id
            """,
            (case_id, kind),
        )
        return [resolve_detail(row) for row in rows]

    def record(self, case_id, record_id, kind=None):
        row = self.db.execute(
            'SELECT * FROM records WHERE case_id = ? AND id = ?',
            (case_id, record_id),
        ).fetchone()
        if row is None or (kind is not None and row['kind'] != kind):
            raise MissingRecord('This item is unavailable in the selected examination.', field=kind or 'record')
        return resolve_detail(row)

    def blob_info(self, case_id, record_id):
        self.record(case_id, record_id)
        rows = self.db.execute(
            """
            SELECT record_blobs.name, blobs.hash, blobs.size
            FROM record_blobs
            JOIN blobs ON record_blobs.hash = blobs.hash
            WHERE record_blobs.record_id = ?
            ORDER BY record_blobs.name
            """,
            (record_id,),
        )
        return {row['name']: {'sha256': row['hash'], 'bytes': row['size']} for row in rows}

    def blob_path(self, case_id, record_id, name):
        blob = self.blob_info(case_id, record_id).get(name)
        if blob is None:
            raise MissingRecord('This image representation is unavailable.')
        path = self.blobs.path(blob['sha256'])
        if not path.is_file() or path.stat().st_size != blob['bytes']:
            raise MissingRecord('Stored image bytes are unavailable or damaged.')
        return path, blob

    def inputs(self, case_id, record_id, *, history=False):
        self.record(case_id, record_id)
        rows = self.db.execute(
            """
            SELECT records.id, records.name, records.kind, records.revision,
                   provenance.relation, provenance.source_revision, provenance.source_snapshot
            FROM provenance
            JOIN records ON provenance.source_id = records.id
            WHERE provenance.case_id = ? AND provenance.derived_id = ?
            ORDER BY provenance.relation, records.id
            """,
            (case_id, record_id),
        )
        results = []
        for row in rows:
            item = dict(row)
            captured = item.pop('source_snapshot')
            if history and captured is not None:
                item['captured'] = json.loads(captured)
            results.append(item)
        return results

    def descendants(self, case_id, record_id):
        rows = self.db.execute(
            """
            SELECT records.id, records.name, records.kind
            FROM provenance
            JOIN records ON provenance.derived_id = records.id
            WHERE provenance.case_id = ? AND provenance.source_id = ?
            ORDER BY records.created, records.id
            """,
            (case_id, record_id),
        )
        return [dict(row) for row in rows]

    def lineage(self, case_id, record_id):
        current = self.record(case_id, record_id)
        seen = {}
        pending = [(current['id'], current['revision'], None)]
        while pending:
            item_id, revision, captured = pending.pop()
            identity = item_id, revision
            previous = seen.get(identity)
            if previous is not None and (captured is None or previous['revision_status'] != 'unresolved_revision'):
                continue
            current = self.record(case_id, item_id)
            if current['revision'] == revision:
                entry = current
                state = 'recorded'
            elif captured is not None and captured['revision'] == revision:
                entry = captured
                state = 'captured_revision'
            else:
                entry = {'id': item_id, 'kind': current['kind'], 'revision': revision}
                state = 'unresolved_revision'
            entry['revision_status'] = state
            entry['inputs'] = []
            entry['blobs'] = {}
            if state == 'recorded':
                references = self.inputs(case_id, item_id, history=True)
                for source in references:
                    captured_source = source.pop('captured', None)
                    pending.append((source['id'], source['source_revision'], captured_source))
                    if source['revision'] != source['source_revision']:
                        source['name'] = captured_source['name'] if captured_source else None
                        source['revision'] = source['source_revision']
                entry['inputs'] = references
                entry['blobs'] = self.blob_info(case_id, item_id)
            seen[identity] = entry
        return list(seen.values())

    def events(self, case_id, limit=35):
        rows = self.db.execute(
            'SELECT * FROM events WHERE case_id = ? ORDER BY seq DESC LIMIT ?',
            (case_id, limit),
        )
        return [resolve_detail(row) for row in rows]

    def event(self, case_id, action, details, record_id=None, timestamp=None):
        self.db.execute(
            'INSERT INTO events(case_id, action, record_id, occurred, details) VALUES (?, ?, ?, ?, ?)',
            (case_id, action, record_id, timestamp or now(), json_text(details)),
        )

    def create_case(self, payload):
        count = self.db.execute('SELECT COUNT(*) FROM cases').fetchone()[0]
        if count >= MAX_CASES:
            raise CapacityExceeded('This workstation reached its examination limit.')
        case_id = new_id()
        timestamp = now()
        self.db.execute(
            """
            INSERT INTO cases(id, title, accession, collection, notes, width, height, scale,
                              synthetic, created, updated, revision)
            VALUES (:id, :title, :accession, :collection, :notes, :width, :height, :scale,
                    :synthetic, :created, :updated, 1)
            """,
            payload | {'id': case_id, 'created': timestamp, 'updated': timestamp},
        )
        self.event(case_id, 'case_created', {'title': payload['title']}, timestamp=timestamp)
        return self.case(case_id)

    def update_case(self, case_id, payload, expected_revision):
        case = self.case(case_id)
        self.require_revision(case, expected_revision)
        stamp = now()
        self.db.execute(
            """
            UPDATE cases
            SET title = :title, accession = :accession, collection = :collection,
                notes = :notes, updated = :stamp, revision = revision + 1
            WHERE id = :case
            """,
            payload | {'stamp': stamp, 'case': case_id},
        )
        self.event(case_id, 'case_updated', {'title': payload['title']}, timestamp=stamp)
        return self.case(case_id)

    def touch_case(self, case_id, timestamp):
        self.db.execute('UPDATE cases SET updated = ? WHERE id = ?', (timestamp, case_id))

    def set_working(self, case_id, field, run_id, expected_revision):
        if field not in CASE_STATE_COLUMNS:
            raise ValueError('Unknown working selection')
        case = self.case(case_id)
        self.require_revision(case, expected_revision)
        if case[field] == run_id:
            return case
        timestamp = now()
        self.db.execute(
            f'UPDATE cases SET {field} = ?, updated = ?, revision = revision + 1 WHERE id = ?',
            (run_id, timestamp, case_id),
        )
        self.event(case_id, 'selection_updated', {'field': field, 'run': run_id}, run_id, timestamp)
        return self.case(case_id)

    def require_revision(self, record, expected):
        if record['revision'] != expected:
            raise RevisionConflict(
                'This record was revised while the editor was open. Reload its current values.',
                facts={'current_revision': record['revision']},
            )

    def assert_capacity(self, case_id, kind, incoming_blobs):
        count = self.db.execute(
            'SELECT COUNT(*) FROM records WHERE case_id = ? AND kind = ?',
            (case_id, kind),
        ).fetchone()[0]
        if count >= COLLECTION_LIMITS[kind]:
            raise CapacityExceeded(f'This examination reached its limit for {kind} records.')
        rows = self.db.execute(
            """
            SELECT blobs.hash, blobs.size
            FROM records
            JOIN record_blobs ON records.id = record_blobs.record_id
            JOIN blobs ON record_blobs.hash = blobs.hash
            WHERE records.case_id = ?
            GROUP BY blobs.hash
            """,
            (case_id,),
        )
        sizes = {row['hash']: row['size'] for row in rows}
        for blob in incoming_blobs:
            sizes.setdefault(blob.hash, len(blob.data))
        if sum(sizes.values()) > MAX_CASE_BYTES:
            raise CapacityExceeded('This examination reached its image storage allowance.')

    def add(self, case_id, kind, name, details, *, blob_data=(), dependencies=()):
        self.case(case_id)
        incoming = [IncomingBlob(label, data) for label, data in blob_data]
        if len({blob.name for blob in incoming}) != len(incoming):
            raise ValueError('Duplicate blob labels')
        self.assert_capacity(case_id, kind, incoming)
        verified = self.verify_dependencies(case_id, dependencies)
        record_id = new_id()
        stamp = now()
        self.db.execute(
            """
            INSERT INTO records(id, case_id, kind, name, details, revision, created, updated)
            VALUES (?, ?, ?, ?, ?, 1, ?, ?)
            """,
            (record_id, case_id, kind, name, json_text(details), stamp, stamp),
        )
        self.attach(case_id, record_id, verified)
        for blob in incoming:
            self.save_blob(record_id, blob)
        self.touch_case(case_id, stamp)
        self.event(case_id, f'{kind}_created', {'name': name}, record_id, stamp)
        return self.record(case_id, record_id)

    def verify_dependencies(self, case_id, dependencies):
        verified = []
        known = set()
        for dependency, relation in dependencies:
            record = self.record(case_id, dependency['id'], dependency['kind'])
            self.require_revision(record, dependency['revision'])
            key = record['id'], relation
            if key in known:
                continue
            known.add(key)
            verified.append((record, relation))
        return verified

    def attach(self, case_id, record_id, verified):
        self.db.executemany(
            """
            INSERT INTO provenance(case_id, derived_id, source_id, relation, source_revision, source_snapshot)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    case_id, record_id, source['id'], relation, source['revision'],
                    json_text(source) if source['kind'] in {'region', 'session'} else None,
                )
                for source, relation in verified
            ],
        )

    def save_blob(self, record_id, blob):
        digest = blob.hash
        if self.db.execute('SELECT 1 FROM blobs WHERE hash = ?', (digest,)).fetchone() is None:
            self.blobs.write(digest, blob.data)
            self.new_blob_hashes.add(digest)
            self.retired_blob_hashes.discard(digest)
            self.db.execute('INSERT INTO blobs(hash, size) VALUES (?, ?)', (digest, len(blob.data)))
        self.db.execute(
            'INSERT INTO record_blobs(record_id, name, hash) VALUES (?, ?, ?)',
            (record_id, blob.name, digest),
        )

    def revise(self, case_id, record_id, kind, expected, name, details):
        record = self.record(case_id, record_id, kind)
        if kind not in {'region', 'session'}:
            raise ExamError('This record is immutable.')
        self.require_revision(record, expected)
        stamp = now()
        self.db.execute(
            """
            UPDATE records
            SET name = ?, details = ?, revision = revision + 1, updated = ?
            WHERE case_id = ? AND id = ?
            """,
            (name, json_text(details), stamp, case_id, record_id),
        )
        self.touch_case(case_id, stamp)
        self.event(case_id, f'{kind}_revised', {'name': name}, record_id, stamp)
        return self.record(case_id, record_id, kind)

    def remove(self, case_id, record_id, kind):
        record = self.record(case_id, record_id, kind)
        case = self.case(case_id)
        if record_id in {case['working_run'], case['registration_run']}:
            raise RevisionConflict('Switch the working surface before removing this record.')
        downstream = self.descendants(case_id, record_id)
        if downstream:
            raise RevisionConflict(
                'Saved examination work still refers to this record.',
                facts={'dependent_records': downstream[:10]},
            )
        blobs = self.blob_info(case_id, record_id)
        self.db.execute('DELETE FROM records WHERE case_id = ? AND id = ?', (case_id, record_id))
        for blob in blobs.values():
            self.remove_unlinked_blob(blob['sha256'])
        stamp = now()
        self.touch_case(case_id, stamp)
        self.event(case_id, f'{kind}_removed', {'name': record['name']}, record_id, stamp)

    def remove_unlinked_blob(self, digest):
        references = self.db.execute(
            'SELECT 1 FROM record_blobs WHERE hash = ? LIMIT 1',
            (digest,),
        ).fetchone()
        if references is None:
            self.db.execute('DELETE FROM blobs WHERE hash = ?', (digest,))
            self.retired_blob_hashes.add(digest)
