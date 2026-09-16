#!/usr/bin/env python3
"""Audit/import a stopped sequential run into a runner-prepared private run.

No inference or runner imports. Dry-run is the default. The runner must prepare
the target with its normal reproducible manifest (generation.concurrency=256),
empty claims/records and .lock; --resume must use identical preparation flags.
Both supervisors must be stopped, not merely between lock acquisitions.

A crash leaves claims/.continuation-incomplete, which the runner must reject as
an unexpected artifact before scheduling anything. Do not delete that sentinel
or resume a partial target: investigate and prepare a different fresh target.
Latency consumers must group imported measurement_epoch='sequential' separately
from new concurrent measurements. This command never edits either manifest.
"""

import argparse
import ast
from contextlib import ExitStack
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
TERMINAL = {'success', 'truncated', 'http_error', 'timeout', 'invalid_response', 'error'}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def load(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    return json.loads(path.read_bytes(), object_pairs_hook=unique,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))


def snapshot(root):
    result = {}
    for path in [root, *sorted(root.rglob('*'))]:
        info = path.lstat()
        directory = stat.S_ISDIR(info.st_mode)
        if (info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) != (0o700 if directory else 0o600)
                or not (directory or stat.S_ISREG(info.st_mode))
                or (not directory and info.st_nlink != 1)):
            raise ValueError('Artifacts must be private, owned, and not links')
        result[str(path.relative_to(root))] = dict(
            sha256=None if directory else hashlib.sha256(path.read_bytes()).hexdigest(),
            inode=info.st_ino, mode=info.st_mode, size=info.st_size,
            mtime_ns=info.st_mtime_ns, ctime_ns=info.st_ctime_ns)
    return result


def sync_dir(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_new(path, value):
    # A partial write is safe only while the durable import sentinel exists.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump(value, stream, sort_keys=True, ensure_ascii=True, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    sync_dir(path.parent)


def validate_manifests(old, new):
    for manifest in (old, new):
        body = {key: value for key, value in manifest.items() if key != 'resume_fingerprint'}
        if manifest.get('resume_fingerprint') != fingerprint(body):
            raise ValueError('Manifest canonical fingerprint mismatch')
    runner = ROOT / 'scripts/translate_flores.py'
    source = runner.read_bytes()
    versions = [node.value.value for node in ast.parse(source).body
                if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant)
                and any(isinstance(t, ast.Name) and t.id == 'VERSION' for t in node.targets)]
    if (versions != [new['program_version']] or new['code_hashes']['scripts/translate_flores.py']
            != hashlib.sha256(source).hexdigest()):
        raise ValueError('Target must be prepared by the current runner')
    if (old['generation'].get('concurrency', 1) != 1
            or old['generation']['batch_size'] != 1
            or new['generation'].get('concurrency') != 256
            or new['generation']['batch_size'] not in (1, 256)
            or old['timeouts']['call_seconds'] != 120
            or new['timeouts']['call_seconds'] != 900):
        raise ValueError('Expected sequential 120s to concurrency 256 / 900s transition')
    left, right = copy.deepcopy(old), copy.deepcopy(new)
    changes = []
    for path in [('resume_fingerprint',), ('program_version',),
                 ('code_hashes', 'scripts/translate_flores.py'),
                 ('generation', 'batch_size'), ('generation', 'concurrency'),
                 ('timeouts', 'call_seconds')]:
        a, b = left, right
        for key in path[:-1]:
            a, b = a[key], b[key]
        before, after = a.pop(path[-1], None), b.pop(path[-1], None)
        if before != after:
            changes.append(dict(path='.'.join(path), old=before, new=after))
    if left != right:
        raise ValueError('Non-allowlisted manifest drift')
    jobs = {job['job_id']: job for job in old['jobs']}
    if len(jobs) != len(old['jobs']) or any(
            not key or Path(key).name != key or key in ('.', '..') for key in jobs):
        raise ValueError('Ambiguous job identities')
    return jobs, changes


def migrate(source, target, *, expected_completed=4357, apply=False, confirm_source_stopped=False):
    source, target = Path(source).absolute(), Path(target).absolute()
    if source == target or source in target.parents or target in source.parents:
        raise ValueError('Run directories must be separate')
    if not confirm_source_stopped:
        raise ValueError('Explicit stopped-supervisor confirmation required')
    with ExitStack() as stack:
        for root in sorted((source, target)):
            if any(p.is_symlink() for p in [root, *root.parents]):
                raise ValueError('Symlink in run path')
            snapshot(root)
            fd = os.open(root / '.lock', os.O_RDONLY | os.O_NOFOLLOW)
            stack.callback(os.close, fd)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        original = snapshot(source)
        target_before = snapshot(target)
        if set(target_before) != {'.', '.lock', 'manifest.json', 'claims', 'records'}:
            raise ValueError('Target must be freshly prepared with empty claims/records')
        old, new = load(source / 'manifest.json'), load(target / 'manifest.json')
        jobs, changes = validate_manifests(old, new)
        artifacts = {}
        unpublished = []
        for name in ('claims', 'records'):
            rows = {}
            for path in (source / name).iterdir():
                pending = name == 'claims' and re.fullmatch(r'\.pending-[0-9a-f]{32}', path.name)
                if not path.is_file() or (not pending and (path.suffix != '.json' or path.stem not in jobs)):
                    raise ValueError('Unexpected source artifact')
                row = load(path)
                key = row.get('job_id') if pending else path.stem
                if (key not in jobs or any(row.get(k) != v for k, v in jobs[key].items())
                        or row.get('resume_fingerprint') != old['resume_fingerprint']):
                    raise ValueError('Artifact identity/fingerprint mismatch')
                if any(k in row for k in ('imported_from', 'measurement_epoch',
                                         'execution_concurrency', 'execution_call_timeout')):
                    raise ValueError('Source is not an original sequential run')
                if name == 'records':
                    if row.get('status') not in TERMINAL:
                        raise ValueError('Invalid terminal status')
                    if row['status'] == 'success':
                        if (not isinstance(row.get('output'), str) or not row['output'].strip()
                                or row.get('finish_reason') == 'length' or row.get('http_requests') != 1):
                            raise ValueError('Invalid successful record')
                    elif row.get('output') is not None:
                        raise ValueError('Failed record contains output')
                elif (row.get('acknowledge_flores_terms') is not True
                      or row.get('confirm_training_data_clean') is not True):
                    raise ValueError('Missing source claim attestations')
                if pending:
                    # write_new cannot return (and transport cannot start) before
                    # publication. Never infer this for a published or recorded job.
                    if (any((source / directory / (key + '.json')).exists()
                            for directory in ('claims', 'records'))
                            or any(item['job_id'] == key for item in unpublished)):
                        raise ValueError('Pending source claim is not unambiguously unpublished')
                    relative = str(path.relative_to(source))
                    unpublished.append(dict(jobs[key], source_relative_path=relative,
                        source_resume_fingerprint=old['resume_fingerprint'],
                        source_sha256=original[relative]['sha256']))
                    continue
                rows[path.stem] = row
            artifacts[name] = rows
        if (artifacts['claims'].keys() != artifacts['records'].keys()
                or len(artifacts['claims']) != expected_completed or expected_completed < 1):
            raise ValueError('Source must have exactly the expected terminal claims and zero active claims')
        provenance = {key: dict(
            source_resume_fingerprint=old['resume_fingerprint'],
            source_claim_sha256=original[f'claims/{key}.json']['sha256'],
            source_record_sha256=original[f'records/{key}.json']['sha256'])
            for key in artifacts['claims']}
        history = dict(schema_version=1, source_run=str(source),
            migration_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            source_manifest_sha256=original['manifest.json']['sha256'],
            source_resume_fingerprint=old['resume_fingerprint'],
            target_resume_fingerprint=new['resume_fingerprint'],
            imported_count=expected_completed, execution_concurrency=1,
            execution_call_timeout=old['timeouts']['call_seconds'],
            measurement_epoch='sequential', changes=changes, artifacts=provenance,
            unpublished_source_claims=sorted(unpublished, key=lambda row: row['source_relative_path']),
            source_snapshot_fingerprint=fingerprint(original))
        if snapshot(source) != original or snapshot(target) != target_before:
            raise ValueError('Run changed during validation')
        if not apply:
            return history
        sentinel = target / 'claims/.continuation-incomplete'
        sentinel.mkdir(mode=0o700)
        sync_dir(sentinel.parent)
        # Leave the sentinel on ANY failure, including a failed write/fsync.
        # Recovery is deliberately a new prepared target, never guessing which
        # part of an interrupted import is safe to resume.
        for name, rows in artifacts.items():
            for key, row in rows.items():
                write_new(target / name / (key + '.json'), dict(row,
                    resume_fingerprint=new['resume_fingerprint'],
                    imported_from=provenance[key], execution_concurrency=1,
                    execution_call_timeout=old['timeouts']['call_seconds'],
                    measurement_epoch='sequential'))
        if snapshot(source) != original:
            raise ValueError('Source changed during import; target is blocked')
        history['imported_at_unix'] = time.time()
        write_new(target / 'continuation.json', history)
        sentinel.rmdir()
        sync_dir(sentinel.parent)
        return history


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--expected-completed', type=int, default=4357)
    parser.add_argument('--confirm-source-stopped', action='store_true')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args(argv)
    try:
        result = migrate(args.source, args.target, expected_completed=args.expected_completed,
                         apply=args.apply, confirm_source_stopped=args.confirm_source_stopped)
        print(f"{'IMPORTED' if args.apply else 'VALIDATED'}: {result['imported_count']} terminal jobs; "
              'concurrency 1 -> 256; call timeout 120 -> 900; no inference started.')
        return 0
    except Exception as error:
        # Do not print JSON values, corpus text, or potentially sensitive paths.
        print('Continuation refused: ' + type(error).__name__, file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
