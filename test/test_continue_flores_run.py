import ast
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any
import unittest
from unittest.mock import patch

from scripts import continue_flores_run as migration


class ContinueFloresTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.source = Path(self.tmp.name) / 'old'
        self.target = Path(self.tmp.name) / 'new'
        runner = (migration.ROOT / 'scripts/translate_flores.py').read_bytes()
        version = next(n.value.value for n in ast.parse(runner).body
                       if isinstance(n, ast.Assign) and isinstance(n.value, ast.Constant)
                       and any(isinstance(t, ast.Name) and t.id == 'VERSION' for t in n.targets))
        self.old: dict[str, Any] = dict(schema_version=1, program_version='old',
            dataset=dict(revision='fixed', files={'dev': 'hash'},
                         evaluation=[{'id': '1', 'source': 'PRIVATE'}],
                         demonstrations={'en': [{'id': '2', 'source': 'demo', 'target': 'example'}]}),
            models={'en': dict(model_id='edukate_csen_v1', revision='unknown',
                config_fingerprint='cfg', effective_endpoint_hash='endpoint',
                generation=dict(temperature=0.2, max_completion_tokens=512, stop=['###'],
                                prompt_hash='prompt', provider='openai'))},
            conditions=[0, 5, 10],
            jobs=[dict(job_id=f'en-0-{s}', target='en', shots=s, eval_id='1') for s in (0, 5, 10)],
            generation=dict(retry_attempts=1, max_resplit_depth=0, batch_size=1),
            timeouts=dict(call_seconds=120, overall_seconds=86400),
            code_hashes={'scripts/translate_flores.py': 'old', 'app/models/model.py': 'unchanged'})
        self.new: dict[str, Any] = copy.deepcopy(self.old)
        self.new['program_version'] = version
        self.new['code_hashes']['scripts/translate_flores.py'] = hashlib.sha256(runner).hexdigest()
        self.new['generation'].update(concurrency=256, batch_size=256)
        self.new['timeouts']['call_seconds'] = 900
        for root, manifest in ((self.source, self.old), (self.target, self.new)):
            root.mkdir(mode=0o700)
            (root / 'claims').mkdir(mode=0o700)
            (root / 'records').mkdir(mode=0o700)
            migration.write_new(root / '.lock', {})
            self.save_manifest(root, manifest)
        for job, status in zip(self.old['jobs'][:2], ('success', 'timeout')):
            base = dict(job, resume_fingerprint=self.old['resume_fingerprint'])
            migration.write_new(self.source / 'claims' / (job['job_id'] + '.json'), dict(base,
                claimed_at_unix=42, acknowledge_flores_terms=True, confirm_training_data_clean=True))
            migration.write_new(self.source / 'records' / (job['job_id'] + '.json'), dict(base,
                status=status, output='PRIVATE output' if status == 'success' else None,
                finish_reason='stop', duration_seconds=1.25, http_requests=1, usage={'total_tokens': 12}))

    def save_manifest(self, root, manifest):
        manifest.pop('resume_fingerprint', None)
        manifest['resume_fingerprint'] = migration.fingerprint(manifest)
        path = root / 'manifest.json'
        if path.exists():
            path.write_text(json.dumps(manifest))
        else:
            migration.write_new(path, manifest)

    def run_migration(self, **kwargs):
        return migration.migrate(self.source, self.target, expected_completed=2,
                                 confirm_source_stopped=True, **kwargs)

    def test_dry_run_is_read_only_and_apply_preserves_original_fields(self):
        before = migration.snapshot(self.source)
        target_before = migration.snapshot(self.target)
        self.assertEqual(self.run_migration()['imported_count'], 2)
        self.assertEqual(migration.snapshot(self.source), before)
        self.assertEqual(migration.snapshot(self.target), target_before)
        history = self.run_migration(apply=True)
        self.assertEqual(migration.snapshot(self.source), before)
        self.assertEqual(migration.load(self.target / 'manifest.json'), self.new)
        self.assertEqual(history, migration.load(self.target / 'continuation.json'))
        self.assertEqual(history['execution_concurrency'], 1)
        self.assertEqual(history['execution_call_timeout'], 120)
        for name in ('claims', 'records'):
            for path in (self.source / name).iterdir():
                original = migration.load(path)
                imported = migration.load(self.target / name / path.name)
                for key, value in original.items():
                    if key != 'resume_fingerprint':
                        self.assertEqual(imported[key], value)
                self.assertEqual(imported['resume_fingerprint'], self.new['resume_fingerprint'])
                self.assertEqual(imported['execution_concurrency'], 1)
                self.assertEqual(imported['execution_call_timeout'], 120)
                self.assertEqual(imported['measurement_epoch'], 'sequential')
                self.assertEqual(imported['imported_from'], history['artifacts'][path.stem])
        self.assertFalse((self.target / 'claims/.continuation-incomplete').exists())
        with self.assertRaises(ValueError):
            self.run_migration(apply=True)

    def test_semantic_drift_is_not_allowlisted(self):
        mutations = [lambda m: m['dataset']['evaluation'].reverse(),
            lambda m: m['dataset']['demonstrations']['en'][0].update(target='changed'),
            lambda m: m['dataset'].update(revision='changed'),
            lambda m: m['dataset']['files'].update(dev='changed'),
            lambda m: m['models']['en'].update(config_fingerprint='changed'),
            lambda m: m['models']['en'].update(effective_endpoint_hash='changed'),
            lambda m: m['models']['en'].update(revision='changed'),
            lambda m: m['models']['en']['generation'].update(temperature=0.9),
            lambda m: m['models']['en']['generation'].update(provider='changed'),
            lambda m: m['models']['en']['generation'].update(prompt_hash='changed'),
            lambda m: m['models']['en']['generation'].update(stop=[]),
            lambda m: m['code_hashes'].update({'app/models/model.py': 'changed'}),
            lambda m: m['jobs'].reverse(),
            lambda m: m['timeouts'].update(overall_seconds=12),
            lambda m: m.update(continuation={'unreproducible': True})]
        # Two evaluation IDs ensure order is actually exercised.
        self.old['dataset']['evaluation'].append({'id': '3', 'source': 'another'})
        self.new['dataset']['evaluation'].append({'id': '3', 'source': 'another'})
        self.save_manifest(self.source, self.old)
        self.save_manifest(self.target, self.new)
        for mutate in mutations:
            candidate = copy.deepcopy(self.new)
            mutate(candidate)
            self.save_manifest(self.target, candidate)
            with self.assertRaises(ValueError):
                migration.validate_manifests(self.old, candidate)

    def test_manifest_fingerprint_and_current_runner_pin(self):
        for field in ('resume_fingerprint', 'program_version'):
            candidate = copy.deepcopy(self.new)
            candidate[field] = 'wrong'
            if field != 'resume_fingerprint':
                self.save_manifest(self.target, candidate)
            with self.assertRaises(ValueError):
                migration.validate_manifests(self.old, candidate)
        self.old['resume_fingerprint'] = 'wrong'
        with self.assertRaises(ValueError):
            migration.validate_manifests(self.old, self.new)

    def test_active_claim_or_orphan_record_refused(self):
        for name in ('claims', 'records'):
            path = next((self.source / name).iterdir())
            content = path.read_bytes()
            path.unlink()
            with self.assertRaises(ValueError):
                self.run_migration()
            migration.write_new(path, json.loads(content))

    def test_corrupt_identity_fingerprint_status_and_attestations(self):
        for name, field, value in [('claims', 'resume_fingerprint', 'bad'),
                                  ('records', 'resume_fingerprint', 'bad'),
                                  ('records', 'eval_id', 'bad'),
                                  ('records', 'status', 'pending'),
                                  ('records', 'execution_call_timeout', 900),
                                  ('claims', 'acknowledge_flores_terms', False)]:
            path = next((self.source / name).iterdir())
            original = migration.load(path)
            path.write_text(json.dumps(dict(original, **{field: value})))
            with self.assertRaises(ValueError):
                self.run_migration()
            path.write_text(json.dumps(original))

    def test_both_nonblocking_locks(self):
        for root in (self.source, self.target):
            with (root / '.lock').open('rb') as stream:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaises(BlockingIOError):
                    self.run_migration(apply=True)

    def test_failure_leaves_durable_fail_closed_sentinel(self):
        original = migration.snapshot(self.source)
        write = migration.write_new
        calls = 0

        def fail(path, value):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError('simulated interruption')
            write(path, value)

        with patch.object(migration, 'write_new', side_effect=fail):
            with self.assertRaises(OSError):
                self.run_migration(apply=True)
        self.assertEqual(migration.snapshot(self.source), original)
        self.assertTrue((self.target / 'claims/.continuation-incomplete').is_dir())
        self.assertFalse((self.target / 'continuation.json').exists())
        with self.assertRaises(ValueError):
            self.run_migration(apply=True)

    def test_unsafe_files_and_unexpected_claims_refused(self):
        path = self.source / 'claims/.pending-crash'
        migration.write_new(path, {})
        with self.assertRaises(ValueError):
            self.run_migration()
        path.unlink()
        (self.target / 'manifest.json').chmod(0o644)
        with self.assertRaises(ValueError):
            self.run_migration()

    def test_requires_stop_confirmation_and_exact_count(self):
        with self.assertRaises(ValueError):
            migration.migrate(self.source, self.target, expected_completed=2)
        with self.assertRaises(ValueError):
            migration.migrate(self.source, self.target, confirm_source_stopped=True)

    def test_unpublished_claim_is_audited_not_imported(self):
        job = self.old['jobs'][2]
        path = self.source / 'claims' / ('.pending-' + 'a' * 32)
        migration.write_new(path, dict(job, resume_fingerprint=self.old['resume_fingerprint'],
            claimed_at_unix=43, acknowledge_flores_terms=True, confirm_training_data_clean=True))
        before = migration.snapshot(self.source)
        history = self.run_migration(apply=True)
        self.assertEqual(migration.snapshot(self.source), before)
        self.assertEqual(history['imported_count'], 2)
        pending, = history['unpublished_source_claims']
        self.assertEqual(pending['job_id'], job['job_id'])
        self.assertEqual(pending['source_resume_fingerprint'], self.old['resume_fingerprint'])
        self.assertEqual(pending['source_sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(pending['source_relative_path'], str(path.relative_to(self.source)))
        for name in ('claims', 'records'):
            self.assertFalse((self.target / name / (job['job_id'] + '.json')).exists())
            self.assertEqual(len(list((self.target / name).iterdir())), 2)
        self.assertEqual(history, migration.load(self.target / 'continuation.json'))

    def test_ambiguous_or_invalid_pending_claim_refused(self):
        job = self.old['jobs'][2]
        valid = dict(job, resume_fingerprint=self.old['resume_fingerprint'],
            claimed_at_unix=43, acknowledge_flores_terms=True, confirm_training_data_clean=True)
        cases = [('claims', dict(valid, **self.old['jobs'][0])),
                 ('claims', dict(valid, resume_fingerprint='wrong')),
                 ('claims', dict(valid, job_id='unknown')),
                 ('claims', dict(valid, eval_id='wrong')),
                 ('claims', dict(valid, acknowledge_flores_terms=False)),
                 ('records', valid)]
        for directory, row in cases:
            path = self.source / directory / ('.pending-' + 'a' * 32)
            migration.write_new(path, row)
            with self.assertRaises(ValueError):
                self.run_migration()
            path.unlink()
        path = self.source / 'claims' / ('.pending-' + 'a' * 32)
        migration.write_new(path, valid)
        path.write_text('{')
        with self.assertRaises(ValueError):
            self.run_migration()


if __name__ == '__main__':
    unittest.main()
