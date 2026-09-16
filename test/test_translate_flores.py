import contextlib
import asyncio
import copy
import hashlib
import importlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import httpx


class FloresRunnerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = self.root / 'models.json'
        self.config.write_text(json.dumps([dict(
            model=f'edukate_cs{target}_v1', model_framework='oai_llm', source=['cs'],
            target=[target], provider='openai',
            server='http://benchmark.invalid', token='private-token',
            prompt='{sentence}', temperature=0.2) for target in ['en', 'de', 'uk']]))
        languages = {'ces_Latn': 'czec1258', 'eng_Latn': 'stan1293',
                     'deu_Latn': 'stan1295', 'ukr_Cyrl': 'ukra1253'}
        for split, count in [('dev', 997), ('devtest', 1012)]:
            (self.root / split).mkdir()
            for code, glotto in languages.items():
                language, script = code.split('_')
                rows = [dict(id=f'{i:04}', text=(f'source{i}' if code == 'ces_Latn'
                    else f'HELD_OUT_REFERENCE_{code}_{i}') if split == 'devtest'
                    else f'demo_{code}_{i}', iso_639_3=language, iso_15924=script,
                    glottocode=glotto, variant='', split=split,
                    url=f'https://synthetic.invalid/{split}/{i}') for i in range(count)]
                (self.root / split / f'{code}.jsonl').write_text(
                    ''.join(json.dumps(row) + '\n' for row in rows))
        self.addCleanup(patch.stopall)
        self.payloads = []
        self.replies = []
        async def handle(transport, request):
            self.payloads.append(json.loads(request.content))
            reply = self.replies.pop(0) if self.replies else 'stop'
            if reply == 'timeout':
                await asyncio.sleep(1)
            if reply == 'invalid':
                return httpx.Response(200, text='SECRET_SERVER_BODY')
            if isinstance(reply, int):
                return httpx.Response(reply, text='SECRET_SERVER_BODY')
            content = (json.loads(request.content)['messages'][-1]['content']
                       if reply == 'echo-source' else 'translation')
            return httpx.Response(200, json={'choices': [dict(
                finish_reason='stop' if reply == 'echo-source' else reply,
                message={'content': content})],
                'usage': {'total_tokens': 12}})
        patch.object(httpx.AsyncHTTPTransport, 'handle_async_request', handle).start()

    def invoke(self, *extra, model_id='edukate_csen_v1'):
        from scripts.translate_flores import main
        args = ['--data-dir', str(self.root), '--revision', 'a' * 40,
                '--models-config', str(self.config), '--targets', 'en',
                '--model', f'en={model_id}', '--model-revision', 'en=unknown',
                '--run-dir', str(self.root / 'run'), *extra]
        output = io.StringIO()
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
            result = main(args)
        self.assertNotIn('private-token', output.getvalue())
        return result, output.getvalue()

    def test_offline_plan_has_no_hidden_limit_or_artifacts(self):
        code, output = self.invoke()
        self.assertEqual(code, 0)
        self.assertIn('3036', output)
        self.assertEqual(self.payloads, [])
        self.assertFalse((self.root / 'run').exists())

    def test_prepare_only_is_offline_and_resumes_with_identical_manifest(self):
        blocked = AssertionError('Preparation must not initialize networking')
        with patch('socket.socket', side_effect=blocked), \
                patch.object(httpx.AsyncClient, '__init__', side_effect=blocked), \
                patch.object(httpx.Client, '__init__', side_effect=blocked):
            self.assertEqual(self.invoke('--prepare-only', '--limit', '1')[0], 0)
        run = self.root / 'run'
        self.assertEqual({p.name for p in run.iterdir()},
                         {'manifest.json', '.lock', 'claims', 'records'})
        self.assertEqual(list((run / 'claims').iterdir()), [])
        self.assertEqual(list((run / 'records').iterdir()), [])
        self.assertEqual((run / '.lock').read_bytes(), b'')
        for path in [run, *run.rglob('*')]:
            self.assertEqual(path.stat().st_mode & 0o777, 0o700 if path.is_dir() else 0o600)
        original = (run / 'manifest.json').read_bytes()
        manifest = json.loads(original)
        self.assertNotIn('prepare_only', manifest)
        self.assertEqual(self.payloads, [])
        self.assertEqual(self.invoke('--execute', '--resume', '--limit', '1',
                                     '--max-requests', '3')[0], 1)
        self.assertEqual(list((run / 'claims').iterdir()), [])
        self.assertEqual(self.execute('--resume')[0], 0)
        self.assertEqual((run / 'manifest.json').read_bytes(), original)
        rows = [json.loads(p.read_text()) for p in (run / 'records').glob('*.json')]
        self.assertEqual({row['job_id'] for row in rows}, {'en-0-0', 'en-0-5', 'en-0-10'})
        self.assertEqual(len(self.payloads), 3)
        for row in rows:
            self.assertEqual(row['status'], 'success')
            self.assertEqual(row['resume_fingerprint'], manifest['resume_fingerprint'])
            self.assertEqual(row['execution_call_timeout'], 900)

    def test_prepare_only_refuses_existing_directory_without_changes(self):
        run = self.root / 'run'
        run.mkdir(mode=0o700)
        self.assertEqual(self.invoke('--prepare-only')[0], 1)
        self.assertEqual(list(run.iterdir()), [])
        run.rmdir()
        self.assertEqual(self.invoke('--prepare-only')[0], 0)
        original = (run / 'manifest.json').read_bytes()
        self.assertEqual(self.invoke('--prepare-only', '--seed', '1730')[0], 1)
        self.assertEqual((run / 'manifest.json').read_bytes(), original)
        self.assertEqual(list((run / 'claims').iterdir()), [])
        self.assertEqual(self.payloads, [])

    def test_prepare_only_requires_run_dir_and_refuses_execute_or_resume(self):
        from scripts.translate_flores import main
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(main(['--prepare-only', '--data-dir', str(self.root),
                                   '--revision', 'a' * 40]), 1)
        for flag in ['--execute', '--resume']:
            self.assertEqual(self.invoke('--prepare-only', flag)[0], 1)
        self.assertFalse((self.root / 'run').exists())
        self.assertEqual(self.payloads, [])

    def prepare_import(self):
        from scripts.continue_flores_run import migrate
        from scripts.translate_flores import fingerprint, write_new
        self.assertEqual(self.invoke('--prepare-only', '--limit', '1')[0], 0)
        target = self.root / 'run'
        old = copy.deepcopy(json.loads((target / 'manifest.json').read_text()))
        old.update(program_version='1.0', generation=dict(
            batch_size=1, retry_attempts=1, max_resplit_depth=0))
        old['timeouts']['call_seconds'] = 120
        old['code_hashes']['scripts/translate_flores.py'] = 'a' * 64
        old.pop('resume_fingerprint')
        old['resume_fingerprint'] = fingerprint(old)
        source = self.root / 'synthetic-old'
        source.mkdir(mode=0o700)
        for directory in ['claims', 'records']:
            (source / directory).mkdir(mode=0o700)
        write_new(source / '.lock', {})
        write_new(source / 'manifest.json', old)
        job = old['jobs'][0]
        write_new(source / 'claims' / (job['job_id'] + '.json'), dict(job,
            resume_fingerprint=old['resume_fingerprint'], acknowledge_flores_terms=True,
            confirm_training_data_clean=True))
        write_new(source / 'records' / (job['job_id'] + '.json'), dict(job,
            resume_fingerprint=old['resume_fingerprint'], status='success',
            output='imported translation', finish_reason='stop', http_requests=1))
        migrate(source, target, expected_completed=1, apply=True, confirm_source_stopped=True)
        return target

    def test_actual_import_resumes_without_replaying_imported_job(self):
        target = self.prepare_import()
        original = (target / 'records/en-0-0.json').read_bytes()
        self.assertEqual(self.execute('--resume')[0], 0)
        self.assertEqual(len(self.payloads), 2)
        self.assertEqual((target / 'records/en-0-0.json').read_bytes(), original)

    def test_deleted_import_pair_blocks_resume_before_http(self):
        target = self.prepare_import()
        for directory in ['claims', 'records']:
            (target / directory / 'en-0-0.json').unlink()
        self.assertEqual(self.execute('--resume')[0], 1)
        self.assertEqual(self.payloads, [])

    def test_import_audit_and_provenance_corruption_block_before_http(self):
        target = self.prepare_import()
        path = target / 'continuation.json'
        original = json.loads(path.read_text())
        mutations = [
            {'imported_count': 2}, {'artifacts': {}},
            {'target_resume_fingerprint': 'b' * 64},
            {'source_resume_fingerprint': 'b' * 64},
            {'source_manifest_sha256': ''},
            {'artifacts': {'unknown-job': original['artifacts']['en-0-0']}},
            {'artifacts': {'en-0-0': dict(original['artifacts']['en-0-0'],
                                        source_record_sha256='')}},
        ]
        for mutation in mutations:
            with self.subTest(mutation=mutation):
                path.write_text(json.dumps(dict(original, **mutation)))
                self.assertEqual(self.execute('--resume')[0], 1)
                self.assertEqual(self.payloads, [])
        path.write_text(json.dumps(original))
        for name in ['claims', 'records']:
            artifact = target / name / 'en-0-0.json'
            row = json.loads(artifact.read_text())
            corrupt = copy.deepcopy(row)
            corrupt['imported_from']['source_claim_sha256'] = 'b' * 64
            artifact.write_text(json.dumps(corrupt))
            self.assertEqual(self.execute('--resume')[0], 1)
            artifact.write_text(json.dumps(row))
        path.unlink()
        self.assertEqual(self.execute('--resume')[0], 1)
        self.assertEqual(self.payloads, [])

    def test_uncertain_request_pauses_targets_and_requires_drain_confirmation(self):
        flags = ['--concurrency', '1', '--call-timeout', '0.03', '--targets', 'en', 'de',
                 '--model', 'de=edukate_csde_v1', '--model-revision', 'de=unknown',
                 '--max-requests', '6']
        self.replies = ['timeout']
        self.assertEqual(self.execute(*flags)[0], 2)
        self.assertEqual(len(self.payloads), 1)
        run = self.root / 'run'
        original = (run / 'records/en-0-0.json').read_bytes()
        self.assertTrue(json.loads(original)['remote_completion_uncertain'])
        self.assertEqual(self.execute('--resume', *flags)[0], 1)
        self.assertEqual(len(self.payloads), 1)
        self.assertEqual(self.execute('--resume', '--confirm-backend-drained', *flags)[0], 2)
        self.assertEqual(len(self.payloads), 6)
        self.assertEqual((run / 'records/en-0-0.json').read_bytes(), original)
        confirmations = list(run.glob('backend-drained-*.json'))
        self.assertEqual(len(confirmations), 1)
        self.assertEqual(confirmations[0].stat().st_mode & 0o777, 0o600)

    def test_transport_disconnect_stops_dispatch_but_drains_inflight(self):
        active = calls = 0
        ready = asyncio.Event()

        async def handle(transport, request):
            nonlocal active, calls
            active += 1
            calls += 1
            if active == 3:
                ready.set()
            try:
                await ready.wait()
                body = json.loads(request.content)
                self.assertEqual(body['model'], 'edukate_csen_v1')
                if len(body['messages']) == 1:
                    raise httpx.RemoteProtocolError('Synthetic disconnect')
                await asyncio.sleep(0.02)
                return httpx.Response(200, json={'choices': [dict(
                    finish_reason='stop', message={'content': 'translated'})]})
            finally:
                active -= 1

        with patch.object(httpx.AsyncHTTPTransport, 'handle_async_request', handle):
            self.assertEqual(self.execute('--concurrency', '3', '--targets', 'en', 'de',
                '--model', 'de=edukate_csde_v1', '--model-revision', 'de=unknown',
                '--limit', '2', '--max-requests', '12')[0], 2)
        self.assertEqual((active, calls), (0, 3))
        rows = [json.loads(p.read_text()) for p in (self.root / 'run/records').glob('*.json')]
        self.assertEqual(len(rows), 3)
        self.assertEqual(sum(row['status'] == 'success' for row in rows), 2)
        self.assertEqual(sum(row['remote_completion_uncertain'] for row in rows), 1)

    def test_complete_http_errors_do_not_pause_target_switch(self):
        self.replies = [400, 503, 'stop']
        flags = ['--concurrency', '1', '--targets', 'en', 'de',
                 '--model', 'de=edukate_csde_v1', '--model-revision', 'de=unknown',
                 '--max-requests', '6']
        self.assertEqual(self.execute(*flags)[0], 2)
        self.assertEqual(len(self.payloads), 6)
        self.assertEqual(self.execute('--resume', *flags)[0], 2)
        self.assertEqual(len(self.payloads), 6)
        self.assertEqual(self.invoke('--confirm-backend-drained')[0], 1)
        self.assertEqual(self.execute('--confirm-backend-drained', *flags)[0], 1)

    def test_global_concurrency_and_out_of_order_association(self):
        active = peak = 0
        clients = {}

        async def handle(transport, request):
            nonlocal active, peak
            body = json.loads(request.content)
            shots = (len(body['messages']) - 1) // 2
            clients.setdefault(shots, set()).add(id(transport))
            source = body['messages'][-1]['content']
            active += 1
            peak = max(peak, active)
            try:
                await asyncio.sleep(0.02 if shots == 0 else 0.001)
                return httpx.Response(200, headers={'x-request-id': f'{source}-{shots}'},
                    json={'choices': [dict(finish_reason='stop',
                        message={'content': f'{source}-{shots}'})]})
            finally:
                active -= 1

        with patch.object(httpx.AsyncHTTPTransport, 'handle_async_request', handle):
            self.assertEqual(self.execute('--concurrency', '3', '--limit', '5',
                                          '--max-requests', '15')[0], 0)
        self.assertEqual(peak, 3)
        self.assertEqual(active, 0)
        self.assertEqual([len(values) for values in clients.values()], [1, 1, 1])
        self.assertEqual(len(set.union(*clients.values())), 3)
        for path in (self.root / 'run/records').glob('*.json'):
            row = json.loads(path.read_text())
            expected = f"source{int(row['eval_id'])}-{row['shots']}"
            self.assertEqual(row['output'], expected)
            self.assertEqual(row['response_request_id'], expected)
            self.assertEqual(row['http_requests'], 1)
            self.assertEqual(row['execution_concurrency'], 3)
            self.assertLessEqual(row['started_at_unix'], row['completed_at_unix'])
        manifest = json.loads((self.root / 'run/manifest.json').read_text())
        self.assertEqual(manifest['generation'], dict(
            retry_attempts=1, max_resplit_depth=0, batch_size=3, concurrency=3))
        self.assertEqual(manifest['timeouts']['call_seconds'], 900)

    def execute(self, *extra):
        return self.invoke('--execute', '--acknowledge-flores-terms',
                           '--confirm-training-data-clean', '--max-requests', '3',
                           '--limit', '1', *extra)

    def test_independent_arms_and_durable_resume(self):
        self.replies = ['length', 503, 'stop']
        self.assertEqual(self.execute('--concurrency', '1')[0], 2)
        self.assertEqual([len(p['messages']) for p in self.payloads], [1, 11, 21])
        self.assertNotIn('HELD_OUT_REFERENCE', json.dumps(self.payloads))
        self.assertTrue(all(p['temperature'] == 0.2 for p in self.payloads))
        records = [json.loads(p.read_text()) for p in sorted(
            (self.root / 'run/records').glob('*.json'))]
        self.assertEqual({r['status'] for r in records}, {'truncated', 'http_error', 'success'})
        self.assertTrue(all(r['output'] is None for r in records if r['status'] != 'success'))
        self.assertNotIn('SECRET_SERVER_BODY', json.dumps(records))
        self.assertEqual(self.execute('--resume', '--concurrency', '1')[0], 2)
        self.assertEqual(len(self.payloads), 3)
        for path in (self.root / 'run').rglob('*'):
            self.assertEqual(path.stat().st_mode & 0o777, 0o700 if path.is_dir() else 0o600)
            if path.is_file():
                self.assertNotIn('private-token', path.read_text())

    def test_execution_requires_attestations_and_budget(self):
        self.assertEqual(self.invoke('--execute')[0], 1)
        self.assertEqual(self.payloads, [])

    def test_full_three_language_plan(self):
        code, output = self.invoke('--targets', 'en', 'de', 'uk',
            '--model', 'de=edukate_csde_v1', '--model', 'uk=edukate_csuk_v1',
            '--model-revision', 'de=immutable-sha', '--model-revision', 'uk=immutable-sha')
        self.assertEqual(code, 0)
        self.assertIn('9108 requests', output)
        self.assertEqual(self.payloads, [])

    def test_budget_and_uncertain_claim_are_never_replayed(self):
        self.assertEqual(self.execute('--max-requests', '1')[0], 2)
        (self.root / 'run/records/en-0-0.json').unlink()
        self.assertEqual(self.execute('--resume')[0], 1)
        self.assertEqual(len(self.payloads), 1)
        code, output = self.execute('--resume', '--confirm-backend-drained')
        self.assertEqual(code, 2)
        self.assertIn('scoring_eligible=false', output)
        self.assertEqual([len(p['messages']) for p in self.payloads], [1, 11, 21])
        self.assertFalse((self.root / 'run/records/en-0-0.json').exists())
        confirmation = next((self.root / 'run').glob('backend-drained-*.json'))
        self.assertEqual(json.loads(confirmation.read_text())['uncertain_jobs'], ['en-0-0'])
        self.assertEqual(self.execute('--resume')[0], 1)
        self.assertEqual(self.execute('--resume', '--confirm-backend-drained')[0], 2)
        self.assertEqual(len(self.payloads), 3)
        self.assertFalse((self.root / 'run/records/en-0-0.json').exists())

    def test_timeout_and_invalid_response_have_no_output(self):
        self.replies = ['timeout', 'invalid', 'stop']
        self.assertEqual(self.execute('--call-timeout', '0.02')[0], 2)
        for shots, status in [(0, 'timeout'), (5, 'invalid_response')]:
            row = json.loads((self.root / f'run/records/en-0-{shots}.json').read_text())
            self.assertEqual(row['status'], status)
            self.assertIsNone(row['output'])
            self.assertEqual(row['http_requests'], 1)

    def test_success_and_resume_configuration_drift(self):
        self.assertEqual(self.execute()[0], 0)
        self.assertEqual(self.execute()[0], 1)
        self.assertEqual(self.execute('--resume')[0], 0)
        configs = json.loads(self.config.read_text())
        configs[0]['temperature'] = 0.9
        self.config.write_text(json.dumps(configs))
        self.assertEqual(self.execute('--resume')[0], 1)
        self.assertEqual(len(self.payloads), 3)

    def test_effective_settings_endpoint_changes_refuse_resume(self):
        configs = json.loads(self.config.read_text())
        configs[0]['server'] = '{FLORES_TEST_SERVER}'
        self.config.write_text(json.dumps(configs))
        settings = self.root / 'settings.py'
        settings.write_text("FLORES_TEST_SERVER = 'http://first.invalid'\n")
        with patch.dict('os.environ', {'LOCAL_SETTINGS': str(settings)}):
            self.assertEqual(self.execute()[0], 0)
            settings.write_text("FLORES_TEST_SERVER = 'http://different.invalid'\n")
            self.assertEqual(self.execute('--resume')[0], 1)
        self.assertEqual(len(self.payloads), 3)

    def test_unsafe_artifacts_and_symlink_run_root_refused(self):
        (self.root / 'elsewhere').mkdir()
        (self.root / 'run').symlink_to(self.root / 'elsewhere', target_is_directory=True)
        self.assertEqual(self.execute('--resume')[0], 1)
        (self.root / 'run').unlink()
        self.assertEqual(self.execute()[0], 0)
        (self.root / 'run/manifest.json').chmod(0o644)
        self.assertEqual(self.execute('--resume')[0], 1)
        self.assertEqual(len(self.payloads), 3)

    def test_pairs_are_consecutive_and_sources_unchanged(self):
        self.assertEqual(self.execute('--concurrency', '1', '--targets', 'en', 'de', '--model', 'de=edukate_csde_v1',
            '--model-revision', 'de=unknown', '--limit', '2', '--max-requests', '12')[0], 0)
        self.assertEqual([len(p['messages']) for p in self.payloads], [1, 11, 21] * 4)
        self.assertEqual([p['messages'][-1]['content'] for p in self.payloads],
                         ['source0'] * 3 + ['source1'] * 3 + ['source0'] * 3 + ['source1'] * 3)

    def test_corrupt_claim_fingerprint_refuses_resume(self):
        self.assertEqual(self.execute()[0], 0)
        path = self.root / 'run/claims/en-0-0.json'
        claim = json.loads(path.read_text())
        claim['resume_fingerprint'] = 'incorrect'
        path.write_text(json.dumps(claim))
        self.assertEqual(self.execute('--resume')[0], 1)

    def test_cross_run_record_substitution_refuses_resume_without_http(self):
        self.assertEqual(self.execute()[0], 0)
        other = self.root / 'other-run'
        self.assertEqual(self.execute('--run-dir', str(other), '--seed', '1730')[0], 0)
        path = self.root / 'run/records/en-0-0.json'
        path.write_bytes((other / 'records/en-0-0.json').read_bytes())
        self.assertEqual(self.execute('--resume')[0], 1)
        self.assertEqual(len(self.payloads), 6)

    def test_manifest_claims_and_all_terminal_records_share_canonical_fingerprint(self):
        self.replies = ['length', 503, 'stop']
        self.assertEqual(self.execute()[0], 2)
        manifest = json.loads((self.root / 'run/manifest.json').read_text())
        declared = manifest.pop('resume_fingerprint')
        expected = hashlib.sha256(json.dumps(
            manifest, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        self.assertEqual(declared, expected)
        for directory in ['claims', 'records']:
            for path in (self.root / 'run' / directory).glob('*.json'):
                self.assertEqual(json.loads(path.read_text())['resume_fingerprint'], expected)

    def test_missing_record_fingerprint_and_corrupt_manifest_refuse_resume(self):
        self.assertEqual(self.execute()[0], 0)
        path = self.root / 'run/records/en-0-0.json'
        original = path.read_text()
        record = json.loads(original)
        record.pop('resume_fingerprint', None)
        path.write_text(json.dumps(record))
        self.assertEqual(self.execute('--resume')[0], 1)
        path.write_text(original)
        path = self.root / 'run/manifest.json'
        manifest = json.loads(path.read_text())
        manifest['resume_fingerprint'] = 'incorrect'
        path.write_text(json.dumps(manifest))
        self.assertEqual(self.execute('--resume')[0], 1)
        self.assertEqual(len(self.payloads), 3)

    def test_successful_identity_output_keeps_normal_model_acceptance(self):
        self.replies = ['echo-source'] * 3
        self.assertEqual(self.execute()[0], 0)
        for path in (self.root / 'run/records').glob('*.json'):
            row = json.loads(path.read_text())
            self.assertEqual(row['status'], 'success')
            self.assertEqual(row['output'], 'source0')
        generation = [{k: v for k, v in p.items() if k != 'messages'} for p in self.payloads]
        self.assertEqual(generation, [generation[0]] * 3)

    def test_overall_deadline_stops_without_claiming_next_arm(self):
        self.replies = ['timeout']
        self.assertEqual(self.execute('--concurrency', '1', '--overall-timeout', '0.05')[0], 2)
        self.assertEqual(len(self.payloads), 1)
        self.assertEqual(len(list((self.root / 'run/claims').glob('*.json'))), 1)

    def test_default_concurrency_caps_requests_and_budget(self):
        active = peak = calls = 0
        release = asyncio.Event()

        async def handle(transport, request):
            nonlocal active, peak, calls
            calls += 1
            active += 1
            peak = max(peak, active)
            if active == 256:
                release.set()
            try:
                await asyncio.wait_for(release.wait(), 5)
                await asyncio.sleep(0)
                return httpx.Response(200, json={'choices': [dict(
                    finish_reason='stop', message={'content': 'translated'})]})
            finally:
                active -= 1

        with patch.object(httpx.AsyncHTTPTransport, 'handle_async_request', handle):
            self.assertEqual(self.execute('--limit', '100', '--max-requests', '263')[0], 2)
            self.assertEqual(self.execute('--limit', '100', '--max-requests', '263', '--resume')[0], 2)
        self.assertEqual((peak, active, calls), (256, 0, 263))
        self.assertEqual(len(list((self.root / 'run/claims').glob('*.json'))), 263)
        self.assertEqual(len(list((self.root / 'run/records').glob('*.json'))), 263)

    def test_cli_bounds_and_concurrency_resume_drift(self):
        for flags in [('--concurrency', '0'), ('--concurrency', '513'),
                      ('--call-timeout', '3601'), ('--call-timeout', 'nan')]:
            self.assertEqual(self.invoke(*flags)[0], 1)
        self.assertEqual(self.execute('--concurrency', '512', '--call-timeout', '3600')[0], 0)
        self.assertEqual(self.execute('--resume', '--concurrency', '256', '--call-timeout', '3600')[0], 1)
        self.assertEqual(len(self.payloads), 3)

    def test_targets_drain_and_clients_close(self):
        active = 0
        closed = []
        original_close = httpx.AsyncClient.__aexit__

        async def close(client, *args):
            self.assertEqual(active, 0)
            closed.append(client)
            await original_close(client, *args)

        async def handle(transport, request):
            nonlocal active
            body = json.loads(request.content)
            if body['model'] == 'edukate_csde_v1':
                self.assertEqual(len(closed), 3)
                self.assertTrue(all(client.is_closed for client in closed))
                self.assertEqual(len(list((self.root / 'run/records').glob('en-*.json'))), 6)
            active += 1
            try:
                await asyncio.sleep(0.005)
                return httpx.Response(200, json={'choices': [dict(
                    finish_reason='stop', message={'content': 'translated'})]})
            finally:
                active -= 1

        with patch.object(httpx.AsyncHTTPTransport, 'handle_async_request', handle), \
                patch.object(httpx.AsyncClient, '__aexit__', close):
            self.assertEqual(self.execute('--concurrency', '3', '--targets', 'en', 'de',
                '--model', 'de=edukate_csde_v1', '--model-revision', 'de=unknown',
                '--limit', '2', '--max-requests', '12')[0], 0)
        self.assertEqual(len(closed), 6)
        self.assertTrue(all(client.is_closed for client in closed))

    def test_partial_response_deadline_finalizes_all_claims(self):
        active = 0

        class SlowBody(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield b'{"choices": ['
                await asyncio.sleep(10)

        async def handle(transport, request):
            nonlocal active
            active += 1
            return httpx.Response(200, stream=SlowBody())

        with patch.object(httpx.AsyncHTTPTransport, 'handle_async_request', handle):
            self.assertEqual(self.execute('--concurrency', '3', '--limit', '5',
                '--max-requests', '15', '--overall-timeout', '0.1')[0], 2)
        self.assertEqual(active, 3)
        claims = list((self.root / 'run/claims').glob('*.json'))
        records = list((self.root / 'run/records').glob('*.json'))
        self.assertEqual(len(claims), 3)
        self.assertEqual(len(records), 3)
        for path in records:
            row = json.loads(path.read_text())
            self.assertEqual(row['status'], 'timeout')
            self.assertIsNone(row['output'])

    def test_cancellation_drains_workers_and_finalizes_claims(self):
        active = 0
        ready = asyncio.Event()

        async def handle(transport, request):
            nonlocal active
            active += 1
            if active == 3:
                ready.set()
            try:
                await ready.wait()
                if len(json.loads(request.content)['messages']) == 1:
                    raise asyncio.CancelledError()
                await asyncio.sleep(10)
            finally:
                active -= 1

        with patch.object(httpx.AsyncHTTPTransport, 'handle_async_request', handle):
            with self.assertRaises(asyncio.CancelledError):
                self.execute('--concurrency', '3', '--limit', '5', '--max-requests', '15')
        self.assertEqual(active, 0)
        records = list((self.root / 'run/records').glob('*.json'))
        self.assertEqual(len(records), 3)
        self.assertEqual(len(list((self.root / 'run/claims').glob('*.json'))), 3)
        for path in records:
            row = json.loads(path.read_text())
            self.assertEqual(row['status'], 'error')
            self.assertEqual(row['error_type'], 'CancelledError')
            self.assertIsNone(row['output'])

    def test_extra_http_per_job_blocked_concurrently(self):
        original_send = httpx.AsyncClient.send

        async def send(client, request, **kwargs):
            await original_send(client, request, **kwargs)
            return await original_send(client, request, **kwargs)

        with patch.object(httpx.AsyncClient, 'send', send):
            self.assertEqual(self.execute('--concurrency', '3')[0], 2)
        self.assertEqual(len(self.payloads), 3)
        for path in (self.root / 'run/records').glob('*.json'):
            row = json.loads(path.read_text())
            self.assertNotEqual(row['status'], 'success')
            self.assertEqual(row['http_requests'], 1)
            self.assertIsNone(row['output'])

    def test_claim_failure_drains_already_claimed_jobs(self):
        from scripts.translate_flores import write_new
        ready = asyncio.Event()
        active = 0

        def write(path, value):
            if path.parent.name == 'claims' and path.stem == 'en-1-0':
                raise OSError('Synthetic claim publication failure')
            write_new(path, value)

        async def handle(transport, request):
            nonlocal active
            active += 1
            if active == 3:
                ready.set()
            try:
                await ready.wait()
                if len(json.loads(request.content)['messages']) != 1:
                    await asyncio.sleep(10)
                return httpx.Response(200, json={'choices': [dict(
                    finish_reason='stop', message={'content': 'translated'})]})
            finally:
                active -= 1

        with patch.object(httpx.AsyncHTTPTransport, 'handle_async_request', handle), \
                patch('scripts.translate_flores.write_new', write):
            self.assertEqual(self.execute('--concurrency', '3', '--limit', '2',
                                          '--max-requests', '6')[0], 1)
        self.assertEqual(active, 0)
        claims = {p.name for p in (self.root / 'run/claims').glob('*.json')}
        records = {p.name for p in (self.root / 'run/records').glob('*.json')}
        self.assertEqual(len(claims), 3)
        self.assertEqual(claims, records)

    def test_generation_prefix_and_scorer_schema(self):
        self.assertEqual(self.execute('--concurrency', '3')[0], 0)
        arms = {len(p['messages']): p for p in self.payloads}
        self.assertEqual(arms[11]['messages'][:-1], arms[21]['messages'][:10])
        for payload in self.payloads:
            self.assertEqual(payload['temperature'], 0.2)
            self.assertEqual(payload['max_tokens'], 512)
            self.assertEqual(payload['stop'], ['###'])
        try:
            sacrebleu = importlib.import_module('sacrebleu')
        except ImportError:
            self.skipTest('Scorer requires optional sacrebleu==2.5.1')
        if sacrebleu.__version__ != '2.5.1':
            self.skipTest('Scorer requires optional sacrebleu==2.5.1')
        from scripts.score_flores import score_run
        report = score_run(self.root / 'run', bootstrap_samples=0)
        self.assertIsInstance(report, dict)

    def test_no_automatic_model_selection_or_ambiguous_configuration(self):
        configs = json.loads(self.config.read_text())
        configs[0]['model'] = 'different-model'
        self.config.write_text(json.dumps(configs))
        self.assertEqual(self.invoke()[0], 1)
        configs[0]['model'] = 'edukate_csen_v1'
        self.config.write_text(json.dumps(configs * 2))
        self.assertEqual(self.invoke()[0], 1)
        self.assertEqual(self.payloads, [])

    def test_compatible_unapproved_models_rejected_in_preview_and_execution(self):
        configs = json.loads(self.config.read_text())
        names = ['EuroLLM-9B-Instruct', 'CohereForAI/aya-expanse-8b', 'edukate_csde_v1']
        self.config.write_text(json.dumps(configs + [dict(configs[0], model=name) for name in names]))
        for index, name in enumerate(names):
            for execute in [False, True]:
                with self.subTest(model=name, execute=execute):
                    flags = ['--execute', '--acknowledge-flores-terms',
                             '--confirm-training-data-clean', '--max-requests', '3'] if execute else []
                    run_dir = self.root / f'unapproved-{index}-{execute}'
                    self.assertEqual(self.invoke('--limit', '1', '--run-dir', str(run_dir),
                                                 *flags, model_id=name)[0], 1)
                    self.assertEqual(self.payloads, [])
                    self.assertFalse(run_dir.exists())


if __name__ == '__main__':
    unittest.main()
