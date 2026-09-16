import contextlib
import asyncio
import hashlib
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

    def execute(self, *extra):
        return self.invoke('--execute', '--acknowledge-flores-terms',
                           '--confirm-training-data-clean', '--max-requests', '3',
                           '--limit', '1', *extra)

    def test_independent_arms_and_durable_resume(self):
        self.replies = ['length', 503, 'stop']
        self.assertEqual(self.execute()[0], 2)
        self.assertEqual([len(p['messages']) for p in self.payloads], [1, 11, 21])
        self.assertNotIn('HELD_OUT_REFERENCE', json.dumps(self.payloads))
        self.assertTrue(all(p['temperature'] == 0.2 for p in self.payloads))
        records = [json.loads(p.read_text()) for p in sorted(
            (self.root / 'run/records').glob('*.json'))]
        self.assertEqual({r['status'] for r in records}, {'truncated', 'http_error', 'success'})
        self.assertTrue(all(r['output'] is None for r in records if r['status'] != 'success'))
        self.assertNotIn('SECRET_SERVER_BODY', json.dumps(records))
        self.assertEqual(self.execute('--resume')[0], 2)
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
        self.assertEqual(self.execute('--resume')[0], 2)
        self.assertEqual([len(p['messages']) for p in self.payloads], [1, 11, 21])
        self.assertFalse((self.root / 'run/records/en-0-0.json').exists())
        self.assertEqual(self.execute('--resume')[0], 2)
        self.assertEqual(len(self.payloads), 3)

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
        self.assertEqual(self.execute('--targets', 'en', 'de', '--model', 'de=edukate_csde_v1',
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
        self.assertEqual(self.execute('--overall-timeout', '0.02')[0], 2)
        self.assertEqual(len(self.payloads), 1)
        self.assertEqual(len(list((self.root / 'run/claims').glob('*.json'))), 1)

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
