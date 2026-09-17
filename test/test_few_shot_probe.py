import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import httpx
from flask import Flask

from app.models.oai_llm_model import OaiLLMModel
from scripts.probe_few_shot import CORPUS, ROOT, experiment, trial, write_new


class FewShotProbeTests(unittest.TestCase):
    def test_adaptive_cap_survives_preflight_and_few_shot_prompt_growth(self):
        for shots in (0, 5):
            examples = [{'source': 'long example ' * 40, 'target': 'translation'}] * shots
            model = OaiLLMModel({
                'source': ['cs'], 'target': ['de'], 'model': 'public-model',
                'provider': 'openai', 'server': 'https://offline.invalid',
                'prompt': 'PRIVATE instruction ' * 40 + '{sentence}',
                'adaptive_completion_budget': {}, 'max_completion_tokens': 512,
                'few_shot_attempts': {'cs-de': [examples]},
                'retry_attempts': 1, 'max_resplit_depth': 0,
            })
            for offline in (True, False):
                with self.subTest(shots=shots, offline=offline):
                    model._create_client = lambda server: httpx.AsyncClient(
                        base_url=server, transport=httpx.MockTransport(lambda request:
                            httpx.Response(200, json={'choices': [{
                                'finish_reason': 'stop', 'message': {'content': 'translated'}}]})))
                    with Flask('offline-test').app_context():
                        observed = asyncio.run(trial(model, 'x', 'de', examples, 1, offline=offline))
                    self.assertEqual(observed['http_requests'], 1)
                    self.assertTrue(observed['app_translated'])
                    self.assertEqual(observed['effective_completion_tokens'], 64)
                    generation = observed['generation']
                    assert isinstance(generation, dict)
                    self.assertEqual(generation['max_tokens'], 64)
                    self.assertNotIn('PRIVATE', json.dumps(observed))

    def test_cli_requires_execute_without_loading_config(self):
        result = subprocess.run([
            sys.executable, str(ROOT / 'scripts/probe_few_shot.py'),
            '--models-config', '/nonexistent/private-config.json',
        ], capture_output=True, text=True, check=True)
        self.assertIn('NOT EXECUTED', result.stdout)
        self.assertEqual(result.stderr, '')

    def test_cli_rejects_excess_budget(self):
        result = subprocess.run([
            sys.executable, str(ROOT / 'scripts/probe_few_shot.py'),
            '--max-requests', '25',
        ], capture_output=True, text=True)
        self.assertEqual(result.returncode, 2)

    def test_fixture_is_held_out(self):
        fixture = json.loads((ROOT / 'examples/few_shot_attempts.json').read_text())
        for stages in fixture['few_shot_attempts'].values():
            self.assertEqual(len(stages), 1)
            self.assertEqual(len(stages[0]), 5)
            self.assertFalse(set(CORPUS) & {example['source'] for example in stages[0]})

    def test_artifacts_are_private_and_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'claim.json'
            write_new(path, {'claimed': True})
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            with self.assertRaises(FileExistsError):
                write_new(path, {'claimed': False})
            self.assertEqual(json.loads(path.read_text()), {'claimed': True})

    def test_real_model_with_mock_http_records_truncation_without_retry(self):
        requests = []

        def respond(request):
            requests.append(request)
            return httpx.Response(200, json={
                'id': 'completion-test', 'model': 'public-model',
                'usage': {'prompt_tokens': 30, 'completion_tokens': 512, 'total_tokens': 542},
                'choices': [{'finish_reason': 'length', 'message': {'content': 'partial'}}],
            })

        model = OaiLLMModel({
            'source': ['cs'], 'target': ['de'], 'model': 'public-model',
            'provider': 'openai', 'server': 'https://offline.invalid',
            'prompt': 'PRIVATE {sentence}', 'token': 'SECRET',
            'few_shot_attempts': {'cs-de': [[]]},
            'retry_attempts': 1, 'max_resplit_depth': 0,
        })
        model._create_client = lambda server: httpx.AsyncClient(
            base_url=server + '/v1', transport=httpx.MockTransport(respond))
        with Flask('offline-test').app_context():
            observed = asyncio.run(trial(model, 'F', 'de', [], 1))
        self.assertEqual(len(requests), 1)
        self.assertEqual(observed['termination'], 'truncated')
        self.assertEqual(observed['output'], 'partial')
        self.assertFalse(observed['app_translated'])
        self.assertEqual(observed['correctness'], 'pending_human_review')
        self.assertNotIn('PRIVATE', json.dumps(observed))
        self.assertNotIn('SECRET', json.dumps(observed))

    def test_message_mismatch_blocks_transport(self):
        requests = []

        def respond(request):
            requests.append(request)
            return httpx.Response(200)

        model = OaiLLMModel({
            'source': ['cs'], 'target': ['de'], 'model': 'public-model',
            'provider': 'openai', 'server': 'https://offline.invalid',
            'few_shot_attempts': {'cs-de': [[]]},
            'retry_attempts': 1, 'max_resplit_depth': 0,
        })
        model._create_client = lambda server: httpx.AsyncClient(
            base_url=server + '/v1', transport=httpx.MockTransport(respond))
        with Flask('offline-test').app_context(), self.assertRaises(RuntimeError):
            asyncio.run(trial(model, 'F', 'de', [{'source': 'K', 'target': 'K'}], 1))
        self.assertEqual(requests, [])

    def test_resume_rejects_changed_resolved_endpoint(self):
        for server in (None, '{PROBE_SERVER}'):
            with self.subTest(server=server), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config = root / 'models.json'
                cfg = {
                    'source': ['cs'], 'target': ['de', 'uk'], 'model': 'public-model',
                    'provider': 'openai', 'model_framework': 'oai_llm',
                }
                if server is not None:
                    cfg['server'] = server
                write_new(config, [cfg])
                settings = []
                for index in range(2):
                    path = root / f'settings-{index}.py'
                    path.write_text(
                        f'DEFAULT_SERVER = PROBE_SERVER = "https://private-{index}.invalid"\n')
                    settings.append(str(path))
                output = root / 'artifacts'
                output.mkdir(mode=0o700)
                args = SimpleNamespace(
                    models_config=str(config), examples=str(ROOT / 'examples/few_shot_attempts.json'),
                    model_de=None, model_uk=None, input_tokens=512, output_tokens=512,
                    call_timeout=120, overall_timeout=2880, max_requests=1, resume=False,
                )
                with patch('scripts.probe_few_shot.trial') as mocked_trial:
                    with patch.dict(os.environ, LOCAL_SETTINGS=settings[0]):
                        asyncio.run(experiment(args, output))
                    manifest = (output / 'manifest.json').read_text()
                    self.assertNotIn('private-0', manifest)
                    self.assertEqual(set(json.loads(manifest)['endpoint_fingerprints']), {'de', 'uk'})
                    args.resume = True
                    mocked_trial.reset_mock()
                    with patch.dict(os.environ, LOCAL_SETTINGS=settings[1]):
                        with self.assertRaisesRegex(ValueError, 'Resume manifest mismatch'):
                            asyncio.run(experiment(args, output))
                    mocked_trial.assert_not_called()
                    self.assertEqual((output / 'manifest.json').read_text(), manifest)

    def test_low_input_budget_rejected_before_preflight_or_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'models.json'
            write_new(config, [{
                'source': ['cs'], 'target': ['de', 'uk'], 'model': 'public-model',
                'provider': 'openai', 'model_framework': 'oai_llm',
                'server': 'https://offline.invalid',
            }])
            args = SimpleNamespace(
                models_config=str(config), examples=str(ROOT / 'examples/few_shot_attempts.json'),
                model_de=None, model_uk=None, input_tokens=1, output_tokens=512,
                call_timeout=120, overall_timeout=2880, max_requests=24, resume=False,
            )
            with patch('scripts.probe_few_shot.trial') as mocked_trial:
                with self.assertRaisesRegex(ValueError, 'Input budget'):
                    asyncio.run(experiment(args, root))
                mocked_trial.assert_not_called()
            self.assertFalse((root / 'manifest.json').exists())
            self.assertEqual(list(root.glob('*.claim.json')), [])

    def test_resume_skips_completed_and_uncertain_claims(self):
        calls = []

        async def fake_trial(model, source, target, examples, timeout,
                             artifact=None, offline=False):
            if not offline:
                calls.append((target, source, len(examples)))
                write_new(artifact, {'termination': 'stopped',
                                     'correctness': 'pending_human_review'})

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'models.json'
            write_new(config, [{
                'source': ['cs'], 'target': ['de', 'uk'], 'model': 'public-model',
                'provider': 'openai', 'model_framework': 'oai_llm',
                'server': 'https://offline.invalid',
            }])
            output = root / 'artifacts'
            output.mkdir(mode=0o700)
            # Simulate a crash after claiming the very first request.
            write_new(output / 'cs-de-0-zero.claim.json', {'uncertain': True})
            args = SimpleNamespace(
                models_config=str(config), examples=str(ROOT / 'examples/few_shot_attempts.json'),
                model_de=None, model_uk=None, input_tokens=512, output_tokens=512,
                call_timeout=120, overall_timeout=2880, max_requests=24, resume=False,
            )
            with patch('scripts.probe_few_shot.trial', side_effect=fake_trial):
                asyncio.run(experiment(args, output))
                args.resume = True
                asyncio.run(experiment(args, output))
            self.assertEqual(len(calls), 23)
            self.assertEqual(len(list(output.glob('*.claim.json'))), 24)
            self.assertEqual(len(list(output.glob('report-*.json'))), 2)
            self.assertFalse((output / 'cs-de-0-zero.result.json').exists())


if __name__ == '__main__':
    unittest.main()
