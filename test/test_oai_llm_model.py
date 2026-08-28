import asyncio
import json
import logging
import threading
import unittest
import xml.etree.ElementTree as ET

from flask import Flask
import httpx
from unittest.mock import patch

from app.factory import create_app
from app.main.api.translation.endpoints.MyAbstractResource import MyAbstractResource
from app.main.translatable import Translatable
from app.main.translate import translate_from_to
from app.model_settings import models as configured_models
from app.models.async_llm_runner import AsyncLLMRunner, close_async_llm_runner
from app.models.llm_errors import (
    LLMBackendError,
    LLMBackendRejected,
    LLMBackendTimeout,
    LLMBackendUnavailable,
)
from app.models.oai_llm_model import OaiLLMModel
from app.models.llm_request_state import (
    get_request_llm_state,
    initialize_request_llm_state,
)
from app.text_utils import extract_text


class OaiLLMModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.api_app = create_app()

    def setUp(self):
        self.app = Flask(__name__)
        self.context = self.app.app_context()
        self.context.push()
        self.llm_logger = logging.getLogger('app.models.oai_llm_model')
        self.logger_was_disabled = self.llm_logger.disabled
        self.llm_logger.disabled = True

    def tearDown(self):
        close_async_llm_runner()
        self.llm_logger.disabled = self.logger_was_disabled
        self.context.pop()

    def make_model(self, handler, **overrides):
        cfg = {
            'source': ['en'],
            'target': ['cs'],
            'provider': 'openai',
            'model': self.id(),
            'model_framework': 'oai_llm',
            'server': 'http://llm.test',
            'batch_size': 2,
            'retry_attempts': 3,
            'retry_wait_min': 0,
            'retry_wait_max': 0,
        }
        cfg.update(overrides)
        model = OaiLLMModel(cfg)
        transport = httpx.MockTransport(handler)
        model._create_client = lambda server: httpx.AsyncClient(
            base_url=f'{server}/v1',
            transport=transport,
        )
        return model

    def translate(self, model, sentences):
        return model.send_sentences_to_backend(sentences, src='en', tgt='cs')

    def test_translates_concurrently_up_to_batch_size_and_preserves_order(self):
        active = 0
        maximum = 0

        async def handler(request):
            nonlocal active, maximum
            payload = json.loads(request.content)
            value = payload['messages'][0]['content'].rsplit(' ', 1)[-1]
            active += 1
            maximum = max(maximum, active)
            try:
                await asyncio.sleep((5 - int(value)) * 0.005)
            finally:
                active -= 1
            return httpx.Response(200, json={
                'choices': [{'message': {'content': f'target-{value}'}}],
            })

        model = self.make_model(
            handler,
            prompt='{sentence}',
            batch_size=2,
        )
        result = self.translate(model, ['0', '1', '2', '3', '4'])

        self.assertEqual(result, [
            'target-0', 'target-1', 'target-2', 'target-3', 'target-4'
        ])
        self.assertEqual(maximum, 2)

    def test_retries_transient_status_and_honors_success(self):
        calls = 0

        async def handler(request):
            nonlocal calls
            calls += 1
            if calls < 3:
                return httpx.Response(503, headers={'Retry-After': '0'})
            return httpx.Response(200, json={
                'choices': [{'message': {'content': 'translated'}}],
            })

        model = self.make_model(handler)
        self.assertEqual(self.translate(model, ['source']), ['translated'])
        self.assertEqual(calls, 3)

    def test_retries_transient_transport_failure(self):
        calls = 0

        async def handler(request):
            nonlocal calls
            calls += 1
            if calls < 3:
                raise httpx.ReadTimeout('timed out', request=request)
            return httpx.Response(200, json={
                'choices': [{'message': {'content': 'translated'}}],
            })

        model = self.make_model(handler)
        self.assertEqual(self.translate(model, ['source']), ['translated'])
        self.assertEqual(calls, 3)

    def test_does_not_retry_permanent_rejection(self):
        calls = 0

        async def handler(request):
            nonlocal calls
            calls += 1
            return httpx.Response(422, json={'error': 'too long'})

        model = self.make_model(handler)
        with self.assertRaises(LLMBackendRejected):
            self.translate(model, ['source'])
        self.assertEqual(calls, 1)

    def test_does_not_retry_invalid_response(self):
        calls = 0

        async def handler(request):
            nonlocal calls
            calls += 1
            return httpx.Response(200, content=b'not-json')

        model = self.make_model(handler)
        with self.assertRaises(LLMBackendError):
            self.translate(model, ['source'])
        self.assertEqual(calls, 1)

    def test_does_not_retry_whitespace_completion(self):
        calls = 0

        async def handler(request):
            nonlocal calls
            calls += 1
            return httpx.Response(200, json={
                'choices': [{'message': {'content': '   '}}],
            })

        model = self.make_model(handler)
        with self.assertRaises(LLMBackendError):
            self.translate(model, ['source'])
        self.assertEqual(calls, 1)

    def test_unsplittable_token_truncated_completion_fails(self):
        calls = 0

        async def handler(request):
            nonlocal calls
            calls += 1
            return httpx.Response(200, json={
                'choices': [{
                    'finish_reason': 'length',
                    'message': {'content': 'truncated'},
                }],
            })

        model = self.make_model(handler)
        with self.assertRaises(LLMBackendError):
            self.translate(model, ['source'])
        self.assertEqual(calls, 1)

    def test_retries_proxy_failure_then_reports_unavailable(self):
        calls = 0

        async def handler(request):
            nonlocal calls
            calls += 1
            raise httpx.ProxyError('proxy unavailable', request=request)

        model = self.make_model(handler)
        with self.assertRaises(LLMBackendUnavailable):
            self.translate(model, ['source'])
        self.assertEqual(calls, 3)

    def test_exhausted_failure_retains_source_and_does_not_cancel_siblings(self):
        async def handler(request):
            payload = json.loads(request.content)
            prompt = payload['messages'][0]['content']
            if prompt == 'fail':
                return httpx.Response(503)
            await asyncio.sleep(0.01)
            return httpx.Response(200, json={
                'choices': [{'message': {'content': f'target-{prompt}'}}],
            })

        model = self.make_model(
            handler,
            batch_size=4,
            retry_attempts=1,
            prompt='{sentence}',
        )
        self.assertEqual(
            self.translate(model, ['fail', 'slow-1', 'slow-2']),
            ['fail', 'target-slow-1', 'target-slow-2'],
        )

    def test_source_fallback_preserves_markup_exactly(self):
        async def handler(request):
            payload = json.loads(request.content)
            prompt = payload['messages'][0]['content']
            if 'bad' in prompt:
                return httpx.Response(422)
            return httpx.Response(200, json={
                'choices': [{'message': {'content': 'translated'}}],
            })

        model = self.make_model(
            handler,
            prompt='{sentence}',
            retry_attempts=1,
        )
        source = '<g id="format">bad</g>'
        self.assertEqual(
            self.translate(model, [source, 'good']),
            [source, 'translated'],
        )

    def test_unexpected_failure_still_cancels_sibling_requests(self):
        cancelled = 0

        async def handler(request):
            nonlocal cancelled
            payload = json.loads(request.content)
            prompt = payload['messages'][0]['content']
            if prompt == 'fail':
                raise RuntimeError('unexpected')
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled += 1
                raise

        model = self.make_model(
            handler,
            batch_size=3,
            retry_attempts=1,
            prompt='{sentence}',
        )
        with self.assertRaisesRegex(RuntimeError, 'unexpected'):
            self.translate(model, ['fail', 'slow-1', 'slow-2'])
        self.assertEqual(cancelled, 2)

    def test_overall_deadline_cancels_batch(self):
        cancelled = 0

        async def handler(request):
            nonlocal cancelled
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled += 1
                raise
            return httpx.Response(200, json={
                'choices': [{'message': {'content': 'late'}}],
            })

        model = self.make_model(handler, overall_timeout=0.01)
        with self.assertRaises(LLMBackendTimeout):
            self.translate(model, ['slow-1', 'slow-2'])
        self.assertEqual(cancelled, 2)

    def test_overall_deadline_retains_completed_segments(self):
        cancelled = 0

        async def handler(request):
            nonlocal cancelled
            payload = json.loads(request.content)
            prompt = payload['messages'][0]['content']
            if prompt == 'fast':
                return httpx.Response(200, json={
                    'choices': [{'message': {'content': 'translated-fast'}}],
                })
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled += 1
                raise

        model = self.make_model(
            handler,
            prompt='{sentence}',
            overall_timeout=0.01,
        )
        self.assertEqual(
            self.translate(model, ['fast', 'slow']),
            ['translated-fast', 'slow'],
        )
        self.assertEqual(cancelled, 1)

    def test_token_truncation_recursively_splits_source(self):
        prompts = []

        async def handler(request):
            payload = json.loads(request.content)
            prompt = payload['messages'][0]['content']
            prompts.append(prompt)
            if prompt == 'one two three four':
                return httpx.Response(200, json={
                    'choices': [{
                        'finish_reason': 'length',
                        'message': {'content': 'truncated'},
                    }],
                })
            return httpx.Response(200, json={
                'choices': [{'message': {'content': prompt.upper()}}],
            })

        model = self.make_model(handler, prompt='{sentence}')
        self.assertEqual(
            self.translate(model, ['one two three four']),
            ['ONE TWO THREE FOUR'],
        )
        self.assertEqual(
            set(prompts),
            {'one two three four', 'one two', 'three', 'four'},
        )

    def test_request_matches_openai_chat_completions_format(self):
        captured = None

        async def handler(request):
            nonlocal captured
            captured = request
            return httpx.Response(200, json={
                'choices': [{'message': {'content': 'line one\nline two'}}],
            })

        model = self.make_model(
            handler,
            token='secret',
            max_completion_tokens=123,
            max_input_tokens=64,
            temperature=0.2,
        )
        result = self.translate(model, ['source'])
        payload = json.loads(captured.content)

        self.assertEqual(captured.url.path, '/v1/chat/completions')
        self.assertEqual(payload['model'], model.model)
        self.assertEqual(payload['max_tokens'], 123)
        self.assertEqual(payload['temperature'], 0.2)
        self.assertEqual(result, ['line one line two'])

    def test_input_chunks_preserve_whitespace_when_reconstructed(self):
        async def handler(request):
            return httpx.Response(200, json={
                'choices': [{'message': {'content': 'unused'}}],
            })

        model = self.make_model(handler, max_input_tokens=2)
        source = 'one two   six ten\n'
        chunks, formatting = model.extract_sentences(source, 'en')

        self.assertEqual(chunks, ['one two', 'six ten'])
        self.assertEqual(model.reconstruct_formatting(chunks, formatting), [
            'one two   ', 'six ten\n'
        ])
        self.assertEqual(
            extract_text(model.reconstruct_formatting(['A', 'B'], formatting)),
            'A   B\n',
        )

    def test_hard_splitting_only_uses_balanced_markup_boundaries(self):
        async def handler(request):
            return httpx.Response(200, json={
                'choices': [{'message': {'content': 'unused'}}],
            })

        model = self.make_model(handler, max_input_tokens=2)
        source = '<g id="format">one</g> two three four'
        chunks, formatting = model.extract_sentences(source, 'en')

        self.assertGreater(len(chunks), 1)
        self.assertEqual(
            extract_text(model.reconstruct_formatting(chunks, formatting)),
            source,
        )
        for chunk in chunks:
            ET.fromstring(f'<root>{chunk}</root>')

    def test_html_void_tags_do_not_block_splitting(self):
        async def handler(request):
            return httpx.Response(200, json={
                'choices': [{'message': {'content': 'unused'}}],
            })

        model = self.make_model(handler, max_input_tokens=2)
        source = 'one<br> two three four'
        chunks, formatting = model.extract_sentences(source, 'en')

        self.assertGreater(len(chunks), 1)
        self.assertEqual(
            extract_text(model.reconstruct_formatting(chunks, formatting)),
            source,
        )

    def test_failed_final_model_hop_is_not_masked_by_earlier_success(self):
        async def first_handler(request):
            return httpx.Response(200, json={
                'choices': [{'message': {'content': 'intermediate'}}],
            })

        async def second_handler(request):
            return httpx.Response(503)

        first = self.make_model(first_handler, prompt='{sentence}')
        second = self.make_model(
            second_handler,
            prompt='{sentence}',
            retry_attempts=1,
        )
        path = [
            {'model': first, 'src': 'en', 'tgt': 'cs'},
            {'model': second, 'src': 'cs', 'tgt': 'de'},
        ]
        with self.api_app.test_request_context('/'), patch(
                'app.main.translate.models.get_model_list', return_value=path):
            initialize_request_llm_state()
            result = translate_from_to('en', 'de', 'original')

            self.assertEqual(result, ['original'])
            self.assertEqual(get_request_llm_state().successful_segments, 0)
            with self.assertRaises(LLMBackendUnavailable):
                Translatable.finalize_llm_translation()

    def test_llm_input_token_limit_defaults_to_256(self):
        async def handler(request):
            return httpx.Response(200, json={
                'choices': [{'message': {'content': 'unused'}}],
            })

        model = self.make_model(handler)
        del model._batch_size
        self.assertEqual(model.max_input_tokens, 256)
        self.assertEqual(model.max_completion_tokens, 512)
        self.assertEqual(model.batch_size, 256)

    def test_input_limit_must_be_smaller_than_completion_limit(self):
        async def handler(request):
            return httpx.Response(200, json={
                'choices': [{'message': {'content': 'unused'}}],
            })

        with self.assertRaisesRegex(
                ValueError,
                'max_input_tokens must be smaller than max_completion_tokens'):
            self.make_model(
                handler,
                max_input_tokens=512,
                max_completion_tokens=512,
            )

    def test_logs_batch_and_request_metadata_without_content(self):
        async def handler(request):
            return httpx.Response(200, json={
                'choices': [{'message': {'content': 'secret translation'}}],
            })

        model = self.make_model(handler)
        self.llm_logger.disabled = False
        with self.assertLogs(self.llm_logger.name, level='INFO') as captured:
            self.translate(model, ['secret source'])
        output = '\n'.join(captured.output)

        self.assertIn('LLM batch started', output)
        self.assertIn('LLM request started', output)
        self.assertIn('LLM request completed', output)
        self.assertIn('LLM batch completed', output)
        self.assertNotIn('secret source', output)
        self.assertNotIn('secret translation', output)

    def test_runner_shutdown_cancels_active_submission(self):
        runner = AsyncLLMRunner()
        started = threading.Event()
        finished = threading.Event()

        async def operation():
            started.set()
            await asyncio.sleep(60)

        def submit():
            try:
                runner.submit(operation())
            except BaseException:
                pass
            finally:
                finished.set()

        thread = threading.Thread(target=submit)
        thread.start()
        self.assertTrue(started.wait(timeout=2))
        runner.close()
        thread.join(timeout=2)

        self.assertTrue(finished.is_set())
        self.assertFalse(thread.is_alive())

    def test_api_maps_exhausted_backend_failure_to_plain_text_503(self):
        app = self.api_app
        model = configured_models.get_model('EuroLLM-9B-Instruct')
        self.llm_logger.disabled = False
        with self.assertLogs(app.logger.name, level='INFO') as captured, \
                self.assertLogs(self.llm_logger.name, level='INFO') as llm_logs, \
                patch.object(model, '_server', 'http://127.0.0.1:9'), patch.object(
                model, 'retry_wait_min', 0), patch.object(
                model, 'retry_wait_max', 0), patch.object(
                MyAbstractResource, 'log_request'):
            response = app.test_client().post(
                f'/api/v2/models/{model.name}',
                data={'input_text': 'Hello', 'src': 'en', 'tgt': 'cs'},
                headers={'Accept': 'text/plain'},
            )

        self.assertEqual(response.status_code, 503)
        self.assertEqual(
            response.get_data(as_text=True),
            'The translation backend is temporarily unavailable.',
        )
        output = '\n'.join(captured.output)
        self.assertIn('API request started', output)
        self.assertIn('LLM backend failure status=503', output)
        self.assertIn('API request completed', output)
        self.assertIn('status=503', output)
        self.assertIn('LLM segment source fallback', '\n'.join(llm_logs.output))

    def test_api_returns_partial_translation_headers(self):
        async def handler(request):
            payload = json.loads(request.content)
            prompt = payload['messages'][0]['content']
            if 'bad' in prompt:
                return httpx.Response(503)
            return httpx.Response(200, json={
                'choices': [{'message': {'content': 'translated-good'}}],
            })

        app = self.api_app
        model = configured_models.get_model('EuroLLM-9B-Instruct')
        transport = httpx.MockTransport(handler)
        client_factory = lambda server: httpx.AsyncClient(
            base_url=f'{server}/v1', transport=transport
        )
        with patch.object(model, '_create_client', side_effect=client_factory), \
                patch.object(model, 'retry_attempts', 1), patch.object(
                MyAbstractResource, 'log_request'):
            response = app.test_client().post(
                f'/api/v2/models/{model.name}',
                data={'input_text': 'good\nbad', 'src': 'en', 'tgt': 'cs'},
                headers={'Accept': 'text/plain'},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.get_data(as_text=True), 'translated-good\nbad')
        self.assertEqual(response.headers['X-Translation-Partial'], 'true')
        self.assertEqual(
            response.headers['X-Translation-Source-Fallback-Segments'], '1'
        )
        self.assertEqual(response.headers['X-Translation-Total-Segments'], '2')
        self.assertIn('X-Translation-Fallback-Token-Ratio', response.headers)
