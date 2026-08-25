import asyncio
import json
import threading
import unittest

from flask import Flask
import httpx
from unittest.mock import patch

from app.factory import create_app
from app.main.api.translation.endpoints.MyAbstractResource import MyAbstractResource
from app.model_settings import models as configured_models
from app.models.async_llm_runner import AsyncLLMRunner, close_async_llm_runner
from app.models.llm_errors import (
    LLMBackendError,
    LLMBackendRejected,
    LLMBackendTimeout,
    LLMBackendUnavailable,
)
from app.models.oai_llm_model import OaiLLMModel


class OaiLLMModelTests(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        self.context = self.app.app_context()
        self.context.push()

    def tearDown(self):
        close_async_llm_runner()
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

    def test_exhausted_failure_cancels_sibling_requests(self):
        cancelled = 0

        async def handler(request):
            nonlocal cancelled
            payload = json.loads(request.content)
            prompt = payload['messages'][0]['content']
            if prompt.endswith('fail '):
                return httpx.Response(503)
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                cancelled += 1
                raise
            return httpx.Response(200, json={
                'choices': [{'message': {'content': 'late'}}],
            })

        model = self.make_model(
            handler,
            batch_size=4,
            retry_attempts=1,
        )
        with self.assertRaises(LLMBackendUnavailable):
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
            temperature=0.2,
        )
        result = self.translate(model, ['source'])
        payload = json.loads(captured.content)

        self.assertEqual(captured.url.path, '/v1/chat/completions')
        self.assertEqual(payload['model'], model.model)
        self.assertEqual(payload['max_tokens'], 123)
        self.assertEqual(payload['temperature'], 0.2)
        self.assertEqual(result, ['line one line two'])

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
        app = create_app()
        app.logger.disabled = True
        model = configured_models.get_model('EuroLLM-9B-Instruct')
        with patch.object(model, '_server', 'http://127.0.0.1:9'), patch.object(
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
