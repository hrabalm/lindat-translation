import asyncio
import base64
import json
import tempfile
import threading
import unittest
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from html import escape
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from flask import Flask

from app.main.document import Document, TikalRunner
from app.models.async_llm_runner import close_async_llm_runner
from app.models.llm_errors import LLMBackendError
from app.models.llm_request_state import suppress_few_shot
from app.models.oai_llm_model import OaiLLMModel


class DocumentFewShotTests(unittest.TestCase):
    def setUp(self):
        self.app = Flask(__name__)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.payloads = []
        self.replies = []
        self.recover = False
        self.model = OaiLLMModel(dict(
            source=['cs'], target=['en'], provider='openai',
            model='document-few-shot-test', server='http://llm.test',
            prompt='{sentence}', prompt_terms='TERMS {terms}: {sentence}',
            batch_size=2, retry_attempts=1, max_resplit_depth=0,
            few_shot_attempts={'all': [[{'source': 'example', 'target': 'demo'}], []]},
        ))

        async def handle(transport, request):
            payload = json.loads(request.content)
            self.payloads.append(payload)
            reply = (self.replies.pop(0) if self.replies else
                     payload['messages'][-1]['content'].removeprefix('CUSTOM '))
            if isinstance(reply, Exception):
                raise reply
            if isinstance(reply, int):
                return httpx.Response(reply)
            return httpx.Response(200, json={'choices': [{
                'finish_reason': 'length' if reply == 'length' else 'stop',
                'message': {'content': reply},
            }]})

        def align(url, headers, json):
            return SimpleNamespace(status_code=200, json=lambda: {'alignment': [
                [] if self.recover else [[i, i] for i in range(min(len(source), len(target)))]
                for source, target in zip(json['src_tokens'], json['trg_tokens'])
            ]})

        for boundary in (patch.object(httpx.AsyncHTTPTransport, 'handle_async_request', handle),
                         patch('requests.post', side_effect=align)):
            boundary.start()
            self.addCleanup(boundary.stop)
        self.addCleanup(close_async_llm_runner)

    def translate_document(self, method='with_model', fraus=False, **options):
        source = self.directory / ('input.xml' if fraus else 'input.html')
        source.write_text(
            '<DOC><Questions><RA><ExText>Before </ExText><InputOption>'
            '<SelectOption Correct="true"><ExText>one</ExText></SelectOption>'
            '</InputOption><ExText> after.</ExText></RA></Questions></DOC>'
            if fraus else '<g id="1">Hello</g>\n', encoding='utf-8')

        def tikal(runner, mode, input_path, output_path, src, tgt=None,
                  translation_path=None, expected_output_path=None, **kwargs):
            # Replace only the external Okapi adapter, preserving the real pipeline.
            output = Path(expected_output_path or output_path)
            value = Path(translation_path or input_path).read_text()
            if fraus:
                if input_path.endswith('.fixed'):
                    tree = ET.parse(input_path)
                    payloads = list(tree.iter('ExText'))
                    if mode == '-xm':
                        value = ''.join(escape(escape(
                            (node.text or '') + ''.join(
                                ET.tostring(child, encoding='unicode') for child in node
                            ) + '\n')) for node in payloads)
                    else:
                        for node, line in zip(payloads, value.splitlines()):
                            translated = ET.fromstring(f'<ExText>{line}</ExText>')
                            node.text = translated.text
                            node[:] = list(translated)
                        value = ET.tostring(tree.getroot(), encoding='unicode')
                elif mode == '-xm':
                    value = ''.join(line[3:-4] + '\n' for line in value.splitlines())
                else:
                    value = ''.join(f'<p>{line}</p>\n' for line in value.splitlines())
            output.write_text(value, encoding='utf-8')
            return str(output)

        document = Document(str(source))
        query = '/?debug=true' + ('&fraus=true&xmlTransform=fraus_v2' if fraus else '')
        with self.app.test_request_context(query), patch.object(
                TikalRunner, 'run', tikal), patch(
                'app.main.translate.models.get_model_list',
                return_value=[{'model': self.model, 'src': 'cs', 'tgt': 'en'}]):
            if method == 'with_model':
                document.translate_with_model(self.model, 'cs', 'en', **options)
            else:
                document.translate_from_to('cs', 'en', **options)
            translation = document.get_translation()
            response = document.create_response({})
            payload = response.get_json()
            self.response_payload = payload
            self.response_headers = dict(response.headers)
            if not fraus:
                self.assertEqual(base64.b64decode(payload['output_base64']).decode(), translation)
                self.assertNotIn('X-FRAUS-Recovery', response.headers)
                self.assertEqual(response.headers['X-Billing-Input-NFC-Len'], '20')
            response.close()
        return translation

    def test_document_terms_suppress_examples_without_rendering_terminology(self):
        for method in ('with_model', 'from_to'):
            with self.subTest(method=method):
                self.payloads.clear()
                self.assertEqual(self.translate_document(
                    method, terms=[[['Hello', 'Changed']]]), '<g id="1">Hello</g>\n')
                self.assertEqual([p['messages'] for p in self.payloads], [
                    [{'role': 'user', 'content': 'Hello'}],
                ])

    def test_custom_prompt_and_empty_overrides_preserve_normal_document_behavior(self):
        for method in ('with_model', 'from_to'):
            for options, count, content in (
                ({'custom_prompt': 'CUSTOM {sentence}'}, 1, 'CUSTOM Hello'),
                ({'terms': [[]]}, 1, 'Hello'),
                ({'terms': [], 'custom_prompt': ''}, 3, 'Hello'),
            ):
                with self.subTest(method=method, options=options):
                    self.payloads.clear()
                    self.assertEqual(self.translate_document(method, fraus=False, **options),
                                     '<g id="1">Hello</g>\n')
                    self.assertEqual(len(self.payloads), 1)
                    self.assertEqual(len(self.payloads[0]['messages']), count)
                    self.assertEqual(self.payloads[0]['messages'][-1]['content'], content)

    def test_suppression_does_not_leak_into_following_document_request(self):
        self.translate_document(terms=[[['Hello', 'Ignored']]])
        self.translate_document()
        self.assertEqual([len(p['messages']) for p in self.payloads], [1, 3])

    def test_errors_reset_suppression_and_truncation_cannot_advance_few_shot(self):
        for reply, error in ((422, LLMBackendError), ('length', LLMBackendError),
                             (RuntimeError('transport failed'), RuntimeError)):
            with self.subTest(reply=reply):
                self.payloads.clear()
                self.replies = [reply]
                with self.assertRaises(error):
                    self.translate_document(terms=[[['Hello', 'Ignored']]])
                self.assertEqual([len(p['messages']) for p in self.payloads], [1])
                self.translate_document()
                self.assertEqual([len(p['messages']) for p in self.payloads], [1, 3])

    def test_successful_unrelated_output_remains_accepted_without_retry(self):
        self.replies = ['Unrelated']
        self.assertEqual(self.translate_document(terms=[[]]), '<g id="1">Unrelated</g>\n')
        self.assertEqual(len(self.payloads), 1)

    def test_concurrent_no_request_context_calls_keep_suppression_isolated(self):
        entered = threading.Event()
        release = threading.Event()

        async def handle(transport, request):
            payload = json.loads(request.content)
            self.payloads.append(payload)
            content = payload['messages'][-1]['content']
            if content == 'suppressed':
                entered.set()
                while not release.is_set():
                    await asyncio.sleep(0.001)
            return httpx.Response(200, json={'choices': [{
                'finish_reason': 'stop', 'message': {'content': content},
            }]})

        def suppressed_call():
            with self.app.app_context(), suppress_few_shot():
                with suppress_few_shot(False):
                    return self.model.translate('suppressed', src='cs', tgt='en')

        with patch.object(httpx.AsyncHTTPTransport, 'handle_async_request', handle):
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(suppressed_call)
                try:
                    self.assertTrue(entered.wait(5))
                    with self.app.app_context():
                        self.assertEqual(self.model.translate('unrelated', src='cs', tgt='en'),
                                         ['unrelated'])
                finally:
                    release.set()
                self.assertEqual(future.result(timeout=5), ['suppressed'])
        self.assertEqual([len(p['messages']) for p in self.payloads], [1, 3])

    def test_fraus_context_and_option_recovery_keep_bypass_and_metadata(self):
        for method in ('with_model', 'from_to'):
            for options in ({'terms': [[['one', 'Do not render']]]},
                            {'custom_prompt': 'CUSTOM {sentence}'},
                            {'terms': [[['one', 'Do not render']]],
                             'custom_prompt': 'CUSTOM {sentence}'}):
                with self.subTest(method=method, options=options):
                    self.payloads.clear()
                    self.recover = True
                    self.replies = ['Lost', 'uno', 'Start __BLANK__ end.']
                    self.translate_document(method, fraus=True, **options)
                    messages = [p['messages'] for p in self.payloads]
                    self.assertEqual([len(m) for m in messages], [1, 1, 1])
                    prefix = 'CUSTOM ' if options.get('custom_prompt') else ''
                    self.assertTrue(messages[0][0]['content'].startswith(prefix + 'Before '))
                    self.assertEqual(messages[1:], [
                        [{'role': 'user', 'content': prefix + 'one'}],
                        [{'role': 'user', 'content': prefix + 'Before __BLANK__ after.'}],
                    ])
                    payload = self.response_payload
                    root = ET.fromstring(base64.b64decode(payload['output_base64']))
                    self.assertEqual([(node.text or '').strip() for node in root.iter('ExText')],
                                     ['Start', 'uno', 'end.'])
                    self.assertEqual(self.response_headers['X-FRAUS-Recovery'], 'recovered')
                    self.assertEqual(payload['trace']['fraus_recovery'], payload['fraus_recovery'])
                    self.assertIn('option_retry', [r['type'] for r in payload['fraus_recovery']])
                    self.assertIn('context_retry', [r['type'] for r in payload['fraus_recovery']])
