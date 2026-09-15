import asyncio
import json
import unittest
from unittest.mock import patch

import httpx
from flask import Flask

from app.models.async_llm_runner import close_async_llm_runner
from app.models.llm_errors import LLMBackendError, LLMBackendTimeout
from app.models.oai_llm_model import OaiLLMModel


class FewShotTests(unittest.TestCase):
    def setUp(self):
        self.context = Flask(__name__).app_context()
        self.context.push()
        self.payloads = []
        self.replies = []

        async def handle(transport, request):
            self.payloads.append(json.loads(request.content))
            reply = self.replies.pop(0) if self.replies else 'translated'
            if isinstance(reply, tuple):
                delay, reply = reply
                await asyncio.sleep(delay)
            if isinstance(reply, int):
                return httpx.Response(reply)
            if isinstance(reply, httpx.Response):
                return reply
            if reply == 'timeout':
                await asyncio.sleep(1)
            return httpx.Response(200, json={'choices': [{
                'finish_reason': 'length' if reply == 'length' else 'stop',
                'message': {'content': reply},
            }]})

        self.http = patch.object(httpx.AsyncHTTPTransport, 'handle_async_request', handle)
        self.http.start()

    def tearDown(self):
        close_async_llm_runner()
        self.http.stop()
        self.context.pop()

    def model(self, **overrides):
        cfg = dict(source=['cs'], target=['de'], provider='openai',
                   model='few-shot-test', server='http://llm.test', batch_size=1,
                   prompt='{src}>{tgt}: {sentence}', retry_attempts=2,
                   retry_wait_min=0, retry_wait_max=0)
        cfg.update(overrides)
        return OaiLLMModel(cfg)

    def send(self, model, **kwargs):
        return model.send_sentences_to_backend(['real'], src='cs', tgt='de', **kwargs)

    def test_ordered_exchanges_specific_replaces_all_and_copies_config(self):
        attempts = {'all': [[{'source': 'unused', 'target': 'unused'}]],
                    'cs-de': [[{'source': 'one', 'target': '{literal}\n'},
                               {'source': 'two', 'target': 'second'}]]}
        model = self.model(few_shot_attempts=attempts)
        attempts['cs-de'][0][0]['source'] = 'mutated'
        self.assertEqual(self.send(model), ['translated'])
        self.assertEqual(self.payloads[0]['messages'], [
            {'role': 'user', 'content': 'Czech>German: one'},
            {'role': 'assistant', 'content': '{literal}\n'},
            {'role': 'user', 'content': 'Czech>German: two'},
            {'role': 'assistant', 'content': 'second'},
            {'role': 'user', 'content': 'Czech>German: real'},
        ])

    def test_invalid_config_fails_construction(self):
        invalid = [None, [], 'all', {'cs-de': []}, {'cs-de': {}},
                   {'cs-de': [None]}, {'cs-de': [[None]]},
                   {'cs-de': [[{'source': 'x'}]]},
                   {'cs-de': [[{'source': 'x', 'target': 'y', 'extra': 'z'}]]},
                   {'cs-de': [[{'source': ' ', 'target': 'y'}]]},
                   {'cs-de': [[{'source': 1, 'target': 'y'}]]},
                   {'cs-de': [[{'source': 'x', 'target': '\t'}]]},
                   {'cs-de': [[{'source': 'x', 'target': 1}]]},
                   {'CS-de': [[]]}, {'cs_de': [[]]}, {'de-cs': [[]]},
                   {'cs-en': [[]]}, {1: [[]]}, {'all': ()}]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.model(few_shot_attempts=value)

    def test_reverse_capability_and_zero_shot_override(self):
        model = self.model(target_to_source=True, few_shot_attempts={
            'all': [[{'source': 'one', 'target': 'eins'}]], 'de-cs': [[]]})
        model.send_sentences_to_backend(['real'], src='de', tgt='cs')
        self.assertEqual(self.payloads[0]['messages'], [
            {'role': 'user', 'content': 'German>Czech: real'}])
        self.send(model)
        self.assertEqual(len(self.payloads[1]['messages']), 3)

    def test_truncation_advances_stages_before_resplit(self):
        model = self.model(few_shot_attempts={'all': [[], [
            {'source': 'example', 'target': 'literal'}], []]})
        self.replies = ['length', 'length', 'final']
        with self.assertLogs('app.models.oai_llm_model', level='INFO') as logs:
            self.assertEqual(model.translate('one two', src='cs', tgt='de'), ['final'])
        events = [getattr(r, 'few_shot') for r in logs.records if hasattr(r, 'few_shot')]
        self.assertEqual(events[-1]['stage_index'], 2)
        self.assertEqual(events[-1]['prior_truncations'], 2)
        self.assertEqual(events[-1]['outcome'], 'success')
        self.assertEqual([len(p['messages']) for p in self.payloads], [1, 3, 1])
        self.assertEqual([p['messages'][-1]['content'] for p in self.payloads],
                         ['Czech>German: one two'] * 3)

    def test_exhausted_parent_resplits_and_children_restart_stage_zero(self):
        model = self.model(few_shot_attempts={'all': [[
            {'source': 'example', 'target': 'literal'}], []]})
        self.replies = ['length', 'length', 'first', 'second']
        self.assertEqual(model.translate('one two', src='cs', tgt='de'), ['first second'])
        self.assertEqual([len(p['messages']) for p in self.payloads], [3, 1, 3, 3])
        self.assertEqual([p['messages'][-1]['content'] for p in self.payloads], [
            'Czech>German: one two', 'Czech>German: one two',
            'Czech>German: one', 'Czech>German: two'])

    def test_custom_prompt_and_terms_bypass_entire_sequence(self):
        for options, expected in [
            ({'custom_prompt': 'custom {sentence}'}, 'custom real'),
            ({'custom_prompt': 'custom {sentence}', 'allow': False}, 'Czech>German: real'),
            ({'custom_prompt': ' '}, ' '),
            ({'terms': [[('a', 'b')]]}, 'terms a -> b\n: real'),
            ({'terms': [[]]}, 'Czech>German: real'),
            ({'custom_prompt': 'custom {sentence}', 'terms': [[('a', 'b')]]}, 'custom real'),
        ]:
            with self.subTest(options=options):
                self.payloads.clear()
                model = self.model(
                    allow_custom_prompt=options.pop('allow', True),
                    prompt_terms='terms {terms}: {sentence}', max_resplit_depth=0,
                    few_shot_attempts={'all': [[{'source': 'one', 'target': 'eins'}], []]})
                self.replies = ['length']
                with self.assertRaises(LLMBackendError):
                    self.send(model, **options)
                self.assertEqual(len(self.payloads), 1)
                self.assertEqual(self.payloads[0]['messages'], [
                    {'role': 'user', 'content': expected}])

    def test_empty_overrides_do_not_bypass_and_nontruncation_does_not_advance(self):
        model = self.model(few_shot_attempts={'all': [[], [
            {'source': 'one', 'target': 'eins'}]]})
        self.replies = ['length', 503, 'ok']
        self.assertEqual(self.send(model, custom_prompt='', terms=[]), ['ok'])
        self.assertEqual([len(p['messages']) for p in self.payloads], [1, 3, 3])
        self.assertEqual(self.payloads[1], self.payloads[2])
        for replies in [[422], [503, 503], ['   ']]:
            with self.subTest(replies=replies):
                self.payloads.clear()
                count = len(replies)
                self.replies = list(replies)
                with self.assertRaises(LLMBackendError):
                    self.send(model)
                self.assertEqual(len(self.payloads), count)
                self.assertTrue(all(len(p['messages']) == 1 for p in self.payloads))

    def test_stage_logs_preserve_parent_history_and_hide_content(self):
        model = self.model(few_shot_attempts={'all': [[
            {'source': 'private-example', 'target': 'private-target'}], []]})
        self.replies = ['length', 'length', 'first', 'second']
        with self.assertLogs('app.models.oai_llm_model', level='INFO') as logs:
            model.translate('one two', src='cs', tgt='de')
        events = [getattr(r, 'few_shot') for r in logs.records if hasattr(r, 'few_shot')]
        self.assertEqual([(e['segment'], e['stage_index'], e['outcome'],
                           e['prior_truncations'], e['depth']) for e in events], [
            ('0', 0, 'truncated', 0, 0), ('0', 1, 'truncated', 1, 0),
            ('0.0', 0, 'success', 0, 1), ('0.1', 0, 'success', 0, 1)])
        self.assertEqual(events[1]['example_fingerprint'],
                         '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945')
        self.assertEqual(events[0]['example_fingerprint'], events[2]['example_fingerprint'])
        self.assertNotEqual(events[0]['example_fingerprint'], events[1]['example_fingerprint'])
        self.assertEqual(len({e['batch_id'] for e in events}), 1)
        for private in ['private-example', 'private-target', 'one two']:
            self.assertNotIn(private, '\n'.join(logs.output) + repr(events))

    def test_failed_stage_after_truncation_keeps_safe_provenance(self):
        for replies, error_type in [
            ([httpx.Response(422, text='private-response-body')], 'LLMBackendRejected'),
            ([503, 503], 'LLMBackendUnavailable'),
            ([httpx.Response(200, text='private-response-body')], 'LLMBackendError'),
        ]:
            with self.subTest(error_type=error_type):
                self.payloads.clear()
                model = self.model(few_shot_attempts={'all': [[
                    {'source': 'private-example', 'target': 'private-target'}], [], []]})
                self.replies = ['length', *replies]
                with self.assertLogs('app.models.oai_llm_model', level='INFO') as logs:
                    with self.assertRaises(LLMBackendError) as raised:
                        model.send_sentences_to_backend(
                            ['private-source'], src='cs', tgt='de')
                self.assertEqual(type(raised.exception).__name__, error_type)
                self.assertEqual(len(self.payloads), 1 + len(replies))
                events = [getattr(r, 'few_shot') for r in logs.records
                          if hasattr(r, 'few_shot')]
                self.assertEqual([(e['stage_index'], e['outcome']) for e in events],
                                 [(0, 'truncated'), (1, 'failed')])
                self.assertEqual(events[1], {
                    'batch_id': events[0]['batch_id'], 'segment': '0', 'depth': 0,
                    'stage_index': 1, 'prior_truncations': 1, 'outcome': 'failed',
                    'error': error_type,
                    'example_fingerprint':
                        '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945',
                })
                for private in ['private-source', 'private-example',
                                'private-target', 'private-response-body']:
                    self.assertNotIn(private, '\n'.join(logs.output) + repr(events))

    def test_deadline_does_not_reset_between_stages_and_keeps_truncation_logs(self):
        model = self.model(overall_timeout=0.15, few_shot_attempts={'all': [[
            {'source': 'private-example', 'target': 'private-target'}], [], []]})
        self.replies = [(0.09, 'length'), (0.09, 'would-succeed-with-reset')]
        with self.assertLogs('app.models.oai_llm_model', level='INFO') as logs:
            with self.assertRaises(LLMBackendTimeout):
                model.send_sentences_to_backend(['private-source'], src='cs', tgt='de')
        self.assertEqual(len(self.payloads), 2)
        events = [getattr(r, 'few_shot') for r in logs.records if hasattr(r, 'few_shot')]
        self.assertEqual([(e['stage_index'], e['outcome']) for e in events],
                         [(0, 'truncated'), (1, 'cancelled')])
        self.assertEqual(events[1], {
            'batch_id': events[0]['batch_id'], 'segment': '0', 'depth': 0,
            'stage_index': 1, 'prior_truncations': 1, 'outcome': 'cancelled',
            'error': 'CancelledError',
            'example_fingerprint':
                '4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945',
        })
        for private in ['private-source', 'private-example', 'private-target',
                        'would-succeed-with-reset']:
            self.assertNotIn(private, '\n'.join(logs.output) + repr(events))

    def test_parent_truncations_remain_logged_when_child_times_out(self):
        model = self.model(overall_timeout=0.1, few_shot_attempts={'all': [[], []]})
        self.replies = ['length', 'length', 'timeout', 'timeout']
        with self.assertLogs('app.models.oai_llm_model', level='INFO') as logs:
            with self.assertRaises(LLMBackendTimeout):
                model.translate('one two', src='cs', tgt='de')
        events = [getattr(r, 'few_shot') for r in logs.records if hasattr(r, 'few_shot')]
        self.assertEqual([(e['segment'], e['stage_index'], e['outcome'])
                          for e in events[:2]],
                         [('0', 0, 'truncated'), ('0', 1, 'truncated')])
        self.assertCountEqual([
            (e['segment'], e['stage_index'], e['outcome'], e['prior_truncations'],
             e['error']) for e in events[2:]], [
            ('0.0', 0, 'cancelled', 0, 'CancelledError'),
            ('0.1', 0, 'cancelled', 0, 'CancelledError')])

    def test_unsplittable_exhaustion_preserves_source_and_default_is_zero_shot(self):
        for config in [{}, {'few_shot_attempts': {}},
                       {'few_shot_attempts': {'all': [[], []]}}]:
            with self.subTest(config=config):
                self.payloads.clear()
                stages = 2 if config.get('few_shot_attempts') else 1
                self.replies = ['ok'] + ['length'] * stages
                model = self.model(**config)
                self.assertEqual(model.send_sentences_to_backend(
                    ['first', '<b>original</b>'], src='cs', tgt='de'),
                    ['ok', '<b>original</b>'])
                self.assertEqual(len(self.payloads), stages + 1)

    def test_resplit_bypass_children_retain_original_terminology_index(self):
        model = self.model(prompt_terms='{terms}{sentence}', few_shot_attempts={
            'all': [[{'source': 'example', 'target': 'literal'}], []]})
        self.replies = ['ok', 'length', 'first', 'second']
        self.assertEqual(model.send_sentences_to_backend(
            ['first', 'one two'], src='cs', tgt='de',
            terms=[[('a', 'b')], [('x', 'y')]]), ['ok', 'first second'])
        self.assertEqual([p['messages'] for p in self.payloads], [
            [{'role': 'user', 'content': 'a -> b\nfirst'}],
            [{'role': 'user', 'content': 'x -> y\none two'}],
            [{'role': 'user', 'content': 'x -> y\none'}],
            [{'role': 'user', 'content': 'x -> y\ntwo'}]])

    def test_success_acceptance_is_unchanged_and_stops_the_sequence(self):
        model = self.model(few_shot_attempts={'all': [[], []]})
        for output in ['real', 'unrelated output', '<broken markup']:
            with self.subTest(output=output):
                self.payloads.clear()
                self.replies = [output]
                self.assertEqual(self.send(model), [output])
                self.assertEqual(len(self.payloads), 1)

    def test_stages_keep_shared_concurrency_limit_and_result_order(self):
        active = maximum = 0

        async def handle(transport, request):
            nonlocal active, maximum
            payload = json.loads(request.content)
            self.payloads.append(payload)
            active += 1
            maximum = max(maximum, active)
            try:
                await asyncio.sleep(0.01)
            finally:
                active -= 1
            return httpx.Response(200, json={'choices': [{
                'finish_reason': 'length' if len(payload['messages']) == 1 else 'stop',
                'message': {'content': payload['messages'][-1]['content']},
            }]})

        model = self.model(batch_size=2, prompt='{sentence}', few_shot_attempts={
            'all': [[], [{'source': 'example', 'target': 'literal'}]]})
        with patch.object(httpx.AsyncHTTPTransport, 'handle_async_request', handle):
            self.assertEqual(model.send_sentences_to_backend(
                ['a', 'b', 'c'], src='cs', tgt='de'), ['a', 'b', 'c'])
        self.assertEqual(maximum, 2)
        for source in ['a', 'b', 'c']:
            self.assertEqual([len(p['messages']) for p in self.payloads
                              if p['messages'][-1]['content'] == source], [1, 3])
