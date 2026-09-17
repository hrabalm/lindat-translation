import asyncio
import json
import unittest
from unittest.mock import patch

import httpx
from flask import Flask

from app.models.async_llm_runner import close_async_llm_runner
from app.models.llm_errors import LLMBackendError, LLMCompletionTruncated
from app.models.oai_llm_model import OaiLLMModel


class CompletionBudgetTests(unittest.TestCase):
    def setUp(self):
        self.context = Flask(__name__).app_context()
        self.context.push()
        self.payloads = []
        self.replies = []
        self.active = self.maximum_active = 0

        async def handle(transport, request):
            self.payloads.append(json.loads(request.content))
            reply = self.replies.pop(0) if self.replies else 'translated'
            self.active += 1
            self.maximum_active = max(self.maximum_active, self.active)
            await asyncio.sleep(0.001)
            self.active -= 1
            if isinstance(reply, int):
                return httpx.Response(reply)
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
                   model='budget-test', server='http://llm.test', batch_size=256,
                   prompt='{sentence}', retry_attempts=2,
                   retry_wait_min=0, retry_wait_max=0)
        cfg.update(overrides)
        return OaiLLMModel(cfg)

    def send(self, model, sources=None, **kwargs):
        return model.send_sentences_to_backend(
            sources if sources is not None else ['real'], src='cs', tgt='de', **kwargs)

    def test_empty_object_enables_defaults_on_actual_http_request(self):
        model = self.model(adaptive_completion_budget={})
        self.assertEqual(self.send(model), ['translated'])
        self.assertEqual(self.payloads[0]['max_tokens'], 64)
        self.assertEqual(model.completion_token_limit('real'), 64)
        self.assertEqual(model.max_completion_tokens, 512)

    def test_invalid_opt_in_configuration_is_rejected(self):
        invalid = [None, True, False, 'enabled', [], 1, {'unknown': 1}]
        for key, values in [
            ('min_tokens', [None, True, False, '64', 0, -1, 1.5, float('nan'), float('inf')]),
            ('extra_tokens', [None, True, False, '32', -1, 1.5, float('nan'), float('inf')]),
            ('tokens_per_input_token', [None, True, False, '4', 0, -1, float('nan'),
                                        float('inf'), -float('inf'), [], {}]),
        ]:
            invalid.extend({key: value} for value in values)
        for budget in invalid:
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                self.model(adaptive_completion_budget=budget)
        for cap in [True, False, '512', 0, -1, 1.5, float('nan'), float('inf'), [], {}]:
            with self.subTest(cap=cap), self.assertRaises(ValueError):
                self.model(adaptive_completion_budget={}, max_completion_tokens=cap)

    def test_stage_and_request_diagnostics_report_source_budget_without_content(self):
        model = self.model(adaptive_completion_budget={}, max_completion_tokens=32,
                           few_shot_attempts={'all': [[], [], []]})
        self.replies = ['length', 503, 'length', 'translated']
        with self.assertLogs('app.models.oai_llm_model', level='INFO') as logs:
            self.send(model, ['private-source'])
        events = [getattr(r, 'few_shot') for r in logs.records if hasattr(r, 'few_shot')]
        self.assertEqual(len(events), 3)
        for event in events:
            self.assertEqual(event['estimated_input_tokens'], 5)
            self.assertEqual(event['adaptive_completion_bound'], 64)
            self.assertEqual(event['effective_completion_tokens'], 32)
        started = [r.getMessage() for r in logs.records
                   if r.getMessage().startswith('LLM request started')]
        self.assertEqual(len(started), 4)
        for message in started:
            self.assertIn('estimated_input_tokens=5', message)
            self.assertIn('adaptive_completion_bound=64', message)
            self.assertIn('effective_completion_tokens=32', message)
        self.assertNotIn('private-source', '\n'.join(logs.output))
        self.assertEqual([p['max_tokens'] for p in self.payloads], [32] * 4)
        self.assertEqual(self.payloads[1], self.payloads[2])

    def test_formula_floor_ceiling_global_precedence_and_unlimited_global(self):
        for config, source, expected in [
            ({}, 'real', 512),
            ({'max_completion_tokens': None}, 'real', None),
            ({'max_completion_tokens': 77}, 'real', 77),
            ({'adaptive_completion_budget': {}, 'max_completion_tokens': None}, 'real', 64),
            ({'adaptive_completion_budget': {}, 'max_completion_tokens': 32}, 'real', 32),
            ({'adaptive_completion_budget': {}}, 'a ' * 200, 512),
            ({'adaptive_completion_budget': {}, 'max_completion_tokens': None}, 'a ' * 200, 832),
            ({'adaptive_completion_budget': {'min_tokens': 1, 'tokens_per_input_token': 1.5,
                                             'extra_tokens': 0}}, 'a bb ccc', 5),
            ({'adaptive_completion_budget': {'min_tokens': 100}}, 'real', 100),
            ({'adaptive_completion_budget': {'extra_tokens': 70}}, 'real', 74),
        ]:
            with self.subTest(config=config):
                model = self.model(**config)
                self.send(model, [source])
                if expected is None:
                    self.assertNotIn('max_tokens', self.payloads[-1])
                else:
                    self.assertEqual(self.payloads[-1]['max_tokens'], expected)
                self.assertEqual(model.completion_token_limit(source), expected)

    def test_large_finite_multiplier_respects_global_cap_without_overflow(self):
        model = self.model(adaptive_completion_budget={'tokens_per_input_token': 1e308})
        self.assertEqual(self.send(model, ['a b']), ['translated'])
        self.assertEqual(self.payloads[0]['max_tokens'], 512)
        self.assertEqual(model.completion_token_limit('a b'), 512)

    def test_large_multiplier_without_global_cap_returns_unbounded_integer(self):
        for multiplier in [1e308, 10 ** 400]:
            with self.subTest(multiplier=multiplier):
                model = self.model(adaptive_completion_budget={'tokens_per_input_token': multiplier},
                                   max_completion_tokens=None)
                limit = model.completion_token_limit('a b')
                self.assertIsInstance(limit, int)
                # Both multipliers are integral, so this conversion is lossless.
                self.assertEqual(limit, 2 * int(multiplier) + 32)

    def test_fractional_multiplier_ceils_exact_stored_float_conservatively(self):
        model = self.model(adaptive_completion_budget={
            'tokens_per_input_token': 0.1, 'min_tokens': 1, 'extra_tokens': 0})
        # Binary 0.1 is slightly above one tenth; its exact product with 10 exceeds 1.
        source = 'a b c d e f g h i j'
        self.assertEqual(self.send(model, [source]), ['translated'])
        self.assertEqual(self.payloads[0]['max_tokens'], 2)
        self.assertEqual(model.completion_token_limit(source), 2)

    def test_long_zero_five_ten_shot_prompts_keep_identical_source_only_cap(self):
        example = {'source': 'private-demo ' * 1000, 'target': 'private-target ' * 1000}
        model = self.model(adaptive_completion_budget={},
                           prompt='private-instructions ' * 1000 + '{sentence}',
                           few_shot_attempts={'all': [[], [example] * 5, [example] * 10]})
        self.replies = ['length', 'length', 'translated']
        source = 'a b c d e f g h i j k l'
        self.assertEqual(self.send(model, [source]), ['translated'])
        self.assertEqual([len(p['messages']) for p in self.payloads], [1, 11, 21])
        self.assertEqual([p['max_tokens'] for p in self.payloads], [80, 80, 80])
        self.assertEqual(model.completion_token_limit(source), 80)

    def test_custom_prompt_and_terminology_do_not_inflate_budget_or_enable_stages(self):
        for options in [
            {'custom_prompt': 'private-instructions ' * 1000 + '{sentence}'},
            {'terms': [[('private-term ' * 1000, 'private-target ' * 1000)]]},
        ]:
            with self.subTest(options=list(options)):
                self.payloads.clear()
                model = self.model(adaptive_completion_budget={'min_tokens': 1},
                                   few_shot_attempts={'all': [[], [], []]}, max_resplit_depth=0)
                self.replies = ['length']
                with self.assertRaises(LLMCompletionTruncated):
                    self.send(model, **options)
                self.assertEqual(len(self.payloads), 1)
                self.assertEqual(len(self.payloads[0]['messages']), 1)
                self.assertGreater(len(self.payloads[0]['messages'][0]['content']), 1000)
                self.assertEqual(self.payloads[0]['max_tokens'], 36)

    def test_markup_is_counted_and_preserved_in_source_sent_to_backend(self):
        model = self.model(adaptive_completion_budget={'min_tokens': 1})
        self.assertEqual(self.send(model, ['<b>real</b>']), ['translated'])
        self.assertEqual(self.payloads[0]['messages'], [
            {'role': 'user', 'content': '<b>real</b>'}])
        self.assertEqual(self.payloads[0]['max_tokens'], 44)

    def test_exhausted_parent_recomputes_smaller_child_budgets(self):
        model = self.model(adaptive_completion_budget={}, few_shot_attempts={'all': [[], [], []]})
        self.replies = ['length', 'length', 'length', 'first', 'second']
        with self.assertLogs('app.models.oai_llm_model', level='INFO') as logs:
            self.assertEqual(model.translate('a b c d e f g h i j k l m n o p', src='cs', tgt='de'),
                             ['first second'])
        self.assertEqual([p['max_tokens'] for p in self.payloads], [96, 96, 96, 64, 64])
        self.assertEqual([p['messages'][-1]['content'] for p in self.payloads[3:]],
                         ['a b c d e f g h', 'i j k l m n o p'])
        events = [getattr(r, 'few_shot') for r in logs.records if hasattr(r, 'few_shot')]
        self.assertEqual([(e['segment'], e['stage_index'], e['estimated_input_tokens'],
                           e['adaptive_completion_bound'], e['effective_completion_tokens'])
                          for e in events], [
            ('0', 0, 16, 96, 96), ('0', 1, 16, 96, 96), ('0', 2, 16, 96, 96),
            ('0.0', 0, 8, 64, 64), ('0.1', 0, 8, 64, 64)])

    def test_256_concurrent_segments_keep_individual_limits_and_global_unchanged(self):
        model = self.model(adaptive_completion_budget={'min_tokens': 1}, max_completion_tokens=2048)
        sources = [' '.join(['a'] * count) for count in range(1, 257)]
        self.assertEqual(self.send(model, sources), ['translated'] * 256)
        self.assertEqual(self.maximum_active, 256)
        self.assertEqual([p['max_tokens'] for p in self.payloads], list(range(36, 1057, 4)))
        self.assertEqual([p['messages'][-1]['content'] for p in self.payloads], sources)
        self.assertEqual(model.max_completion_tokens, 2048)

    def test_exhausted_resplit_children_still_fail_strictly(self):
        model = self.model(adaptive_completion_budget={'min_tokens': 1, 'extra_tokens': 0},
                           few_shot_attempts={'all': [[], [], []]})
        self.replies = ['length'] * 9
        with self.assertRaises(LLMCompletionTruncated):
            model.translate('one two', src='cs', tgt='de')
        self.assertEqual([p['max_tokens'] for p in self.payloads], [8] * 3 + [4] * 6)

    def test_empty_and_truncated_completions_remain_failures_without_linguistic_filtering(self):
        model = self.model(adaptive_completion_budget={}, few_shot_attempts={'all': [[], [], []]})
        for replies, error in [(['   '], LLMBackendError),
                               (['length'] * 3, LLMCompletionTruncated)]:
            with self.subTest(replies=replies):
                self.payloads.clear()
                self.replies = list(replies)
                with self.assertRaises(error):
                    model.translate('real', src='cs', tgt='de')
                self.assertEqual(len(self.payloads), len(replies))
                self.assertTrue(all(p['max_tokens'] == 64 for p in self.payloads))
        for output in ['real', 'unrelated output', '<broken markup']:
            with self.subTest(output=output):
                self.payloads.clear()
                self.replies = [output]
                self.assertEqual(self.send(model), [output])
                self.assertEqual(len(self.payloads), 1)
