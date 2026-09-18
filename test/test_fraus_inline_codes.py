"""Offline Document/transform regressions for Okapi inline-code serialization."""

import os
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import tempfile
from typing import cast
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from flask import Flask

from app.main.document import Document, FrausTranslationError, FrausV2XmlTransform
from test.test_fraus_tag_reinsertion import word_alignment


@unittest.skipUnless(os.environ.get('RUN_FRAUS_OKAPI') == '1',
                     'Set RUN_FRAUS_OKAPI=1 for real offline Okapi')
class InlineCodeDocumentTests(unittest.TestCase):
    def roundtrip(self, source, force_recovery=False):
        tikal = Path(os.environ.get('TIKAL_PATH', '/tmp/opencode/okapi/'))
        if not (tikal / 'tikal.sh').is_file():
            self.skipTest(f'Okapi not found: {tikal}')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.xml'
            path.write_text(source, encoding='utf-8')
            document = Document(str(path))
            with Flask(__name__).test_request_context('/?debug=true', method='POST', data={
                    'fraus': 'true', 'xmlTransform': 'fraus_v2',
                    'forceSentenceLevel': 'true', 'maxSegmentTokens': '100',
            }), patch('app.main.document.TIKAL_PATH', str(tikal.resolve()) + os.sep), patch(
                    'app.main.document.translate_from_to', side_effect=lambda src, tgt, text, **kwargs:
                    ([text], [text.replace('word', 'house')])) as backend, patch(
                    'app.main.document.LindatAligner.align',
                    side_effect=AssertionError('forced alignment failure') if force_recovery
                    else word_alignment), patch(
                    'socket.socket.connect', side_effect=AssertionError('Network forbidden')), patch(
                    'socket.create_connection', side_effect=AssertionError('Network forbidden')):
                with redirect_stderr(StringIO()):
                    document.translate_from_to('cs', 'en')
            restored = ET.parse(document.get_translated_path('en')).getroot()
            return restored, document, backend.call_count

    def test_minimal_empty_math_pair_roundtrips_without_model_or_recovery(self):
        restored, document, calls = self.roundtrip(
            '<DOC><Equation>&lt;mtd&gt;&lt;mrow&gt;&lt;/mrow&gt;&lt;/mtd&gt;'
            '&lt;mfrac&gt; &lt;/mfrac&gt;</Equation></DOC>')
        self.assertEqual(restored.findtext('Equation'),
                         '<mtd><mrow></mrow></mtd><mfrac> </mfrac>')
        self.assertEqual(calls, 0)
        self.assertNotIn('fraus_recovery', document.debug_trace)
        self.assertIn('<g id="2"></g>', document.get_translation())


    def test_native_empty_elements_and_translated_prose_are_preserved(self):
        restored, document, calls = self.roundtrip(
            '<DOC><Image Id="image" Src="a.png"/>'
            '<ExText>&lt;b&gt;word&lt;/b&gt;&lt;br /&gt;</ExText>'
            '<Equation>&lt;math&gt;&lt;mrow /&gt;&lt;mi&gt;x&lt;/mi&gt;'
            '&lt;mo&gt;+&lt;/mo&gt;&lt;/math&gt;</Equation></DOC>')
        image = restored.find('Image')
        assert image is not None
        self.assertEqual(image.attrib, {'Id': 'image', 'Src': 'a.png'})
        self.assertEqual(len(image), 0)
        self.assertIsNone(image.text)
        self.assertEqual(restored.findtext('ExText'), '<b>house </b><br />')
        self.assertEqual(restored.findtext('Equation'),
                         '<math><mrow /><mi>x</mi><mo>+</mo></math>')
        self.assertEqual(calls, 1)
        self.assertNotIn('fraus_recovery', document.debug_trace)
        self.assertIn('<x ', document.get_translation())
        self.assertNotIn('</x>', document.get_translation())

    def test_alignment_recovery_retains_pairs_and_standalone_codes(self):
        restored, document, calls = self.roundtrip(
            '<DOC><ExText>&lt;b&gt;&lt;i&gt;&lt;/i&gt;&lt;/b&gt;word&lt;br /&gt;</ExText>'
            '</DOC>', force_recovery=True)
        self.assertEqual(restored.findtext('ExText'), '<b><i></i></b>house<br />')
        self.assertGreater(calls, 0)
        recovery = cast(list[dict[str, object]], document.debug_trace['fraus_recovery'])
        self.assertTrue(any(item['strategy'] == 'plain_text_nodes'
                            for item in recovery))
        self.assertNotRegex(document.get_translation(), r'<g\b[^>]*\s/>')
        self.assertNotIn('</x>', document.get_translation())


class InlineCodeTransformTests(unittest.TestCase):
    def test_accepted_context_fallback_retains_empty_pairs_and_literals(self):
        transform = FrausV2XmlTransform(language='en')
        transform.variant_sequence = [['a']]
        source = ('<g id="empty" title="a &gt; b"><g id="nested"></g></g>'
                  'word &lt;g id="literal" /&gt; &amp; '
                  '<x id="standalone"/><g id="fraus-option-a">one</g>\n')
        target = source.replace('word', 'house')
        result = transform.fallback(source, target, lambda text: self.fail('Unexpected retry'))
        self.assertIn('<g id="nested"></g>', result)
        self.assertIn('title="a &gt; b"', result)
        self.assertIn('house &lt;g id="literal" /&gt; &amp; ', result)
        self.assertIn('<x id="standalone" />', result)
        self.assertNotIn('</x>', result)
        self.assertEqual(transform.fallback_diagnostics, [])

    def test_context_retry_preserves_empty_pairs_and_translates_visible_text(self):
        transform = FrausV2XmlTransform(language='en')
        transform.variant_sequence = [['a']]
        source = ('<g id="empty"></g>word<x id="standalone"/>'
                  '<g id="fraus-option-a">one</g> ending\n')
        calls = []

        def translate(text):
            calls.append(text)
            return text.replace('word', 'house').replace('one', 'two').replace('ending', 'finish')

        result = transform.fallback(source, '\n', translate)
        self.assertEqual(result, '<g id="empty"></g>house<x id="standalone" />'
                         '<g id="fraus-option-a">two</g> finish\n')
        self.assertTrue(any('__BLANK__' in call for call in calls))
        self.assertTrue(any(item.get('accepted') for item in transform.fallback_diagnostics))

    def test_empty_codes_do_not_allow_lost_visible_context(self):
        transform = FrausV2XmlTransform(language='en')
        transform.variant_sequence = [['a']]
        source = '<g id="empty"></g>word<g id="fraus-option-a">one</g>\n'
        target = '<g id="empty"></g><g id="fraus-option-a">two</g>\n'
        with self.assertRaisesRegex(FrausTranslationError, 'no usable translation'):
            transform.fallback(source, target, lambda text: '')

    def test_unmapped_context_restores_inert_formatting_placeholders(self):
        transform = FrausV2XmlTransform(language='en')
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.xml'
            prepared = Path(directory) / 'prepared.xml'
            output = Path(directory) / 'output.xml'
            source.write_text(
                '<DOC><Questions><RA><ExText>&lt;b&gt;word </ExText>'
                '<InputOption><SelectOption Correct="true"><ExText>one</ExText>'
                '</SelectOption></InputOption><ExText> ending&lt;/b&gt;</ExText>'
                '</RA></Questions></DOC>', encoding='utf-8')
            transform.preprocess(str(source), str(prepared))
            calls = []

            def translate(text):
                calls.append(text)
                return text.replace('word', 'house').replace('one', 'two').replace('ending', 'finish')

            self.assertEqual(transform.fallback('unmapped\n', 'translated\n', translate),
                             'translated\n')
            transform.postprocess(str(prepared), str(output))
            restored = ET.parse(output).getroot()
            self.assertEqual([node.text for node in restored.iter('ExText')],
                             ['<b>house ', 'two', ' finish</b>'])
            self.assertNotIn('fraus-format-', output.read_text(encoding='utf-8'))
            self.assertTrue(any('__BLANK__' in call for call in calls))
            self.assertTrue(any(item['type'] == 'ambiguous_ra_recovery'
                                for item in transform.fallback_diagnostics))


if __name__ == '__main__':
    unittest.main()
