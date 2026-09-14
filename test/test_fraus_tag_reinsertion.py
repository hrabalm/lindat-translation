"""Tag preservation through Document translation with offline service doubles."""

from pathlib import Path
import os
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET

from flask import Flask

from app.main.document import Document, DocumentPipeline
from app.main.markuptranslator import MarkupTranslator
from document_translation.markuptranslator import Aligner, Translator
from document_translation.regextokenizer import RegexTokenizer


def word_alignment(source, target):
    assert len(source) == len(target)
    assert all(len(left) == len(right) for left, right in zip(source, target))
    return [[(i, i) for i in range(len(sentence))] for sentence in source]


class FrausTagReinsertionTests(unittest.TestCase):
    def translate_document(self, source, expected_calls=(), expected_recovery=False):
        def run(pipeline, input_path, output_path, src, tgt, translate, debug=False):
            result = translate(source)
            return SimpleNamespace(text=result, trace={})

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.xml'
            path.write_text('<DOC/>', encoding='utf-8')
            document = Document(str(path))
            with Flask(__name__).test_request_context('/?debug=true'), patch(
                    'app.main.document.text_input_with_src_tgt.parse_args',
                    return_value={'fraus': True, 'xmlTransform': 'fraus_v2'}), patch.object(
                    DocumentPipeline, 'run', run), patch(
                    'app.main.document.translate_from_to',
                    side_effect=lambda src, tgt, text, **kwargs:
                    ([text], [text.replace('word', 'house')])) as backend, patch(
                    'app.main.document.LindatAligner.align', side_effect=word_alignment) as align, patch(
                    'socket.socket.connect', side_effect=AssertionError('Network forbidden')):
                document.translate_from_to('cs', 'en')
            self.assertEqual([call.args[2] for call in backend.call_args_list], list(expected_calls))
            if not expected_calls:
                align.assert_called_once_with([[]], [[]])
            self.assertEqual('fraus_recovery' in document.debug_trace, expected_recovery)
            return document.get_translation()

    def test_whitespace_only_pair_is_retained_without_translation_or_recovery(self):
        self.assertEqual(self.translate_document('<g id="1"> </g>'), '<g id="1"> </g>')

    def test_whitespace_and_nested_pairs_keep_their_aligned_span(self):
        for source in ('<g id="1">\t</g>', '<g id="1">  \t </g>',
                       '<g id="1">\xa0 </g>', '<g id="1"><g id="2"> </g></g>',
                       '<g id="1"> </g><g id="2">\t</g>'):
            with self.subTest(source=source):
                self.assertEqual(self.translate_document(source), source.replace('\xa0', ' '))

    def test_empty_pair_and_self_closing_tag_survive(self):
        for source in ('<g id="1"></g>', '<x id="1"/>'):
            with self.subTest(source=source):
                # Document serializes an empty g as <g ... />, which upstream
                # treats as an unclosed pair; existing structural recovery retains it.
                result = ET.fromstring('<root>' + self.translate_document(
                    source, expected_recovery=source.startswith('<g')) + '</root>')
                self.assertEqual([(node.tag, node.attrib) for node in result.iter()],
                                 [('root', {}), (source[1:2], {'id': '1'})])
                self.assertFalse(''.join(result.itertext()).strip())

    def test_document_keeps_line_endings_nested_tags_and_translated_words(self):
        source = '<g id="1"> </g>\r\n<g id="2"><g id="3">word</g></g>\n<g id="4">\t</g>'
        self.assertEqual(self.translate_document(source, ('word',)),
                         '<g id="1"> </g>\r\n<g id="2"><g id="3">house </g></g>\n<g id="4">\t</g>')

    def test_public_markup_api_does_not_scan_or_expand_into_neighboring_lines(self):
        # Strict Document splits lines first; exercise the package's multiline API too.
        class WordTranslator(Translator):
            def translate(self, input_text: str):
                return [input_text], [input_text.replace('word', 'house')]

        class WordAligner(Aligner):
            def align(self, src_batch, tgt_batch):
                return word_alignment(src_batch, tgt_batch)

        markup = MarkupTranslator(WordTranslator(), WordAligner(), RegexTokenizer())
        for source in ('<g id="1"> </g>\nword', 'word\n<g id="1"> </g>',
                       '<g id="1"> </g> \nword', 'word\n<g id="1"> </g> ',
                       'word\n <g id="1"><g id="2">\t</g></g> \nword',
                       '<g id="1"> </g>\n<g id="2">\t</g>'):
            with self.subTest(source=source):
                self.assertEqual(markup.translate(source), source.replace('word', 'house'))

    def test_public_markup_api_still_expands_whole_text_line_to_include_unaligned_words(self):
        class ExpandedTranslator(Translator):
            def translate(self, input_text: str):
                return [input_text], ['new house']

        class PartialAligner(Aligner):
            def align(self, src_batch, tgt_batch):
                return [[(0, 1)]]

        markup = MarkupTranslator(ExpandedTranslator(), PartialAligner(), RegexTokenizer())
        self.assertEqual(markup.translate('<g id="1">word</g>'), '<g id="1">new house</g>')

    @unittest.skipUnless(os.environ.get('RUN_FRAUS_OKAPI') == '1',
                         'Set RUN_FRAUS_OKAPI=1 for real offline Okapi')
    def test_offline_okapi_roundtrip_preserves_bold_space_and_translates_word(self):
        tikal = Path(os.environ.get('TIKAL_PATH', '/tmp/opencode/okapi/'))
        if not (tikal / 'tikal.sh').is_file():
            self.skipTest(f'Okapi not found: {tikal}')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.xml'
            path.write_text(
                '<DOC><ExText Id="space">&amp;lt;b&amp;gt; &amp;lt;/b&amp;gt;</ExText>'
                '<ExText Id="word">&amp;lt;b&amp;gt;word&amp;lt;/b&amp;gt;</ExText></DOC>',
                encoding='utf-8')
            document = Document(str(path))
            with Flask(__name__).test_request_context('/?debug=true', method='POST', data={
                    'fraus': 'true', 'xmlTransform': 'fraus_v2',
                    'forceSentenceLevel': 'false', 'maxSegmentTokens': '0',
            }), patch('app.main.document.TIKAL_PATH', str(tikal.resolve()) + os.sep), patch(
                    'app.main.document.translate_from_to', side_effect=lambda src, tgt, text, **kwargs:
                    ([text], [text.replace('word', 'house')])) as backend, patch(
                    'app.main.document.LindatAligner.align', side_effect=word_alignment), patch(
                    'socket.socket.connect', side_effect=AssertionError('Network forbidden')), patch(
                    'socket.create_connection', side_effect=AssertionError('Network forbidden')):
                document.translate_from_to('cs', 'en')
            restored = ET.parse(document.get_translated_path('en')).getroot()
            space = restored.find('./ExText[@Id="space"]')
            word = restored.find('./ExText[@Id="word"]')
            assert space is not None
            assert word is not None
            self.assertEqual(space.text, '<b> </b>')
            self.assertEqual(word.text, '<b>house </b>')
            self.assertEqual([call.args[2] for call in backend.call_args_list], ['word'])
            self.assertNotIn('fraus_recovery', document.debug_trace)
            self.assertTrue({'extract_xml', 'extract_html', 'merge_html', 'merge_xml',
                             'postprocessed_xml'} <= document.debug_trace.keys())


if __name__ == '__main__':
    unittest.main()
