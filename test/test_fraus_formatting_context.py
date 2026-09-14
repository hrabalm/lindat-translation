import html
import os
import tempfile
import unittest
import xml.etree.ElementTree as ET
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from flask import Flask

from app.main.document import (
    Document, FrausTranslationError, FrausV2XmlTransform, sanitize_generated_markup,
)


class FrausFormattingContextTests(unittest.TestCase):
    def round_trip(self, before, after, translate, candidate=None):
        with tempfile.TemporaryDirectory() as directory:
            source, prepared, restored = [Path(directory) / name for name in
                                          ('source.xml', 'prepared.xml', 'restored.xml')]
            source.write_text(
                '<DOC><Questions><RA><ExText Id="before">'
                + html.escape(before) + '</ExText>'
                '<InputOption><SelectOption Correct="true"><ExText>A</ExText>'
                '</SelectOption></InputOption><ExText Id="after">'
                + html.escape(after) + '</ExText></RA></Questions></DOC>',
                encoding='utf-8',
            )
            transform = FrausV2XmlTransform()
            transform.preprocess(str(source), str(prepared))
            if candidate is None:
                # Extraction cannot associate this line with the original variant.
                self.assertEqual(transform.fallback('unmapped\n', 'target\n', translate),
                                 'target\n')
            else:
                tree = ET.parse(prepared)
                payload = tree.find('.//RA/ExText')
                assert payload is not None
                payload.text = candidate
                tree.write(prepared, encoding='utf-8')
            transform.postprocess(str(prepared), str(restored))
            return ET.parse(restored)

    def test_unmapped_context_preserves_table_across_option(self):
        tree = self.round_trip('<table><tr><td>', '</td></tr></table>',
                               lambda text: 'translated option\n')
        self.assertEqual([node.text for node in tree.findall('.//RA/ExText')],
                         ['<table><tr><td>', '</td></tr></table>'])
        self.assertEqual(tree.findtext('.//SelectOption/ExText'), 'translated option')

    def test_unmapped_layout_fragments_keep_exact_source_formatting(self):
        fragments = [
            '<br/>', '<br/><br/>', '<br/>      <br/>',
            '<div><br></div><div><br></div><div><br></div>',
            '<span style="line-height: 2em;" > ', '<b>',
            '&lt;table&gt;&lt;tr&gt;&lt;td&gt;',
        ]
        for fragment in fragments:
            with self.subTest(fragment=fragment):
                tree = self.round_trip(fragment, '', lambda text: 'option target\n')
                self.assertEqual(tree.findtext('.//RA/ExText'), fragment)

    def test_unmapped_mixed_context_keeps_tags_and_translates_visible_text(self):
        def translate(text):
            return text.replace('Before', 'Start').replace('inside', 'within').replace(
                'after', 'end').replace('A\n', 'option target\n')

        tree = self.round_trip(
            '<span style="line-height: 2em;"> Before <b>inside</b> ',
            ' after</span><br/>', translate,
        )
        self.assertEqual([node.text for node in tree.findall('.//RA/ExText')], [
            '<span style="line-height: 2em;"> Start <b>within</b> ',
            ' end</span><br/>',
        ])
        self.assertEqual(tree.findtext('.//SelectOption/ExText'), 'option target')

    def test_missing_visible_context_is_not_satisfied_by_markup(self):
        for candidate in (' ', '<b></b>', '&lt;b&gt;&lt;/b&gt;',
                          '<span title="a &quot; > b"></span>'):
            with self.subTest(candidate=candidate), self.assertRaisesRegex(
                    FrausTranslationError, 'context reconstruction has no usable translation'):
                self.round_trip('<b>Visible context</b>', '', None, candidate=candidate)

    def test_quoted_tag_delimiters_never_reach_context_prompts(self):
        for quote in ('"', "'"):
            for encoding_depth in range(3):
                for visible in ('Before', ''):
                    with self.subTest(quote=quote, encoding_depth=encoding_depth, visible=visible):
                        opening = f'<span title={quote}a > b{quote}>'
                        closing = '</span>'
                        for _ in range(encoding_depth):
                            opening = html.escape(opening)
                            closing = html.escape(closing)
                        prompts = []

                        def translate(text):
                            prompts.append(text)
                            self.assertIn(text, ('Before__BLANK__\n', 'A\n'))
                            return {'Before__BLANK__\n': 'Start__BLANK__\n',
                                    'A\n': 'Option target\n'}[text]

                        # The opening and closing tags span the option boundary.
                        tree = self.round_trip(opening + visible, closing, translate)
                        self.assertEqual([node.text for node in tree.findall('.//RA/ExText')],
                                         [opening + ('Start' if visible else ''), closing])
                        self.assertEqual(prompts, ['Before__BLANK__\n', 'A\n'] if visible else ['A\n'])

    def test_unmapped_missing_visible_translation_still_fails(self):
        with self.assertRaisesRegex(FrausTranslationError, 'context recovery produced no usable translation'):
            self.round_trip('<b>Visible context</b>', '<br/>', lambda text: ' \n')

    def test_quote_entities_in_attribute_values_do_not_leak_into_prompts(self):
        for attribute in ('title=Bob&#39;s', 'title = Bob&#39;s',
                          'title = "Bob&#34;s > book"', "title = 'Bob&#39;s > book'"):
            for encoding_depth in range(3):
                with self.subTest(attribute=attribute, encoding_depth=encoding_depth):
                    opening, closing = f'<span {attribute}>', '</span>'
                    for _ in range(encoding_depth):
                        opening, closing = html.escape(opening), html.escape(closing)
                    prompts = []

                    def translate(text):
                        prompts.append(text)
                        self.assertIn(text, ('Before__BLANK__\n', 'A\n'))
                        return text.replace('Before', 'Start')

                    tree = self.round_trip(opening + 'Before', closing, translate)
                    self.assertEqual([node.text for node in tree.findall('.//RA/ExText')],
                                     [opening + 'Start', closing])
                    self.assertEqual(prompts, ['Before__BLANK__\n', 'A\n'])

    def test_sanitizer_removes_generated_tags_with_unquoted_quote_entities(self):
        target = '<span title=it&#39;s>Before</span>'
        for encoding_depth in range(3):
            with self.subTest(encoding_depth=encoding_depth):
                self.assertEqual(sanitize_generated_markup('plain', target), 'Before')
                self.assertEqual(sanitize_generated_markup('<span>source</span>', target), target)
            target = html.escape(target)

    def test_attribute_entities_are_data_at_each_tag_encoding_layer(self):
        for value in ('a&gt;b', '&#39;Bob', '&quot;Bob'):
            for depth in range(3):
                for visible in ('Before', ''):
                    with self.subTest(value=value, depth=depth, visible=visible):
                        opening, closing = f'<span title={value}>', '</span>'
                        for _ in range(depth):
                            opening, closing = html.escape(opening), html.escape(closing)
                        prompts = []

                        def translate(text):
                            prompts.append(text)
                            self.assertIn(text, ('Before__BLANK__\n', 'A\n'))
                            return text.replace('Before', 'Start')

                        tree = self.round_trip(opening + visible, closing, translate)
                        self.assertEqual([node.text for node in tree.findall('.//RA/ExText')],
                                         [opening + ('Start' if visible else ''), closing])
                        self.assertEqual(prompts, ['Before__BLANK__\n', 'A\n'] if visible else ['A\n'])

    def test_sanitizer_uses_delimiters_from_the_tag_encoding_layer(self):
        for value in ('a&gt;b', '&#39;Bob', '&quot;Bob'):
            target = f'<span title={value}>Before</span>'
            for depth in range(3):
                with self.subTest(value=value, depth=depth):
                    self.assertEqual(sanitize_generated_markup('plain', target), 'Before')
                    self.assertEqual(sanitize_generated_markup('<span>source</span>', target), target)
                target = html.escape(target)

    def test_encoded_tags_can_keep_raw_attribute_quotes(self):
        for quote in ('"', "'"):
            opening, closing = f'<span title={quote}a > b{quote}>', '</span>'
            for depth in range(1, 3):
                opening = html.escape(opening, quote=False)
                closing = html.escape(closing, quote=False)
                with self.subTest(quote=quote, depth=depth):
                    def translate(text):
                        self.assertIn(text, ('Before__BLANK__\n', 'A\n'))
                        return text.replace('Before', 'Start')

                    tree = self.round_trip(opening + 'Before', closing, translate)
                    self.assertEqual([node.text for node in tree.findall('.//RA/ExText')],
                                     [opening + 'Start', closing])
                    self.assertEqual(sanitize_generated_markup('plain', opening + 'Before' + closing),
                                     'Before')

    @unittest.skipUnless(os.environ.get('RUN_FRAUS_OKAPI') == '1',
                         'Set RUN_FRAUS_OKAPI=1 and TIKAL_PATH for real Okapi')
    def test_document_preserves_table_and_reports_recovery_with_real_okapi(self):
        tikal = Path(os.environ['TIKAL_PATH'])
        if tikal.is_file():
            tikal = tikal.parent
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source.xml'
            source.write_text(
                '<DOC><Questions><Question><RA><ExText Id="before">'
                '&amp;lt;table&amp;gt;&amp;lt;tr&amp;gt;&amp;lt;td&amp;gt;'
                '</ExText><InputOption><SelectOption Correct="true">'
                '<ExText>A</ExText></SelectOption></InputOption><ExText Id="after">'
                '&amp;lt;/td&amp;gt;&amp;lt;/tr&amp;gt;&amp;lt;/table&amp;gt;'
                '</ExText></RA></Question></Questions></DOC>', encoding='utf-8',
            )
            document = Document(str(source))
            with Flask(__name__).test_request_context('/?debug=true'), patch(
                    'app.main.document.TIKAL_PATH', str(tikal) + os.sep), patch(
                    'app.main.document.text_input_with_src_tgt.parse_args',
                    return_value={'fraus': True, 'xmlTransform': 'fraus_v2'}), patch(
                    'app.main.document.MarkupTranslator.translate', side_effect=lambda text: text), patch(
                    'app.main.document.translate_from_to',
                    side_effect=lambda src, tgt, text, **kwargs: ([text], ['Option target'])), patch(
                    'socket.socket.connect', side_effect=AssertionError('Network forbidden')), patch(
                    'socket.create_connection', side_effect=AssertionError('Network forbidden')), redirect_stderr(StringIO()):
                document.translate_from_to('cs', 'en')
                tree = ET.parse(document.translated_path)
                self.assertEqual([html.unescape(node.text or '') for node in
                                  tree.findall('.//RA/ExText')],
                                 ['<table><tr><td>', '</td></tr></table>'])
                self.assertEqual(tree.findtext('.//SelectOption/ExText'), 'Option target')
                response = document.create_response({})
                self.assertEqual(response.headers['X-FRAUS-Recovery'], 'recovered')
                self.assertTrue(any(item['type'] == 'ambiguous_ra_recovery'
                                    for item in response.get_json()['fraus_recovery']))
