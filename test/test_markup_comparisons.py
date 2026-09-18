"""Synthetic regressions derived from captured sin/f comparison syntax."""

import html
import os
import re
import tempfile
import unittest
import xml.etree.ElementTree as ET
from contextlib import redirect_stderr
from difflib import SequenceMatcher
from io import StringIO
from pathlib import Path
from unittest.mock import patch

from app.main.document import Document, sanitize_generated_markup
from test import test_fraus_formatting_context


class MarkupComparisonTests(unittest.TestCase):
    def test_html_attribute_names_are_protected_in_formatting_context(self):
        context = test_fraus_formatting_context.FrausFormattingContextTests()
        for name in ('\u010d', '@click', '.value', '1value'):
            opening, closing = f'<span {name}="a < b > c">', '</span>'
            for depth in range(3):
                with self.subTest(name=name, depth=depth):
                    prompts = []

                    def translate(text):
                        prompts.append(text)
                        self.assertIn(text, ('Before__BLANK__\n', 'A\n'))
                        return text.replace('Before', 'Start')

                    tree = context.round_trip(opening + 'Before', closing, translate)
                    self.assertEqual([node.text for node in tree.findall('.//RA/ExText')],
                                     [opening + 'Start', closing])
                    self.assertEqual(prompts, ['Before__BLANK__\n', 'A\n'])
                opening, closing = map(html.escape, (opening, closing))

    def test_html_attribute_names_remove_whole_unknown_tags_and_keep_allowed_tags(self):
        for name in ('\u010d', '@click', '.value', '1value'):
            for attribute in (name, f'{name}="a < b > c"'):
                opening, closing = f'<unknown {attribute}>', '</unknown>'
                allowed = f'<g {attribute}>kept</g>'
                source = f'<g {attribute}/>'
                for depth in range(3):
                    with self.subTest(attribute=attribute, depth=depth):
                        self.assertEqual(sanitize_generated_markup('<g>source</g>',
                                         opening + 'text' + closing + allowed), 'text' + allowed)
                        # An attributed standalone source tag must also authorize g.
                        self.assertEqual(sanitize_generated_markup(
                            source, '<g>target</g>'), '<g>target</g>')
                    opening, closing, allowed, source = map(
                        html.escape, (opening, closing, allowed, source))

    def test_captured_comparisons_keep_real_opening_g(self):
        source = '<g id="5">left &lt; <g id="7"><g id="8">right</g></g></g>'
        for comparison in ('< sin ', '< f(', '< sin(t) '):
            with self.subTest(comparison=comparison):
                target = (f'<g id="5">left {comparison}'
                          '<g id="7"><g id="8">right</g></g></g>')
                self.assertEqual(sanitize_generated_markup(source, target), target)

    def test_attribute_names_stop_at_layer_delimiters_and_reject_controls(self):
        for prefix in ('<unknown @click', '<unknown @click"broken>',
                       "<unknown .value'broken>", '<unknown 1value\x00bad>',
                       '<unknown 1value\x01bad>', '<unknown 1value\x7fbad>'):
            for depth in range(3):
                with self.subTest(prefix=prefix, depth=depth):
                    target = prefix + '<g id="7">kept</g>'
                    for _ in range(depth):
                        target = html.escape(target)
                    self.assertEqual(sanitize_generated_markup('<g/>', target), target)

        target = '<unknown @click>text <g id="7">kept</g></unknown>'
        expected = 'text <g id="7">kept</g>'
        for depth in range(3):
            with self.subTest(depth=depth):
                self.assertEqual(sanitize_generated_markup('<g/>', target), expected)
            target, expected = map(html.escape, (target, expected))

    def test_non_tags_do_not_swallow_following_markup_at_any_layer(self):
        for prefix in ('< sin ', '< f(', '<sin(t) ', '<f(t) ', '<fake ',
                       '<fake broken=', '<fake value=bad',
                       '<fake value=a=b ', '<fake=bad ', '< 3 ', '<x', 'a < b > c '):
            for depth in range(3):
                with self.subTest(prefix=prefix, depth=depth):
                    source = '<g>source</g><x/>'
                    kept = prefix + '<g id="7">right</g><x id="8"/>'
                    target = kept + '<new-tag data-id=7>text</new-tag>'
                    expected = kept + 'text'
                    for _ in range(depth):
                        source, target, expected = map(html.escape, (source, target, expected))
                    self.assertEqual(sanitize_generated_markup(source, target), expected)

    def test_valid_tags_and_attribute_delimiters_keep_exact_encoding(self):
        for attribute in ('disabled', 'data-id=7', 'title="a < b > c"',
                          "title='a < b > c'", 'title=&#39;Bob', 'title=&quot;Bob',
                          'title=a&lt;b&gt;c', 'title = "Bob&#34;s < book"'):
            for quote_escape in (True, False):
                opening, closing = f'<custom-tag {attribute}>', '</custom-tag>'
                for depth in range(3):
                    with self.subTest(attribute=attribute, depth=depth,
                                      quote_escape=quote_escape):
                        target = opening + 'a &amp; b' + closing
                        self.assertEqual(sanitize_generated_markup('plain', target), 'a &amp; b')
                        self.assertEqual(sanitize_generated_markup('<custom-tag/>', target), target)
                    opening = html.escape(opening, quote=quote_escape)
                    closing = html.escape(closing, quote=quote_escape)


@unittest.skipUnless(os.environ.get('RUN_FRAUS_OKAPI') == '1',
                     'Set RUN_FRAUS_OKAPI=1 and TIKAL_PATH for real offline Okapi')
class MarkupComparisonOkapiTests(unittest.TestCase):
    def test_generated_operators_survive_document_merge(self):
        from flask import Flask

        tikal = Path(os.environ['TIKAL_PATH'])
        if tikal.is_file():
            tikal = tikal.parent
        # Only syntax is derived from captures, not private document prose.
        fragments = [
            '<b>cos <i>t</i> &lt; sin <b><i>u</i></b>?</b>',
            '<b>f(<i>t</i>) &lt; f(<b><i>u</i></b>).</b>',
        ]
        calls = []

        def backend(src, tgt, text, **kwargs):
            self.assertEqual((src, tgt), ('en', 'de'))
            target = html.unescape(text)
            self.assertIn('<', target)
            calls.append(target)
            return [text], [target]

        def align(source, target):
            self.assertEqual(len(source), len(target))
            # Entity spelling tokenizes differently from the generated literal <.
            return [[(match.a + i, match.b + i)
                     for match in SequenceMatcher(a=left, b=right, autojunk=False).get_matching_blocks()
                     for i in range(match.size)] for left, right in zip(source, target)]

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'comparisons.xml'
            root = ET.Element('DOC')
            for index, fragment in enumerate(fragments):
                ET.SubElement(root, 'ExText', Id=str(index)).text = fragment
            ET.ElementTree(root).write(source, encoding='utf-8')
            document = Document(str(source))
            with Flask(__name__).test_request_context('/?debug=true', method='POST', data={
                    'fraus': 'true', 'xmlTransform': 'fraus_v2',
                    'forceSentenceLevel': 'true', 'maxSegmentTokens': '100',
            }), patch('app.main.document.TIKAL_PATH', str(tikal) + os.sep), \
                    patch('app.main.document.translate_from_to', side_effect=backend), \
                    patch('app.main.document.translate_with_model',
                          side_effect=AssertionError('Unexpected model route')), \
                    patch('app.main.document.LindatAligner.align_request', side_effect=align), \
                    patch('socket.socket.connect', side_effect=AssertionError('Network forbidden')), \
                    patch('socket.socket.connect_ex', side_effect=AssertionError('Network forbidden')), \
                    patch('socket.create_connection', side_effect=AssertionError('Network forbidden')), \
                    redirect_stderr(StringIO()):
                document.translate_from_to('en', 'de')
            restored = ET.parse(document.translated_path)

        self.assertEqual(len(calls), 2)
        self.assertRegex(document.translation, r'< sin <g id="\d+">')
        self.assertRegex(document.translation, r'< f\(<g id="\d+">')
        depth = 0
        for tag in re.findall(r'</?g\b[^>]*>', document.translation):
            depth += -1 if tag.startswith('</') else 1
            self.assertGreaterEqual(depth, 0)
        self.assertEqual(depth, 0)
        self.assertTrue({'merge_html', 'merge_xml', 'postprocessed_xml'}
                        <= document.debug_trace.keys())
        self.assertEqual(len(restored.findall('ExText')), 2)
        for actual, expected in zip(restored.findall('ExText'), fragments):
            # Okapi normalizes boundary whitespace, not the operators or paired tags.
            self.assertEqual(re.sub(r'\s+', '', html.unescape(actual.text or '')),
                             re.sub(r'\s+', '', html.unescape(expected)))
