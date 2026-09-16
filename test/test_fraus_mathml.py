"""Offline full-Document tests; enable with RUN_FRAUS_OKAPI=1 and TIKAL_PATH.

Only the model and external aligner are stubbed. XML/HTML extraction, sentence
ownership, markup translation and both Okapi merges run normally.
"""

from contextlib import redirect_stderr
from io import StringIO
import os
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch
import xml.etree.ElementTree as ET


@unittest.skipUnless(os.environ.get('RUN_FRAUS_OKAPI') == '1',
                     'Set RUN_FRAUS_OKAPI=1 for real offline Okapi tests')
class FrausMathMLTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = Path(os.environ.get(
            'TIKAL_PATH', 'fraus_test_data_translations_2026-09-08/.dependencies/okapi-1.48.0'))
        if not (directory / 'tikal.sh').is_file():
            raise RuntimeError(f'Okapi not found in {directory}')
        cls.tikal_directory = str(directory.resolve()) + os.sep

    def roundtrip(self, fragments: list[tuple[str, str]]):
        from flask import Flask
        from app.main.document import Document

        root = ET.Element('DOC')
        for index, (tag, text) in enumerate(fragments):
            ET.SubElement(root, tag, Id=str(index)).text = text
        calls: list[str] = []

        def backend(src, tgt, text, **kwargs):
            self.assertEqual((src, tgt), ('en', 'de'))
            calls.append(text)
            return [text], [text.replace('SOURCE', 'TARGET')]

        def align(source, target):
            self.assertEqual(len(source), len(target))
            for left, right in zip(source, target):
                self.assertEqual(len(left), len(right))
            return [[(i, i) for i in range(len(sentence))] for sentence in source]

        log = StringIO()
        with tempfile.TemporaryDirectory(prefix='fraus-mathml-') as directory:
            path = Path(directory) / 'input.xml'
            ET.ElementTree(root).write(path, encoding='utf-8', xml_declaration=True)
            document = Document(str(path))
            with Flask(__name__).test_request_context('/?debug=true', method='POST', data={
                    'fraus': 'true', 'xmlTransform': 'fraus_v2',
                    'forceSentenceLevel': 'true', 'maxSegmentTokens': '100',
            }), patch('app.main.document.TIKAL_PATH', self.tikal_directory), \
                    patch('app.main.document.translate_from_to', side_effect=backend), \
                    patch('app.main.document.translate_with_model',
                          side_effect=AssertionError('Unexpected model route')), \
                    patch('app.main.document.LindatAligner.align_request', side_effect=align), \
                    patch('socket.socket.connect', side_effect=AssertionError('Network forbidden')), \
                    patch('socket.socket.connect_ex', side_effect=AssertionError('Network forbidden')), \
                    patch('socket.create_connection', side_effect=AssertionError('Network forbidden')), \
                    patch.dict(os.environ, {'JAVA_TOOL_OPTIONS': '-Xmx256m -XX:ActiveProcessorCount=2'}), \
                    redirect_stderr(log):
                try:
                    document.translate_from_to('en', 'de')
                except Exception as error:
                    self.fail(f'{type(error).__name__}: {error}\nCalls: {calls!r}\n{log.getvalue()}')
            restored = ET.parse(document.get_translated_path('de')).getroot()
        self.assertTrue({'extract_xml', 'extract_html', 'merge_html', 'merge_xml',
                         'postprocessed_xml'} <= document.debug_trace.keys())
        self.assertEqual([(node.tag, node.attrib) for node in root.iter()],
                         [(node.tag, node.attrib) for node in restored.iter()])
        return [node.text or '' for node in restored], calls

    def test_symbolic_equation_never_calls_model(self):
        formula = '<math><mrow><msub><mi>F</mi><mi>vz</mi></msub><mo>=</mo><mn>42</mn></mrow></math>'
        output, calls = self.roundtrip([('Equation', formula)])
        self.assertEqual(calls, [])
        self.assertEqual(output[0].strip(), formula)

    def test_symbolic_prefixes_and_mismatched_root_in_both_xml_carriers(self):
        formulas = []
        for prefix in ('', 'm:', 'mml:'):
            formula = (f'<{prefix}math><{prefix}mfrac><{prefix}mi>x</{prefix}mi>'
                       f'<{prefix}mn>17</{prefix}mn></{prefix}mfrac>'
                       f'<{prefix}mo>+</{prefix}mo></{prefix}math>')
            formulas.append(formula)
        # Legal outer XML carries malformed MathML as text; do not repair it.
        formulas.append(formulas[2].replace('<mml:math>', '<math>'))
        fragments = [(tag, formula) for tag in ('Equation', 'ExText') for formula in formulas]
        output, calls = self.roundtrip(fragments)
        self.assertEqual(calls, [])
        self.assertEqual([text.strip() for text in output], [text for _, text in fragments])

    def test_mtext_and_surrounding_prose_translate_but_annotations_do_not(self):
        fragments = []
        for tag in ('Equation', 'ExText'):
            for prefix in ('', 'm:', 'mml:'):
                formula = (
                    f'SOURCEbefore <b>SOURCEbold</b> <{prefix}math display="inline">'
                    f'<{prefix}semantics><{prefix}mrow>'
                    f'<{prefix}mtext>SOURCEforce label</{prefix}mtext>'
                    f'<{prefix}mi>F</{prefix}mi><{prefix}mo>=</{prefix}mo>'
                    f'<{prefix}msup><{prefix}mi>v</{prefix}mi>'
                    f'<{prefix}mn>2</{prefix}mn></{prefix}msup>'
                    f'<{prefix}mtext>kg</{prefix}mtext></{prefix}mrow>'
                    f'<{prefix}annotation encoding="text/plain">SECRETplain</{prefix}annotation>'
                    f'<{prefix}annotation-xml encoding="MathML-Presentation">'
                    f'<{prefix}mtext>SECRETnested</{prefix}mtext>'
                    f'<mtext>SECRETunprefixed</mtext>'
                    f'</{prefix}annotation-xml></{prefix}semantics></{prefix}math>'
                    ' <i>SOURCEafter</i>')
                fragments.append((tag, formula))
                if prefix == 'mml:':
                    fragments.append((tag, formula.replace(
                        '<mml:math display="inline">', '<math display="inline">')))
        output, calls = self.roundtrip(fragments)
        self.assertEqual(len(calls), len(fragments), calls)
        for text in calls:
            self.assertEqual(text.split(), ['SOURCEbefore', 'SOURCEbold',
                                           'SOURCEforce', 'label', 'kg', 'SOURCEafter'])
        for actual, (_, source) in zip(output, fragments):
            # Compare ordered tags/attributes and each text run. XML extraction
            # already collapses whitespace; alignment can move boundary spaces.
            self.assertEqual(
                [part.strip() for part in re.split(r'(<[^>]*>)', actual) if part.strip()],
                [part.strip() for part in re.split(r'(<[^>]*>)', source.replace('SOURCE', 'TARGET'))
                 if part.strip()])

    def test_short_labels_and_ordinary_prose_outside_math_still_translate(self):
        output, calls = self.roundtrip([
            ('ExText', 'SOURCElabel'), ('ExText', 'SOURCEordinary prose.'),
            ('ExText', '<b>SOURCEx</b>'), ('ExText', 'x'), ('ExText', 'kg'),
        ])
        self.assertEqual([text.strip() for text in calls],
                         ['SOURCElabel', 'SOURCEordinary prose.', 'SOURCEx', 'x', 'kg'])
        self.assertEqual([text.strip() for text in output],
                         ['TARGETlabel', 'TARGETordinary prose.', '<b>TARGETx </b>', 'x', 'kg'])


if __name__ == '__main__':
    unittest.main()
