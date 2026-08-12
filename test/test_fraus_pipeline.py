import os
import tempfile
import unittest
from contextlib import redirect_stderr
from io import StringIO
from types import SimpleNamespace

from app.main.document import (
    DocumentPipeline,
    FrausDocumentFormat,
    StandardDocumentFormat,
    TikalRunner,
    fix_fraus_encoding,
    unwrap_and_escape,
    unescape_extracted_line,
    wrap_paragraph,
)


class FrausTransformTests(unittest.TestCase):
    def test_fix_encoding_only_changes_xml_declaration(self):
        self.assertEqual(
            fix_fraus_encoding('<?xml version="1.0" encoding="utf-16"?>\n'),
            '<?xml version="1.0" encoding="utf-8"?>\n',
        )
        line = '<text encoding="utf-16">content</text>\n'
        self.assertEqual(fix_fraus_encoding(line), line)

    def test_unescape_and_paragraph_transforms_preserve_current_behavior(self):
        self.assertEqual(unescape_extracted_line('&amp;lt;i&amp;gt;A&amp;nbsp;B'), '<i>A\xa0B')
        self.assertEqual(wrap_paragraph('hello\n'), '<p>hello</p>\n')
        self.assertEqual(unwrap_and_escape('<p>a &lt; b</p>\n'), 'a &amp;amp;lt; b\n')


def fake_tikal(calls):
    def run(command, stdout):
        calls.append(command)
        output_path = command[command.index('-to') + 1]
        if command[1] == '-xm' and output_path.endswith('.fixed'):
            with open(output_path + '.en', 'w', encoding='utf-8') as file:
                file.write('Hello &amp;nbsp;\n')
        elif command[1] == '-xm':
            with open(output_path + '.en', 'w', encoding='utf-8') as file:
                file.write('Hello\n')
        else:
            with open(output_path, 'w', encoding='utf-8') as file:
                file.write('<p>Ahoj</p>\n')
        return SimpleNamespace(returncode=0)
    return run


class PipelineTests(unittest.TestCase):
    def make_input(self, directory):
        path = os.path.join(directory, 'input.xml')
        with open(path, 'w', encoding='utf-8') as file:
            file.write('<?xml version="1.0" encoding="utf-16"?>\nHello\n')
        return path

    def test_fraus_pipeline_preserves_original_stage_order_and_result(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.make_input(directory)
            output = os.path.join(directory, 'input.cs.xml')
            calls = []
            pipeline = DocumentPipeline(
                FrausDocumentFormat(
                    TikalRunner('/tikal/', run=fake_tikal(calls)),
                    'xml-profile', 'html-profile',
                )
            )
            translated = []

            result = pipeline.run(source, output, 'en', 'cs',
                                  lambda text: translated.append(text) or 'Ahoj\n')

            self.assertEqual(result.output_path, output)
            self.assertEqual(result.text, 'Ahoj\n')
            self.assertEqual(translated, ['Hello\n'])
            self.assertEqual(len(calls), 4)
            self.assertEqual([call[1] for call in calls], ['-xm', '-xm', '-lm', '-lm'])
            self.assertEqual(calls[0][3:], ['-fc', 'xml-profile', '-sl', 'en', '-to', source + '.fixed'])
            self.assertEqual(calls[1][3:], ['-fc', 'html-profile', '-sl', 'en', '-to', source + '.fixed.en.html.p'])
            self.assertTrue(os.path.exists(output))
            self.assertFalse(os.path.exists(source))

    def test_standard_pipeline_has_same_extract_translate_merge_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.make_input(directory)
            output = os.path.join(directory, 'input.cs.xml')
            calls = []
            pipeline = DocumentPipeline(
                StandardDocumentFormat(TikalRunner('/tikal/', run=fake_tikal(calls)))
            )
            translated = []

            result = pipeline.run(source, output, 'en', 'cs',
                                  lambda text: translated.append(text) or 'Ahoj\n')

            self.assertEqual(result.text, 'Ahoj\n')
            self.assertEqual(translated, ['Hello\n'])
            self.assertEqual(len(calls), 2)
            self.assertEqual([call[1] for call in calls], ['-xm', '-lm'])
            self.assertEqual(calls[0][3:], ['-sl', 'en', '-to', source])
            self.assertTrue(os.path.exists(output))

    def test_debug_trace_is_per_run_and_written_to_stderr(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.make_input(directory)
            output = os.path.join(directory, 'input.cs.xml')
            pipeline = DocumentPipeline(
                StandardDocumentFormat(TikalRunner('/tikal/', run=fake_tikal([]))),
                debug=True,
            )
            stderr = StringIO()
            with redirect_stderr(stderr):
                result = pipeline.run(source, output, 'en', 'cs', lambda text: 'Ahoj\n')

            self.assertEqual(list(result.trace), ['extract', 'translate', 'merge'])
            self.assertIn('stage=extract', stderr.getvalue())
            self.assertIn('stage=translate', stderr.getvalue())
            self.assertIn('stage=merge', stderr.getvalue())

    def test_debug_can_be_enabled_per_run(self):
        with tempfile.TemporaryDirectory() as directory:
            source = self.make_input(directory)
            output = os.path.join(directory, 'input.cs.xml')
            pipeline = DocumentPipeline(
                StandardDocumentFormat(TikalRunner('/tikal/', run=fake_tikal([])))
            )
            result = pipeline.run(source, output, 'en', 'cs',
                                  lambda text: 'Ahoj\n', debug=True)
            self.assertEqual(list(result.trace), ['extract', 'translate', 'merge'])


if __name__ == '__main__':
    unittest.main()
