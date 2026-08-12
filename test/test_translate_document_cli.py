import importlib.util
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path


CLI_PATH = Path(__file__).parents[1] / 'scripts' / 'translate_document.py'
SPEC = importlib.util.spec_from_file_location('translate_document_cli', CLI_PATH)
cli = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(cli)


class TranslateDocumentCliTests(unittest.TestCase):
    def test_print_segments_pairs_source_and_target_lines(self):
        output = StringIO()
        with redirect_stdout(output):
            cli.print_segments({
                'extract_html': 'one\ntwo\n',
                'translate': 'jedna\ndvě\n',
            })
        self.assertIn('[1] SOURCE: one', output.getvalue())
        self.assertIn('TARGET: jedna', output.getvalue())
        self.assertIn('[2] SOURCE: two', output.getvalue())
        self.assertIn('TARGET: dvě', output.getvalue())

    def test_print_segments_supports_standard_document_trace(self):
        output = StringIO()
        with redirect_stdout(output):
            cli.print_segments({'extract': 'one\n', 'translate': 'jedna\n'})
        self.assertIn('SOURCE: one', output.getvalue())


if __name__ == '__main__':
    unittest.main()
