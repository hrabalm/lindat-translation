import os
import stat
import tempfile
import unittest
import xml.etree.ElementTree as ET
from app.settings import (
    FRAUS_V2_FORCE_SENTENCE_LEVEL,
    FRAUS_V2_MAX_SEGMENT_TOKENS,
    _cleanup_upload_folder,
    _create_upload_folder,
)
from contextlib import redirect_stderr
from flask import Flask
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

from app.main.document import (
    Document,
    DocumentPipeline,
    FrausV2XmlTransform,
    FrausDocumentFormat,
    StandardDocumentFormat,
    TikalError,
    TikalRunner,
    fix_fraus_encoding,
    unwrap_and_escape,
    unescape_extracted_line,
    sanitize_generated_markup,
    translate_with_line_fallback,
    wrap_paragraph,
)
from app.main.api.translation.parsers import text_input_with_src_tgt
from app.models.llm_errors import LLMBackendUnavailable
from app.models.llm_request_state import (
    LLMSegmentRecord,
    get_request_llm_state,
)


class FrausTransformTests(unittest.TestCase):
    def test_markup_translation_retries_each_line_after_alignment_failure(self):
        calls = []

        def translate(text):
            calls.append(text)
            if len(calls) == 1:
                raise AssertionError('alignment failed')
            return text.upper()

        self.assertEqual(
            translate_with_line_fallback('first\nsecond\n', translate),
            'FIRST\nSECOND\n',
        )
        self.assertEqual(calls, ['first\nsecond\n', 'first\n', 'second\n'])

    def test_fraus_v2_settings_have_configurable_defaults(self):
        self.assertIsInstance(FRAUS_V2_FORCE_SENTENCE_LEVEL, bool)
        self.assertIsInstance(FRAUS_V2_MAX_SEGMENT_TOKENS, int)

    def test_fraus_v2_transform_round_trips_extext_segmentation(self):
        transform = FrausV2XmlTransform()
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, 'input.xml')
            prepared = os.path.join(directory, 'prepared.xml')
            restored = os.path.join(directory, 'restored.xml')
            original = '''<DOC><Questions><Question><RA>
<ExText Id="text-1">Before&#160;</ExText>
<InputOption Id="input-1">
  <SelectOption ParentId="select-1"><ExText>one</ExText></SelectOption>
  <SelectOption ParentId="select-2"><ExText>two</ExText></SelectOption>
</InputOption>
<ExText Id="text-2"> after.</ExText>
</RA></Question></Questions></DOC>'''
            with open(source, 'w', encoding='utf-8') as file:
                file.write(original)

            transform.preprocess(source, prepared)
            prepared_text = open(prepared, encoding='utf-8').read()
            self.assertEqual(prepared_text.count('fraus:variant'), 2)
            self.assertIn('fraus:source-ra', prepared_text)
            transform.postprocess(prepared, restored)
            restored_root = ET.parse(restored).getroot()
            self.assertEqual(
                [x.text for x in restored_root.findall('.//InputOption/SelectOption/ExText')],
                ['one', 'two'],
            )

    def test_fraus_v2_preserves_consecutive_extext_boundaries(self):
        transform = FrausV2XmlTransform()
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, 'input.xml')
            prepared = os.path.join(directory, 'prepared.xml')
            restored = os.path.join(directory, 'restored.xml')
            with open(source, 'w', encoding='utf-8') as file:
                file.write('''<DOC><Questions><Question><RA>
<ExText Id="first">First part.</ExText><ExText Id="second">Second part.</ExText>
<InputOption><SelectOption ParentId="choice"><ExText>answer</ExText></SelectOption></InputOption>
</RA></Question></Questions></DOC>''')

            transform.preprocess(source, prepared)
            tree = ET.parse(prepared)
            payload = tree.find('.//Questions//RA/ExText')
            payload.find('./g[@id="fraus-text-1"]').text = 'Translated second.'
            payload.text = 'Translated first.'
            tree.write(prepared, encoding='utf-8', xml_declaration=True)
            transform.postprocess(prepared, restored)

            self.assertEqual(
                [node.text for node in ET.parse(restored).findall('.//Questions//RA/ExText')],
                ['Translated first.', 'Translated second.'],
            )

    def test_fraus_v2_preserves_extext_identity_attributes(self):
        transform = FrausV2XmlTransform()
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, 'input.xml')
            prepared = os.path.join(directory, 'prepared.xml')
            restored = os.path.join(directory, 'restored.xml')
            with open(source, 'w', encoding='utf-8') as file:
                file.write('''<DOC><Questions><Question><RA>
<ExText Id="first" ParentId="first-parent">Before </ExText>
<InputOption><SelectOption ParentId="choice"><ExText>answer</ExText></SelectOption></InputOption>
<ExText Id="second"> after.</ExText>
</RA></Question></Questions></DOC>''')

            transform.preprocess(source, prepared)
            transform.postprocess(prepared, restored)

            first, second = ET.parse(restored).findall('.//Questions//RA/ExText')
            self.assertEqual((first.get('Id'), first.get('ParentId')),
                             ('first', 'first-parent'))
            self.assertEqual((second.get('Id'), second.get('ParentId')),
                             ('second', None))

    def test_fraus_v2_keeps_equal_translations_for_distinct_options(self):
        transform = FrausV2XmlTransform()
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, 'input.xml')
            prepared = os.path.join(directory, 'prepared.xml')
            translated = os.path.join(directory, 'translated.xml')
            restored = os.path.join(directory, 'restored.xml')
            with open(source, 'w', encoding='utf-8') as file:
                file.write('''<DOC><Questions><Question><RA>
<ExText ParentId="parent">Choose </ExText>
<InputOption Id="input"><SelectOption ParentId="correct" Correct="true"><ExText>one</ExText></SelectOption><SelectOption ParentId="wrong"><ExText>two</ExText></SelectOption></InputOption>
</RA></Question></Questions></DOC>''')
            transform.preprocess(source, prepared)
            tree = ET.parse(prepared)
            for payload in tree.findall('.//Questions//RA/ExText'):
                for marker in payload.iter('g'):
                    marker.text = 'same-target-form'
            tree.write(translated, encoding='utf-8', xml_declaration=True)
            transform.postprocess(translated, restored)
            values = [x.text for x in ET.parse(restored).findall('.//InputOption/SelectOption/ExText')]
            self.assertEqual(values, ['same-target-form', 'same-target-form'])

    def test_fraus_v2_maps_options_without_unique_parent_ids(self):
        transform = FrausV2XmlTransform()
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, 'input.xml')
            prepared = os.path.join(directory, 'prepared.xml')
            restored = os.path.join(directory, 'restored.xml')
            with open(source, 'w', encoding='utf-8') as file:
                file.write('''<DOC><Questions><Question><RA>
<ExText>Choose </ExText>
<InputOption><SelectOption><ExText>a</ExText></SelectOption><SelectOption><ExText>b</ExText></SelectOption></InputOption>
<InputOption><SelectOption ParentId="duplicate"><ExText>c</ExText></SelectOption><SelectOption ParentId="duplicate"><ExText>d</ExText></SelectOption></InputOption>
</RA></Question></Questions></DOC>''')

            transform.preprocess(source, prepared)
            tree = ET.parse(prepared)
            for variant_index, payload in enumerate(
                    tree.findall('.//Questions//RA/ExText')):
                for option_index, marker in enumerate(payload.findall('./g')):
                    marker.text = f'target-{variant_index}-{option_index}'
            tree.write(prepared, encoding='utf-8', xml_declaration=True)
            transform.postprocess(prepared, restored)

            self.assertEqual(
                [node.text for node in ET.parse(restored).findall(
                    './/InputOption/SelectOption/ExText')],
                ['target-0-0', 'target-1-0', 'target-0-1', 'target-1-1'],
            )

    def test_fraus_v2_uses_correct_then_each_wrong_choice_for_three_variants(self):
        transform = FrausV2XmlTransform()
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, 'input.xml')
            prepared = os.path.join(directory, 'prepared.xml')
            with open(source, 'w', encoding='utf-8') as file:
                file.write('''<DOC><Questions><Question><RA>
<ExText>Choose </ExText>
<InputOption Id="input"><SelectOption ParentId="wrong-1"><ExText>alpha</ExText></SelectOption><SelectOption ParentId="correct" Correct="true"><ExText>beta</ExText></SelectOption><SelectOption ParentId="wrong-2"><ExText>gamma</ExText></SelectOption></InputOption>
</RA></Question></Questions></DOC>''')
            transform.preprocess(source, prepared)
            payloads = ET.parse(prepared).findall('.//Questions//RA/ExText')
            self.assertEqual(len(payloads), 3)
            self.assertEqual(
                [next(payload.iter('g')).text for payload in payloads],
                ['beta', 'alpha', 'gamma'],
            )

    def test_fraus_v2_gives_each_correct_choice_its_own_variant(self):
        transform = FrausV2XmlTransform()
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, 'input.xml')
            prepared = os.path.join(directory, 'prepared.xml')
            with open(source, 'w', encoding='utf-8') as file:
                file.write('''<DOC><Questions><Question><RA>
<ExText>Choose </ExText>
<InputOption><SelectOption ParentId="correct-1" Correct="true"><ExText>one</ExText></SelectOption><SelectOption ParentId="wrong"><ExText>three</ExText></SelectOption><SelectOption ParentId="correct-2" Correct="true"><ExText>two</ExText></SelectOption></InputOption>
</RA></Question></Questions></DOC>''')
            transform.preprocess(source, prepared)
            payloads = ET.parse(prepared).findall('.//Questions//RA/ExText')
            self.assertEqual(
                [next(payload.iter('g')).text for payload in payloads],
                ['one', 'two', 'three'],
            )

    def test_fraus_v2_cycles_shorter_option_lists_across_variants(self):
        transform = FrausV2XmlTransform()
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, 'input.xml')
            prepared = os.path.join(directory, 'prepared.xml')
            with open(source, 'w', encoding='utf-8') as file:
                file.write('''<DOC><Questions><Question><RA>
<ExText>Choose </ExText>
<InputOption><SelectOption Correct="true"><ExText>a</ExText></SelectOption><SelectOption><ExText>b</ExText></SelectOption></InputOption>
<InputOption><SelectOption Correct="true"><ExText>c</ExText></SelectOption><SelectOption><ExText>d</ExText></SelectOption><SelectOption><ExText>e</ExText></SelectOption><SelectOption><ExText>f</ExText></SelectOption></InputOption>
</RA></Question></Questions></DOC>''')
            transform.preprocess(source, prepared)
            payloads = ET.parse(prepared).findall('.//Questions//RA/ExText')
            values = [[marker.text for marker in payload.iter('g')] for payload in payloads]
            self.assertEqual(values, [['a', 'c'], ['b', 'd'], ['a', 'e'], ['b', 'f']])

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

    def test_sanitize_generated_markup_matches_tag_names_only(self):
        source = 'Hello <g id="source">world</g> &lt;br/&gt;.'
        target = '<p>Привіт <g id="target">світе</g> &amp;lt;br/&amp;gt;.</p>'
        self.assertEqual(
            sanitize_generated_markup(source, target),
            'Привіт <g id="target">світе</g> &amp;lt;br/&amp;gt;.',
        )

    def test_sanitize_generated_markup_removes_double_escaped_unknown_tags(self):
        source = 'Text &amp;lt;g id="source"&amp;gt;word&amp;lt;/g&amp;gt;.'
        target = '&amp;lt;p&amp;gt;Text &amp;lt;g id="target"&amp;gt;word&amp;lt;/g&amp;gt;.&amp;lt;/p&amp;gt;'
        self.assertEqual(
            sanitize_generated_markup(source, target),
            'Text &amp;lt;g id="target"&amp;gt;word&amp;lt;/g&amp;gt;.',
        )


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
    @staticmethod
    def make_upload(filename, content):
        class Upload:
            def __init__(self):
                self.filename = filename

            def save(self, path):
                with open(path, 'wb') as file:
                    file.write(content)

        return Upload()

    def make_input(self, directory):
        path = os.path.join(directory, 'input.xml')
        with open(path, 'w', encoding='utf-8') as file:
            file.write('<?xml version="1.0" encoding="utf-16"?>\nHello\n')
        return path

    def test_upload_base_directories_are_private_and_unique(self):
        with tempfile.TemporaryDirectory() as parent:
            first = _create_upload_folder(parent)
            second = _create_upload_folder(parent)

            self.assertNotEqual(first, second)
            prefix = f'lindat-translation-{getattr(os, "getuid", os.getpid)()}-'
            self.assertTrue(os.path.basename(first).startswith(prefix))
            self.assertEqual(stat.S_IMODE(os.stat(first).st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(os.stat(second).st_mode), 0o700)

    def test_upload_base_cleanup_only_runs_in_owner_process(self):
        with tempfile.TemporaryDirectory() as parent:
            path = _create_upload_folder(parent)
            owner_pid = os.getpid()

            with patch('app.settings.os.getpid', return_value=owner_pid + 1):
                _cleanup_upload_folder(path, owner_pid)
            self.assertTrue(os.path.isdir(path))

            _cleanup_upload_folder(path, owner_pid)
            self.assertFalse(os.path.exists(path))

    def test_document_uploads_with_same_name_use_isolated_directories(self):
        with tempfile.TemporaryDirectory() as directory, patch(
                'app.main.document.UPLOAD_FOLDER', directory):
            first = Document.from_file(self.make_upload('same.xml', b'first'))
            second = Document.from_file(self.make_upload('same.xml', b'second'))

            self.assertNotEqual(first._work_dir, second._work_dir)
            self.assertNotEqual(first.orig_full_path, second.orig_full_path)
            self.assertEqual(stat.S_IMODE(os.stat(first._work_dir).st_mode),
                             0o700)
            self.assertEqual(stat.S_IMODE(os.stat(second._work_dir).st_mode),
                             0o700)
            self.assertEqual(os.path.basename(first.orig_full_path), 'same.xml')
            self.assertEqual(os.path.basename(second.orig_full_path), 'same.xml')
            self.assertEqual(first._input_file_name, 'same.xml')
            self.assertEqual(second._input_file_name, 'same.xml')
            with open(first.orig_full_path, 'rb') as file:
                self.assertEqual(file.read(), b'first')
            with open(second.orig_full_path, 'rb') as file:
                self.assertEqual(file.read(), b'second')

            first._cleanup_work_dir()
            self.assertFalse(os.path.exists(first._work_dir))
            self.assertTrue(os.path.exists(second._work_dir))
            second._cleanup_work_dir()

    def test_document_translation_failure_removes_only_its_work_directory(self):
        with tempfile.TemporaryDirectory() as directory, patch(
                'app.main.document.UPLOAD_FOLDER', directory):
            failed = Document.from_file(self.make_upload('same.xml', b'failed'))
            other = Document.from_file(self.make_upload('same.xml', b'other'))

            with patch.object(
                    failed, '_extract_translate_merge',
                    side_effect=RuntimeError('translation failed')):
                with self.assertRaisesRegex(RuntimeError, 'translation failed'):
                    failed.translate_from_to('cs', 'uk')

            self.assertFalse(os.path.exists(failed._work_dir))
            self.assertTrue(os.path.exists(other.orig_full_path))
            other._cleanup_work_dir()

    def test_document_response_keeps_public_filename_and_cleans_work_directory(self):
        app = Flask(__name__)
        with tempfile.TemporaryDirectory() as directory, patch(
                'app.main.document.UPLOAD_FOLDER', directory):
            document = Document.from_file(self.make_upload('exercise.xml', b'input'))
            document.translated_path = document.get_translated_path('uk')
            with open(document.translated_path, 'wb') as file:
                file.write(b'translated')

            with app.test_request_context('/'):
                response = document.create_response({})
                self.assertEqual(response.headers['X-Billing-Filename'], 'exercise.xml')
                self.assertIn('exercise.uk.xml', response.headers['Content-Disposition'])
                self.assertFalse(os.path.exists(document._work_dir))
                response.close()

            self.assertFalse(os.path.exists(document._work_dir))

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

    def test_fraus_v2_preprocesses_and_postprocesses_the_document(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, 'input.xml')
            with open(source, 'w', encoding='utf-8') as file:
                file.write('<DOC><Questions><Question><RA><ExText Id="text">Hello</ExText><InputOption Id="input"><SelectOption><ExText>one</ExText></SelectOption></InputOption></RA></Question></Questions></DOC>')
            output = os.path.join(directory, 'input.cs.xml')
            calls = []
            pipeline = DocumentPipeline(
                FrausDocumentFormat(
                    TikalRunner('/tikal/', run=fake_tikal(calls)),
                    'xml-profile', 'html-profile', FrausV2XmlTransform(),
                )
            )
            result = pipeline.run(source, output, 'en', 'cs',
                                  lambda text: 'Ahoj\n')
            self.assertEqual(result.text, 'Ahoj\n')
            self.assertTrue(os.path.exists(output))
            self.assertFalse(os.path.exists(source + '.preprocessed'))

    def test_fraus_v2_failure_restores_source_and_retries_legacy_pipeline(self):
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, 'input.xml')
            original = b'<DOC><RA><ExText>Original</ExText></RA></DOC>'
            with open(source, 'wb') as file:
                file.write(original)
            document = Document(source)
            calls = []

            def run(document_format, src, tgt, method, model,
                    custom_prompt, terms, split):
                calls.append(document_format)
                if len(calls) == 1:
                    os.remove(source)
                    with open(document.get_translated_path(tgt), 'w', encoding='utf-8') as file:
                        file.write('partial')
                    raise TikalError('merge failed')
                self.assertTrue(os.path.exists(source))
                with open(source, 'rb') as file:
                    self.assertEqual(file.read(), original)
                self.assertFalse(os.path.exists(document.get_translated_path(tgt)))

            with patch.object(
                    text_input_with_src_tgt, 'parse_args',
                    return_value={'xmlTransform': 'fraus_v2'}), patch.object(
                    document, '_run_document_pipeline', side_effect=run):
                document._extract_translate_merge_fraus(
                    'cs', 'uk', 'from_to', None
                )

            self.assertEqual(len(calls), 2)
            self.assertIsInstance(calls[0].xml_transform, FrausV2XmlTransform)
            self.assertIsNone(calls[1].xml_transform)
            self.assertIsNone(document.xml_transform)
            self.assertEqual(document._fallback_diagnostics, [{
                'type': 'document_pipeline_fallback',
                'strategy': 'legacy_fraus',
                'reason': 'TikalError',
            }])

    def test_document_adds_fallback_details_only_to_debug_trace(self):
        app = Flask(__name__)
        document = Document('/tmp/input.xml')
        document._fallback_diagnostics = [{
            'type': 'option_retry',
            'strategy': 'standalone',
        }]
        result = SimpleNamespace(text='translated', trace={'translate': 'translated'})
        document_format = StandardDocumentFormat(TikalRunner('/tikal/'))

        with app.test_request_context('/?debug=true'), patch.object(
                DocumentPipeline, 'run', return_value=result):
            get_request_llm_state().add([LLMSegmentRecord(
                segment='0',
                estimated_tokens=2,
                translated=False,
                error=LLMBackendUnavailable('unavailable'),
            )])
            document._run_document_pipeline(
                document_format, 'cs', 'uk', 'from_to', None, None, None, True
            )
            self.assertEqual(document.debug_trace['fallbacks'],
                             document._fallback_diagnostics)
            self.assertEqual(document.debug_trace['llm_fallbacks'], [{
                'segment': '0',
                'estimated_tokens': 2,
                'strategy': 'original_source',
                'resplit_depth': 0,
                'error': 'LLMBackendUnavailable',
                'status': 503,
            }])

        with app.test_request_context('/'), patch.object(
                DocumentPipeline, 'run', return_value=result):
            document._run_document_pipeline(
                document_format, 'cs', 'uk', 'from_to', None, None, None, True
            )
            self.assertNotIn('fallbacks', document.debug_trace)

    def test_fraus_v2_profile_keeps_contextual_g_tags_inline(self):
        profile = open('app/okapi_profiles/okf_html@fraus.fprm', encoding='utf-8').read()
        self.assertIn('  g:\n    ruleTypes: [INLINE]', profile)

    def test_fraus_v2_restores_space_between_option_and_following_text(self):
        transform = FrausV2XmlTransform()
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, 'input.xml')
            prepared = os.path.join(directory, 'prepared.xml')
            restored = os.path.join(directory, 'restored.xml')
            with open(source, 'w', encoding='utf-8') as file:
                file.write('<DOC><Questions><Question><RA><ExText Id="before" ParentId="parent">ve</ExText><InputOption Id="input"><SelectOption><ExText>dvou</ExText></SelectOption></InputOption><ExText Id="after">sportovkyně.</ExText></RA></Question></Questions></DOC>')
            transform.preprocess(source, prepared)
            transform.postprocess(prepared, restored)
            self.assertEqual(ET.parse(restored).find('.//InputOption/SelectOption/ExText').text, 'dvou')
            self.assertEqual(len(ET.parse(restored).findall('.//Questions//RA')), 1)

    def test_fraus_v2_preserves_non_text_ra_children_in_place(self):
        transform = FrausV2XmlTransform()
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, 'input.xml')
            prepared = os.path.join(directory, 'prepared.xml')
            restored = os.path.join(directory, 'restored.xml')
            with open(source, 'w', encoding='utf-8') as file:
                file.write('''<DOC><Questions><Question><RA>
<ExText>Before </ExText><Equation Id="equation"/><LineBreak Id="break"/>
<InputOption><SelectOption ParentId="choice" Correct="true"><ExText>one</ExText></SelectOption></InputOption>
<Image Id="image"/><ExText> after.</ExText>
</RA></Question></Questions></DOC>''')
            transform.preprocess(source, prepared)
            transform.postprocess(prepared, restored)
            ra = ET.parse(restored).find('.//Questions//RA')
            self.assertEqual(
                [(child.tag, child.get('Id')) for child in ra],
                [('ExText', ra[0].get('Id')), ('Equation', 'equation'),
                 ('LineBreak', 'break'), ('InputOption', None),
                 ('Image', 'image'), ('ExText', ra[-1].get('Id'))],
            )

    def test_fraus_v2_preserves_original_option_when_translation_is_empty(self):
        transform = FrausV2XmlTransform()
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, 'input.xml')
            prepared = os.path.join(directory, 'prepared.xml')
            restored = os.path.join(directory, 'restored.xml')
            with open(source, 'w', encoding='utf-8') as file:
                file.write('''<DOC><Questions><Question><RA>
<ExText>Choose </ExText><InputOption><SelectOption ParentId="choice"><ExText>original</ExText></SelectOption></InputOption>
</RA></Question></Questions></DOC>''')
            transform.preprocess(source, prepared)
            tree = ET.parse(prepared)
            tree.find('.//Questions//RA/ExText/g').text = ''
            tree.write(prepared, encoding='utf-8', xml_declaration=True)
            transform.postprocess(prepared, restored)
            self.assertEqual(
                ET.parse(restored).findtext('.//InputOption/SelectOption/ExText'),
                'original',
            )

    def test_fraus_v2_preserves_translated_documents_without_option_ras(self):
        transform = FrausV2XmlTransform()
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, 'input.xml')
            translated = os.path.join(directory, 'translated.xml')
            restored = os.path.join(directory, 'restored.xml')
            with open(source, 'w', encoding='utf-8') as file:
                file.write('<DOC><RA><ExText>Original</ExText></RA></DOC>')
            with open(translated, 'w', encoding='utf-8') as file:
                file.write('<DOC><RA><ExText>Translated</ExText></RA></DOC>')
            transform.preprocess(source, os.path.join(directory, 'prepared.xml'))
            transform.postprocess(translated, restored)
            self.assertEqual(ET.parse(restored).find('.//ExText').text, 'Translated')

    def test_fraus_v2_skips_ras_with_empty_input_options(self):
        transform = FrausV2XmlTransform()
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, 'input.xml')
            prepared = os.path.join(directory, 'prepared.xml')
            with open(source, 'w', encoding='utf-8') as file:
                file.write('<DOC><Questions><Question><RA><ExText>Context</ExText><InputOption Id="empty"/></RA></Question></Questions></DOC>')
            transform.preprocess(source, prepared)
            root = ET.parse(prepared).getroot()
            self.assertEqual(len(root.findall('.//Questions//RA')), 1)
            self.assertIsNone(root.find('.//Questions//RA').get('{urn:lindat:fraus-v2}variant'))

    def test_fraus_v2_preserves_anonymized_non_inline_document_shapes(self):
        fixture_dir = os.path.join(os.path.dirname(__file__), 'fixtures', 'fraus')
        names = [
            'no_questions_items.xml',
            'questions_options.xml',
            'matching_items.xml',
            'options_only.xml',
            'marking_text.xml',
        ]
        for name in names:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                source = os.path.join(fixture_dir, name)
                prepared = os.path.join(directory, 'prepared.xml')
                translated = os.path.join(directory, 'translated.xml')
                restored = os.path.join(directory, 'restored.xml')
                transform = FrausV2XmlTransform()
                transform.preprocess(source, prepared)
                translated_text = open(source, encoding='utf-8').read().replace(
                    'Event', 'TranslatedEvent'
                ).replace('Choose', 'TranslatedChoose').replace(
                    'Match', 'TranslatedMatch'
                ).replace('Mark', 'TranslatedMark')
                with open(translated, 'w', encoding='utf-8') as file:
                    file.write(translated_text)
                transform.postprocess(translated, restored)
                output = ET.parse(restored).getroot()
                self.assertEqual(len(list(output.iter('RA'))), len(list(ET.parse(source).getroot().iter('RA'))))
                self.assertEqual(len(list(output.iter('ExText'))), len(list(ET.parse(source).getroot().iter('ExText'))))
                self.assertFalse(any('fraus' in str(element.tag) for element in output.iter()))

    def test_fraus_v2_ignores_outer_nested_option_markers(self):
        transform = FrausV2XmlTransform()
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, 'input.xml')
            prepared = os.path.join(directory, 'prepared.xml')
            translated = os.path.join(directory, 'translated.xml')
            restored = os.path.join(directory, 'restored.xml')
            with open(source, 'w', encoding='utf-8') as file:
                file.write('''<DOC><Questions><Question><RA>
<ExText ParentId="parent">Before </ExText>
<InputOption Id="first"><SelectOption ParentId="first-select" Correct="true"><ExText>one</ExText></SelectOption></InputOption>
<ExText> after </ExText>
<InputOption Id="second"><SelectOption ParentId="second-select" Correct="true"><ExText>two</ExText></SelectOption></InputOption>
</RA></Question></Questions></DOC>''')
            transform.preprocess(source, prepared)
            tree = ET.parse(prepared)
            payload = tree.find('.//Questions//RA/ExText')
            first, second = list(payload)
            first.text = 'context '
            second.text = 'translated-two'
            payload.remove(second)
            first.append(second)
            tree.write(translated, encoding='utf-8', xml_declaration=True)
            transform.postprocess(translated, restored)
            values = [x.text for x in ET.parse(restored).findall('.//InputOption/SelectOption/ExText')]
            self.assertEqual(values, ['one', 'translated-two'])

    def test_fraus_v2_fallback_retries_variants_when_line_counts_change(self):
        transform = FrausV2XmlTransform()
        transform.variant_sequence = [['select-1'], ['select-2']]
        source = '<g id="1">one</g>\n<g id="1">two</g>\n'
        translated = '<g id="1"><g id="2">one target</g></g>\n'
        calls = []

        def translate_one(text):
            calls.append(text)
            return '<g id="1">clean target</g>\n'

        result = transform.fallback(source, translated, translate_one)
        self.assertEqual(result, source)
        self.assertEqual(len(calls), 2)
        self.assertEqual(transform.fallback_values, {
            'select-1': 'clean target',
            'select-2': 'clean target',
        })

    def test_fraus_v2_fallback_uses_standalone_option_translation(self):
        transform = FrausV2XmlTransform()
        transform.variant_sequence = [['select-1']]
        source = '<g id="1">one</g>\n'
        translated = '<g id="1"><g id="2">contaminated context</g></g>\n'
        calls = []

        def translate_one(text):
            calls.append(text)
            if '<g' in text:
                return '<g id="1"><g id="2">too much context</g></g>\n'
            return 'standalone target\n'

        result = transform.fallback(source, translated, translate_one)
        self.assertEqual(result, source)
        self.assertEqual(transform.fallback_values['select-1'], 'standalone target')
        self.assertEqual(calls, ['<g id="1">one</g>\n', 'one\n'])
        self.assertEqual(transform.fallback_diagnostics[-1]['strategy'],
                         'standalone')

    def test_fraus_v2_fallback_rejects_excessive_standalone_translation(self):
        transform = FrausV2XmlTransform()
        transform.variant_sequence = [['select-1']]

        def translate_one(text):
            if '<g' in text:
                return '<g id="1"><g id="2">contaminated</g></g>\n'
            return 'one two three four five six seven eight nine ten\n'

        transform.fallback(
            '<g id="1">one</g>\n',
            '<g id="1"><g id="2">contaminated</g></g>\n',
            translate_one,
        )
        self.assertIsNone(transform.fallback_values['select-1'])

    def test_fraus_v2_fallback_retries_missing_or_extra_flat_markers(self):
        source = '<g id="first">one</g> and <g id="second">two</g>\n'
        targets = [
            '<g id="first">uno</g> and two\n',
            '<g id="first">uno</g> and <g id="second">dos</g> <g id="extra">extra</g>\n',
        ]
        for target in targets:
            with self.subTest(target=target):
                transform = FrausV2XmlTransform()
                transform.variant_sequence = [['select-1', 'select-2']]
                calls = []

                def translate_one(text):
                    calls.append(text)
                    marker_id = 'first' if 'first' in text else 'second'
                    return f'<g id="{marker_id}">clean target</g>\n'

                transform.fallback(source, target, translate_one)
                self.assertEqual(len(calls), 2)
                self.assertEqual(transform.fallback_values, {
                    'select-1': 'clean target',
                    'select-2': 'clean target',
                })

    def test_fraus_v2_fallback_retries_empty_markers(self):
        transform = FrausV2XmlTransform()
        transform.variant_sequence = [['select-1']]
        calls = []

        def translate_one(text):
            calls.append(text)
            return '<g id="first">clean target</g>\n'

        transform.fallback(
            '<g id="first">one</g>\n',
            '<g id="first"></g>\n',
            translate_one,
        )
        self.assertEqual(calls, ['<g id="first">one</g>\n'])
        self.assertEqual(transform.fallback_values['select-1'], 'clean target')

    def test_fraus_v2_fallback_ignores_consecutive_text_markers(self):
        transform = FrausV2XmlTransform()
        transform.variant_sequence = [['choice']]
        source = (
            'First<g id="fraus-text-1">Second</g>'
            '<g id="fraus-option-choice">one</g>\n'
        )
        calls = []

        def translate_one(text):
            calls.append(text)
            return '<g id="fraus-option-choice">clean target</g>\n'

        transform.fallback(
            source,
            '<g id="fraus-text-1">First Second</g>'
            '<g id="fraus-option-choice"><g id="nested">bad</g></g>\n',
            translate_one,
        )

        self.assertEqual(len(calls), 1)
        self.assertIn('<g id="fraus-option-choice">one</g>', calls[0])
        self.assertEqual(transform.fallback_values['choice'], 'clean target')

    def test_fraus_v2_fallback_rejects_markup_absorbed_by_option(self):
        transform = FrausV2XmlTransform()
        transform.variant_sequence = [['select-1']]
        calls = []

        def translate_one(text):
            calls.append(text)
            if '<g' in text:
                return '<g id="first">target<br/>context</g>\n'
            return 'standalone target\n'

        transform.fallback(
            '<g id="first">one</g> context<br/>\n',
            '<g id="first">target<br/>context</g>\n',
            translate_one,
        )
        self.assertEqual(transform.fallback_values['select-1'], 'standalone target')
        self.assertEqual(calls, [
            '<g id="first">one</g> context\n',
            'one\n',
        ])

    def test_fraus_v2_retry_escapes_literal_comparison_symbols(self):
        retry = FrausV2XmlTransform._keep_one_marker(
            '<g id="first">25 321 &lt;</g> 52 213 &gt; 24 695\n', 0
        )
        self.assertEqual(
            retry,
            '<g id="first">25 321 &lt;</g> 52 213 &gt; 24 695\n',
        )
        ET.fromstring(f'<root>{retry}</root>')

    def test_fraus_v2_preserves_ambiguous_source_markup(self):
        transform = FrausV2XmlTransform()
        transform.variant_sequence = [['select-1']]
        source = '<g id="format"><g id="option">one</g></g>\n'
        result = transform.fallback(
            source,
            '<g id="format"><g id="option"><g id="extra">target</g></g></g>\n',
            lambda text: text,
        )
        self.assertEqual(result, source)
        self.assertIsNone(transform.fallback_values['select-1'])

    def test_fraus_v2_can_force_sentence_boundaries(self):
        transform = FrausV2XmlTransform(force_sentence_level=True)
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, 'input.xml')
            prepared = os.path.join(directory, 'prepared.xml')
            with open(source, 'w', encoding='utf-8') as file:
                file.write('<DOC><Questions><Question><RA><ExText Id="before">First sentence. Second sentence.</ExText><InputOption Id="input"><SelectOption><ExText>one</ExText></SelectOption></InputOption></RA></Question></Questions></DOC>')
            transform.preprocess(source, prepared)
            self.assertIn('\n', ET.parse(prepared).find('.//Questions//RA/ExText').text)

    def test_fraus_v2_uses_sentence_splitter_prefix_rules(self):
        transform = FrausV2XmlTransform(force_sentence_level=True)
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, 'input.xml')
            prepared = os.path.join(directory, 'prepared.xml')
            with open(source, 'w', encoding='utf-8') as file:
                file.write('<DOC><Questions><Question><RA><ExText Id="before">Dr. Novák přijel. To je test.</ExText><InputOption Id="input"><SelectOption><ExText>one</ExText></SelectOption></InputOption></RA></Question></Questions></DOC>')
            transform.preprocess(source, prepared)
            text = ET.parse(prepared).find('.//Questions//RA/ExText').text
            self.assertIn('Dr. Novák přijel.\nTo je test.', text)

    def test_fraus_v2_packs_sentences_under_max_tokens(self):
        transform = FrausV2XmlTransform(force_sentence_level=True, max_segment_tokens=3)
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, 'input.xml')
            prepared = os.path.join(directory, 'prepared.xml')
            with open(source, 'w', encoding='utf-8') as file:
                file.write('<DOC><Questions><Question><RA><ExText Id="before">One sentence. Two sentence.</ExText><InputOption Id="input"><SelectOption><ExText>one</ExText></SelectOption></InputOption></RA></Question></Questions></DOC>')
            transform.preprocess(source, prepared)
            text = ET.parse(prepared).find('.//Questions//RA/ExText').text
            self.assertIn('One sentence.\nTwo sentence.', text)

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
