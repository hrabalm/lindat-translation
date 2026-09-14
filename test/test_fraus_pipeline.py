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
    FrausTranslationError,
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
                    if marker.get('id', '').startswith('fraus-option-'):
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
            values = [[
                marker.text for marker in payload.iter('g')
                if marker.get('id', '').startswith('fraus-option-')
            ] for payload in payloads]
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

    def test_fraus_v2_failure_propagates_without_legacy_retry(self):
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
                raise TikalError('merge failed')

            with patch.object(
                    text_input_with_src_tgt, 'parse_args',
                    return_value={'xmlTransform': 'fraus_v2'}), patch.object(
                    document, '_run_document_pipeline', side_effect=run):
                with self.assertRaisesRegex(TikalError, 'merge failed'):
                    document._extract_translate_merge_fraus(
                        'cs', 'uk', 'from_to', None
                    )

            self.assertEqual(len(calls), 1)
            self.assertIsInstance(calls[0].xml_transform, FrausV2XmlTransform)
            self.assertIs(document.xml_transform, calls[0].xml_transform)
            self.assertEqual(document._fallback_diagnostics, [])

    def test_document_retains_transform_diagnostics_when_recovery_fails(self):
        app = Flask(__name__)
        document = Document('/tmp/input.xml')
        document.text = 'source\n'

        class FailingTransform:
            fallback_diagnostics = [{
                'type': 'unsafe_context_variant',
                'variant_index': 0,
            }]

            def fallback(self, source, target, translate):
                raise FrausTranslationError('context recovery produced no usable translation')

        document.xml_transform = FailingTransform()
        with app.test_request_context('/'), patch(
                'app.main.document.translate_with_line_fallback',
                return_value='target\n'):
            with self.assertRaisesRegex(FrausTranslationError, 'context recovery'):
                document._translate('cs', 'en', 'from_to')

        self.assertEqual(document._fallback_diagnostics, [{
            'type': 'unsafe_context_variant',
            'variant_index': 0,
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

    def test_fraus_v2_postprocess_rejects_empty_option_without_source_copy(self):
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
            with self.assertRaises(FrausTranslationError):
                transform.postprocess(prepared, restored)
            self.assertFalse(os.path.exists(restored))

    def test_fraus_v2_postprocess_rejects_missing_context_without_source_copy(self):
        for missing in ('before', 'after', 'payload'):
            with self.subTest(missing=missing), tempfile.TemporaryDirectory() as directory:
                transform = FrausV2XmlTransform()
                source = os.path.join(directory, 'input.xml')
                prepared = os.path.join(directory, 'prepared.xml')
                restored = os.path.join(directory, 'restored.xml')
                with open(source, 'w', encoding='utf-8') as file:
                    file.write('''<DOC><Questions><Question><RA>
<ExText Id="before">Before </ExText>
<InputOption><SelectOption><ExText>one</ExText></SelectOption></InputOption>
<ExText Id="after"> after.</ExText>
</RA></Question></Questions></DOC>''')
                transform.preprocess(source, prepared)
                tree = ET.parse(prepared)
                ra = tree.find('.//Questions//RA')
                payload = ra.find('ExText')
                marker = payload.find('g')
                payload.text = 'Translated before '
                marker.text = 'accepted option'
                marker.tail = ' translated after.'
                if missing == 'before':
                    payload.text = None
                elif missing == 'after':
                    marker.tail = None
                else:
                    ra.remove(payload)
                tree.write(prepared, encoding='utf-8', xml_declaration=True)
                with self.assertRaises(FrausTranslationError):
                    transform.postprocess(prepared, restored)
                self.assertFalse(os.path.exists(restored))

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

    def test_fraus_v2_postprocess_rejects_missing_outer_nested_option(self):
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
            first, second = [
                marker for marker in payload
                if marker.get('id', '').startswith('fraus-option-')
            ]
            first.text = 'context '
            second.text = 'translated-two'
            payload.remove(second)
            first.append(second)
            tree.write(translated, encoding='utf-8', xml_declaration=True)
            with self.assertRaises(FrausTranslationError):
                transform.postprocess(translated, restored)
            self.assertFalse(os.path.exists(restored))

    def test_fraus_v2_fallback_rejects_variants_when_line_counts_change(self):
        transform = FrausV2XmlTransform()
        transform.variant_sequence = [['select-1'], ['select-2']]
        source = ('<g id="fraus-option-select-1">one</g>\n'
                  '<g id="fraus-option-select-2">two</g>\n')
        translated = ('<g id="fraus-option-select-1">'
                      '<g id="nested">one target</g></g>\n')
        with self.assertRaisesRegex(AssertionError, 'line structure'):
            transform.fallback(source, translated, lambda text: text)

    def test_fraus_v2_fallback_uses_standalone_option_translation(self):
        transform = FrausV2XmlTransform()
        transform.variant_sequence = [['select-1']]
        source = '<g id="fraus-option-select-1">one</g>\n'
        translated = ('<g id="fraus-option-select-1">'
                      '<g id="nested">contaminated context</g></g>\n')
        calls = []

        def translate_one(text):
            calls.append(text)
            return 'standalone target\n'

        result = transform.fallback(source, translated, translate_one)
        self.assertEqual(result, '<g id="fraus-option-select-1">standalone target</g>\n')
        self.assertEqual(transform.fallback_values['select-1'], 'standalone target')
        self.assertEqual(calls, [
            'one\n'
        ])
        self.assertEqual(
            [item['strategy'] for item in transform.fallback_diagnostics
             if item['type'] == 'option_retry'], ['standalone'],
        )

    def test_fraus_v2_fallback_isolates_option_with_wrong_marker_id(self):
        transform = FrausV2XmlTransform()
        transform.variant_sequence = [['select-1']]

        calls = []

        def translate_one(text):
            calls.append(text)
            return 'standalone target\n'

        transform.fallback(
            '<g id="fraus-option-select-1">one</g>\n',
            '<g id="wrong">wrong target</g>\n',
            translate_one,
        )

        self.assertEqual(transform.fallback_values['select-1'], 'standalone target')
        self.assertEqual(calls, ['one\n'])
        self.assertEqual(
            [item['strategy'] for item in transform.fallback_diagnostics
             if item['type'] == 'option_retry'], ['standalone'],
        )

    def test_fraus_v2_fallback_accepts_long_plain_option_translation(self):
        transform = FrausV2XmlTransform()
        transform.variant_sequence = [['select-1']]

        value = 'one two three four five six seven eight nine ten'
        result = transform.fallback(
            '<g id="fraus-option-select-1">one</g>\n',
            '<g id="fraus-option-select-1"><g id="nested">contaminated</g></g>\n',
            lambda text: value + '\n',
        )
        self.assertEqual(result, f'<g id="fraus-option-select-1">{value}</g>\n')
        self.assertEqual(transform.fallback_values['select-1'], value)

    def test_fraus_v2_fallback_accepts_unchanged_multiword_option(self):
        transform = FrausV2XmlTransform(language='cs')
        transform.variant_sequence = [['select-1']]

        transform.fallback(
            '<g id="fraus-option-select-1">Los Angeles</g>\n',
            '<g id="fraus-option-select-1"><g id="nested">bad</g></g>\n',
            lambda text: text,
        )

        self.assertEqual(transform.fallback_values['select-1'], 'Los Angeles')

    def test_fraus_v2_fallback_allows_unchanged_measurement_unit(self):
        transform = FrausV2XmlTransform(language='cs')
        transform.variant_sequence = [['select-1']]

        transform.fallback(
            '<g id="fraus-option-select-1">km</g>\n',
            '<g id="fraus-option-select-1"><g id="nested">bad</g></g>\n',
            lambda text: text,
        )

        self.assertEqual(transform.fallback_values['select-1'], 'km')

    def test_fraus_v2_fallback_retains_valid_options_with_missing_or_extra_tags(self):
        source = ('<g id="fraus-option-select-1">one</g> and '
                  '<g id="fraus-option-select-2">two</g>\n')
        targets = [
            '<g id="fraus-option-select-1">uno</g> and two\n',
            ('<g id="fraus-option-select-1">uno</g> and '
             '<g id="fraus-option-select-2">dos</g> '
             '<g id="fraus-option-extra">extra</g>\n'),
        ]
        for target in targets:
            with self.subTest(target=target):
                transform = FrausV2XmlTransform()
                transform.variant_sequence = [['select-1', 'select-2']]
                calls = []

                def translate_one(text):
                    calls.append(text)
                    if '__BLANK__' in text:
                        return text.replace('and', 'and target')
                    self.assertEqual(text, 'two\n')
                    return 'clean target\n'

                result = transform.fallback(source, target, translate_one)
                missing = 'fraus-option-select-2' not in target
                self.assertEqual(calls, (['two\n'] if missing else []) + [
                    '__BLANK__ and __BLANK__\n',
                ])
                root = ET.fromstring(f'<root>{result}</root>')
                self.assertEqual(
                    [(node.get('id'), node.text) for node in root],
                    [('fraus-option-select-1', 'uno'),
                     ('fraus-option-select-2', 'clean target' if missing else 'dos')],
                )
                self.assertEqual(root[0].tail, ' and target ')

    def test_fraus_v2_fallback_retries_empty_markers(self):
        transform = FrausV2XmlTransform()
        transform.variant_sequence = [['select-1']]
        calls = []

        def translate_one(text):
            calls.append(text)
            return 'clean target\n'

        transform.fallback(
            '<g id="fraus-option-select-1">one</g>\n',
            '<g id="fraus-option-select-1"></g>\n',
            translate_one,
        )
        self.assertEqual(calls, [
            'one\n'
        ])
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
            if '__BLANK__' in text or '__PLACEHOLDER__' in text:
                return text.replace('First', 'First target').replace(
                    'Second', 'Second target'
                )
            return {'one\n': 'clean target\n',
                    'First\n': 'First target\n',
                    'Second\n': 'Second target\n'}[text]

        result = transform.fallback(
            source,
            '<g id="fraus-text-1">First Second</g>'
            '<g id="fraus-option-choice"><g id="nested">bad</g></g>\n',
            translate_one,
        )

        self.assertEqual(calls, [
            'one\n', 'FirstSecond__BLANK__\n',
            'FirstSecond__PLACEHOLDER__\n', 'First\n', 'Second\n',
        ])
        self.assertEqual(result, 'First target<g id="fraus-text-1">Second target</g>'
                         '<g id="fraus-option-choice">clean target</g>\n')
        self.assertEqual(transform.fallback_values['choice'], 'clean target')

    def test_fraus_v2_fallback_rejects_markup_absorbed_by_option(self):
        transform = FrausV2XmlTransform()
        transform.variant_sequence = [['select-1']]
        calls = []

        def translate_one(text):
            calls.append(text)
            if '__BLANK__' in text:
                return text.replace('context', 'translated context')
            return 'standalone target\n'

        transform.fallback(
            '<g id="fraus-option-select-1">one</g> context<br/>\n',
            '<g id="fraus-option-select-1">target<br/>context</g>\n',
            translate_one,
        )
        self.assertEqual(transform.fallback_values['select-1'], 'standalone target')
        self.assertEqual(calls, [
            'one\n',
            '__BLANK__ context\n',
        ])

    def test_fraus_v2_fallback_preserves_literal_comparisons_and_entities(self):
        transform = FrausV2XmlTransform()
        transform.variant_sequence = [['choice']]
        calls = []

        def translate_one(text):
            calls.append(text)
            return text

        source = ('<g id="fraus-option-choice">25 321 &lt;</g>'
                  ' 52 213 &gt; 24 695 &amp; A\n')
        result = transform.fallback(source, '<g id="wrong">bad</g>\n', translate_one)
        self.assertEqual(calls, ['25 321 &lt;\n',
                                '__BLANK__ 52 213 &gt; 24 695 &amp; A\n'])
        self.assertEqual(result, source)
        root = ET.fromstring(f'<root>{result}</root>')
        self.assertEqual(root[0].text, '25 321 <')
        self.assertEqual(root[0].tail, ' 52 213 > 24 695 & A\n')

    def test_fraus_v2_ignores_unrelated_g_markup(self):
        transform = FrausV2XmlTransform()
        transform.variant_sequence = [['select-1']]
        source = '<g id="format"><g id="option">one</g></g>\n'
        translated = '<g id="format"><g id="option">target</g></g>\n'
        self.assertEqual(
            transform.fallback(source, translated, lambda text: text),
            translated,
        )
        self.assertEqual(transform.fallback_diagnostics, [])

    def test_fraus_v2_fallback_translates_source_context_segments(self):
        transform = FrausV2XmlTransform()
        transform.variant_sequence = [['choice']]
        source = (
            'Český začátek <g id="fraus-option-choice">volba</g>'
            ' český konec.\n'
        )

        calls = []

        def translate_one(text):
            calls.append(text)
            if '__BLANK__' in text or '__PLACEHOLDER__' in text:
                return 'marker removed\n'
            return {'volba\n': 'choice\n',
                    'Český začátek\n': 'English start\n',
                    'český konec.\n': 'English end.\n'}[text]

        result = transform.fallback(
            source,
            '<g id="fraus-option-choice"><g id="nested">bad</g></g>\n',
            translate_one,
        )

        self.assertEqual(
            result,
            'English start <g id="fraus-option-choice">choice</g> English end.\n',
        )
        self.assertEqual(transform.fallback_values['choice'], 'choice')
        self.assertNotIn('Český', result)
        self.assertNotIn('český', result)
        self.assertEqual(calls, [
            'volba\n', 'Český začátek __BLANK__ český konec.\n',
            'Český začátek __PLACEHOLDER__ český konec.\n',
            'Český začátek\n', 'český konec.\n',
        ])
        self.assertEqual(
            [item['strategy'] for item in transform.fallback_diagnostics
             if item['type'] == 'context_retry'],
            ['__BLANK__', '__PLACEHOLDER__', 'isolated_context', 'isolated_context'],
        )

    def test_fraus_v2_recovery_round_trips_context_and_options(self):
        transform = FrausV2XmlTransform()
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, 'input.xml')
            prepared = os.path.join(directory, 'prepared.xml')
            translated = os.path.join(directory, 'translated.xml')
            restored = os.path.join(directory, 'restored.xml')
            with open(source, 'w', encoding='utf-8') as file:
                file.write('''<DOC><Questions><Question><RA>
<ExText Id="before">Before </ExText>
<InputOption><SelectOption><ExText Id="one">one</ExText></SelectOption><SelectOption><ExText Id="two">two</ExText></SelectOption></InputOption>
<ExText Id="after"> after.</ExText>
</RA></Question></Questions></DOC>''')

            transform.preprocess(source, prepared)
            tree = ET.parse(prepared)
            payloads = tree.findall('.//Questions//RA/ExText')

            def inner_xml(payload):
                return (payload.text or '') + ''.join(
                    ET.tostring(child, encoding='unicode')
                    for child in payload
                )

            source_text = '\n'.join(inner_xml(payload) for payload in payloads) + '\n'
            bad_lines = []
            for line in source_text.splitlines():
                root = ET.fromstring(f'<root>{line}</root>')
                marker = next(
                    item for item in root.iter('g')
                    if item.get('id', '').startswith('fraus-option-')
                )
                nested = ET.SubElement(marker, 'g', {'id': 'bad'})
                nested.text, marker.text = marker.text, None
                serialized = ET.tostring(root, encoding='unicode')
                bad_lines.append(serialized[len('<root>'):-len('</root>')])
            bad_translation = '\n'.join(bad_lines) + '\n'

            def translate_one(text):
                if '__BLANK__' in text:
                    return text.replace('Before', 'Target before').replace(
                        'after.', 'target after.'
                    )
                self.assertIn(text, ['one\n', 'two\n'])
                return 'target ' + text

            recovered = transform.fallback(
                source_text, bad_translation, translate_one
            )
            for payload, line in zip(payloads, recovered.splitlines()):
                root = ET.fromstring(f'<root>{line}</root>')
                payload.text = root.text
                payload[:] = list(root)
            tree.write(translated, encoding='utf-8', xml_declaration=True)
            transform.postprocess(translated, restored)

            root = ET.parse(restored).getroot()
            self.assertEqual(
                [item.text for item in root.findall('.//Questions//RA/ExText')],
                ['Target before ', ' target after.'],
            )
            self.assertEqual(
                [item.text for item in root.findall(
                    './/InputOption/SelectOption/ExText')],
                ['target one', 'target two'],
            )
            self.assertEqual(
                [item.get('Id') for item in root.findall('.//Questions//RA/ExText')],
                ['before', 'after'],
            )
            self.assertEqual(
                [item.get('Id') for item in root.findall('.//InputOption/SelectOption/ExText')],
                ['one', 'two'],
            )

    def test_fraus_v2_fallback_rejects_empty_context(self):
        transform = FrausV2XmlTransform()
        transform.variant_sequence = [['choice']]

        def translate_one(text):
            if '__BLANK__' in text:
                return '__BLANK__\n'
            if '__PLACEHOLDER__' in text:
                return '__PLACEHOLDER__\n'
            if text == 'volba\n':
                return 'choice\n'
            return '\n'

        with self.assertRaisesRegex(FrausTranslationError, 'context recovery'):
            transform.fallback(
                'Český text <g id="fraus-option-choice">volba</g>\n',
                '<g id="fraus-option-choice"><g id="nested">bad</g></g>\n',
                translate_one,
            )

    def test_fraus_v2_fallback_accepts_unchanged_model_context(self):
        transform = FrausV2XmlTransform()
        transform.variant_sequence = [['choice']]

        def translate_one(text):
            if '__BLANK__' in text:
                return text
            if text == 'volba\n':
                return 'choice\n'
            return text

        result = transform.fallback(
            'Český text <g id="fraus-option-choice">volba</g>\n',
            '<g id="fraus-option-choice"><g id="nested">bad</g></g>\n',
            translate_one,
        )
        self.assertEqual(
            result,
            'Český text <g id="fraus-option-choice">choice</g>\n',
        )

    def test_fraus_v2_context_recovery_retries_deleted_moved_or_changed_blank(self):
        for damaged in ('Translated text with the answer\n',
                        '__BLANK__Translated text\n',
                        'Translated __ANSWER__\n'):
            with self.subTest(damaged=damaged):
                transform = FrausV2XmlTransform()
                transform.variant_sequence = [['choice']]
                calls = []

                def translate_one(text):
                    calls.append(text)
                    if '__BLANK__' in text:
                        return damaged
                    self.assertEqual(text, 'Text before __PLACEHOLDER__\n')
                    return 'Translated text __PLACEHOLDER__\n'

                result = transform.fallback(
                    'Text before <g id="fraus-option-choice">one</g>\n',
                    '<g id="fraus-option-choice">accepted option</g>\n',
                    translate_one,
                )
                self.assertEqual(calls, ['Text before __BLANK__\n',
                                         'Text before __PLACEHOLDER__\n'])
                self.assertEqual(result, 'Translated text '
                                 '<g id="fraus-option-choice">accepted option</g>\n')

    def test_fraus_v2_fallback_rejects_empty_or_markup_model_outputs(self):
        for output in ('', ' \n', '<br/>\n', '<b>invented</b>\n'):
            for stage in ('option', 'context'):
                with self.subTest(output=output, stage=stage):
                    transform = FrausV2XmlTransform()
                    transform.variant_sequence = [['choice']]
                    calls = []

                    def translate_one(text):
                        calls.append(text)
                        return output

                    source = '<g id="fraus-option-choice">one</g>\n'
                    target = '<g id="fraus-option-choice"></g>\n'
                    if stage == 'context':
                        source = 'Before ' + source
                        target = '<g id="fraus-option-choice">accepted</g>\n'
                    with self.assertRaises(FrausTranslationError):
                        transform.fallback(source, target, translate_one)
                    self.assertEqual(calls, ['one\n' if stage == 'option'
                                             else 'Before __BLANK__\n'])

    def test_fraus_v2_context_recovery_tries_second_placeholder(self):
        transform = FrausV2XmlTransform()
        transform.variant_sequence = [['first', 'second']]
        calls = []

        def translate_one(text):
            calls.append(text)
            if '__BLANK__' in text:
                return '__BLANK__ target first __BLANK__ answer\n'
            return '__PLACEHOLDER__ translated context __PLACEHOLDER__\n'

        result = transform.fallback(
            '<g id="fraus-option-first">one</g> first second '
            '<g id="fraus-option-second">two</g>\n',
            '<g id="fraus-option-first">uno</g><g id="fraus-option-second">dos</g>\n',
            translate_one,
        )

        self.assertEqual(
            result,
            '<g id="fraus-option-first">uno</g> translated context '
            '<g id="fraus-option-second">dos</g>\n',
        )
        self.assertEqual(len(calls), 2)
        self.assertEqual(
            transform.fallback_diagnostics[-1]['strategy'],
            '__PLACEHOLDER__',
        )

    def test_fraus_v2_context_recovery_unescapes_model_entities(self):
        transform = FrausV2XmlTransform()
        transform.variant_sequence = [['choice']]
        result = transform.fallback(
            'A &amp; B <g id="fraus-option-choice">one</g>\n',
            '<g id="fraus-option-choice">uno</g>\n',
            lambda text: 'C &amp; D __BLANK__\n',
        )

        self.assertEqual(result, 'C &amp; D <g id="fraus-option-choice">uno</g>\n')
        self.assertEqual(ET.fromstring(f'<root>{result}</root>').text, 'C & D ')

    def test_fraus_v2_context_recovery_uses_isolated_context(self):
        calls = []

        def translate_one(text):
            calls.append(text)
            if '__BLANK__' in text or '__PLACEHOLDER__' in text:
                return 'marker was removed\n'
            return 'unprotected target\n'

        transform = FrausV2XmlTransform()
        transform.variant_sequence = [['first', 'second']]
        result = transform.fallback(
            '<g id="fraus-option-first">one</g> source context '
            '<g id="fraus-option-second">two</g>\n',
            '<g id="fraus-option-first">uno</g><g id="fraus-option-second">dos</g>\n',
            translate_one,
        )

        self.assertEqual(
            result,
            '<g id="fraus-option-first">uno</g> unprotected target '
            '<g id="fraus-option-second">dos</g>\n',
        )
        self.assertEqual(calls, ['__BLANK__ source context __BLANK__\n',
                                 '__PLACEHOLDER__ source context __PLACEHOLDER__\n',
                                 'source context\n'])
        self.assertEqual(
            transform.fallback_diagnostics[-1]['strategy'], 'isolated_context'
        )

    def test_fraus_v2_context_recovery_caches_only_model_results(self):
        transform = FrausV2XmlTransform()
        transform.variant_sequence = [['first'], ['second']]
        source = (
            'Before <g id="fraus-option-first">one</g> after.\n'
            'Before <g id="fraus-option-second">two</g> after.\n'
        )
        translated = (
            '<g id="fraus-option-first"><g id="nested">bad</g></g>\n'
            '<g id="fraus-option-second"><g id="nested">bad</g></g>\n'
        )
        context_calls = []

        def translate_one(text):
            if '__BLANK__' in text:
                context_calls.append(text)
                return text.replace('Before', 'Target before').replace(
                    'after.', 'target after.'
                )
            self.assertIn(text, ['one\n', 'two\n'])
            return 'target option\n'

        transform.fallback(source, translated, translate_one)

        self.assertEqual(context_calls, ['Before __BLANK__ after.\n'])
        types = [item['type'] for item in transform.fallback_diagnostics]
        self.assertEqual(types.count('context_retry'), 1)
        self.assertEqual(types.count('context_cache_hit'), 1)

    def test_fraus_v2_fallback_maps_okapi_renumbered_markers(self):
        transform = FrausV2XmlTransform()
        transform.variant_sequence = [['choice']]
        transform.variant_marker_kinds = [['option']]
        transform.variant_source_payloads = ['<g id="fraus-option-choice">choice</g> context']

        def translate_one(text):
            if '__BLANK__' in text:
                return text.replace('context', 'target context')
            return 'target choice\n'

        result = transform.fallback(
            '<g id="1">choice</g> context\n',
            '<g id="1"><g id="2">bad</g></g>\n',
            translate_one,
        )

        self.assertEqual(
            result,
            '<g id="1">target choice</g> target context\n',
        )
        self.assertEqual(transform.fallback_values['choice'], 'target choice')

    def test_fraus_v2_fallback_rejects_multiline_marker_structure(self):
        transform = FrausV2XmlTransform()
        transform.variant_sequence = [['choice']]
        transform.variant_marker_kinds = [['text', 'option']]

        with self.assertRaisesRegex(AssertionError, 'ambiguous'):
            transform.fallback(
                '<g id="1">First sentence.\nSecond sentence.</g>'
                '<g id="2">choice</g>\n',
                'translated\n',
                lambda text: text,
            )

    def test_fraus_v2_fallback_validates_generated_option_ids(self):
        transform = FrausV2XmlTransform()
        transform.variant_sequence = [['expected']]

        with self.assertRaisesRegex(AssertionError, 'marker identity'):
            transform.fallback(
                '<g id="fraus-option-unexpected">one</g>\n',
                '<g id="fraus-option-unexpected"><g id="bad">one</g></g>\n',
                lambda text: text,
            )

    def test_fraus_v2_fallback_preserves_comments_during_recovery(self):
        transform = FrausV2XmlTransform()
        transform.variant_sequence = [['choice']]

        def translate_one(text):
            if '__BLANK__' in text:
                return text.replace('Before', 'Target before')
            return 'target choice\n'

        result = transform.fallback(
            'Before<!--keep--><g id="fraus-option-choice">one</g>\n',
            '<g id="fraus-option-choice"><g id="bad">one</g></g>\n',
            translate_one,
        )

        self.assertEqual(
            result,
            'Target before<!--keep--><g id="fraus-option-choice">target choice</g>\n',
        )

    def test_fraus_v2_fallback_rejects_redistributed_marker_lines(self):
        transform = FrausV2XmlTransform()
        transform.variant_sequence = [['choice']]
        source = (
            'Český úvod.\n'
            '<g id="fraus-option-choice">volba</g>\n'
        )

        def translate_one(text):
            self.fail('Line validation must precede isolated translation')

        with self.assertRaisesRegex(AssertionError, 'line positions'):
            transform.fallback(
                source,
                '<g id="fraus-option-choice">choice</g>\nEnglish introduction.\n',
                translate_one,
            )

    def test_fraus_v2_fallback_rejects_shifted_non_variant_lines(self):
        transform = FrausV2XmlTransform()
        transform.variant_sequence = [['choice']]
        source = (
            '<g id="fraus-option-choice">volba</g>\n'
            'První řádek.\n'
            'Druhý řádek.\n'
            'Třetí řádek.\n'
        )
        def translate_one(text):
            self.fail('Line validation must precede isolated translation')

        with self.assertRaisesRegex(AssertionError, 'line positions'):
            transform.fallback(
                source,
                'First line.\nSecond line.\nThird line.\n'
                '<g id="fraus-option-choice">choice</g>\n',
                translate_one,
            )

    def test_fraus_v2_fallback_rejects_total_line_changes(self):
        transform = FrausV2XmlTransform()
        transform.variant_sequence = [['choice']]

        def translate_one(text):
            self.fail('Line validation must precede isolated translation')

        with self.assertRaisesRegex(AssertionError, 'line structure'):
            transform.fallback(
                '<g id="fraus-option-choice">volba</g>\nČeský závěr.\n',
                '<g id="fraus-option-choice">choice</g>\n',
                translate_one,
            )

    def test_fraus_v2_fallback_preserves_crlf_line_endings(self):
        transform = FrausV2XmlTransform()
        transform.variant_sequence = [['choice']]

        def translate_one(text):
            if '__BLANK__' in text:
                return 'English context__BLANK__\n'
            self.assertEqual(text, 'volba\n')
            return 'choice\n'

        result = transform.fallback(
            'Český text <g id="fraus-option-choice">volba</g>\r\n',
            '<g id="fraus-option-choice"><g id="nested">bad</g></g>\r\n',
            translate_one,
        )

        self.assertEqual(
            result,
            'English context <g id="fraus-option-choice">choice</g>\r\n',
        )

    def test_fraus_v2_fallback_rejects_changed_crlf_line_count(self):
        transform = FrausV2XmlTransform()
        transform.variant_sequence = [['choice']]

        def translate_one(text):
            self.fail('Line validation must precede isolated translation')

        with self.assertRaisesRegex(AssertionError, 'line structure'):
            transform.fallback(
                '<g id="fraus-option-choice">volba</g>\r\nČeský závěr.\r\n',
                '<g id="fraus-option-choice">choice</g>\n',
                translate_one,
            )

    def test_fraus_v2_can_force_sentence_boundaries(self):
        transform = FrausV2XmlTransform(force_sentence_level=True)
        with tempfile.TemporaryDirectory() as directory:
            source = os.path.join(directory, 'input.xml')
            prepared = os.path.join(directory, 'prepared.xml')
            with open(source, 'w', encoding='utf-8') as file:
                file.write('<DOC><Questions><Question><RA><ExText Id="before">First sentence. Second sentence.</ExText><InputOption Id="input"><SelectOption><ExText>one</ExText></SelectOption></InputOption></RA></Question></Questions></DOC>')
            transform.preprocess(source, prepared)
            prepared_text = open(prepared, encoding='utf-8').read()
            self.assertNotIn('__FRAUS_VARIANT_', prepared_text)
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
