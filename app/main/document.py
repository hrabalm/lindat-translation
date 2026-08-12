import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from html import escape, unescape
from typing import Callable, Dict, List, Optional, Tuple
from unicodedata import normalize

from flask import request, send_from_directory
from werkzeug.utils import secure_filename

from app.main.api.restplus import api
from app.main.api.translation.parsers import text_input_with_src_tgt
from app.main.translate import translate_from_to, translate_with_model
from app.main.translatable import Translatable
from app.settings import ALLOWED_EXTENSIONS, MAX_TEXT_LENGTH, TIKAL_PATH, UPLOAD_FOLDER
from app.text_utils import count_words
from document_translation.lindat_services.align import LindatAligner
from document_translation.markuptranslator import MarkupTranslator, Translator
from document_translation.pdf_tools.pdfeditor import PdfEditor
from document_translation.regextokenizer import RegexTokenizer


def fix_fraus_encoding(line: str) -> str:
    if line.startswith('<?xml version="1.0" encoding="utf-16"?>'):
        return line.replace("utf-16", "utf-8")
    return line


def unescape_extracted_line(line: str) -> str:
    return unescape(unescape(line)).replace("&nbsp;", " ")


def wrap_paragraph(line: str) -> str:
    return f"<p>{line.rstrip(chr(10))}</p>\n"


def unwrap_and_escape(line: str) -> str:
    stripped = line.rstrip("\n")
    return escape(escape(stripped[3:-4] + "\n"))


def transform_file(input_path: str, output_path: str,
                   transform: Callable[[str], str]) -> None:
    with open(input_path, "r", encoding="utf-8") as source, open(
            output_path, "w", encoding="utf-8") as destination:
        for line in source:
            destination.write(transform(line))


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as source:
        return source.read()


class TikalError(RuntimeError):
    pass


class TikalRunner:
    def __init__(self, tikal_path: str, run=subprocess.run):
        self.tikal_path = tikal_path
        self._run = run

    def run(self, mode: str, input_path: str, output_path: str, src: str,
            tgt: Optional[str] = None, profile: Optional[str] = None,
            translation_path: Optional[str] = None,
            profile_at_end: bool = False,
            stdout=None,
            expected_output_path: Optional[str] = None) -> str:
        command = [self.tikal_path + "tikal.sh", mode, input_path]
        if profile and not profile_at_end:
            command.extend(["-fc", profile])
        command.extend(["-sl", src])
        if tgt:
            command.extend(["-tl", tgt, "-overtrg"])
        if translation_path:
            command.extend(["-from", translation_path])
        command.extend(["-to", output_path])
        if profile and profile_at_end:
            command.extend(["-fc", profile])
        result = self._run(command, stdout=subprocess.DEVNULL if stdout is None else stdout)
        if result.returncode != 0:
            raise TikalError(f"Tikal failed with exit code {result.returncode}")
        expected = expected_output_path or output_path
        if not os.path.exists(expected):
            raise TikalError(f"Tikal did not create expected output: {expected}")
        return expected


@dataclass
class PipelineResult:
    output_path: str
    text: str
    trace: Dict[str, str] = field(default_factory=dict)


@dataclass
class PipelineContext:
    input_path: str
    output_path: str
    src: str
    tgt: str
    artifacts: Dict[str, str]


@dataclass
class PipelineStage:
    name: str
    run: Callable[[PipelineContext], Optional[str]]


class DocumentFormat:
    def stages(self, context: PipelineContext) -> List[PipelineStage]:
        raise NotImplementedError


class StandardDocumentFormat(DocumentFormat):
    def __init__(self, runner: TikalRunner, profile: Optional[str] = None):
        self.runner = runner
        self.profile = profile

    def stages(self, context: PipelineContext) -> List[PipelineStage]:
        source = context.input_path
        extracted = source + "." + context.src
        translated = source + "." + context.tgt

        def extract(ctx):
            self.runner.run("-xm", source, source, ctx.src, profile=self.profile,
                            profile_at_end=True, expected_output_path=extracted)
            ctx.artifacts["extracted"] = extracted
            return read_text(extracted)

        def translate(ctx):
            ctx.artifacts["translation"] = translated
            return read_text(extracted)

        def merge(ctx):
            self.runner.run("-lm", source, ctx.output_path, ctx.src, ctx.tgt,
                            profile=self.profile, translation_path=translated,
                            profile_at_end=True, stdout=sys.stderr)
            # Standard document outputs may be binary (for example ODT/DOCX).
            return ctx.output_path

        return [PipelineStage("extract", extract),
                PipelineStage("translate", translate),
                PipelineStage("merge", merge)]


class FrausDocumentFormat(DocumentFormat):
    def __init__(self, runner: TikalRunner, xml_profile: str, html_profile: str):
        self.runner = runner
        self.xml_profile = xml_profile
        self.html_profile = html_profile

    def stages(self, context: PipelineContext) -> List[PipelineStage]:
        source = context.input_path
        fixed = source + ".fixed"
        xml_extracted = fixed + "." + context.src
        html_unescaped = xml_extracted + ".html"
        paragraphs = html_unescaped + ".p"
        extracted = paragraphs + "." + context.src
        translated = paragraphs + "." + context.tgt
        html_translated = paragraphs + ".translated"
        xml_translated = fixed + ".translated"

        def fix_encoding(ctx):
            transform_file(source, fixed, fix_fraus_encoding)
            ctx.artifacts["fixed_xml"] = fixed
            return read_text(fixed)

        def extract_xml(ctx):
            self.runner.run("-xm", fixed, fixed, ctx.src, profile=self.xml_profile,
                            expected_output_path=xml_extracted)
            ctx.artifacts["xml_extracted"] = xml_extracted
            return read_text(xml_extracted)

        def unescape_html(ctx):
            transform_file(xml_extracted, html_unescaped, unescape_extracted_line)
            ctx.artifacts["unescaped_html"] = html_unescaped
            return read_text(html_unescaped)

        def wrap(ctx):
            transform_file(html_unescaped, paragraphs, wrap_paragraph)
            ctx.artifacts["paragraph_html"] = paragraphs
            return read_text(paragraphs)

        def extract_html(ctx):
            self.runner.run("-xm", paragraphs, paragraphs, ctx.src,
                            profile=self.html_profile,
                            expected_output_path=extracted)
            ctx.artifacts["html_extracted"] = extracted
            return read_text(extracted)

        def translate(ctx):
            ctx.artifacts["translation"] = translated
            return read_text(extracted)

        def merge_html(ctx):
            self.runner.run("-lm", paragraphs, html_translated, ctx.src, ctx.tgt,
                            profile=self.html_profile, translation_path=translated)
            return read_text(html_translated)

        def escape_xml(ctx):
            transform_file(html_translated, xml_translated, unwrap_and_escape)
            ctx.artifacts["xml_translation"] = xml_translated
            return read_text(xml_translated)

        def merge_xml(ctx):
            self.runner.run("-lm", fixed, ctx.output_path, ctx.src, ctx.tgt,
                            profile=self.xml_profile, translation_path=xml_translated)
            return read_text(ctx.output_path)

        return [PipelineStage("fix_encoding", fix_encoding),
                PipelineStage("extract_xml", extract_xml),
                PipelineStage("unescape_html", unescape_html),
                PipelineStage("wrap_paragraphs", wrap),
                PipelineStage("extract_html", extract_html),
                PipelineStage("translate", translate),
                PipelineStage("merge_html", merge_html),
                PipelineStage("escape_xml", escape_xml),
                PipelineStage("merge_xml", merge_xml)]


class DocumentPipeline:
    def __init__(self, document_format: DocumentFormat, debug: bool = False):
        self.document_format = document_format
        self.debug = debug

    def run(self, input_path: str, output_path: str, src: str, tgt: str,
            translate: Callable[[str], str], debug: Optional[bool] = None) -> PipelineResult:
        if debug is not None:
            self.debug = debug
        context = PipelineContext(input_path, output_path, src, tgt, {})
        trace: Dict[str, str] = {}
        translation_text = ""
        generated = set()
        try:
            for stage in self.document_format.stages(context):
                if stage.name == "translate":
                    extracted = stage.run(context) or ""
                    translation_text = translate(extracted)
                    path = context.artifacts["translation"]
                    with open(path, "w", encoding="utf-8") as destination:
                        destination.write(translation_text)
                    value = translation_text
                else:
                    value = stage.run(context) or ""
                generated.update(context.artifacts.values())
                if self.debug:
                    trace[stage.name] = value
                    print(f"[document] stage={stage.name}\n{value}", file=sys.stderr)
            return PipelineResult(output_path, translation_text, trace)
        finally:
            generated.update(context.artifacts.values())
            for path in generated:
                if path != output_path and os.path.exists(path):
                    os.remove(path)
            if os.path.exists(input_path) and os.path.exists(output_path):
                os.remove(input_path)


class InnerLindatTranslator(Translator):
    def __init__(self, method, src, tgt, model=None, custom_prompt=None, terms=None, split=True):
        self.method = method
        self.src = src
        self.tgt = tgt
        self.model = model
        self.split = split
        self.custom_prompt = custom_prompt

    def translate(self, input_text: str, split=True) -> Tuple[List[str], List[str]]:
        num_prefix_newlines = 0
        if input_text.startswith("\n"):
            while input_text[num_prefix_newlines] == "\n":
                num_prefix_newlines += 1
        if num_prefix_newlines:
            input_text = input_text[num_prefix_newlines:]

        num_suffix_newlines = 0
        if input_text.endswith("\n"):
            while input_text[-1 - num_suffix_newlines] == "\n":
                num_suffix_newlines += 1
        if num_suffix_newlines:
            input_text = input_text[:-num_suffix_newlines]

        if self.method == "with_model":
            src_sentences, tgt_sentences = translate_with_model(
                self.model, input_text, self.src, self.tgt,
                return_source_sentences=True, custom_prompt=self.custom_prompt,
                split=split,
            )
        else:
            src_sentences, tgt_sentences = translate_from_to(
                self.src, self.tgt, input_text, return_source_sentences=True,
                custom_prompt=self.custom_prompt, split=split,
            )

        if tgt_sentences:
            tgt_sentences = [
                src if re.match(r"^\s+$", src) else tgt
                for src, tgt in zip(src_sentences, tgt_sentences)
            ]
            src_sentences[0] = "\n" * num_prefix_newlines + src_sentences[0]
            tgt_sentences[0] = "\n" * num_prefix_newlines + tgt_sentences[0]
            src_sentences[-1] = src_sentences[-1].rstrip("\n") + "\n" * num_suffix_newlines
            tgt_sentences[-1] = tgt_sentences[-1].rstrip("\n") + "\n" * num_suffix_newlines
            src_sentences = [s + " " if not s.endswith("\n") else s for s in src_sentences]
            tgt_sentences = [s + " " if not s.endswith("\n") else s for s in tgt_sentences]
        return src_sentences, tgt_sentences


class Document(Translatable):
    def __init__(self, orig_full_path):
        self.orig_full_path = orig_full_path
        self._input_file_name = os.path.basename(orig_full_path)
        self._input_word_count = 0
        self._output_word_count = 0
        self._input_nfc_len = 0
        self.debug_trace = {}

    @classmethod
    def from_file(cls, request_file):
        if not request_file:
            api.abort(code=400, message='Empty file')
        if not cls.allowed_file(request_file.filename):
            api.abort(code=415, message='Unsupported file type for translation')
        filename = secure_filename(request_file.filename)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        orig_full_path = os.path.join(UPLOAD_FOLDER, filename)
        request_file.save(orig_full_path)
        return cls(orig_full_path)

    @classmethod
    def allowed_file(cls, filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    def translate_from_to(self, src, tgt, custom_prompt=None, terms=None, split=True):
        self._extract_translate_merge(src, tgt, "from_to", None, custom_prompt, terms, split)

    def translate_with_model(self, model, src, tgt, custom_prompt=None, terms=None, split=True):
        self._extract_translate_merge(src, tgt, "with_model", model, custom_prompt, terms, split)

    def _extract_translate_merge(self, src, tgt, method, model, custom_prompt=None, terms=None, split=True):
        if self.orig_full_path.endswith('.pdf'):
            return self._extract_translate_merge_pdf(src, tgt, method, model, custom_prompt, terms, split)
        args = text_input_with_src_tgt.parse_args(request)
        if args.get('fraus', False):
            return self._extract_translate_merge_fraus(src, tgt, method, model, custom_prompt, terms, split)
        return self._extract_translate_merge_document(src, tgt, method, model, custom_prompt, terms, split)

    def get_translated_path(self, tgt):
        orig_root, file_extension = os.path.splitext(self.orig_full_path)
        return f"{orig_root}.{tgt}{file_extension}"

    def _extract_translate_merge_fraus(self, src, tgt, method, model, custom_prompt=None, terms=None, split=True):
        app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        document_format = FrausDocumentFormat(
            TikalRunner(TIKAL_PATH),
            os.path.join(app_dir, 'okapi_profiles', 'okf_xml@fraus.fprm'),
            os.path.join(app_dir, 'okapi_profiles', 'okf_html@fraus.fprm'),
        )
        self._run_document_pipeline(document_format, src, tgt, method, model, custom_prompt, terms, split)

    def _extract_translate_merge_document(self, src, tgt, method, model, custom_prompt=None, terms=None, split=True):
        profile = None
        if self.orig_full_path.endswith(".inxml"):
            profile = TIKAL_PATH + "okf_xml@all_inline"
        elif self.orig_full_path.endswith(".innopxml"):
            profile = TIKAL_PATH + "okf_xml@all_inline_not_paragraphs"
        self._run_document_pipeline(
            StandardDocumentFormat(TikalRunner(TIKAL_PATH), profile),
            src, tgt, method, model, custom_prompt, terms, split,
        )

    def _run_document_pipeline(self, document_format, src, tgt, method, model, custom_prompt, terms, split):
        self.translated_path = self.get_translated_path(tgt)

        def translate(text):
            self.text = text
            self._translate(src, tgt, method, model, custom_prompt=custom_prompt, terms=terms, split=split)
            return self.translation

        result = DocumentPipeline(document_format).run(
            self.orig_full_path, self.translated_path, src, tgt, translate,
            debug=str(request.values.get('debug', '')).lower() in {'1', 'true', 'yes'},
        )
        self.translation = result.text
        self.debug_trace = result.trace

    def _extract_translate_merge_pdf(self, src, tgt, method, model=None, custom_prompt=None, terms=None, split=True):
        self.pdf_editor = PdfEditor(self.orig_full_path)
        lines = self.pdf_editor.extract_text()
        input_text = "<lb />".join(lines)
        assert "\n" not in input_text
        self.text = input_text.replace("<page-break />", "\n")
        self._translate(src, tgt, method, model, custom_prompt=custom_prompt, terms=terms, split=split)
        translated_lines = self.translation.replace("\n", "<page-break />").split("<lb />")
        assert len(lines) == len(translated_lines), f"{len(lines)} != {len(translated_lines)}"
        self.translated_path = self.get_translated_path(tgt)
        self.pdf_editor.merge_text(translated_lines, self.translated_path)

    def _translate(self, src, tgt, method, model=None, custom_prompt=None, terms=None, split=True):
        text_without_tags = re.sub(r'<[^>]*>', '', self.text)
        self._input_word_count = count_words(text_without_tags)
        self._input_nfc_len = len(normalize('NFC', self.text))
        args = text_input_with_src_tgt.parse_args(request)
        if self._input_nfc_len >= MAX_TEXT_LENGTH and not args.get('ignoreSizeLimit', False):
            api.abort(code=413, message='The total text length in the document exceeds the translation limit.')
        translator = InnerLindatTranslator(method, src, tgt, model, custom_prompt=custom_prompt, terms=terms, split=split)
        mt = MarkupTranslator(translator, LindatAligner(src, tgt, show_progress=False), RegexTokenizer())
        self.translation = mt.translate(self.text)
        self._output_word_count = len(self.translation.split())

    def get_text(self):
        return self.text

    def get_translation(self):
        return self.translation

    def create_response(self, extra_headers):
        if str(request.values.get('debug', '')).lower() in {'1', 'true', 'yes'}:
            from flask import jsonify
            import base64

            with open(self.translated_path, 'rb') as translated_file:
                output = base64.b64encode(translated_file.read()).decode('ascii')
            response = jsonify({
                'filename': os.path.basename(self.translated_path),
                'output_base64': output,
                'trace': self.debug_trace,
            })
            response.headers.extend({**self.prep_billing_headers(), **extra_headers})
            os.remove(self.translated_path)
            return response
        response = send_from_directory(UPLOAD_FOLDER, os.path.basename(self.translated_path))
        response.headers.extend({**self.prep_billing_headers(), **extra_headers})
        os.remove(self.translated_path)
        return response
