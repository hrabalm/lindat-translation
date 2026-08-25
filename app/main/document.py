import os
import re
import subprocess
import sys
import copy
import html
import shutil
import xml.etree.ElementTree as ET
import uuid
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
from app.settings import (
    ALLOWED_EXTENSIONS,
    FRAUS_V2_FORCE_SENTENCE_LEVEL,
    FRAUS_V2_MAX_SEGMENT_TOKENS,
    MAX_TEXT_LENGTH,
    TIKAL_PATH,
    UPLOAD_FOLDER,
)
from app.text_utils import count_words
from app.text_utils import split_text_into_sentences
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


def unwrap_paragraph_preserve_markup(line: str) -> str:
    stripped = line.rstrip("\n")
    return stripped[3:-4] + "\n"


_TAG_TOKEN_RE = re.compile(
    r"(?:<|&lt;|&amp;lt;)\s*/?\s*[A-Za-z][\w:.-]*\b.*?(?:>|&gt;|&amp;gt;)",
    re.IGNORECASE | re.DOTALL,
)
_TAG_NAME_RE = re.compile(r"<\s*/?\s*([A-Za-z][\w:.-]*)", re.IGNORECASE)


def _tag_name(token: str) -> Optional[str]:
    decoded = unescape(unescape(token))
    match = _TAG_NAME_RE.match(decoded)
    return match.group(1).lower() if match else None


def sanitize_generated_markup(source: str, target: str) -> str:
    """Remove generated tags whose names are absent from the source."""
    allowed = {
        name for name in (_tag_name(token) for token in _TAG_TOKEN_RE.findall(source))
        if name is not None
    }

    def replace(match):
        token = match.group(0)
        return token if _tag_name(token) in allowed else ""

    return _TAG_TOKEN_RE.sub(replace, target)


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


def translate_with_line_fallback(text, translate, on_fallback=None):
    try:
        return translate(text)
    except AssertionError:
        if on_fallback is not None:
            on_fallback(len(text.splitlines()))
        return "".join(translate(line) for line in text.splitlines(keepends=True))


class XmlTransform:
    def preprocess(self, input_path: str, output_path: str) -> None:
        raise NotImplementedError

    def postprocess(self, input_path: str, output_path: str) -> None:
        raise NotImplementedError


class FrausV2XmlTransform(XmlTransform):
    """Translate FRAUS answer options in the context of their exercise."""

    namespace = "urn:lindat:fraus-v2"
    text_tag = f"{{{namespace}}}text"
    option_tag = f"{{{namespace}}}option"
    synthetic_prefix = "fraus-v2-"

    def __init__(self, force_sentence_level=False, max_segment_tokens=None,
                 language="cs"):
        self.original_tree = None
        self.original_root = None
        self.force_sentence_level = force_sentence_level
        self.max_segment_tokens = max_segment_tokens
        self.language = language
        self.variant_records = {}
        self.variant_sequence = []
        self.fallback_values = {}
        self.fallback_diagnostics = []

    def _split_text(self, text):
        if not self.force_sentence_level and not self.max_segment_tokens:
            return text
        leading = text[:len(text) - len(text.lstrip())]
        trailing = text[len(text.rstrip()):]
        body = text.strip()
        if not body:
            return text
        sentences = split_text_into_sentences(body, self.language)
        if self.max_segment_tokens:
            packed = []
            current = ""
            for sentence in sentences:
                candidate = f"{current} {sentence}".strip() if current else sentence
                if current and len(candidate.split()) > self.max_segment_tokens:
                    packed.append(current)
                    current = sentence
                else:
                    current = candidate
            if current:
                packed.append(current)
            sentences = packed
        return leading + "\n".join(sentences) + trailing

    def preprocess(self, input_path: str, output_path: str) -> None:
        ET.register_namespace("fraus", self.namespace)
        source = read_text(input_path)
        source = source.replace('encoding="utf-16"', 'encoding="utf-8"')
        source = source.replace("&nbsp;", "&#160;")
        self.original_root = ET.fromstring(source)
        root = copy.deepcopy(self.original_root)
        self.variant_records = {}
        self.variant_sequence = []
        self.fallback_values = {}
        self.fallback_diagnostics = []

        for ra in list(root.iter("RA")):
            if not any(ancestor.tag == "Questions" for ancestor in self._ancestors(root, ra)):
                continue
            children = list(ra)
            options = [child for child in children if child.tag == "InputOption"]
            if not options:
                continue
            if any(not option.findall("SelectOption") for option in options):
                continue
            max_options = max(len(list(option.findall("SelectOption"))) for option in options)
            if not max_options:
                continue

            direct_text = [child for child in children if child.tag == "ExText"]
            if not direct_text:
                continue
            record = {
                "original": copy.deepcopy(ra),
                "source_id": str(id(ra)),
                "variants": max_options,
                "path": self._element_path(root, ra),
            }
            self.variant_records[id(ra)] = record
            # ElementTree has no parent pointers; replace the RA in its owner.
            owner = self._find_parent(root, ra)
            if owner is None:
                continue
            insert_at = list(owner).index(ra)
            owner.remove(ra)
            for variant in range(max_options):
                variant_ra = copy.deepcopy(ra)
                variant_ra.set(f"{{{self.namespace}}}source-ra", str(id(ra)))
                variant_ra.set(f"{{{self.namespace}}}variant", str(variant))
                self.variant_sequence.append(
                    self._flatten_context(variant_ra, variant)
                )
                owner.insert(insert_at + variant, variant_ra)

        ET.ElementTree(root).write(output_path, encoding="utf-8", xml_declaration=True)

    def postprocess(self, input_path: str, output_path: str) -> None:
        translated_root = ET.parse(input_path).getroot()
        root = copy.deepcopy(translated_root)
        for source_id, record in self.variant_records.items():
            variants = [ra for ra in root.iter("RA")
                        if ra.get(f"{{{self.namespace}}}source-ra") == str(source_id)]
            if not variants:
                continue
            correct = next((ra for ra in variants if ra.get(f"{{{self.namespace}}}variant") == "0"), variants[0])
            restored = self._restore_ra(correct, variants, record)
            owner = self._find_parent(root, variants[0])
            if owner is None:
                continue
            index = list(owner).index(variants[0])
            for variant in variants:
                owner.remove(variant)
            owner.insert(index, restored)

        ET.ElementTree(root).write(output_path, encoding="utf-8", xml_declaration=True)

    @staticmethod
    def _ancestors(root, target):
        path = []
        def visit(parent):
            for child in parent:
                if child is target:
                    path.extend([parent])
                    return True
                if visit(child):
                    path.append(parent)
                    return True
            return False
        visit(root)
        return path

    @staticmethod
    def _find_parent(root, target):
        if target is None:
            return None
        for parent in root.iter():
            if target in list(parent):
                return parent
        return None

    @staticmethod
    def _element_path(root, target):
        path = []
        def visit(parent):
            for index, child in enumerate(parent):
                path.append(index)
                if child is target:
                    return True
                if visit(child):
                    return True
                path.pop()
            return False
        return tuple(path) if visit(root) else None

    @staticmethod
    def _element_at_path(root, path):
        element = root
        try:
            for index in path:
                element = list(element)[index]
        except (IndexError, TypeError):
            return None
        return element

    @staticmethod
    def _selected(selects, variant):
        correct = [x for x in selects if x.get("Correct") == "true"]
        wrong = [x for x in selects if x not in correct]
        ordered = correct + wrong
        return ordered[variant % len(ordered)]

    def _flatten_context(self, ra, variant):
        pieces = []
        selected_keys = []
        children = list(ra)
        text_index = 0
        option_index = 0
        for child in children:
            if child.tag == "ExText":
                pieces.append(("text", text_index, self._split_text(child.text or "")))
                text_index += 1
                ra.remove(child)
            elif child.tag == "InputOption":
                selects = child.findall("SelectOption")
                selected = self._selected(selects, variant)
                select_index = selects.index(selected)
                source_id = ra.get(f"{{{self.namespace}}}source-ra")
                option_key = f"{source_id}-{option_index}-{select_index}"
                option_text = selected.find("ExText")
                pieces.append(("option", option_key,
                               option_text.text if option_text is not None else ""))
                selected_keys.append(option_key)
                option_index += 1
                ra.remove(child)
        payload = ET.Element("ExText", {"Id": f"{self.synthetic_prefix}{id(ra)}-{variant}"})
        last = None
        for index, piece in enumerate(pieces):
            if piece[0] == "text":
                if index and pieces[index - 1][0] == "option":
                    text = piece[2]
                    if (text and not text[0].isspace()
                            and not text.lstrip().startswith((".", ",", ";", ":", "!", "?", ")"))):
                        piece = ("text", piece[1], " " + text)
                if index and pieces[index - 1][0] == "text":
                    last = ET.SubElement(payload, "g", {"id": f"fraus-text-{piece[1]}"})
                    last.text = piece[2]
                elif last is None:
                    payload.text = (payload.text or "") + piece[2]
                else:
                    last.tail = (last.tail or "") + piece[2]
            else:
                last = ET.SubElement(payload, "g", {"id": f"fraus-option-{piece[1]}"})
                last.text = piece[2]
        ra.insert(0, payload)
        return selected_keys

    def _restore_ra(self, translated_ra, variants, record):
        restored = copy.deepcopy(record["original"])
        for attr in (f"{{{self.namespace}}}source-ra", f"{{{self.namespace}}}variant"):
            restored.attrib.pop(attr, None)
        payload = translated_ra.find("ExText")
        runs = []
        if payload is not None:
            if payload.text:
                runs.append(payload.text)
            for marker in payload:
                marker_id = marker.get("id", "")
                if marker_id.startswith("fraus-text-"):
                    runs.append("".join(marker.itertext()))
                if marker.tail:
                    runs.append(marker.tail)
        option_values = {}
        for variant in variants:
            number = int(variant.get(f"{{{self.namespace}}}variant", "0"))
            option_values[number] = {}
            variant_payload = variant.find("ExText")
            if variant_payload is not None:
                for marker in variant_payload.iter("g"):
                    marker_id = marker.get("id", "")
                    if marker_id.startswith("fraus-option-"):
                        nested = any(
                            child.tag == "g"
                            and child.get("id", "").startswith("fraus-option-")
                            for child in marker.iter()
                            if child is not marker
                        )
                        if not nested:
                            option_values[number][marker_id[len("fraus-option-"):]] = "".join(marker.itertext())
        selected_values = {}
        for values in option_values.values():
            selected_values.update(values)
        selected_values.update(self.fallback_values)
        new_children = []
        run_index = 0
        option_index = 0
        for child in list(restored):
            if child.tag == "ExText":
                value = runs[run_index] if run_index < len(runs) else None
                if value is not None and (value.strip() or not (child.text or "").strip()):
                    child.text = value
                new_children.append(child)
                run_index += 1
            elif child.tag == "InputOption":
                for select_index, select in enumerate(child.findall("SelectOption")):
                    option_text = select.find("ExText")
                    option_key = f"{record['source_id']}-{option_index}-{select_index}"
                    value = selected_values.get(option_key)
                    if option_text is not None and value and value.strip():
                        option_text.text = value
                new_children.append(child)
                option_index += 1
            else:
                new_children.append(child)
        for position, child in enumerate(new_children):
            child.set("Position", str(position))
        restored[:] = new_children
        return restored

    def fallback(self, source_text, translated_text, translate_one):
        source_lines = source_text.splitlines(keepends=True)
        target_lines = translated_text.splitlines(keepends=True)
        source_variant_indices = [index for index, line in enumerate(source_lines)
                                  if "<g" in line]
        target_variant_indices = [index for index, line in enumerate(target_lines)
                                  if "<g" in line]
        source_variants = [source_lines[index] for index in source_variant_indices]
        target_variants = [target_lines[index] for index in target_variant_indices]
        if not source_variants:
            return translated_text

        unsafe_indices = set()
        unsafe_reasons = {}
        if len(source_variants) == len(target_variants):
            pairs = zip(source_variants, target_variants)
            for index, (source_line, target_line) in enumerate(pairs):
                try:
                    source_fragment = ET.fromstring(f"<root>{source_line.strip()}</root>")
                    fragment = ET.fromstring(f"<root>{target_line.strip()}</root>")
                    source_markers = list(source_fragment.iter("g"))
                    markers = list(fragment.iter("g"))
                    reasons = []
                    if ([marker.get("id") for marker in markers]
                            != [marker.get("id") for marker in source_markers]):
                        reasons.append("marker_identity")
                    if any(
                            [child.tag for child in marker.iter() if child is not marker]
                            != [child.tag for child in source_marker.iter()
                                 if child is not source_marker]
                            for source_marker, marker in zip(source_markers, markers)):
                        reasons.append("marker_structure")
                    if any(not "".join(marker.itertext()).strip() for marker in markers):
                        reasons.append("empty_marker")
                    if reasons:
                        unsafe_indices.add(index)
                        unsafe_reasons[index] = reasons
                except ET.ParseError:
                    unsafe_indices.add(index)
                    unsafe_reasons[index] = ["invalid_markup"]
        else:
            # Newlines are not stable translation boundaries. Retry every
            # variant when the first pass changes their count.
            unsafe_indices.update(range(len(source_variants)))
            unsafe_reasons.update(
                (index, ["variant_line_count"])
                for index in range(len(source_variants))
            )

        for variant_index in sorted(unsafe_indices):
            if variant_index >= len(self.variant_sequence):
                continue
            self.fallback_diagnostics.append({
                "type": "unsafe_context_variant",
                "variant_index": variant_index,
                "reasons": unsafe_reasons.get(variant_index, ["unknown"]),
                "action": "restore_source_context",
            })
            source_line = source_variants[variant_index]
            select_ids = self.variant_sequence[variant_index]
            source_root = ET.fromstring(f"<root>{source_line.strip()}</root>")
            all_source_markers = list(source_root.iter("g"))
            option_markers = [
                marker for marker in all_source_markers
                if marker.get("id", "").startswith("fraus-option-")
            ]
            source_markers = option_markers or all_source_markers
            if len(source_markers) != len(select_ids):
                for select_id in select_ids:
                    self.fallback_values[select_id] = None
                    self.fallback_diagnostics.append({
                        "type": "option_retry",
                        "variant_index": variant_index,
                        "option_key": select_id,
                        "strategy": "original",
                        "reason": "source_marker_count",
                    })
                continue
            for marker_index, select_id in enumerate(select_ids):
                source_marker = source_markers[marker_index]
                source_value = "".join(source_marker.itertext()).strip()
                retry_source = self._keep_one_marker(
                    source_line, all_source_markers.index(source_marker)
                )
                accepted_value = None
                strategy = "original"
                try:
                    retry_target = translate_one(retry_source)
                    retry_root = ET.fromstring(f"<root>{retry_target.strip()}</root>")
                    retry_markers = list(retry_root.iter("g"))
                    if len(retry_markers) == 1 and not any(
                            child.tag == "g" for child in retry_markers[0].iter()
                            if child is not retry_markers[0]) and (
                            [child.tag for child in retry_markers[0].iter()
                             if child is not retry_markers[0]]
                            == [child.tag for child in source_marker.iter()
                                if child is not source_marker]):
                        value = "".join(retry_markers[0].itertext()).strip()
                        if self._safe_option_value(source_value, value):
                            accepted_value = value
                            strategy = "contextual"
                except (ET.ParseError, AssertionError, ValueError):
                    pass
                if accepted_value is None:
                    try:
                        standalone = self._standalone_value(translate_one(source_value + "\n"))
                        if self._safe_option_value(source_value, standalone):
                            accepted_value = standalone
                            strategy = "standalone"
                    except (AssertionError, ValueError):
                        pass
                # None tells restoration to retain the original option text.
                self.fallback_values[select_id] = accepted_value
                self.fallback_diagnostics.append({
                    "type": "option_retry",
                    "variant_index": variant_index,
                    "option_key": select_id,
                    "strategy": strategy,
                })
        if len(source_variants) != len(target_variants):
            return source_text
        for variant_index in unsafe_indices:
            target_lines[target_variant_indices[variant_index]] = source_variants[variant_index]
        return "".join(target_lines)

    @staticmethod
    def _safe_option_value(source_value, target_value):
        if not target_value or not target_value.strip():
            return False
        source_words = max(1, len(source_value.split()))
        return len(target_value.split()) <= source_words * 5 + 4

    @staticmethod
    def _standalone_value(translation):
        value = translation.strip()
        if not value:
            return ""
        try:
            root = ET.fromstring(f"<root>{value}</root>")
            return "".join(root.itertext()).strip()
        except ET.ParseError:
            return re.sub(r"<[^>]+>", "", value).strip()

    @staticmethod
    def _keep_one_marker(line, marker_index):
        root = ET.fromstring(f"<root>{line.rstrip(chr(10))}</root>")
        markers = list(root.iter("g"))
        selected = markers[marker_index]

        def render(node):
            result = html.escape(node.text or "", quote=False)
            for child in node:
                content = render(child)
                if child is selected:
                    result += f'<g id="{child.get("id", "")}">{content}</g>'
                else:
                    result += content
                result += html.escape(child.tail or "", quote=False)
            return result

        return render(root) + "\n"


XML_TRANSFORMS = {
    "fraus_v2": FrausV2XmlTransform,
}


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
    final_output_path: str
    src: str
    tgt: str
    artifacts: Dict[str, str]


@dataclass
class PipelineStage:
    name: str
    run: Callable[[PipelineContext], Optional[str]]


class DocumentFormat:
    def prepare(self, context: PipelineContext) -> None:
        return None

    def finalize(self, context: PipelineContext) -> None:
        return None

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
    def __init__(self, runner: TikalRunner, xml_profile: str, html_profile: str,
                 xml_transform: Optional[XmlTransform] = None):
        self.runner = runner
        self.xml_profile = xml_profile
        self.html_profile = html_profile
        self.xml_transform = xml_transform

    def prepare(self, context: PipelineContext) -> None:
        if self.xml_transform is None:
            return
        preprocessed = context.input_path + ".preprocessed"
        self.xml_transform.preprocess(context.input_path, preprocessed)
        context.input_path = preprocessed
        context.artifacts["preprocessed_xml"] = preprocessed

    def finalize(self, context: PipelineContext) -> None:
        if self.xml_transform is None:
            return
        postprocessed = context.output_path + ".postprocessed"
        self.xml_transform.postprocess(context.output_path, postprocessed)
        os.replace(postprocessed, context.final_output_path)

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
            transform = (
                unwrap_paragraph_preserve_markup
                if self.xml_transform is not None
                else unwrap_and_escape
            )
            transform_file(html_translated, xml_translated, transform)
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
        context = PipelineContext(input_path, output_path, output_path, src, tgt, {})
        trace: Dict[str, str] = {}
        translation_text = ""
        generated = set()
        try:
            self.document_format.prepare(context)
            generated.update(context.artifacts.values())
            if self.debug and "preprocessed_xml" in context.artifacts:
                trace["preprocessed_xml"] = read_text(context.artifacts["preprocessed_xml"])
                print(f"[document] stage=preprocessed_xml\n{trace['preprocessed_xml']}", file=sys.stderr)
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
            self.document_format.finalize(context)
            if self.debug and self.document_format.__class__ is FrausDocumentFormat and self.document_format.xml_transform:
                trace["postprocessed_xml"] = read_text(output_path)
                print(f"[document] stage=postprocessed_xml\n{trace['postprocessed_xml']}", file=sys.stderr)
            return PipelineResult(output_path, translation_text, trace)
        finally:
            generated.update(context.artifacts.values())
            for path in generated:
                if path not in (output_path, context.final_output_path) and os.path.exists(path):
                    os.remove(path)
            if os.path.exists(input_path) and os.path.exists(context.final_output_path):
                os.remove(input_path)


class InnerLindatTranslator(Translator):
    def __init__(self, method, src, tgt, model=None, custom_prompt=None, terms=None, split=True):
        self.method = method
        self.src = src
        self.tgt = tgt
        self.model = model
        self.split = split
        self.custom_prompt = custom_prompt
        self.debug_segments = []

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

        self.debug_segments.extend(
            {"source": source, "target": target}
            for source, target in zip(src_sentences, tgt_sentences)
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
    def __init__(self, orig_full_path, original_filename=None, work_dir=None):
        self.orig_full_path = orig_full_path
        self._input_file_name = original_filename or os.path.basename(orig_full_path)
        self._work_dir = work_dir
        self._input_word_count = 0
        self._output_word_count = 0
        self._input_nfc_len = 0
        self.debug_trace = {}
        self.debug_segments = []
        self.xml_transform = None
        self._fallback_diagnostics = []

    @classmethod
    def from_file(cls, request_file):
        if not request_file:
            api.abort(code=400, message='Empty file')
        if not cls.allowed_file(request_file.filename):
            api.abort(code=415, message='Unsupported file type for translation')
        filename = secure_filename(request_file.filename)
        work_dir = os.path.join(UPLOAD_FOLDER, str(uuid.uuid4()))
        os.mkdir(work_dir, mode=0o700)
        os.chmod(work_dir, 0o700)
        orig_full_path = os.path.join(work_dir, filename)
        try:
            request_file.save(orig_full_path)
        except Exception:
            shutil.rmtree(work_dir, ignore_errors=True)
            raise
        return cls(orig_full_path, original_filename=filename, work_dir=work_dir)

    @classmethod
    def allowed_file(cls, filename):
        return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

    def translate_from_to(self, src, tgt, custom_prompt=None, terms=None, split=True):
        self._fallback_diagnostics = []
        try:
            self._extract_translate_merge(src, tgt, "from_to", None, custom_prompt, terms, split)
        except Exception:
            self._cleanup_work_dir()
            raise

    def translate_with_model(self, model, src, tgt, custom_prompt=None, terms=None, split=True):
        self._fallback_diagnostics = []
        try:
            self._extract_translate_merge(src, tgt, "with_model", model, custom_prompt, terms, split)
        except Exception:
            self._cleanup_work_dir()
            raise

    def _cleanup_work_dir(self):
        if self._work_dir:
            shutil.rmtree(self._work_dir, ignore_errors=True)

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
        args = text_input_with_src_tgt.parse_args(request)
        transform_name = args.get('xmlTransform')
        if transform_name and transform_name not in XML_TRANSFORMS:
            api.abort(code=400, message=f'Unknown XML transform: {transform_name}')
        document_format = FrausDocumentFormat(
            TikalRunner(TIKAL_PATH),
            os.path.join(app_dir, 'okapi_profiles', 'okf_xml@fraus.fprm'),
            os.path.join(app_dir, 'okapi_profiles', 'okf_html@fraus.fprm'),
            XML_TRANSFORMS[transform_name](
                force_sentence_level=(
                    FRAUS_V2_FORCE_SENTENCE_LEVEL
                    if args.get('forceSentenceLevel') is None
                    else args.get('forceSentenceLevel')
                ),
                max_segment_tokens=(
                    FRAUS_V2_MAX_SEGMENT_TOKENS
                    if args.get('maxSegmentTokens') is None
                    else args.get('maxSegmentTokens')
                ),
                language=src,
            ) if transform_name else None,
        )
        if transform_name == "fraus_v2":
            document_format.xml_profile = os.path.join(
                app_dir, 'okapi_profiles', 'okf_xml@fraus_v2.fprm'
            )
        self.xml_transform = document_format.xml_transform
        source_bytes = None
        if self.xml_transform is not None:
            with open(self.orig_full_path, "rb") as source_file:
                source_bytes = source_file.read()
        try:
            self._run_document_pipeline(
                document_format, src, tgt, method, model, custom_prompt, terms, split
            )
        except Exception as error:
            retryable = isinstance(
                error, (AssertionError, TikalError)
            ) or (isinstance(error, ValueError)
                  and "paired tag" in str(error).lower())
            if source_bytes is None or not retryable:
                raise
            self._fallback_diagnostics.append({
                "type": "document_pipeline_fallback",
                "strategy": "legacy_fraus",
                "reason": (
                    "paired_tag" if isinstance(error, ValueError)
                    else type(error).__name__
                ),
            })
            with open(self.orig_full_path, "wb") as source_file:
                source_file.write(source_bytes)
            translated_path = self.get_translated_path(tgt)
            if os.path.exists(translated_path):
                os.remove(translated_path)
            self.xml_transform = None
            legacy_format = FrausDocumentFormat(
                TikalRunner(TIKAL_PATH),
                os.path.join(app_dir, 'okapi_profiles', 'okf_xml@fraus.fprm'),
                os.path.join(app_dir, 'okapi_profiles', 'okf_html@fraus.fprm'),
            )
            self._run_document_pipeline(
                legacy_format, src, tgt, method, model, custom_prompt, terms, split
            )

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

        debug = str(request.values.get('debug', '')).lower() in {'1', 'true', 'yes'}
        result = DocumentPipeline(document_format).run(
            self.orig_full_path, self.translated_path, src, tgt, translate,
            debug=debug,
        )
        self.translation = result.text
        self.debug_trace = dict(result.trace)
        if self.debug_segments:
            self.debug_trace["llm_segments"] = self.debug_segments
        if debug and self._fallback_diagnostics:
            self.debug_trace["fallbacks"] = copy.deepcopy(self._fallback_diagnostics)

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
        self.debug_segments = []

        def translate_markup(text):
            translator = InnerLindatTranslator(
                method, src, tgt, model, custom_prompt=custom_prompt,
                terms=terms, split=split,
            )
            mt = MarkupTranslator(
                translator, LindatAligner(src, tgt, show_progress=False), RegexTokenizer()
            )
            result = mt.translate(text)
            result = sanitize_generated_markup(text, result)
            self.debug_segments.extend(translator.debug_segments)
            return result

        self.translation = translate_with_line_fallback(
            self.text,
            translate_markup,
            lambda line_count: self._fallback_diagnostics.append({
                "type": "line_alignment_retry",
                "line_count": line_count,
            }),
        )
        if self.xml_transform is not None and hasattr(self.xml_transform, "fallback"):
            self.translation = self.xml_transform.fallback(
                self.text, self.translation, translate_markup
            )
            self._fallback_diagnostics.extend(
                self.xml_transform.fallback_diagnostics
            )
        self._output_word_count = len(self.translation.split())

    def get_text(self):
        return self.text

    def get_translation(self):
        return self.translation

    def create_response(self, extra_headers):
        if str(request.values.get('debug', '')).lower() in {'1', 'true', 'yes'}:
            from flask import jsonify
            import base64

            try:
                with open(self.translated_path, 'rb') as translated_file:
                    output = base64.b64encode(translated_file.read()).decode('ascii')
                response = jsonify({
                    'filename': os.path.basename(self.translated_path),
                    'output_base64': output,
                    'trace': self.debug_trace,
                })
                response.headers.extend({**self.prep_billing_headers(), **extra_headers})
                return response
            finally:
                if self._work_dir:
                    self._cleanup_work_dir()
                elif os.path.exists(self.translated_path):
                    os.remove(self.translated_path)
        directory = self._work_dir or UPLOAD_FOLDER
        try:
            response = send_from_directory(directory, os.path.basename(self.translated_path))
        except Exception:
            self._cleanup_work_dir()
            raise
        response.headers.extend({**self.prep_billing_headers(), **extra_headers})
        if self._work_dir:
            self._cleanup_work_dir()
        else:
            os.remove(self.translated_path)
        return response
