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
from app.models.llm_errors import LLMBackendError
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


class FrausTranslationError(LLMBackendError):
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
    context_placeholders = ("__BLANK__", "__PLACEHOLDER__")

    def __init__(self, force_sentence_level=False, max_segment_tokens=None,
                 language="cs"):
        self.original_tree = None
        self.original_root = None
        self.force_sentence_level = force_sentence_level
        self.max_segment_tokens = max_segment_tokens
        self.language = language
        self.variant_records = {}
        self.variant_sequence = []
        self.variant_marker_kinds = []
        self.variant_source_payloads = []
        self.fallback_values = {}
        self.fallback_diagnostics = []
        self.translated_units = {}

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
        self.variant_marker_kinds = []
        self.variant_source_payloads = []
        self.fallback_values = {}
        self.fallback_diagnostics = []
        self.translated_units = {}

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
                selected_keys, marker_kinds = self._flatten_context(
                    variant_ra, variant
                )
                self.variant_sequence.append(selected_keys)
                self.variant_marker_kinds.append(marker_kinds)
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
        marker_kinds = []
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
                    marker_kinds.append("text")
                    last.text = piece[2]
                elif last is None:
                    payload.text = (payload.text or "") + piece[2]
                else:
                    last.tail = (last.tail or "") + piece[2]
            else:
                last = ET.SubElement(payload, "g", {"id": f"fraus-option-{piece[1]}"})
                marker_kinds.append("option")
                last.text = piece[2]
        ra.insert(0, payload)
        serialized = ET.tostring(payload, encoding="unicode")
        self.variant_source_payloads.append(html.unescape(
            serialized[serialized.index(">") + 1:serialized.rindex("</ExText>")]))
        return selected_keys, marker_kinds

    def _restore_ra(self, translated_ra, variants, record):
        restored = copy.deepcopy(record["original"])
        for attr in (f"{{{self.namespace}}}source-ra", f"{{{self.namespace}}}variant"):
            restored.attrib.pop(attr, None)
        payload = record.get("recovered_context")
        if payload is None:
            payload = translated_ra.find("ExText")
        runs = []
        previous = None
        text_index = option_index = 0
        for child in restored:
            if child.tag == "ExText":
                if previous == "text":
                    marker = (payload.find(f'./g[@id="fraus-text-{text_index}"]')
                              if payload is not None else None)
                    value = "".join(marker.itertext()) if marker is not None else None
                elif previous == "option":
                    marker = (payload.find(f'./g[@id="{last_option_id}"]')
                              if payload is not None else None)
                    value = marker.tail if marker is not None else None
                else:
                    value = payload.text if payload is not None else None
                runs.append(value)
                text_index += 1
                previous = "text"
            elif child.tag == "InputOption":
                selects = child.findall("SelectOption")
                select_index = selects.index(self._selected(selects, 0))
                last_option_id = f"fraus-option-{record['source_id']}-{option_index}-{select_index}"
                option_index += 1
                previous = "option"
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
            for key, value in values.items():
                selected_values.setdefault(key, value)
        selected_values.update(self.fallback_values)
        translated_other = [child for child in translated_ra if child.tag != "ExText"]
        original_other = [child for child in restored if child.tag not in ("ExText", "InputOption")]
        if [child.tag for child in translated_other] != [child.tag for child in original_other]:
            raise FrausTranslationError("FRAUS non-option children changed during reconstruction")
        other_children = iter(translated_other)
        new_children = []
        run_index = 0
        option_index = 0
        for child in list(restored):
            if child.tag == "ExText":
                value = runs[run_index] if run_index < len(runs) else None
                if value is not None and (value.strip() or not (child.text or "").strip()):
                    child.text = value
                elif (child.text or "").strip():
                    raise FrausTranslationError("FRAUS context reconstruction has no usable translation")
                new_children.append(child)
                run_index += 1
            elif child.tag == "InputOption":
                for select_index, select in enumerate(child.findall("SelectOption")):
                    option_text = select.find("ExText")
                    option_key = f"{record['source_id']}-{option_index}-{select_index}"
                    value = selected_values.get(option_key)
                    if option_text is not None and value and value.strip():
                        option_text.text = value
                    elif option_text is not None and (option_text.text or "").strip():
                        raise FrausTranslationError("FRAUS option reconstruction has no usable translation")
                new_children.append(child)
                option_index += 1
            else:
                new_children.append(copy.deepcopy(next(other_children)))
        for position, child in enumerate(new_children):
            child.set("Position", str(position))
        restored[:] = new_children
        return restored

    def fallback(self, source_text, translated_text, translate_one):
        source_lines = source_text.splitlines(keepends=True)
        target_lines = translated_text.splitlines(keepends=True)
        context_translations = {}
        source_variant_indices = self._variant_line_indices(source_lines)
        variant_indices = list(range(len(self.variant_sequence)))
        if self.variant_source_payloads and self.variant_records:
            mapping = self._map_source_variants(source_lines)
            source_variant_indices = [line_index for line_index, _, _ in mapping]
            variant_indices = [variant_index for _, variant_index, _ in mapping]
        source_variants = [source_lines[index] for index in source_variant_indices]
        same_line_count = len(source_lines) == len(target_lines)
        if not self.variant_sequence:
            return translated_text
        if not source_variants and not self.variant_marker_kinds:
            return translated_text
        if len(source_variants) != len(variant_indices):
            raise AssertionError("FRAUS variant structure is ambiguous during recovery")
        if not same_line_count:
            raise AssertionError("FRAUS line structure changed during recovery")
        target_variant_indices = self._variant_line_indices(target_lines)
        if (len(target_variant_indices) == len(source_variant_indices)
                and target_variant_indices != source_variant_indices):
            raise AssertionError("FRAUS variant line positions changed during recovery")
        target_variants = [target_lines[index] for index in source_variant_indices]
        recovered_lines = list(target_lines)

        correct_indices = {}
        offset = 0
        for record in self.variant_records.values():
            for index in range(offset, offset + record["variants"]):
                correct_indices[index] = offset
            offset += record["variants"]
        accepted_contexts = {}
        contextual_values = {}
        for line_index, index, source_line, target_line in zip(
                source_variant_indices, variant_indices, source_variants, target_variants):
            try:
                source_root = self._parse_fragment(source_line)
            except ET.ParseError:
                continue
            target_options = {}
            scopes = self.translated_units.get(line_index, [(source_line, target_line)])
            for source_unit, target_unit in scopes:
                unit_ids = {m.get("id") for m in self._parse_fragment(source_unit).iter("g")}
                candidates = ([target_unit] if line_index in self.translated_units
                              else self._sentence_fragments(target_unit))
                for sentence in candidates:
                    try:
                        target_root = self._parse_fragment(sentence)
                    except ET.ParseError:
                        continue
                    for marker in target_root.iter("g"):
                        if marker.get("id") in unit_ids:
                            target_options.setdefault(marker.get("id"), []).append(marker)
            markers = list(source_root.iter("g"))
            kinds = (self.variant_marker_kinds[index] if self.variant_marker_kinds else [
                "option" if m.get("id", "").startswith("fraus-option-") else "text"
                for m in markers])
            options = [m for m, kind in zip(markers, kinds) if kind == "option"]
            for source_marker, key in zip(options, self.variant_sequence[index]):
                matches = target_options.get(source_marker.get("id"), [])
                if len(matches) == 1 and self._shape(matches[0]) == self._shape(source_marker):
                    value = "".join(matches[0].itertext()).strip()
                    if value:
                        contextual_values.setdefault(key, value)
        pending = set(range(len(self.variant_sequence))) - set(variant_indices)
        offset = 0
        for record in self.variant_records.values():
            record_indices = set(range(offset, offset + record["variants"]))
            offset += record["variants"]
            if not pending & record_indices:
                continue
            pending.update(record_indices)
            self._recover_unmapped_ra(record, translate_one, contextual_values, context_translations)
        for line_index, variant_index, source_line, target_line in zip(
                source_variant_indices, variant_indices, source_variants, target_variants):
            if variant_index in pending:
                continue
            source_root = self._parse_fragment(source_line.rstrip("\r\n"))
            source_markers = list(source_root.iter("g"))
            kinds = (self.variant_marker_kinds[variant_index]
                     if self.variant_marker_kinds else [
                         "option" if marker.get("id", "").startswith("fraus-option-")
                         else "text" for marker in source_markers])
            option_markers = [marker for marker, kind in zip(source_markers, kinds)
                              if kind == "option"]
            select_ids = self.variant_sequence[variant_index]
            if len(option_markers) != len(select_ids):
                raise AssertionError("FRAUS option marker count changed during recovery")
            if not self.variant_marker_kinds and [m.get("id") for m in option_markers] != [
                    f"fraus-option-{key}" for key in select_ids]:
                raise AssertionError("FRAUS option marker identity changed during recovery")
            try:
                target_root = self._parse_fragment(target_line.rstrip("\r\n"))
            except ET.ParseError:
                target_root = ET.Element("root")
            option_ids = {marker.get("id") for marker in option_markers}
            context_valid = self._context_valid(source_root, target_root, option_ids)
            owned_units = self.translated_units.get(line_index)
            if owned_units is not None:
                for source_unit, target_unit in owned_units:
                    try:
                        unit_valid = self._context_valid(
                            self._parse_fragment(source_unit), self._parse_fragment(target_unit), option_ids)
                    except ET.ParseError:
                        unit_valid = False
                    context_valid = context_valid and unit_valid
            values = {}
            for source_marker, select_id in zip(option_markers, select_ids):
                value = contextual_values.get(select_id)
                source_value = "".join(source_marker.itertext()).strip()
                if not value and source_value:
                    value = self.fallback_values.get(select_id)
                    if value is None:
                        value = self._standalone_value(translate_one(
                            html.escape(source_value, quote=False) + "\n"))
                        if not value:
                            raise FrausTranslationError("FRAUS option recovery produced no usable translation")
                        self.fallback_diagnostics.append({
                            "type": "option_retry", "variant_index": variant_index,
                            "option_key": select_id, "strategy": "standalone",
                        })
                values[source_marker.get("id")] = value or ""
                self.fallback_values.setdefault(select_id, value or "")

            correct_index = correct_indices.get(variant_index, variant_index)
            if not context_valid:
                self.fallback_diagnostics.append({
                    "type": "unsafe_context_variant", "variant_index": variant_index,
                    "action": "recover_sentence_context" if correct_index == variant_index
                    else "reuse_correct_context",
                })
                if correct_index in accepted_contexts:
                    target_root = copy.deepcopy(accepted_contexts[correct_index])
                    for marker, source_marker in zip(target_root.iter("g"), source_markers):
                        marker.attrib = dict(source_marker.attrib)
                else:
                    if owned_units is not None:
                        source_sentences = [source for source, _ in owned_units]
                        target_sentences = [target for _, target in owned_units]
                    else:
                        source_sentences = self._sentence_fragments(source_line)
                        target_sentences = self._sentence_fragments(target_line)
                        if (len(source_sentences) != len(target_sentences)
                                or any(not list(self._parse_fragment(sentence).iter("g"))
                                       for sentence in source_sentences)):
                            source_sentences, target_sentences = [source_line], [""]
                        else:
                            for sentence, candidate in zip(source_sentences, target_sentences):
                                expected_ids = {m.get("id") for m in self._parse_fragment(sentence).iter("g")}
                                try:
                                    actual_ids = {m.get("id") for m in self._parse_fragment(candidate).iter("g")}
                                except ET.ParseError:
                                    continue
                                if not actual_ids <= expected_ids:
                                    source_sentences, target_sentences = [source_line], [""]
                                    break
                    recovered_parts = []
                    for index, sentence in enumerate(source_sentences):
                        candidate = (target_sentences[index]
                                     if len(source_sentences) == len(target_sentences) else "")
                        try:
                            valid = self._context_valid(
                                self._parse_fragment(sentence), self._parse_fragment(candidate),
                                option_ids)
                        except ET.ParseError:
                            valid = False
                        recovered_parts.append(candidate if valid else self._recover_context(
                            sentence, translate_one, context_translations,
                            variant_index, option_ids))
                    recovered = "".join(recovered_parts)
                    target_root = self._parse_fragment(recovered.rstrip("\r\n"))
            for marker in target_root.iter("g"):
                if marker.get("id") in values:
                    marker[:] = []
                    marker.text = values[marker.get("id")]
            if correct_index == variant_index:
                accepted_contexts[variant_index] = copy.deepcopy(target_root)
            recovered = ET.tostring(target_root, encoding="unicode")
            recovered_lines[line_index] = self._preserve_line_ending(
                source_line, recovered[len("<root>"):-len("</root>")])
        return "".join(recovered_lines)

    def _recover_unmapped_ra(self, record, translate_one, contextual_values, translations):
        # Extraction ownership is uncertain, but the original RA's XML positions
        # are not. Recover there without changing any unidentified output lines.
        source = copy.deepcopy(record["original"])
        for node in source.iter("ExText"):
            node.text = _TAG_TOKEN_RE.sub("", html.unescape(node.text or ""))
        source.set(f"{{{self.namespace}}}source-ra", record["source_id"])
        temporary = FrausV2XmlTransform()
        temporary._flatten_context(source, 0)
        payload = source.find("ExText")
        source_line = html.escape(payload.text or "", quote=False) + "".join(
            ET.tostring(child, encoding="unicode") for child in payload)
        option_ids = {marker.get("id") for marker in payload.iter("g")
                      if marker.get("id", "").startswith("fraus-option-")}
        self.fallback_diagnostics.append({
            "type": "ambiguous_ra_recovery", "source_id": record["source_id"],
            "strategy": "original_xml_boundaries",
        })
        recovered = self._recover_context(source_line, translate_one, translations, None, option_ids)
        record["recovered_context"] = self._parse_fragment(recovered)
        for option_index, option in enumerate(record["original"].findall("InputOption")):
            for select_index, select in enumerate(option.findall("SelectOption")):
                key = f"{record['source_id']}-{option_index}-{select_index}"
                value = contextual_values.get(key)
                if value is None:
                    source_value = _TAG_TOKEN_RE.sub("", html.unescape(select.findtext("ExText") or ""))
                    value = (self._standalone_value(translate_one(
                        html.escape(source_value, quote=False) + "\n")) if source_value.strip() else source_value)
                    if source_value.strip() and not value:
                        raise FrausTranslationError("FRAUS option recovery produced no usable translation")
                self.fallback_values[key] = value

    @staticmethod
    def _shape(node):
        return (node.tag, node.get("id"), tuple(FrausV2XmlTransform._shape(child) for child in node))

    def _context_valid(self, source, target, option_ids):
        if self._shape(source) != self._shape(target):
            return False
        for source_node, target_node in zip(source.iter(), target.iter()):
            if source_node.get("id") not in option_ids:
                if (source_node.text or "").strip() and not (target_node.text or "").strip():
                    return False
            if (source_node.tail or "").strip() and not (target_node.tail or "").strip():
                return False
        return True

    def _sentence_fragments(self, line):
        """Split only at sentence boundaries outside paired spans."""
        def match_offsets(text, sentence, cursor):
            # The splitter can collapse whitespace; match against the original
            # text so both offsets include the actual whitespace widths.
            pattern = r"\s*(" + r"\s+".join(
                re.escape(word) for word in sentence.split()) + ")"
            match = re.compile(pattern).match(text, cursor)
            return match.span(1) if match else None

        try:
            root = self._parse_fragment(line)
        except ET.ParseError:
            # A malformed later sentence need not invalidate an independently
            # parseable earlier one. Never try to repair the malformed markup.
            sentences = split_text_into_sentences(line.strip(), self.language)
            fragments = []
            cursor = 0
            for sentence in sentences:
                offsets = match_offsets(line, sentence, cursor)
                if offsets is None:
                    return [line]
                start, end = offsets
                if fragments:
                    fragments[-1] += line[cursor:start]
                else:
                    start = 0
                fragments.append(line[start:end])
                cursor = end
            if line[cursor:].strip():
                return [line]
            if fragments:
                fragments[-1] += line[cursor:]
            return fragments or [line]
        plain = "".join(root.itertext())
        sentences = split_text_into_sentences(plain.strip(), self.language)
        starts = [0]
        cursor = 0
        for index, sentence in enumerate(sentences):
            offsets = match_offsets(plain, sentence, cursor)
            if offsets is None:
                return [line]
            start, cursor = offsets
            if index:
                starts.append(start)
        if plain[cursor:].strip():
            return [line]
        starts.append(len(plain))
        spans = []
        cursor = len(root.text or "")
        for child in root:
            if not isinstance(child.tag, str):
                return [line]
            end = cursor + len("".join(child.itertext()))
            # Group inseparable sentences without losing safe boundaries
            # before or after this span (including any nested markup).
            starts = [start for start in starts if not cursor < start < end]
            spans.append((cursor, end, child))
            cursor = end + len(child.tail or "")
        fragments = []
        for start, end in zip(starts, starts[1:]):
            pieces = []
            cursor = start
            for child_start, child_end, child in spans:
                # The final unit also owns terminal empty elements, including
                # tag-only fragments whose text range is [0, 0].
                if (start <= child_start < end
                        or child_start == child_end == end == len(plain)):
                    pieces.append(html.escape(plain[cursor:child_start], quote=False))
                    marker = copy.deepcopy(child)
                    marker.tail = None
                    pieces.append(ET.tostring(marker, encoding="unicode"))
                    cursor = child_end
            pieces.append(html.escape(plain[cursor:end], quote=False))
            fragments.append("".join(pieces))
        return fragments or [line]

    def _recover_context(self, source_line, translate_one, translations,
                         variant_index, option_marker_ids):
        if source_line.endswith("\r\n"):
            body, line_ending = source_line[:-2], "\r\n"
        elif source_line.endswith("\n"):
            body, line_ending = source_line[:-1], "\n"
        else:
            body, line_ending = source_line, ""
        root = self._parse_fragment(body)

        tokens = []

        def is_option(node):
            return node.tag == "g" and node.get("id") in option_marker_ids

        def collect(node):
            if node.text is not None:
                tokens.append(("text", node, "text"))
            for child in node:
                if is_option(child):
                    tokens.append(("option", child, None))
                elif isinstance(child.tag, str):
                    collect(child)
                if child.tail is not None:
                    tokens.append(("text", child, "tail"))

        collect(root)

        groups = [[]]
        for kind, node, attribute in tokens:
            if kind == "option":
                groups.append([])
            elif kind == "text":
                groups[-1].append((node, attribute, getattr(node, attribute)))
        source_groups = ["".join(value for _, _, value in group) for group in groups]
        if not any(value.strip() for value in source_groups):
            return source_line

        # A blank identifies a context run, not boundaries between adjacent
        # ExText nodes or inline formatting. Those require XML-based isolation.
        assignable = all(sum(bool(value.strip()) for _, _, value in group) <= 1
                         for group in groups)
        accepted = None
        for placeholder in self.context_placeholders:
            if any(placeholder in value for value in source_groups):
                continue
            source_value = placeholder.join(source_groups)
            cache_key = ("sentence", placeholder, tuple(source_groups))
            cached = cache_key in translations
            target_value = translations.get(cache_key)
            if target_value is None:
                target_value = self._standalone_value(translate_one(
                    html.escape(source_value, quote=False) + "\n"))
            if not target_value:
                raise FrausTranslationError("FRAUS context recovery produced no usable translation")
            parts = target_value.split(placeholder)
            valid = (assignable and len(parts) == len(groups)
                     and not any(other in target_value for other in self.context_placeholders
                                 if other != placeholder)
                     and all(bool(source.strip()) == bool(target.strip())
                             for source, target in zip(source_groups, parts)))
            self.fallback_diagnostics.append({
                "type": "context_cache_hit" if cached else "context_retry",
                "variant_index": variant_index, "strategy": placeholder,
                "accepted": valid,
            })
            if valid:
                translations[cache_key] = target_value
                accepted = parts
                break

        for group_index, group in enumerate(groups):
            for node, attribute, source_value in group:
                if not source_value.strip():
                    continue
                if accepted is not None:
                    target_value = accepted[group_index].strip()
                else:
                    cache_key = ("isolated_context", source_value.strip())
                    cached = cache_key in translations
                    target_value = translations.get(cache_key)
                    if target_value is None:
                        target_value = self._standalone_value(translate_one(
                            html.escape(source_value.strip(), quote=False) + "\n"))
                        if (not target_value or any(marker in target_value
                                for marker in self.context_placeholders)):
                            raise FrausTranslationError("FRAUS context recovery produced no usable translation")
                        translations[cache_key] = target_value
                    self.fallback_diagnostics.append({
                        "type": "context_cache_hit" if cached else "context_retry",
                        "variant_index": variant_index, "strategy": "isolated_context",
                        "accepted": True,
                    })
                leading = source_value[:len(source_value) - len(source_value.lstrip())]
                trailing = source_value[len(source_value.rstrip()):]
                setattr(node, attribute, leading + target_value + trailing)

        recovered = ET.tostring(root, encoding="unicode")
        return recovered[len("<root>"):-len("</root>")] + line_ending

    @staticmethod
    def _option_marker_ids(line):
        if "fraus-option-" not in line:
            return []
        try:
            root = ET.fromstring(f"<root>{line.strip()}</root>")
        except ET.ParseError:
            return []
        return [
            marker.get("id") for marker in root.iter("g")
            if marker.get("id", "").startswith("fraus-option-")
        ]

    def _variant_line_indices(self, lines):
        if not self.variant_source_payloads:
            return [
                index for index, line in enumerate(lines)
                if self._option_marker_ids(line)
            ]
        mapping = self._map_source_variants(lines)
        # Only full source signatures authorize these roles. Target detection
        # cannot reinterpret a missing option as an unrelated formatting span.
        for _, variant_index, kinds in mapping:
            self.variant_marker_kinds[variant_index] = kinds
        return [line_index for line_index, _, _ in mapping]

    def _map_source_variants(self, lines):
        """Return (source line index, variant list index, g roles) from provenance.

        Okapi renumbers XML codes, then adds HTML codes and renumbers again.
        Match text and nested code boundaries instead of either set of IDs.
        Ambiguous or incomplete associations are deliberately not guessed.
        """
        from html.parser import HTMLParser

        class SignatureParser(HTMLParser):
            def __init__(self):
                super().__init__(convert_charrefs=True)
                self.events = []
                self.kinds = []
                self.stack = []
                self.text = []
                self.valid = True

            def flush(self):
                value = " ".join("".join(self.text).split())
                if value:
                    self.events.append(("text", value))
                self.text = []

            def handle_data(self, data):
                self.text.append(data)

            def handle_starttag(self, tag, attrs):
                self.flush()
                if tag in ("area", "base", "br", "col", "embed", "hr", "img",
                           "input", "link", "meta", "param", "source", "track", "wbr"):
                    self.events.append(("empty",))
                    return
                self.stack.append(tag)
                self.events.append(("start",))
                marker_id = dict(attrs).get("id", "")
                self.kinds.append(
                    "option" if tag == "g" and marker_id.startswith("fraus-option-")
                    else "text" if tag == "g" and marker_id.startswith("fraus-text-")
                    else "format")

            def handle_endtag(self, tag):
                self.flush()
                if not self.stack or self.stack.pop() != tag:
                    self.valid = False
                self.events.append(("end",))

            def handle_startendtag(self, tag, attrs):
                self.flush()
                self.events.append(("empty",))

            def signature(self, value):
                self.feed(value)
                self.close()
                self.flush()
                return tuple(self.events) if self.valid and not self.stack else None

        expected = {}
        for variant_index, payload in enumerate(self.variant_source_payloads):
            parser = SignatureParser()
            signature = parser.signature(payload)
            if signature is None:
                return []
            expected.setdefault(signature, []).append((variant_index, parser.kinds))
        candidates = {signature: [] for signature in expected}
        for line_index, line in enumerate(lines):
            try:
                root = ET.fromstring(f"<root>{line.strip()}</root>")
            except ET.ParseError:
                continue
            signature = SignatureParser().signature(line)
            if (signature in candidates
                    and len(list(root.iter("g"))) == len(expected[signature][0][1])):
                candidates[signature].append(line_index)
        mapping = []
        for signature, variants in expected.items():
            if len(candidates[signature]) != len(variants):
                continue
            mapping.extend((line_index, variant_index, kinds)
                           for line_index, (variant_index, kinds)
                           in zip(candidates[signature], variants))
        mapping.sort()
        indices = [variant_index for _, variant_index, _ in mapping]
        if indices != sorted(indices):
            return []
        return mapping

    @staticmethod
    def _parse_fragment(value):
        parser = ET.XMLParser(target=ET.TreeBuilder(
            insert_comments=True,
            insert_pis=True,
        ))
        return ET.fromstring(f"<root>{value}</root>", parser=parser)

    @staticmethod
    def _preserve_line_ending(source_line, target_line):
        line_ending = (
            "\r\n" if source_line.endswith("\r\n")
            else "\n" if source_line.endswith("\n")
            else ""
        )
        target_body = target_line.rstrip("\r\n")
        if "\n" in target_body or "\r" in target_body:
            raise AssertionError("FRAUS line recovery changed line count")
        return target_body + line_ending

    @staticmethod
    def _standalone_value(translation):
        value = translation.strip()
        if not value:
            return ""
        if _TAG_TOKEN_RE.search(value):
            raise FrausTranslationError("FRAUS recovery returned markup instead of plain text")
        return html.unescape(value)

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
    def __init__(self, method, src, tgt, model=None, custom_prompt=None, terms=None, split=True, strict=False):
        self.method = method
        self.src = src
        self.tgt = tgt
        self.model = model
        self.split = split
        self.custom_prompt = custom_prompt
        self.debug_segments = []
        self.strict = strict

    def translate(self, input_text: str, split=True) -> Tuple[List[str], List[str]]:
        from app.models.llm_request_state import get_request_llm_state

        if self.strict and not input_text.strip():
            return [input_text], [input_text]
        state = get_request_llm_state() if self.strict else None
        record_count = len(state.records) if state is not None else 0
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

        if self.strict:
            if state is not None:
                for record in state.records[record_count:]:
                    if not record.translated:
                        raise record.error or FrausTranslationError(
                            'FRAUS v2 backend retained untranslated source'
                        )
            if len(src_sentences) != len(tgt_sentences):
                raise FrausTranslationError('FRAUS v2 backend returned mismatched sentence counts')
            if input_text.strip() and (
                    not tgt_sentences or not ''.join(tgt_sentences).strip()
                    or any(source.strip() and not target.strip()
                           for source, target in zip(src_sentences, tgt_sentences))):
                raise FrausTranslationError('FRAUS v2 backend returned empty text')

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
            src_sentences = [
                s + " " if s and not s[-1].isspace() else s
                for s in src_sentences
            ]
            tgt_sentences = [
                s + " " if s and not s[-1].isspace() else s
                for s in tgt_sentences
            ]
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
            self.finalize_llm_translation()
        except Exception:
            self._cleanup_work_dir()
            raise

    def translate_with_model(self, model, src, tgt, custom_prompt=None, terms=None, split=True):
        self._fallback_diagnostics = []
        try:
            self._extract_translate_merge(src, tgt, "with_model", model, custom_prompt, terms, split)
            self.finalize_llm_translation()
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
        return self._run_document_pipeline(
            document_format, src, tgt, method, model, custom_prompt, terms, split
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
        if debug and isinstance(self.xml_transform, FrausV2XmlTransform):
            self.debug_trace["fraus_sentence_units"] = [
                {"line_index": line_index, "unit_index": unit_index,
                 "source": source, "target": target}
                for line_index, units in self.xml_transform.translated_units.items()
                for unit_index, (source, target) in enumerate(units)
            ]
        if debug and self._fallback_diagnostics:
            self.debug_trace["fallbacks"] = copy.deepcopy(self._fallback_diagnostics)
            if isinstance(self.xml_transform, FrausV2XmlTransform):
                self.debug_trace['fraus_recovery'] = copy.deepcopy(self._fallback_diagnostics)
        if debug:
            from app.models.llm_request_state import get_request_llm_state

            llm_state = get_request_llm_state()
            if llm_state is not None and llm_state.fallback_segments:
                self.debug_trace["llm_fallbacks"] = copy.deepcopy(
                    llm_state.fallback_diagnostics()
                )

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
        strict = isinstance(self.xml_transform, FrausV2XmlTransform)

        def translate_markup(text):
            from app.models.llm_request_state import (
                llm_state_checkpoint,
                rollback_llm_state,
            )

            checkpoint = llm_state_checkpoint()
            translator = InnerLindatTranslator(
                method, src, tgt, model, custom_prompt=custom_prompt,
                terms=terms, split=split,
                strict=strict,
            )
            mt = MarkupTranslator(
                translator, LindatAligner(src, tgt, show_progress=False), RegexTokenizer()
            )
            try:
                result = mt.translate(text)
                result = sanitize_generated_markup(text, result)
            except Exception as error:
                if not strict or isinstance(error, (AssertionError, ValueError)):
                    rollback_llm_state(checkpoint)
                raise
            self.debug_segments.extend(translator.debug_segments)
            return result

        def translate_plain(text):
            """Translate XML-escaped plain text, returning XML-escaped text."""
            translator = InnerLindatTranslator(
                method, src, tgt, model, custom_prompt=custom_prompt,
                terms=terms, split=split, strict=True,
            )
            _, targets = translator.translate(unescape(text), split=split)
            self.debug_segments.extend(translator.debug_segments)
            return escape(''.join(targets), quote=False)

        if strict:
            lines = []
            source_lines = self.text.splitlines(keepends=True)
            variant_indices = set(self.xml_transform._variant_line_indices(source_lines))
            self.xml_transform.translated_units = {}
            for index, line in enumerate(source_lines):
                body = line.rstrip('\r\n')
                ending = line[len(body):]
                if not body.strip():
                    lines.append(line)
                    continue
                units = []
                for unit_index, source_unit in enumerate(self.xml_transform._sentence_fragments(body)):
                    try:
                        candidate = translate_markup(source_unit)
                    except (AssertionError, ValueError) as error:
                        if isinstance(error, ValueError) and 'paired tag' not in str(error).lower():
                            raise
                        candidate = ''
                        if index not in variant_indices and 'fraus-option-' not in source_unit:
                            try:
                                root = self.xml_transform._parse_fragment(source_unit)
                            except ET.ParseError as parse_error:
                                raise FrausTranslationError(
                                    'Cannot preserve markup during FRAUS sentence recovery'
                                ) from parse_error
                            for node in root.iter():
                                for attribute in ('text', 'tail'):
                                    value = getattr(node, attribute)
                                    if value and value.strip() and isinstance(node.tag, str):
                                        target = unescape(translate_plain(escape(value, quote=False)))
                                        leading = value[:len(value) - len(value.lstrip())]
                                        trailing = value[len(value.rstrip()):]
                                        setattr(node, attribute, leading + target.strip() + trailing)
                            candidate = (escape(root.text or '', quote=False) + ''.join(
                                ET.tostring(child, encoding='unicode') for child in root
                            ))
                        self._fallback_diagnostics.append({
                            'type': 'line_alignment_recovery',
                            'line_index': index, 'unit_index': unit_index,
                            'strategy': 'transform_recovery' if not candidate else 'plain_text_nodes',
                            'reason': type(error).__name__,
                        })
                    else:
                        if unescape(re.sub(r'<[^>]*>', '', source_unit)).strip() and not unescape(
                                re.sub(r'<[^>]*>', '', candidate)).strip():
                            raise FrausTranslationError('FRAUS v2 backend returned empty text')
                    # Output sentence counts do not define ownership. Keep the
                    # entire response in its source unit, including its separators.
                    leading = source_unit[:len(source_unit) - len(source_unit.lstrip())]
                    trailing = source_unit[len(source_unit.rstrip()):]
                    candidate = leading + ' '.join(candidate.strip().splitlines()) + trailing
                    units.append((source_unit, candidate))
                self.xml_transform.translated_units[index] = units
                lines.append(''.join(target for _, target in units) + ending)
            self.translation = ''.join(lines)
        else:
            self.translation = translate_with_line_fallback(
                self.text,
                translate_markup,
                lambda line_count: self._fallback_diagnostics.append({
                    "type": "line_alignment_retry",
                    "line_count": line_count,
                }),
            )
        if self.xml_transform is not None and hasattr(self.xml_transform, "fallback"):
            try:
                self.translation = self.xml_transform.fallback(
                    self.text, self.translation, translate_plain if strict else translate_markup
                )
            finally:
                self._fallback_diagnostics.extend(
                    self.xml_transform.fallback_diagnostics
                )
        self._output_word_count = len(self.translation.split())

    def get_text(self):
        return self.text

    def get_translation(self):
        return self.translation

    def create_response(self, extra_headers):
        recovery = (self._fallback_diagnostics
                    if isinstance(self.xml_transform, FrausV2XmlTransform) else [])
        if recovery:
            extra_headers = {**extra_headers, 'X-FRAUS-Recovery': 'recovered'}
        if str(request.values.get('debug', '')).lower() in {'1', 'true', 'yes'}:
            from flask import jsonify
            import base64

            try:
                with open(self.translated_path, 'rb') as translated_file:
                    output = base64.b64encode(translated_file.read()).decode('ascii')
                payload = {
                    'filename': os.path.basename(self.translated_path),
                    'output_base64': output,
                    'trace': self.debug_trace,
                }
                if recovery:
                    payload['fraus_recovery'] = copy.deepcopy(recovery)
                    payload['trace'] = {**self.debug_trace, 'fraus_recovery': copy.deepcopy(recovery)}
                response = jsonify(payload)
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
