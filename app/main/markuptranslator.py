"""Application-owned fix for whitespace-only paired tag reinsertion.

Adapted from kukas/document-translation, document_translation 0.0.4,
commit e01b1ec10dfdaccdb583d7427547e89616ffc3e8:
https://github.com/kukas/document-translation
Vendored methods: MarkupTranslator.translate and TagReinserter.reinsert_tags.
Released under GNU GPL version 3; see document_translation.LICENSE.
This code is provided WITHOUT ANY WARRANTY.

Modified 2026-09-14: bound both text-anchor scans to the source line and
expand to the whole target line only when a text anchor exists. The translate
override routes paired-tag reinsertion here; all other package behavior is
inherited/delegated unchanged. Upstream hardcodes TagReinserter in translate,
with no constructor parameter or overridable reinsertion hook.

Existing NBSP normalization and whitespace-alignment limitations are unchanged;
this repair preserves the computed alignment span, not unaligned whitespace.

On dependency upgrades, compare these two methods against upstream. Remove
this module and restore the upstream import once its line-local scans are fixed
and test/test_fraus_tag_reinsertion.py passes with the upstream implementation.
"""

from bisect import bisect_left
from collections import defaultdict
import logging
from time import perf_counter
from typing import Dict, List, Set, Tuple

from document_translation.alignedsegments import AlignedSegments
from document_translation.markuptranslator import MarkupTranslator as BaseMarkupTranslator
from document_translation.segmentedtext import PairedTagSegment, SegmentedText, TextSegment
from document_translation.tagreinserter import TagReinserter

logger = logging.getLogger(__name__)


def reinsert_tags(aligned_segments: AlignedSegments) -> AlignedSegments:
    """Reinsert aligned paired tags, including those enclosing only whitespace."""
    tag_stack: List[int] = []
    unique_opening_tags: Dict[int, Tuple[int, PairedTagSegment]] = dict()
    unique_closing_tags: Dict[int, Tuple[int, PairedTagSegment]] = dict()
    tag_to_tgt_indices: defaultdict[int, Set[int]] = defaultdict(set)

    tgt_index_lookup = {tgt: i for i, tgt in enumerate(aligned_segments.tgt)}

    def _find_line_boundaries(segments: SegmentedText):
        line_boundaries = [i for i, seg in enumerate(segments) if str(seg) == "\n"]
        line_boundaries = [-1] + line_boundaries + [len(segments)]
        return line_boundaries
    src_line_boundaries = _find_line_boundaries(aligned_segments.src)
    tgt_line_boundaries = _find_line_boundaries(aligned_segments.tgt)
    assert len(src_line_boundaries) == len(tgt_line_boundaries)

    line = 0
    for src_index, seg in enumerate(aligned_segments.src):
        if str(seg) == "\n":
            line += 1
            if tag_stack:
                raise ValueError(f"Paired tag is not closed in the source text on line {line}.")
        if isinstance(seg, PairedTagSegment):
            if seg.opening_tag:
                tag_stack.append(src_index)
                unique_opening_tags[src_index] = (src_index, seg)
            else:
                tag_src_index = tag_stack.pop()
                unique_closing_tags[tag_src_index] = (src_index, seg)
        else:
            tgt_segments = aligned_segments.alignment.get(seg)
            if len(tgt_segments) > 0:
                for tgt_segment in tgt_segments:
                    for tag_src_index in tag_stack:
                        tag_to_tgt_indices[tag_src_index].add(tgt_index_lookup[tgt_segment])

    if tag_stack:
        raise ValueError("Paired tag is not closed in the source text.")

    assert set(unique_opening_tags.keys()) == set(unique_closing_tags.keys())

    to_insert = defaultdict(list)

    for tag_src_index in unique_opening_tags.keys():
        tagged_tgt_indices = tag_to_tgt_indices[tag_src_index]
        if not tagged_tgt_indices:
            continue
        min_tgt_index = min(tagged_tgt_indices)
        max_tgt_index = max(tagged_tgt_indices)

        opening_src_index, opening_tag = unique_opening_tags[tag_src_index]
        assert opening_src_index == tag_src_index
        closing_src_index, closing_tag = unique_closing_tags[tag_src_index]

        # find the current line
        line_bound_index = bisect_left(src_line_boundaries, tag_src_index)
        left_line_bound = src_line_boundaries[line_bound_index-1]+1
        right_line_bound = src_line_boundaries[line_bound_index]
        assert left_line_bound <= tag_src_index and tag_src_index < right_line_bound

        # Whitespace-only lines have no text anchor; never scan a neighbor line.
        first_text_src_index = left_line_bound
        while (first_text_src_index < right_line_bound
               and not isinstance(aligned_segments.src[first_text_src_index], TextSegment)):
            first_text_src_index += 1
        last_text_src_index = right_line_bound-1
        while (last_text_src_index >= left_line_bound
               and not isinstance(aligned_segments.src[last_text_src_index], TextSegment)):
            last_text_src_index -= 1

        # Without text, retain the aligned span rather than expanding it.
        if (first_text_src_index < right_line_bound
                and opening_src_index <= first_text_src_index
                and closing_src_index >= last_text_src_index):
            logger.info(f"Found a tag {opening_tag.debug_str} that spans the entire line {line_bound_index} in the source.")
            left_tgt_line_bound = tgt_line_boundaries[line_bound_index-1]+1
            right_tgt_line_bound = tgt_line_boundaries[line_bound_index]
            min_tgt_index = left_tgt_line_bound
            max_tgt_index = right_tgt_line_bound-1

        to_insert[min_tgt_index].append(opening_tag)
        # we prepend the closing tag so that it matches the order of the opening tags
        to_insert[max_tgt_index+1].insert(0, closing_tag)

    logger.info("now reinserting paired tags")
    num_inserted = 0
    for reinsert_index in sorted(to_insert.keys()):
        for seg in to_insert[reinsert_index]:
            aligned_segments.insert_segment(reinsert_index + num_inserted, seg)
            aligned_segments.alignment.add(seg, seg)
            num_inserted += 1
    logger.info(f"done reinserting {num_inserted} paired tags")

    return aligned_segments


class MarkupTranslator(BaseMarkupTranslator):
    def translate(self, src: str) -> str:
        timer_start = perf_counter()
        # remove non-breakable spaces
        src = src.replace("\xa0", " ")
        src_segments = SegmentedText.from_string(src)
        src_segments = self.tokenize_segmented_text(src_segments, self.tokenizer)

        src_for_translation, src_segments_to_src_for_translation_alignment = self.translator_view(src_segments)
        src_segments_to_src_for_translation = AlignedSegments(src_segments, src_for_translation, src_segments_to_src_for_translation_alignment)

        logger.info("RUN TRANSLATION")
        timer = perf_counter()
        src_sentences, tgt_sentences = self.translator.translate(str(src_for_translation))
        translation_time = perf_counter() - timer

        src_sentences = SegmentedText.from_sentences(src_sentences)
        src_sentences = self.tokenize_segmented_text(src_sentences, self.tokenizer)
        # prepare source sentences for word alignment
        src_tokens, src_sentences_to_src_tokens_alignment = self.aligner_view(src_sentences)
        src_sentences_to_src_tokens = AlignedSegments(src_sentences, src_tokens, src_sentences_to_src_tokens_alignment)

        # recover the sentence segmentation from src_sentences
        src_for_translation_to_src_sentences = AlignedSegments(src_for_translation, src_sentences)
        logger.debug(":: recover_alignment")
        logger.debug(src_for_translation_to_src_sentences.debug_print())
        src_for_translation_to_src_sentences.recover_alignment()

        tgt_sentences = SegmentedText.from_sentences(tgt_sentences)
        tgt_sentences = self.tokenize_segmented_text(tgt_sentences, self.tokenizer)

        # prepare target sentences for word alignment
        tgt_tokens, tgt_sentences_to_tgt_tokens_alignment = self.aligner_view(tgt_sentences)
        tgt_sentences_to_tgt_tokens = AlignedSegments(tgt_sentences, tgt_tokens, tgt_sentences_to_tgt_tokens_alignment)
        tgt_tokens_to_tgt_sentences = tgt_sentences_to_tgt_tokens.swap_sides()

        logger.info("RUN ALIGNER")
        timer = perf_counter()
        src_tokens_to_tgt_tokens_alignment = self.align_segments(src_tokens, tgt_tokens)
        alignment_time = perf_counter() - timer

        src_tokens_to_tgt_tokens_alignment.recover_newline_alignment()

        src_for_translation_to_tgt_sentences = \
            src_for_translation_to_src_sentences \
            .compose(src_sentences_to_src_tokens) \
            .compose(src_tokens_to_tgt_tokens_alignment) \
            .compose(tgt_tokens_to_tgt_sentences)

        logger.info(":: infer_whitespace_alignment")
        src_for_translation_to_tgt_sentences.infer_whitespace_alignment()

        src_segments_to_tgt_sentences = \
            src_segments_to_src_for_translation \
            .compose(src_for_translation_to_tgt_sentences)

        logger.info("\n:: final alignment before reinserting tags:")
        logger.debug(src_segments_to_tgt_sentences.debug_print())

        logger.info("\n:: reinsert paired tags")
        reinsert_tags(src_segments_to_tgt_sentences)
        logger.debug(src_segments_to_tgt_sentences.debug_print())

        logger.info("\n:: reinsert aligned whitespace")
        TagReinserter.reinsert_whitespace(src_segments_to_tgt_sentences)
        logger.debug(src_segments_to_tgt_sentences.debug_print())

        logger.info("\n:: reinsert missing segments")
        TagReinserter.reinsert_segments(src_segments_to_tgt_sentences)
        logger.debug(src_segments_to_tgt_sentences.debug_print())

        logger.info(f"Translation took {translation_time:.2f} sec")
        logger.info(f"Alignment took {alignment_time:.2f} seconds")
        total = perf_counter() - timer_start
        logger.info(f"Total time {total:.2f} seconds")
        logger.info(f"Total without requests {(total - translation_time - alignment_time):.2f} seconds")
        return str(src_segments_to_tgt_sentences.tgt)
