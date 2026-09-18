# FRAUS Inline Markup Preservation

These fixes address two local reconstruction failures found in the adaptive-budget
run. Completion budgets, model prompts, source XML, MathML exclusion policy, and
the accepted shared-option-frame trade-off are unchanged.

## Paired And Standalone Codes

Okapi's extracted-text protocol distinguishes paired `g` codes from standalone
`x` codes. ElementTree considers `<g></g>` and `<g />` equivalent XML, but Okapi's
merge does not treat these lexical forms interchangeably.

`_serialize_inline_codes` in `app/main/document.py` serializes trusted extracted
code trees and retains explicit empty `g` pairs. It is used at sentence-fragment,
accepted-context, context-recovery, and strict plain-text-recovery output
boundaries. Standalone `x` codes remain self-closing. Native XML serialization and
escaped literal tag text are not rewritten.

The self-closing `fraus-format-*` tokens used during unmapped-context recovery
remain unchanged because that path restores formatting by exact token spelling.
Using `short_empty_elements=False` globally is unsafe: it also expands `x` codes
and can produce parseable but corrupted formulas.

## Comparisons Are Not Tags

The tag-token recognizer requires a tag name immediately after `<` or `</`, with
valid attribute/closing boundaries. Mathematical text such as `< sin(...)` or
`< f(...)` must not consume the next legitimate opening `g` tag. Unquoted
attributes cannot consume another opening delimiter; quoted values retain
quote-aware handling of `<` and `>`.

Attribute names use HTML-compatible boundaries, including Unicode, `@click`,
`.value`, and digit-leading names. Raw, once-encoded, and twice-encoded tags are
recognized at their own encoding layer. Deeper entities remain attribute data.

`sanitize_generated_markup` still removes generated tag names absent from the
source. It is a translation-markup filter, not a general HTML security sanitizer.
The fix does not disable it, parse and reserialize all HTML, or escape entire
merged lines.

## Verification

Regression tests exercise the public sanitizer, FRAUS transform/recovery, and
full Document translation seams. Real offline Okapi tests are enabled with:

```bash
RUN_FRAUS_OKAPI=1 \
TIKAL_PATH="$PWD/fraus_test_data_translations_2026-09-08/.dependencies/okapi-1.48.0/" \
.venv/bin/python -m unittest \
  test.test_fraus_inline_codes test.test_markup_comparisons \
  test.test_fraus_tag_reinsertion test.test_fraus_formatting_context
```

The final implementation passed eleven full offline replays covering all nine
affected sources: seven identity-backend cases with real extraction/merges and
four exact captured-response cases. The latter checked 189 generation and 254
alignment request bodies before returning saved responses. Final outputs matched
the successful diagnostic preservation controls; no live service calls were
needed for those replays.

Evidence is retained privately under
`fraus_test_data_translations_2026-09-17_adaptive/analysis/markup_fix_validation_final/`.
The seven-source comparison checks decoded formula structure, not merely whether
the outer XML parses. The additional mixed-whitespace/MathML-prose preservation
limitation documented during diagnosis is not solved by these two repairs.

Source-data repair, general entity decoding, superscript alignment, and linguistic
quality remain separate issues. A frozen runtime requires an audited deployment
before a live failure-only retry can use this code.
