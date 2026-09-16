# FRAUS MathML Extraction

The FRAUS HTML filter keeps mathematical structure inline and protects symbolic
content from translation. It does not exclude the entire formula: language in
`mtext` remains eligible for translation.

## Rules

`app/okapi_profiles/okf_html@fraus.fprm` uses these rules for unprefixed names and
the explicit `m:` and `mml:` spellings:

- Structural elements such as `math`, `mrow`, `msub`, `mfrac`, and `mtable`:
  `INLINE`.
- `mi`, `mn`, and `mo`: `INLINE, EXCLUDE`.
- `mtext`: `INLINE`, without `INCLUDE`, so it cannot reopen excluded annotations.
- `annotation` and `annotation-xml`: `INLINE, EXCLUDE`.
- Math roots also use `PRESERVE_WHITESPACE`. Global whitespace behavior is not
  changed. This local rule can preserve surrounding prose in the same text unit.

Both legacy FRAUS and FRAUS v2 select this HTML profile. These are literal tag
rules, not a general namespace resolver. Other prefix spellings are not covered.

Pure symbolic formulas become code-only extracted text; the existing Document
translation path makes no model calls for those fixtures. Formula symbols are
retained in Okapi reconstruction data, not replaced by guessed translations.
Embedded prose and ordinary text outside formulas still reach the model. Units
and symbols written in `mtext`, including `kg` or `N`, remain translatable: there
is no new unit-detection heuristic.

## Regression Tests

```bash
RUN_FRAUS_OKAPI=1 \
TIKAL_PATH="$PWD/fraus_test_data_translations_2026-09-08/.dependencies/okapi-1.48.0/" \
.venv/bin/python -m unittest test.test_fraus_mathml
```

The tests use the full Document path and real offline Okapi. Only external model
and aligner services are stubbed. They cover both Equation and ExText carriers,
all three prefix forms, a mismatched math root, mixed prose, nested annotations,
nonidentity translations, and literal short labels outside MathML.

Offline full-document comparisons against the previous profile measured:

| Frozen input | Previous model calls | New model calls |
| --- | ---: | ---: |
| `3941.xml` | 759 | 149 |
| `3944.xml` | 583 | 131 |

These are deterministic model-stub counts, not live translation quality or timing
measurements. Standalone `F` and `G` calls disappear; 20 standalone `N` calls
remain in `3941.xml`, consistent with translatable `mtext` labels. All four replay
outputs passed structural validation with no warnings. Baseline and candidate
identity outputs had no differences under the documented decoded normalization.

## Existing Limits

This profile change does not repair malformed MathML, change entity decoding, or
promise byte-for-byte formula preservation. The existing XML pass can collapse
spaces; unescaping can turn `&#x003C;` into a literal `<` inside an operator; and
Tikal can reorder crossed malformed tags. Those behaviors require separate fixes.

The full-document replay also found one pre-existing superscript-boundary shift
in `3941.xml` and twelve in `3944.xml`, identical with both profiles. Passing
structural validation does not establish mathematical or linguistic correctness.

Diagnostic evidence is under
`fraus_test_data_translations_2026-09-08/analysis/mathml_implementation_replay/` and
`analysis/mathml_tikal_probe/` beneath the same run directory. These artifacts are
not required to run the regression tests apart from the local Okapi installation.

The frozen batch runtime has its own profile copy. A workspace change does not
deploy it to the batch; any deployment and retry should be separately audited.
