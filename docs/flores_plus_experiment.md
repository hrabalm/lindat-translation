# FLORES+ EDUKATE Experiment

Prepared for the three explicit EDUKATE directions. No API configuration, frozen
FRAUS runtime, or existing translation output is changed by this experiment.
No evaluation inference has been executed. A validated full-plan dry run reports
9,108 requests.

## Data

Dataset: `openlanguagedata/flores_plus`, revision
`5fec6c13f9e5a4db2f745d4ec0d7c9721ddc4f06` (card version 4.6), CC BY-SA 4.0.
Downloaded from the official repository using the user's existing HF login.
The data root is:

```text
experiments/flores_plus/data/5fec6c13f9e5a4db2f745d4ec0d7c9721ddc4f06/
  raw/dev/{ces_Latn,eng_Latn,deu_Latn,ukr_Cyrl}.jsonl
  raw/devtest/{ces_Latn,eng_Latn,deu_Latn,ukr_Cyrl}.jsonl
  aligned/dev.jsonl
  aligned/devtest.jsonl
  metadata/README.md
  download_manifest.json
  preparation_manifest.json
```

There are 997 aligned dev rows and 1,012 aligned devtest rows. Aligned rows have
`id`, `split`, `source_language`, `source`, and `references` keyed by `en`, `de`,
and `uk`. Joins use normalized IDs, never file line positions. Raw dataset bytes
and text are preserved; numeric IDs become strings in the aligned views.

The initial download manifest records the initial integer-ID validation blocker;
`preparation_manifest.json` records its resolved, successful offline preparation.
Both are retained rather than rewriting history. File hashes and the resolved
revision are recorded. Recheck them without downloading:

```bash
/home/mhn/workspace/translator/.venv/bin/python scripts/prepare_flores_data.py \
  --revision 5fec6c13f9e5a4db2f745d4ec0d7c9721ddc4f06 \
  --output-dir experiments/flores_plus/data/5fec6c13f9e5a4db2f745d4ec0d7c9721ddc4f06 \
  --verify-local
```

Data and run directories are Git-ignored and permission-restricted. Do not publish
the corpus, reference-bearing manifests, or model outputs to a crawlable location.
Retain OLDI attribution and CC BY-SA obligations. Access acceptance and training
contamination are separate: HF login does not establish that these model weights
are free of FLORES training data. Confirm training provenance and that the backend
does not use submitted prompts for training before running evaluation.

## Comparison

| Direction | Explicit frontend model ID | Backend model |
| --- | --- | --- |
| Czech to English | `edukate_csen_v1` | `ufal/edukate_csen_v1` |
| Czech to German | `edukate_csde_v1` | `ufal/edukate_csde_v1` |
| Czech to Ukrainian | `edukate_csuk_v1` | `ufal/edukate_csuk_v1` |

Each of 1,012 devtest source sentences is translated independently with 0, 5,
and 10 examples. These are three direct conditions, not the production recovery
chain. Failures never escalate to another shot count or trigger resplitting.
Transport attempts are limited to one per job.

Demonstrations come from FLORES+ dev, **not** the earlier hand-written exercise
examples. The same ordered ten dev IDs are used across targets. The five-shot
condition uses their first five. Default seed 1729 selects:

```text
700, 914, 14, 477, 560, 52, 254, 713, 151, 266
```

All 997 dev IDs pass the exact-text and article-URL exclusion checks against full
devtest for these four languages. Every evaluation condition sees the same
ordered test IDs. The current target reference is never sent to the model.
Examples use separate user/assistant exchanges with the same configured
instruction as the actual source. Conditions retain the model's generation
settings, including temperature and completion cap; hashes and safe settings are
recorded in the private manifest.

The runner completes all conditions for English before German, then Ukrainian,
with one request in flight. It does not use the automatic routing graph or Aya.
Changing examples does not change model weights or require a model reload.

## Preview

Run from the repository root:

```bash
.venv/bin/python scripts/translate_flores.py \
  --data-dir experiments/flores_plus/data/5fec6c13f9e5a4db2f745d4ec0d7c9721ddc4f06/raw \
  --revision 5fec6c13f9e5a4db2f745d4ec0d7c9721ddc4f06 \
  --targets en de uk \
  --model en=edukate_csen_v1 \
  --model de=edukate_csde_v1 \
  --model uk=edukate_csuk_v1 \
  --model-revision en=unknown \
  --model-revision de=unknown \
  --model-revision uk=unknown
```

Without `--execute`, this makes no HTTP requests and creates no run artifacts.
The actual weights revisions are not known from the configuration; replace
`unknown` with verified revisions when available. These are operator declarations,
not server-enforced pins. Neither a revision label nor this program certifies
training-data cleanliness.

## Execution

Only after the operator can truthfully confirm the dataset terms and training
provenance, create the ignored parent directory and append execution flags to
the preview command:

```bash
mkdir -p experiments/flores_plus/runs
```

For a 10-sentence-per-direction pilot, append:

```text
--limit 10
--execute
--run-dir experiments/flores_plus/runs/edukate-pilot-0-5-10
--max-requests 90
--acknowledge-flores-terms
--confirm-training-data-clean
```

For full devtest, omit `--limit`, use a fresh run directory, and set
`--max-requests 9108`. The flags record the operator's attestations; do not pass
them merely to bypass a check. Pilot results must not be presented as full
FLORES+ scores.

Default and maximum per-call timeout is 120 seconds; the default and maximum
overall time per invocation is 86,400 seconds. Smaller bounds can be supplied
with `--call-timeout` and `--overall-timeout`. A timeout cannot guarantee that
remote generation stops immediately.

## Resume And Records

The run directory is new and private (0700); files are private (0600). The
manifest freezes dataset hashes, examples, job identities, model declarations,
configuration and resolved-endpoint fingerprints, generation settings, timeouts,
and code hashes. No tokens or plaintext server URLs are saved.

Add `--resume` with otherwise matching parameters to continue. Raising
`--max-requests` increases the cumulative claim budget. Changed data, examples,
configuration, endpoint, code, or decoding settings are rejected, not mixed.
Claims are durably published before requests and never replayed automatically.
An interrupted claim without a terminal record blocks scoring; it requires
explicit operator investigation, not automatic retranslation.

`records/` stores status, accepted output, finish reason, duration, usage, and
request IDs. Failed/truncated outputs are not accepted as translations and are
not replaced by source text. No production files are changed.

## Scoring

Scoring needs only the private run and pinned SacreBLEU; it has no model calls or
application dependencies. An isolated temporary uv environment avoids modifying
the application's requirements:

```bash
uv run --no-project --with sacrebleu==2.5.1 --python 3.11 \
  scripts/score_flores.py experiments/flores_plus/runs/edukate-pilot-0-5-10
```

The scorer requires every planned job to have a terminal record and all three
conditions to cover identical IDs. Terminal failures contribute blank hypotheses
to every metric; no arm-specific successful subset is silently substituted.

Reports include:

- Corpus BLEU (13a) and chrF++ with complete SacreBLEU signatures per direction.
- Success coverage, truncation rate, and latency percentiles for each shot count.
- Pairwise score differences for 0 vs 5, 0 vs 10, and 5 vs 10.
- Paired sentence bootstrap (default 1,000 resamples, seed 1729) with Holm-adjusted
  p-values across the report's target/metric/comparison family.

`--bootstrap-samples 0` disables bootstrap; up to 10,000 samples are supported.
Reports are private and non-overwriting. Each complete report pair is published
atomically under `reports/<timestamp>-<uuid>/scores.json` and `scores.md`.
Report publication requires Linux `renameat2(RENAME_NOREPLACE)` and fails closed
if it is unavailable.
Per-system bootstrap confidence intervals are not confidence intervals of the
score difference. One example set and one decoding run do not establish
robustness across example selections, model versions, or domains.

BLEU-13a is not FLORES-200 spBLEU. These sentence-level general-domain scores
also do not assess FRAUS markup, mathematical preservation, or short puzzle
fragments. Keep a separate error review for hallucination and repetition.

## Offline Verification

Loader, preparation, and runner tests use synthetic data and mocked HTTP:

```bash
.venv/bin/python -m unittest test.test_flores_data test.test_prepare_flores_data test.test_translate_flores
```

Scorer tests use the pinned optional dependency in an isolated environment:

```bash
uv run --no-project --with sacrebleu==2.5.1 --python 3.11 \
  python -m unittest test.test_flores_scores
```

Ordinary test discovery explicitly skips scorer tests when SacreBLEU 2.5.1 is
unavailable. These tests never require the downloaded corpus or live inference.
