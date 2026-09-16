# FLORES+ EDUKATE Experiment

Prepared for the three explicit EDUKATE directions. No API configuration, frozen
FRAUS runtime, or existing translation output is changed by this experiment.
The full plan contains 9,108 requests. The original sequential run was safely
stopped with 4,357 terminal records and no requests in flight; its artifacts are
preserved. This incomplete run is not eligible for quality scoring.

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

Runner 2.0 completes all conditions for English before German, then Ukrainian,
using an async queue with **256 concurrent requests by default**, configurable
with `--concurrency` from 1 through 512. This is a global limit across all three
shot arms of the active model, not 256 per arm; models remain sequential.
It does not use the automatic routing graph or Aya. Changing examples does not
change model weights or require a model reload. The FRAUS document pipeline's
separate four-document limit does not apply to these direct vLLM requests.

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

Without `--execute` or `--prepare-only`, this makes no HTTP requests and creates
no run artifacts. Adding `--prepare-only --run-dir FRESH_RUN_DIR` creates only a
fresh private manifest, lock, and empty claims/records directories, without HTTP
clients or inference. Preparation forbids `--execute` and `--resume` and does not
require execution attestations. It does not authorize later inference.
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

The default per-call timeout is **900 seconds**, with a maximum of **3,600
seconds**, to accommodate vLLM queueing under concurrency. The default and maximum
overall time per invocation is 86,400 seconds. Set bounds with `--call-timeout`
and `--overall-timeout`; the remaining overall deadline may shorten a call.
A timeout cannot guarantee that remote generation stops immediately.
If a sent request's remote completion is uncertain, the runner stops dispatching
new work, drains its local tasks, and does not switch language models. Inspect the
backend before resuming. `--confirm-backend-drained` is an explicit operator
confirmation, valid only with `--execute --resume`; it creates a private audit
record without changing the manifest or replaying the uncertain job.

## Resume And Records

The run directory is new and private (0700); files are private (0600). The
manifest freezes dataset hashes, examples, job identities, model declarations,
configuration and resolved-endpoint fingerprints, generation settings, timeouts,
and code hashes. No tokens or plaintext server URLs are saved.

Add `--resume` with otherwise matching parameters to continue. Raising
`--max-requests` increases the cumulative claim budget. Changed data, examples,
configuration, endpoint, code, decoding settings, concurrency, or call timeout
are rejected, not silently mixed by normal resume.
For imported runs, resume also validates `continuation.json` and the presence and
provenance of every imported claim/record pair. Missing imported results cause
an error rather than retransmitting previously completed inputs.
Claims are durably published before requests and never replayed automatically.
An interrupted claim without a terminal record blocks scoring; it requires
explicit operator investigation, not automatic retranslation.

`records/` stores status, accepted output, finish reason, duration, usage,
request IDs, `execution_concurrency`, and `execution_call_timeout` (seconds).
Failed/truncated outputs are not accepted as translations and are not replaced
by source text. No production files are changed.

## Audited Continuation

The serial-to-parallel transition is an explicit audited import, not a relaxation
of normal resume checks. `scripts/continue_flores_run.py` is a narrow offline
importer for the stopped 4,357-record source: concurrency 1 / 120-second calls to
concurrency 256 / 900-second calls. It checks matching terminal claims and
records, zero unresolved claims, private owned artifacts, manifest fingerprints,
and unchanged data, job identities, examples, model configuration, endpoints,
and decoding settings. Only the runner version/hash and specified execution
policy fields may differ. It is not a general migration tool.

Stop both supervisors before preparation/import, and keep them stopped during
the audit. Finalize and freeze runner/model/data-loader code before preparing
the target; the importer requires the target's runner version/hash to match the
current code. Prepare a **fresh** target with the same full-plan preview arguments
and `--prepare-only --run-dir NEW_RUN --concurrency 256 --call-timeout 900`.
Do not pass `--execute` or `--resume` during preparation.

Audit first, without modifying either run:

```bash
.venv/bin/python scripts/continue_flores_run.py \
  --source OLD_RUN --target NEW_RUN \
  --expected-completed 4357 --confirm-source-stopped
```

Only after reviewing the audit, repeat with `--apply`. The importer preserves
the source and both manifests; imported records/claims bind to the new
fingerprint and carry `execution_concurrency: 1`, `measurement_epoch:
"sequential"`, and `imported_from` containing the old fingerprint and original
claim/record hashes. A private `continuation.json` records the audit. An import
failure leaves `claims/.continuation-incomplete` blocking execution: investigate
and prepare a different fresh target, never remove the sentinel to force resume.

Resume the prepared/imported target with matching plan and execution parameters,
replacing `--prepare-only` with `--execute --resume`, and supply the actual
attestations and `--max-requests 9108`. Imported claims count toward this cumulative
budget and are not retranslated. Do not edit hashed code between preparation,
import, and resume. This procedure is documented here, not executed by scoring.

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
- Execution-policy strata per direction/shot count and overall, with record and
  imported counts, timed/missing counts, and nearest-rank p50/p90/p95/p99 latency,
  grouped by concurrency and call timeout when known.
- Pairwise score differences for 0 vs 5, 0 vs 10, and 5 vs 10.
- Paired sentence bootstrap (default 1,000 resamples, seed 1729) with Holm-adjusted
  p-values across the report's target/metric/comparison family.

Quality scores and paired tests include both audited imported and current
records, using all 1,012 identical evaluation IDs per arm for the full run.
Latency is different: serial and parallel measurements must not be pooled as
comparable performance. The original aggregate latency remains descriptive only,
with an explicit mixed-policy warning and separate execution strata. Counts and
policies come from records, not private application configuration or an import
summary's asserted count. Concurrency is a configured limit, not measured occupancy.

For untouched older runs without record metadata, concurrency falls back to
`manifest.generation.concurrency`, then `generation.batch_size`, then 1; only
integers in 1..512 are accepted (booleans are rejected). Record metadata takes
precedence, so imported serial records are not relabeled with the destination
limit. Call timeout falls back to the manifest only for non-imported records;
an imported record without its own timeout remains unknown (`null`), never
assumed to use the destination's 900 seconds. The scorer does not revalidate the
import migration or require imported records to match current execution limits.
It still refuses incomplete runs before computing any metrics.

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
