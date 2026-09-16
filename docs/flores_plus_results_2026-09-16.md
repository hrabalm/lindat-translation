# FLORES+ Few-Shot Results

## Conclusion

All **9,108 translations succeeded**: 1,012 Czech devtest sentences, three
EDUKATE targets, and independent 0/5/10-shot conditions. There were no recorded
truncations, HTTP failures, or timeouts.

The tested examples did not establish a consistent translation-quality
improvement. No pairwise difference is significant after Holm correction across
the 18 target/metric/comparison tests. This general-domain sentence benchmark
does not assess FRAUS markup or the isolated letters that motivated fallback.

## Scores

SacreBLEU 2.5.1; BLEU uses case-sensitive 13a tokenization and exponential
smoothing. chrF++ uses character order 6, word order 2, and beta 2. Full signatures
are stored in the private score report.

| Target | Shots | BLEU | chrF++ |
| --- | ---: | ---: | ---: |
| English | 0 | 42.12 | 66.23 |
| English | 5 | 42.58 | 65.96 |
| English | 10 | 42.85 | 66.04 |
| German | 0 | 29.72 | 56.73 |
| German | 5 | 30.07 | 56.81 |
| German | 10 | 29.90 | 56.62 |
| Ukrainian | 0 | 23.33 | 50.19 |
| Ukrainian | 5 | 23.27 | 50.17 |
| Ukrainian | 10 | 23.21 | 50.15 |

English BLEU increases while chrF++ slightly decreases. German has a small peak
at five shots; Ukrainian scores slightly decrease. The largest English BLEU
gain, ten versus zero shots, is +0.73, with Holm-adjusted p = 0.161838.

Paired sentence bootstrap uses 1,000 resamples and seed 1729, comparing 0 vs 5,
0 vs 10, and 5 vs 10 for both metrics in each target. Confidence summaries are
per-system, not confidence intervals of the differences. One fixed example set
and one inference run do not establish robustness across demonstration selection
or model revisions.

## Provenance

- Dataset: `openlanguagedata/flores_plus`, pinned revision
  `5fec6c13f9e5a4db2f745d4ec0d7c9721ddc4f06`, CC BY-SA 4.0.
- Models: explicit `edukate_csen_v1`, `edukate_csde_v1`, and `edukate_csuk_v1`,
  routed to `ufal/<public-id>`, never automatic Aya/EuroLLM routing.
- The user confirmed the dataset is absent from training data. Actual served
  weight revisions remain unknown/operator-declared rather than verified pins.
- Ten fixed dev demonstrations; five-shot uses the first five. The same devtest
  IDs and references are used across conditions. No test reference is prompted.
- No retries, resplitting, or automatic shot escalation in benchmark requests.
- All claims, terminal records, identities, source artifact hashes, and dataset
  hashes passed the final offline integrity audit. Every job appears exactly once.

## Concurrency Correction

The original runner incorrectly used concurrency 1. It was stopped with 4,357
completed results and no published claims left in flight. Those results and the
original run were preserved. An unpublished temporary claim was retained as
evidence; it had never reached transport.

An audited continuation changed only runner execution policy/code: concurrency
256, a 900-second per-call limit to accommodate queueing, and async scheduling.
Dataset, ordered demonstrations, evaluation rows, model settings, decoding, and
references remained unchanged. The old 120-second measurements are labeled
separately from the new 900-second execution policy; neither limit was hit.

The remaining **4,751 requests completed in 217.95 seconds**, measured from first
new request start to last completion, about 21.8 requests/second including the
target transition. Peak observed concurrency was exactly **256**, with no overlap
between language models:

| Target | New requests | Span | Peak overlap |
| --- | ---: | ---: | ---: |
| German | 1,715 | 80.10 s | 256 |
| Ukrainian | 3,036 | 137.79 s | 256 |

All English results and 1,321 German results came from the preserved sequential
run. Scoring pools quality on identical IDs but reports latency by concurrency
and deadline. Timing is not a controlled serial-versus-parallel comparison:
the two phases cover different languages/sentences and different queue states.

## Artifacts

Private, Git-ignored final run:
`experiments/flores_plus/runs/edukate-full-20260916-c256-final/`

- `analysis/results-summary.md`: detailed safe aggregate summary and comparisons.
- `analysis/audit.json`: independent coverage, integrity, and overlap checks.
- `reports/20260916T172929.212085Z-d9b1d239edef4fd789e2d371873f08d7/scores.json`:
  signatures, bootstrap outputs, corrected p-values, coverage and timing strata.
- `continuation.json`: imported artifact hashes and explicit execution changes.

The original `edukate-full-20260916/` run remains unchanged. Corpus text,
reference-bearing manifests, and outputs remain private and are not included in
this summary. See [experiment operations](flores_plus_experiment.md) for controls.
