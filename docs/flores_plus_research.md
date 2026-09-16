# FLORES+ Isolated Few-Shot Experiment Research

Checked 2026-09-16 against primary sources only. Research/proposal, not authorization to download data or run the experiment. No inference, dependency installation, or application/configuration changes were performed; no corpus text or credentials are included here.

Implementation follow-up: the user subsequently authorized downloading with their
existing Hugging Face login. The eight pinned JSONL files and pinned README were
downloaded and verified locally. All four languages have 997 dev and 1,012 devtest
rows, with matching ID sets. Contrary to the card's example, actual IDs are
integers. The loader accepts nonnegative integer or nonblank string IDs,
normalizes them to strings, and rejects collisions after normalization. Raw files
remain unchanged. Reusable aligned exports and provenance manifests are private
under `experiments/flores_plus/data/`; no evaluation inference has been run.

## Dataset Identity And Pin

- Use OLDI's `openlanguagedata/flores_plus`, not `facebook/flores` or a FLORES-200 archive. FLORES+ is the actively developed successor to FLORES-200, with corrections and additional varieties. OLDI's GitHub repository is archived (2024-11-12) and directs users to Hugging Face. Historical FLORES-200 scores/data are not automatically interchangeable with FLORES+. [1][3][4]
- The public card currently reports version **4.6**. The official API reports main SHA **`5fec6c13f9e5a4db2f745d4ec0d7c9721ddc4f06`**, last modified **2026-07-27T11:55:51Z**. Recommend this immutable SHA as the candidate experiment pin. This is the observed main revision plus the card's version claim, **not an independently verified stable release tag**: the refs endpoint and changelog returned HTTP 401 without authentication. Verify the pinned README/changelog after authorized access before freezing the manifest. Do not use floating `main` or claim an unverified newer/stable release. [1][2]
- Public splits are `dev` and `devtest`; OLDI does not manage or distribute Meta's separate blind test. There is no training split. Text originates primarily from English Wikipedia-family sources (Wikinews, Wikijunior, Wikivoyage); Czech-source evaluation therefore includes translated source text and, for cs->en, often the original-language reference. [1]

## Languages, Counts, And Schema

The API file inventory confirms `dev/<code>.jsonl` and `devtest/<code>.jsonl` for **each** code below. Counts are the official card's published per-language counts, **not row-count measurements of downloaded files**. The unauthenticated dataset-server size request also returned 401. Actual counts, completeness, and ID equality remain mandatory preflight checks. [1][2]

| Role | Configuration | Glottocode | Published dev | Published devtest |
| --- | --- | --- | ---: | ---: |
| Czech source (`cs`) | `ces_Latn` | `czec1258` | 997 | 1,012 |
| English target (`en`) | `eng_Latn` | `stan1293` | 997 | 1,012 |
| German target (`de`) | `deu_Latn` | `stan1295` | 997 | 1,012 |
| Ukrainian target (`uk`) | `ukr_Cyrl` | `ukra1253` | 997 | 1,012 |

The underscore is part of the real configuration name. None of these four varieties has a split-only exception in the card. Do not infer coverage for other varieties from the general count statement. [1]

Each JSONL record contains `id`, `text`, `iso_639_3`, `iso_15924`, `glottocode`, `variant`, `url`, `domain`, `topic`, `has_image`, `has_hyperlink`, `last_updated`, and `split`. The card shows a string `id`; preserve identifiers rather than treating them as line offsets. `last_updated` is the version where that row changed, not the whole dataset version. A languoid is identified by `(iso_639_3, iso_15924, glottocode, variant)`. **Within a languoid, `(split, id)` uniquely identifies a sentence and aligns translations across languoids.** [1]

Required preflight: select exactly the intended languoid per config; assert unique `(split, id)`, nonempty text, expected split/count, and equal ID sets across the four languages per split. Join on `(split, id)`, never zip raw lines. Fail on duplicates or missing IDs rather than silently taking an inner-join subset; freeze a deterministic joined order for references and every system output. Numeric IDs alone need not identify a sentence across splits.

## Access, License, And Contamination Blockers

- License: **CC BY-SA 4.0**. Preserve attribution/license and identify modifications; sharing adaptations entails ShareAlike obligations. The card also asks users not to publicly redistribute exports unless protected against automatic scraping. Keep corpus, reference-bearing prompts, and text-bearing logs out of public Git/artifacts. Distinguish license terms from the repository's access commitments rather than treating the gate as a different copyright license. [1][5]
- The API reports `gated: auto`. The public landing card is readable, but raw README, changelog, refs, and size metadata requests returned **401** in this unauthenticated research session. That does not establish whether the user's account already has access; no local credentials were inspected. [1][2][6]
- The API exposes two required acknowledgments: **"I agree not to re-host FLORES+ in places where it could be picked up by web crawlers"** and **"If I evaluate using FLORES+, I will ensure that its contents are not in the training data"**. The card says evaluation only, not training. Do not accept these on the user's behalf or bypass the gate with mirrors. [1][2]
- **Unknown model contamination is a real blocker, not proof of cleanliness.** The user's translation models may have been trained/fine-tuned on FLORES, FLORES-200, FLORES+, or their underlying web text; we do not know. Ask for training provenance. If the required assurance cannot be made, seek OLDI clarification or choose another evaluation dataset rather than silently accepting. A dev/devtest split prevents experiment-created leakage, not historical model contamination. Also confirm the inference provider will not retain prompts for training. Report remaining contamination uncertainty with any eventual results.
- Acceptable later fetch: the user reviews/accepts the gate in their browser, authenticates locally with `hf auth login`, then uses official `datasets.load_dataset("openlanguagedata/flores_plus", "ces_Latn", revision=PIN, token=True)` (repeat for the other configs), or official `hf_hub_download`/`snapshot_download` with `repo_type="dataset"`, `revision=PIN`, and only the eight required JSONL files plus provenance documents. Hub download supports full commit hashes and file filters. Keep tokens in the credential store, not code, commands, logs, or this document. No download was executed here. [1][6][7]

## Recommended Experiment Design

Proposal only; obtain language/model and execution approval first.

1. Freeze a manifest with repository, SHA, verified release label, file SHA-256 hashes, actual split counts, languoid identities, ordered evaluation IDs, demonstration IDs/order, selection algorithm/seed, and tool versions. Keep corpus outside the tracked experiment code and production configuration.
2. Use **dev for demonstrations and the entire devtest for evaluation**: expected 1,012 items per target. Choose 10 distinct aligned dev IDs once using a specified deterministic sampler; the 5-shot set is the first five of the same ordered 10; 0-shot has none. Use the same source IDs across targets, with target-specific translations. Do not fine-tune on these examples. Confirm that evaluation-time in-context examples satisfy the accepted terms if unclear.
3. Assert disjoint `(split, id)` demonstration/evaluation keys and check for exact duplicate source text across the sets. Inspect article URLs for overlap; preferentially select demonstrations from articles absent from devtest when feasible, documenting any exceptions. Do not choose examples or tune prompts using devtest references/scores. Freeze demonstrations before evaluation.
4. Hold model revision, prompt template/instruction language, generation parameters, output extraction, input order, context budget, and error/retry policy constant across 0/5/10 shots. Use independent requests without conversational carryover. Never expose the current evaluation target reference in a prompt. Predefine failure handling and report failures; do not drop different rows per condition. Pin model/provider IDs and record whether deterministic decoding is actually supported.
5. Score each direction separately on exactly the same ordered evaluation IDs. If all three targets and all shots are approved, this is **9,108 evaluation translations per model** (3 targets x 3 conditions x 1,012), excluding retries or pilot work. Do not silently run additional demonstration seeds; one frozen set supports a conditional comparison, not a claim robust to demonstration selection.

## Metrics And Paired Comparisons

Use corpus-level **BLEU and chrF++**, with detokenized hypotheses/references and one reference per item. Explicit recommended settings: case-sensitive BLEU, `13a` tokenizer, exponential smoothing, no effective-order adjustment; chrF++ with character order 6, word order 2, beta 2, no character whitespace, case-sensitive, effective-order smoothing (not epsilon smoothing). Default `chrf` alone is **not chrF++**. [8]

Expected ordinary signatures (templates, not measured results; `V` is the eventual pinned installed version):

```text
BLEU|nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:V
chrF2++|nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:V
```

Retain the actual emitted full signatures and scores, not score-only output. BLEU-13a is not FLORES-200 spBLEU: historical comparison would additionally require the explicitly named `flores200` SentencePiece tokenizer and its pinned model/checksum. Do not imply tokenizer choice selects or downloads the FLORES+ evaluation corpus. [4][8]

For later execution only, after a SacreBLEU version is explicitly pinned, an illustrative per-direction command is:

```bash
SACREBLEU_SEED=12345 sacrebleu ref.txt -i hyp_0.txt hyp_5.txt hyp_10.txt \
  -m bleu chrf --tokenize 13a --smooth-method exp \
  --chrf-char-order 6 --chrf-word-order 2 --chrf-beta 2 \
  --paired-bs --paired-bs-n 10000 --paired-jobs 1 -f json
```

SacreBLEU takes the **first system as baseline**, so this tests 0-vs-5 and 0-vs-10, not 5-vs-10; run a separately declared comparison for the latter if desired. The default is 1,000 bootstrap resamples; 10,000 and seed 12345 above are proposed explicit settings. Source code resamples the same sentence indices for baseline and candidate, recomputes corpus scores, and emits p-values plus per-system bootstrap summaries; paired signatures add `bs` and `seed`. Preserve actual scores/deltas, confidence summaries, p-values, seed, and full signatures. Do not substitute overlap of single-system CIs for a paired test or describe per-system CIs as CIs of the difference. [8][9]

Predeclare comparisons and multiplicity handling (e.g. Holm correction across the chosen family). Sentence bootstrap does not model article clustering or variability from different demonstration sets/inference runs; acknowledge that limitation. Treat significance as evidence about score differences, not proof of practical translation quality. SacreBLEU sources were inspected at observed master SHA `c596d9d2072a8f84200574a7a5c56c618e8d37e8`; this is a source citation pin, not a claim about the latest stable package. [9][10]

## Decisions To Ask Before Implementation

- Confirm all three targets (`cs->en`, `cs->de`, `cs->uk`) or a subset, and which model(s)/immutable revisions/endpoints to compare; can each actually consume in-context demonstrations?
- Can the user substantiate the gate's training-data assurance and accept its terms, including provider data handling? Unknown provenance must remain explicit.
- Approve demonstration/instruction language, fixed-example protocol, full devtest versus an initial explicitly labeled pilot, and inference budget. No full run is authorized yet.

## Primary Sources

1. [Official FLORES+ dataset card](https://huggingface.co/datasets/openlanguagedata/flores_plus): version, splits/counts, schema/alignment, uses, license, language coverage, loading instructions.
2. [Official HF dataset API](https://huggingface.co/api/datasets/openlanguagedata/flores_plus): SHA/date, gate acknowledgments, eight file paths. [Compact revision/access metadata](https://huggingface.co/api/datasets/openlanguagedata/flores_plus?expand[]=sha&expand[]=lastModified&expand[]=gated). [Changelog](https://huggingface.co/datasets/openlanguagedata/flores_plus/blob/main/CHANGELOG.md) and [refs](https://huggingface.co/api/datasets/openlanguagedata/flores_plus/refs) were gated; not independently verified.
3. [Official OLDI GitHub repository](https://github.com/openlanguagedata/flores): archived, redirects to HF.
4. [Meta FLORES-200 README](https://github.com/facebookresearch/flores/blob/main/flores200/README.md): historical dataset, hidden test, metrics, successor notice.
5. [CC BY-SA 4.0 official deed and legal-code link](https://creativecommons.org/licenses/by-sa/4.0/).
6. [HF gated-dataset documentation](https://huggingface.co/docs/hub/datasets-gated): browser acceptance and authenticated downloads.
7. [HF official download documentation](https://huggingface.co/docs/huggingface_hub/guides/download): immutable revisions and filtered downloads.
8. [SacreBLEU official README, pinned source](https://github.com/mjpost/sacrebleu/blob/c596d9d2072a8f84200574a7a5c56c618e8d37e8/README.md): metrics, signatures, tokenizers, paired CLI.
9. [SacreBLEU significance implementation, pinned source](https://github.com/mjpost/sacrebleu/blob/c596d9d2072a8f84200574a7a5c56c618e8d37e8/sacrebleu/significance.py): paired resampling, seeds, defaults, results/signatures.
10. [SacreBLEU official commit API](https://api.github.com/repos/mjpost/sacrebleu/commits/master): observed source revision.
