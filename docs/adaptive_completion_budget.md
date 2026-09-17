# Adaptive Completion Budget

Translation models can opt into a generous source-size-based completion limit.
The global `max_completion_tokens` remains an upper ceiling; few-shot examples
and instruction text do not increase the allowance.

## Configuration

Add this field to an individual OAI model entry:

```json
"adaptive_completion_budget": {
  "min_tokens": 64,
  "tokens_per_input_token": 4,
  "extra_tokens": 32
}
```

Absence disables adaptive budgeting. An empty object selects the defaults above.
Explicit null, unknown keys, invalid types, nonpositive multipliers/minimums, and
negative margins are rejected at model initialization. Boolean values are not
accepted as numbers. Enabling it also requires a positive integer global cap,
or null for no global ceiling.

For estimated source tokens `T`:

```text
upper_bound = max(64, ceil(4 * T) + 32)
effective_max_tokens = min(global_max_completion_tokens, upper_bound)
```

If the global cap is null, use `upper_bound`. A smaller global cap still wins,
even when below the 64-token floor. With a global cap of 512:

| Estimated input tokens | Effective output cap |
| ---: | ---: |
| 1 | 64 |
| 5 | 64 |
| 10 | 72 |
| 25 | 132 |
| 50 | 232 |
| 100 | 432 |
| 120 or more | 512 |

## Estimator And Behavior

The implementation uses the existing `OaiLLMModel._estimated_text_tokens`
heuristic on the actual source segment, including source markup when present.
This is not an exact model-tokenizer count. Instructions, terminology text, chat
wrappers, and demonstration pairs are excluded. Normal instructions and examples
are otherwise unchanged.

All zero-/five-/ten-shot attempts for one source segment receive the same cap.
Transport retries resend that cap. Resplit children get independently computed
caps based on their own text. Per-request limits do not mutate shared model
configuration, so concurrent segments cannot overwrite each other's allowances.

Budget exhaustion retains existing `finish_reason=length` handling: try the next
configured stage, then resplit if possible, then report failure/source fallback
under the existing caller policy. Strict FRAUS still rejects untranslated
fallbacks. Truncated text is never silently accepted. There is no new post-hoc
linguistic gate or repetition classifier.

Enabled request/stage diagnostics record `estimated_input_tokens`,
`adaptive_completion_bound`, and `effective_completion_tokens` without source
text. Experiment probes calculate the same cap for their request-contract checks
and record the actual HTTP `max_tokens` value.

## Calibration

Offline calibration checked all **9,108 successful EDUKATE FLORES+ results**
against actual reported completion usage, covering three directions and all
0/5/10-shot conditions. The source estimator was pinned to the version used in
that experiment. No completion met or exceeded the recommended cap; minimum
headroom was 56 tokens and maximum utilization was 35%.

The largest observed completion/source-estimate ratio was 1.9231, below the
multiplier of four even before the extra allowance. Candidate budgets ranged
from 76 to 444 on those evaluation sentences. This is retrospective length
coverage, not a guarantee of future translation quality or output equivalence.
Those sources have estimates 11-103, so the 64-token floor for tiny fragments is
deliberately generous rather than empirically certified by FLORES+.

For the known FRAUS active-voice explanation, the source estimate is 7 and the
new cap is 64. Its repeated saved target estimates to 144 tokens, but exact
completion usage was not retained in that document trace. A short unrelated
hallucination can still fit: `hlodavci` has source estimate 2 and its incorrect
saved response estimates to 16, below the cap. This change bounds generation
length/cost; it is not a semantic quality detector.

Private calibration artifacts are under `experiments/adaptive_completion_budget/`.
They contain aggregate results, per-record hashes, and reproducible offline
checks; FLORES+ text remains private and Git-ignored.

## Rollout Scope

Initial activation is restricted to `edukate_csen_v1`, `edukate_csde_v1`, and
`edukate_csuk_v1`. EuroLLM and Aya retain their existing global-cap behavior.
The approved full FRAUS rerun uses a new output directory and the normalized
source inventory from the previous completed run. Previous outputs, records,
traces, and configuration are preserved for comparison. Model ordering and
four-document aligner-safe concurrency remain unchanged; this is not a direct
vLLM benchmark run.
