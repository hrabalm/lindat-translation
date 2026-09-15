# Few-Shot Experiment Results

## Conclusion

The configurable attempt sequence is implemented but is **not enabled in the
production model configuration or frozen FRAUS runtime**. The 97 failed documents
were not retried in this experiment.

Five-shot improved normal termination on the six German probes, but did not fix
Ukrainian `EN`. These are **AI-assisted observations, not human-reviewed quality
approval**. User review is pending before any rollout.

## Method

Exactly 24 sequential backend requests: six held-out synthetic inputs, German and
Ukrainian, zero-shot and direct five-shot. Each five-shot request contained five
user/assistant exchanges using the existing instruction renderer. Both conditions
used the same model, production prompt, temperature (0), stop configuration, and
512-token completion cap for their direction.

Transport retries, stage escalation, and resplitting were disabled for this
controlled comparison. The per-call timeout was 120 seconds. This isolates prompt
effects rather than measuring the latency of the complete fallback chain. The
ordered fallback chain, including resplitting and cancellation, is covered by
mocked-backend regression tests.

The example sources were `K`, `____`, `nadpis`, and two Czech exercise
instructions. None exactly matched a held-out input.

Private raw results and the versioned manifest are in
`/tmp/opencode/few-shot-trial-20260915/`. They contain synthetic-input completions,
not full document outputs. Do not publish raw artifacts indiscriminately: model
responses could echo private instructions. The manifest records source hashes,
config and resolved-endpoint fingerprints, examples, bounds, and the request plan.

## Termination

| Direction | Condition | Normal stop | Token-limit truncation | No observed response |
| --- | --- | ---: | ---: | ---: |
| Czech to German | Zero-shot | 3/6 | 2/6 | 1/6 |
| Czech to German | Five-shot | 6/6 | 0/6 | 0/6 |
| Czech to Ukrainian | Zero-shot | 5/6 | 1/6 | 0/6 |
| Czech to Ukrainian | Five-shot | 5/6 | 1/6 | 0/6 |

All 19 normally stopped responses were accepted by the application. The four
truncated responses and one case with no observed response were not accepted.
Application-retained source text is not counted as a successful translation.

German zero-shot `F` has no recorded HTTP response, finish reason, or token usage;
the call lasted 120.101 seconds and returned an untranslated application fallback.
This is consistent with the timeout bound, but its artifact does not identify the
exact exception or establish whether the backend itself truncated.

## Output Assessment

Unicode escapes in this table denote the exact decoded synthetic inputs or
outputs; repeated responses are summarized, not accepted or trimmed into success.

| Target | Input | Zero-shot | Five-shot | Assessment |
| --- | --- | --- | --- | --- |
| German | `F` | No observed response | `F` | Literal label preserved by five-shot. |
| German | `pov\u011bst` | `der Ruf` repeated to the cap | `Sage` | Five-shot ends normally and selects the legend sense; source is ambiguous without context. |
| German | `EN` | `EN` | `EN` | Identifier preserved. |
| German | `J` | `J` plus repeated invisible characters to the cap | `J` | Five-shot removes observed looping. |
| German | `ot\u00e1zka` | `Frage` | `Frage` | Plausible translation. |
| German | `Dobr\u00fd den.` | `Guten Tag.` | `Guten Tag.` | Plausible translation. |
| Ukrainian | `F` | `F` | `F` | Literal label preserved. |
| Ukrainian | `pov\u011bst` | `\u0421\u043b\u0430\u0432\u0430` | Same | Means fame/glory; may fit renown but not the legend sense. Needs context. |
| Ukrainian | `EN` | `EN:` repeated to the cap | Identical repetition | Neither condition recovers. |
| Ukrainian | `J` | `J` | `J` | Literal label preserved. |
| Ukrainian | `ot\u00e1zka` | `\u0417\u0430\u043f\u0438\u0442\u0430\u043d\u043d\u044f` | Same word with lowercase initial | Plausible translation; capitalization difference only. |
| Ukrainian | `Dobr\u00fd den.` | `\u0414\u043e\u0431\u0440\u0438\u0439 \u0434\u0435\u043d\u044c.` | Same | Plausible translation. |

German zero-shot `J` contains 128 EN SPACE characters and 127 ZERO WIDTH SPACE
characters after the letter. Both Ukrainian `EN` outputs contain 256 occurrences
of `EN:` and are exactly identical. All four confirmed truncations consumed
512 completion tokens. Normally stopped outputs have no surrounding whitespace
or observed repetition.

## Cost And Limits

Recorded per-call durations total 211.016 seconds for zero-shot and 32.739 seconds
for five-shot. These are sums, not independent wall-clock measurements or a
performance guarantee. Five-shot added 211 prompt tokens per comparable German
request and 212 per Ukrainian request. Lower latency here does not imply lower
total token usage; missing usage for German zero-shot `F` is unknown, not zero.

Only one observation per input and condition was made. The corpus is small,
mostly very short, and contains ambiguous identifiers and words. Five-shot
success on these inputs does not predict recovery of all German documents, and
the Ukrainian failure remains unresolved. No new linguistic acceptance gate,
truncation acceptance, or technical-token bypass was introduced.

## Implementation Verification

- Full suite with real offline Okapi and a running workspace frontend:
  **265 tests: 251 passed, 12 failed, 1 errored, 1 skipped**.
- The 12 failures and Swagger error reproduce the previously observed live-suite
  issues: seven absent legacy model routes, five wording/formatting assertions,
  and a missing Swagger response description. The live-aligner opt-in test was
  skipped. All new tests passed.
- Repeated targeted typechecking: new probe and tests are clean; 33 diagnostics
  remain in existing application code (including two pre-existing request-state
  annotations). No newly introduced diagnostics remain.
- Independent standards and spec reviews have no remaining actionable findings.

See [configuration and probe instructions](few_shot_attempts.md). Deployment or
a new failure-only batch campaign requires a separate decision after reviewing
the results above.
