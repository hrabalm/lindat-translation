# Few-Shot Probe

The probe is ready for an explicitly authorized run. Do not deploy or run document batches until the experiment outputs
and translations have been reviewed. The probe does not edit `app/models.json`,
start a backend, or change batch runtime.

## Config Contract

`examples/few_shot_attempts.json` is a public per-model config fragment, not a
replacement models file. `few_shot_attempts` maps exact pairs (`cs-de`, `cs-uk`)
and optionally `all` to a nonempty list of stages. Each stage is a list of
`{"source": "...", "target": "..."}` examples.

The model constructor validates exact directions against that model's supported
pairs. The public fixture contains both pairs; for a single-pair model, include
only its matching entry rather than copying the entire fragment. The probe
automatically isolates the matching pair in each deep-copied model config.

- A specific pair replaces `all`; stages are not merged.
- `[[]]` means zero-shot.
- `[[example1, example2, example3, example4, example5]]` means direct five-shot.
- `[[], [example1, example2, example3, example4, example5]]` means zero-shot,
  then five-shot only after truncation, before resplitting.
- Custom prompts or terminology disable examples.
- Document terminology remains unapplied under existing document behavior;
  terminology presence suppresses examples for that request scope, not globally.
- Each example is a separate user/assistant exchange. Example user inputs and
  the final held-out user input use the same model prompt renderer.

The probe compares only single-stage zero-shot versus direct five-shot. It does
not evaluate escalation policy. Core privacy-safe diagnostics have no dependency
here: HTTPX hooks observe actual requests and responses without modifying them.
An offline mock preflight validates all 24 message sequences and generation
settings before any live transport. Unsupported or incompatible core code fails
closed; do not bypass this check.

## Operator Commands

From the repository root, preview (no config loading, network, or artifacts):

```bash
.venv/bin/python scripts/probe_few_shot.py
```

Only after explicit authorization, with the existing backend already available:

```bash
.venv/bin/python scripts/probe_few_shot.py --execute --model-de edukate_csde_v1 --model-uk edukate_csuk_v1 --max-requests 24 --call-timeout 120 --overall-timeout 2880 --output-dir /tmp/opencode/few-shot-reviewed-trial
```

Select `--model-de MODEL_ID --model-uk MODEL_ID` if the private models config has
multiple eligible OAI models. Otherwise exactly one eligible model per pair is
required. Config is loaded internally; tokens, server URLs and private prompts
are never printed or intentionally saved. No Flask server is started. A minimal
Flask context loads production settings and `LOCAL_SETTINGS` for server resolution.

The output directory must not exist for a new experiment, and its parent must
exist. Omit `--output-dir` for a unique directory under `/tmp/opencode`.

## Bounds And Controls

- Fixed held-out corpus per pair: `F`, `pov\u011bst`, `EN`, unseen `J`,
  `ot\u00e1zka`, `Dobr\u00fd den.` (Unicode escapes describe actual Czech text).
- Six inputs x two pairs x two arms = maximum 24 live requests. The same source
  is tested zero-shot then five-shot, sequentially. No parallel calls or batch job.
- Five public examples include `K`, `____`, heading and two task instructions.
  Overlap between example sources and held-out inputs is rejected.
- Input budget and completion cap default to 512 each, bounded above by 512.
  Input budget is the app's source-segmentation estimate, not a cap on total
  prompt tokens including examples; actual usage is recorded when supplied.
  Because the probe calls the segment path directly without splitting, it rejects
  any budget smaller than any held-out source estimate before preflight or network.
- Deep-copied model config preserves provider, model, prompt, temperature and
  stop behavior. Overrides affect only examples, budgets, concurrency and bounds.
- Retry attempts = 1, resplit depth = 0, one stage per arm. A request hook blocks
  any second HTTP request within a call, even if core behavior changes.
- Per-call wall timeout <=120 seconds, live-loop deadline <=2880 seconds per
  invocation. Remaining overall time shortens the final call. Preflight/setup and
  artifact I/O are outside that live-loop deadline. Cancellation cannot guarantee
  the remote server stops computing; do not immediately replay timed-out calls.
- `--max-requests` selects a prefix of the fixed 24-call plan, not a fresh budget
  on resume. A smaller prefix is an incomplete, potentially unbalanced trial.

## Artifacts And Restart

Directory permissions are 0700, JSON files 0600. Do not publish raw artifacts:
even synthetic-input completions could echo private prompt text. Request bodies,
authorization, prompts, server URLs and backend error bodies are not recorded.

- `manifest.json`: public examples and fingerprints, private config fingerprints
  (not config contents), resolved endpoint fingerprints (never plaintext), model IDs,
  app model/probe code hashes, corpus, safe
  probe arguments and ordered case IDs.
- `cs-PAIR-INDEX-ARM.claim.json`: durable pre-transport claim, source and model ID.
- `cs-PAIR-INDEX-ARM.result.json`: request/completion IDs, model IDs, allowlisted
  generation settings, status, finish reason, usage, duration, raw completion,
  app output and fallback status. Missing metadata remains null/unknown.
- `report-UUID.json`: append-only sanitized termination summary, no output text,
  and an explicit `pending_human_review` correctness status.
- `.lock`: exclusive process lock; concurrent use of one directory is rejected.

Restart with the exact same arguments, adding `--resume`:

```bash
.venv/bin/python scripts/probe_few_shot.py --execute --resume --model-de edukate_csde_v1 --model-uk edukate_csuk_v1 --max-requests 24 --call-timeout 120 --overall-timeout 2880 --output-dir /tmp/opencode/few-shot-reviewed-trial
```

Any claimed call is skipped, including timeout/error and interrupted calls with
no result. Never delete claims to retry. No existing JSON is overwritten. Changed
code, config, resolved endpoints, examples or bound arguments invalidate resume.
Flask settings, including `LOCAL_SETTINGS`, are loaded before the manifest so a
changed default server or interpolated server value also invalidates resume.
A failed preflight
with newly integrated code therefore needs a new directory; inspect existing
claims before authorizing another experiment. A partial/corrupt manifest needs
operator inspection, not automatic repair. Claims left by a crash may consume a
slot without any HTTP request: conservative undercount avoidance is intentional.

Failures print only exception type to avoid leaking secrets. `RuntimeError`
during preflight usually means the few-shot message contract is not integrated;
`ValueError` can indicate ambiguous models, fixture/manifest mismatch or unsafe
output permissions. Inspect code/config locally, not by printing private config.

Review termination separately from correctness. `stop` is not evidence of a
correct translation; `length` is truncation, and app fallback is not a successful
translation. Human review must assess identity preservation, target language,
meaning and unwanted explanation for both arms. No automatic correctness claim,
promotion, production config change, or batch/deployment action is authorized.
