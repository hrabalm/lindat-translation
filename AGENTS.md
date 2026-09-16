# Direct Translation Experiments

- For experiments calling the vLLM backend directly, use about **256 concurrent
  requests** by default. Up to **512** is acceptable when useful; vLLM can queue
  requests. Account for queueing when setting request deadlines.
- Finish one language/model's work before switching models to avoid expensive
  weight reloads. Concurrency applies within that model, not across models.
- Keep the FRAUS document pipeline's concurrency policy separate: it uses an
  external word aligner that has rate-limited high-concurrency runs. Its
  four-document limit is not a limit on direct vLLM experiments.
- For FLORES+ experiments, read `docs/flores_plus_experiment.md` for data access,
  private artifacts, explicit EDUKATE routing, and resume/provenance safeguards.
