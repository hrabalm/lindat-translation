#!/usr/bin/env python3
"""Score private schema_version=1 FLORES runs without application dependencies.

Run with: uv run --no-project --with sacrebleu==2.5.1 --python 3.11
          scripts/score_flores.py RUN_DIR

The runner writes manifest.json and records/<job_id>.json using the identities
job_id/target/shots/eval_id. Manifest, records, and mandatory durable claims must
share resume_fingerprint: SHA-256 of the manifest without that field, serialized
with json.dumps(sort_keys=True, separators=(',', ':')).encode(). This is run
binding, not cryptographic authentication. Optional duration_seconds is nonnegative.
All planned jobs must have terminal records; claims are never results. Every
non-success status contributes an empty hypothesis, including truncation.
Reports contain aggregates only, not text, IDs, model metadata, or error bodies.
Complete generations publish atomically to reports/<timestamp>-<uuid>/scores.json
and scores.md. Publication requires Linux renameat2(RENAME_NOREPLACE); unsupported
systems fail closed rather than risking replacement or partial reports.
"""

import argparse
from collections import Counter
import ctypes
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import sys
import tempfile
from typing import Any
import uuid


STATUSES = ("success", "truncated", "http_error", "timeout", "invalid_response", "error")
TARGETS = {"en", "de", "uk", "eng_Latn", "deu_Latn", "ukr_Cyrl"}


def read_json(path: Path) -> Any:
    def unique(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate JSON key")
            value[key] = item
        return value
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique)


def score_run(run_dir: Path, bootstrap_samples: int = 1000, seed: int = 1729) -> dict[str, Any]:
    """Validate complete paired coverage before computing any scores."""
    import sacrebleu
    from sacrebleu.metrics.bleu import BLEU
    from sacrebleu.metrics.chrf import CHRF

    if sacrebleu.__version__ != "2.5.1":
        raise ValueError("requires sacrebleu==2.5.1")
    if type(bootstrap_samples) is not int or not 0 <= bootstrap_samples <= 10000:
        raise ValueError("bootstrap samples must be between 0 and 10000")
    if type(seed) is not int or seed < 1:
        raise ValueError("seed must be a positive integer")
    manifest = read_json(run_dir / "manifest.json")
    fingerprint = hashlib.sha256(json.dumps(
        {key: value for key, value in manifest.items() if key != "resume_fingerprint"},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if manifest.get("resume_fingerprint") != fingerprint:
        raise ValueError("manifest fingerprint mismatch")
    if (type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1
            or manifest.get("conditions") != [0, 5, 10]
            or any(type(s) is not int for s in manifest["conditions"])):
        raise ValueError("unsupported manifest schema or conditions")
    models = manifest["models"]
    if not isinstance(models, dict) or not models or not set(models) <= TARGETS:
        raise ValueError("invalid model targets")
    dataset = manifest["dataset"]
    if (dataset.get("repository") != "openlanguagedata/flores_plus"
            or not isinstance(dataset.get("revision"), str)
            or not re.fullmatch(r"[0-9a-fA-F]{40}", dataset["revision"])):
        raise ValueError("invalid dataset provenance claim")
    evaluation = dataset["evaluation"]
    if not isinstance(evaluation, list) or not evaluation:
        raise ValueError("empty evaluation")
    ids = []
    for row in evaluation:
        if (not isinstance(row["id"], str) or not row["id"]
                or set(row["references"]) != set(models)
                or any(not isinstance(text, str) or not text.strip()
                       for text in row["references"].values())):
            raise ValueError("invalid reference coverage")
        ids.append(row["id"])
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate evaluation IDs")
    expected = {(target, shots, identifier) for target in models for shots in (0, 5, 10)
                for identifier in ids}
    jobs = manifest["jobs"]
    planned = {}
    job_ids = set()
    for job in jobs:
        job_id = job["job_id"]
        if (not isinstance(job_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", job_id)
                or job_id in job_ids or type(job["shots"]) is not int):
            raise ValueError("unsafe or duplicate job identity")
        identity = (job["target"], job["shots"], job["eval_id"])
        if identity not in expected or identity in planned:
            raise ValueError("duplicate or unplanned evaluation pair")
        planned[identity] = job
        job_ids.add(job_id)
    if set(planned) != expected:
        raise ValueError("incomplete planned three-arm coverage")
    if {p.name for p in (run_dir / "records").glob("*.json")} != {j + ".json" for j in job_ids}:
        raise ValueError("missing or unplanned terminal records")
    if {p.name for p in (run_dir / "claims").glob("*.json")} != {j + ".json" for j in job_ids}:
        raise ValueError("missing or unplanned durable claim")
    records = {}
    for identity, job in planned.items():
        claim_path = run_dir / "claims" / (job["job_id"] + ".json")
        claim = read_json(claim_path)
        if (any(claim.get(key) != job[key] for key in ("job_id", "target", "shots", "eval_id"))
                or type(claim.get("shots")) is not int
                or claim.get("resume_fingerprint") != fingerprint):
            raise ValueError("claim identity or fingerprint mismatch")
        record = read_json(run_dir / "records" / (job["job_id"] + ".json"))
        if (any(record.get(key) != job[key] for key in ("job_id", "target", "shots", "eval_id"))
                or type(record.get("shots")) is not int
                or record.get("resume_fingerprint") != fingerprint):
            raise ValueError("record identity or fingerprint mismatch")
        if (record.get("status") not in STATUSES or "output" not in record
                or (record["output"] is not None and not isinstance(record["output"], str))
                or "finish_reason" not in record):
            raise ValueError("invalid terminal record")
        if record["status"] == "success" and (
                not isinstance(record["output"], str) or not record["output"].strip()
                or record["finish_reason"] == "length"):
            raise ValueError("invalid success record")
        duration = record.get("duration_seconds")
        if duration is not None and (type(duration) not in (int, float)
                                     or not math.isfinite(duration) or duration < 0):
            raise ValueError("invalid duration")
        records[identity] = record

    def metrics():
        return {"BLEU": BLEU(lowercase=False, tokenize="13a", smooth_method="exp", effective_order=False),
                "chrF++": CHRF(char_order=6, word_order=2, beta=2,
                               lowercase=False, whitespace=False, eps_smoothing=False)}

    report: dict[str, Any] = {
        "schema_version": 1, "sacrebleu_version": sacrebleu.__version__,
        "resume_fingerprint": fingerprint,
        "manifest_sha256": hashlib.sha256((run_dir / "manifest.json").read_bytes()).hexdigest(),
        "pilot": len(evaluation) < 1012, "evaluation_count": len(evaluation),
        "failure_policy": "all non-success records scored as blank hypotheses; no rows dropped",
        "bootstrap": {"samples": bootstrap_samples, "seed": seed,
                      "method": "SacreBLEU paired sentence bootstrap" if bootstrap_samples else "disabled",
                      "multiplicity": "Holm across all target/metric/pair tests in this report",
                      "ties": "Zero observed score difference has p=1; original SacreBLEU p retained separately (its strict comparison can report small p for degenerate ties).",
                      "limitations": "Conditional on one demo set and run; no article clustering. CI summaries are per-system, not delta CIs."},
        "targets": {},
    }
    p_values: list[dict[str, Any]] = []
    for target in sorted(models):
        references = [[row["references"][target] for row in evaluation]]
        hypotheses = {}
        arms: dict[str, Any] = {}
        for shots in (0, 5, 10):
            rows = [records[(target, shots, identifier)] for identifier in ids]
            hypotheses[shots] = [row["output"] if row["status"] == "success" else "" for row in rows]
            counts = Counter(row["status"] for row in rows)
            durations = sorted(row["duration_seconds"] for row in rows if row.get("duration_seconds") is not None)
            scored = {}
            for name, metric in metrics().items():
                score = metric.corpus_score(hypotheses[shots], references)
                scored[name] = {"score": score.score, "signature": f"{score.name}|{metric.get_signature()}"}
            arms[str(shots)] = {
                "metrics": scored, "planned": len(ids), "completed": len(rows),
                "status_counts": {status: counts[status] for status in STATUSES},
                "success_coverage": counts["success"] / len(ids),
                "blank_hypotheses": len(ids) - counts["success"],
                "truncation_rate": counts["truncated"] / len(ids),
                "duration_seconds": {"count": len(durations), "missing": len(ids) - len(durations),
                                     "method": "nearest rank", **{
                                         f"p{p}": durations[math.ceil(p / 100 * len(durations)) - 1] if durations else None
                                         for p in (50, 90, 95, 99)}},
            }
        comparisons = []
        for baseline, candidate in ((0, 5), (0, 10), (5, 10)):
            comparison = {"baseline": baseline, "candidate": candidate,
                          "deltas": {name: arms[str(candidate)]["metrics"][name]["score"]
                                     - arms[str(baseline)]["metrics"][name]["score"] for name in metrics()}}
            if bootstrap_samples:
                from sacrebleu.significance import PairedTest, Result
                previous_seed = os.environ.get("SACREBLEU_SEED")
                os.environ["SACREBLEU_SEED"] = str(seed)
                try:
                    signatures, results = PairedTest(
                        [(str(baseline), hypotheses[baseline]), (str(candidate), hypotheses[candidate])],
                        metrics(), references, test_type="bs", n_samples=bootstrap_samples, n_jobs=1)()
                finally:
                    if previous_seed is None:
                        os.environ.pop("SACREBLEU_SEED", None)
                    else:
                        os.environ["SACREBLEU_SEED"] = previous_seed
                paired = {}
                for name, signature in signatures.items():
                    base, result = results[name]
                    if (not isinstance(base, Result) or not isinstance(result, Result)
                            or result.p_value is None or base.mean is None or base.ci is None
                            or result.mean is None or result.ci is None):
                        raise ValueError("incomplete paired bootstrap result")
                    paired[name] = {"signature": f"{name}|{signature}",
                                    "sacrebleu_p_value": float(result.p_value),
                                    "p_value": 1.0 if base.score == result.score else float(result.p_value),
                                    "baseline": {"mean": float(base.mean), "ci95_half_width": float(base.ci)},
                                    "candidate": {"mean": float(result.mean), "ci95_half_width": float(result.ci)}}
                    p_values.append(paired[name])
                comparison["paired_bootstrap"] = paired
            comparisons.append(comparison)
        report["targets"][target] = {"conditions": arms, "comparisons": comparisons}
    adjusted = 0.0
    for rank, result in enumerate(sorted(p_values, key=lambda item: item["p_value"])):
        adjusted = max(adjusted, min(1.0, result["p_value"] * (len(p_values) - rank)))
        result["p_value_holm"] = adjusted
        result["significant_holm_0_05"] = adjusted < 0.05
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=1000, help="0 disables; maximum 10000")
    parser.add_argument("--seed", type=int, default=1729)
    args = parser.parse_args()
    try:
        root = args.run_dir.absolute()
        if any(path.is_symlink() for path in (*root.parents, root)):
            raise ValueError("symlink in run path")
        root_info = root.stat()
        if not stat.S_ISDIR(root_info.st_mode) or root_info.st_uid != os.getuid() or root_info.st_mode & 0o077:
            raise ValueError("run directory must be private and owned")
        report = score_run(args.run_dir, args.bootstrap_samples, args.seed)
        directory = root / "reports"
        directory.mkdir(mode=0o700, exist_ok=True)
        info = directory.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError("reports directory must be private")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + "-" + uuid.uuid4().hex
        report["created_at"] = datetime.now(timezone.utc).isoformat()
        markdown = ["# FLORES+ Scores", "", f"SacreBLEU: {report['sacrebleu_version']}",
                    f"Evaluation rows per arm: {report['evaluation_count']}; pilot: {report['pilot']}",
                    "", report["failure_policy"], "", "| Target | Shots | BLEU | chrF++ | Success | Truncation |",
                    "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for target, value in report["targets"].items():
            for shots, arm in value["conditions"].items():
                markdown.append(f"| {target} | {shots} | {arm['metrics']['BLEU']['score']:.3f} | "
                                f"{arm['metrics']['chrF++']['score']:.3f} | {arm['success_coverage']:.3%} | "
                                f"{arm['truncation_rate']:.3%} |")
        markdown.extend(["", "Full signatures, paired deltas, coverage, latency, and bootstrap details:",
                         "", "```json", json.dumps(report, indent=2, allow_nan=False), "```", ""])
        with tempfile.TemporaryDirectory(prefix=".pending-", dir=directory) as staging:
            for suffix, content in (("json", json.dumps(report, indent=2, allow_nan=False) + "\n"),
                                    ("md", "\n".join(markdown))):
                fd = os.open(Path(staging) / f"scores.{suffix}", os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    stream.write(content)
                    stream.flush()
                    os.fsync(stream.fileno())
            fd = os.open(staging, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            # Unlike os.rename, this cannot replace even an empty existing directory.
            rename = ctypes.CDLL(None, use_errno=True).renameat2
            rename.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
            rename.restype = ctypes.c_int
            if rename(-100, os.fsencode(staging), -100, os.fsencode(directory / stamp), 1) != 0:
                raise OSError(ctypes.get_errno(), "atomic report publication failed")
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        print("Scored complete paired run; private JSON and Markdown generation published under RUN_DIR/reports.")
        return 0
    except Exception:
        # Exceptions from parsers/metrics may contain private text or backend data.
        print("Scoring refused: check schema, complete record identities, reference coverage, private report permissions, and sacrebleu==2.5.1. No partial scores emitted.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
