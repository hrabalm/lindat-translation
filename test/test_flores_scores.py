"""Exercise the standalone scoring CLI, without application imports/config."""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import hashlib
from importlib.metadata import PackageNotFoundError, version
from io import StringIO
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from typing import Any
import unittest
from unittest.mock import patch

from scripts import score_flores


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/score_flores.py"
try:
    HAS_PINNED_SACREBLEU = version("sacrebleu") == "2.5.1"
except PackageNotFoundError:
    HAS_PINNED_SACREBLEU = False


def fingerprint(manifest):
    return hashlib.sha256(json.dumps(
        {key: value for key, value in manifest.items() if key != "resume_fingerprint"},
        sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@unittest.skipUnless(HAS_PINNED_SACREBLEU, "requires isolated sacrebleu==2.5.1 test environment")
class FloresScoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "records").mkdir()
        (self.root / "claims").mkdir()
        self.manifest: dict[str, Any] = {
            "schema_version": 1, "conditions": [0, 5, 10], "models": {"en": {}},
            "dataset": {"repository": "openlanguagedata/flores_plus", "revision": "a" * 40,
                        "evaluation": [
                            {"id": str(i), "source": "PRIVATE SOURCE",
                             "references": {"en": "The small cat sat on the warm mat ."}}
                            for i in range(3)], "selection": {"pilot": True}},
            "jobs": [],
        }
        for shots in (0, 5, 10):
            for i in range(3):
                job = {"job_id": f"en-{shots}-{i}", "target": "en", "shots": shots,
                       "eval_id": str(i)}
                self.manifest["jobs"].append(job)
                self.write_record(job, "success", "The small cat sat on the warm mat .")
        self.write_manifest()

    def write_manifest(self):
        self.manifest["resume_fingerprint"] = fingerprint(self.manifest)
        (self.root / "manifest.json").write_text(json.dumps(self.manifest), encoding="utf-8")
        for path in (self.root / "records").glob("*.json"):
            record = json.loads(path.read_text())
            record["resume_fingerprint"] = self.manifest["resume_fingerprint"]
            path.write_text(json.dumps(record))
            claim = {key: record[key] for key in ("job_id", "target", "shots", "eval_id", "resume_fingerprint")}
            (self.root / "claims" / path.name).write_text(json.dumps(claim))

    def write_record(self, job, status, output, **extra):
        (self.root / "records" / (job["job_id"] + ".json")).write_text(json.dumps(
            {**job, "status": status, "output": output, "finish_reason": "stop",
             "resume_fingerprint": self.manifest.get("resume_fingerprint"),
             "duration_seconds": 1.5, **extra}), encoding="utf-8")

    def run_score(self, *arguments):
        return subprocess.run([sys.executable, str(SCRIPT), str(self.root),
                               "--bootstrap-samples", "0", *arguments],
                              text=True, capture_output=True, check=False)

    def report(self):
        files = sorted((self.root / "reports").glob("*/scores.json"))
        return json.loads(files[-1].read_text()), files[-1]

    def test_perfect_corpus_scores_signatures_and_private_new_reports(self):
        result = self.run_score()
        self.assertEqual(result.returncode, 0, result.stderr)
        report, path = self.report()
        arm = report["targets"]["en"]["conditions"]["0"]
        self.assertAlmostEqual(arm["metrics"]["BLEU"]["score"], 100)
        self.assertAlmostEqual(arm["metrics"]["chrF++"]["score"], 100)
        self.assertIn("tok:13a", arm["metrics"]["BLEU"]["signature"])
        self.assertIn("nw:2", arm["metrics"]["chrF++"]["signature"])
        self.assertEqual(arm["metrics"]["BLEU"]["signature"],
                         "BLEU|nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.5.1")
        self.assertEqual(arm["metrics"]["chrF++"]["signature"],
                         "chrF2++|nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.5.1")
        self.assertEqual(report["sacrebleu_version"], "2.5.1")
        self.assertEqual(len(report["targets"]["en"]["comparisons"]), 3)
        self.assertEqual(os.stat(path).st_mode & 0o777, 0o600)
        self.assertNotIn("PRIVATE SOURCE", path.read_text() + result.stdout + result.stderr)
        self.assertNotIn("warm mat", path.read_text())
        self.assertEqual(self.run_score().returncode, 0)
        self.assertEqual(len(list((self.root / "reports").glob("*/scores.json"))), 2)
        for generation in (self.root / "reports").iterdir():
            self.assertEqual(generation.stat().st_mode & 0o777, 0o700)
            self.assertEqual({p.name for p in generation.iterdir()}, {"scores.json", "scores.md"})
            self.assertEqual((generation / "scores.md").stat().st_mode & 0o777, 0o600)

    def test_mismatched_claim_is_not_accepted_as_valid_identity(self):
        job = self.manifest["jobs"][0]
        (self.root / "claims" / (job["job_id"] + ".json")).write_text(
            json.dumps({**job, "resume_fingerprint": self.manifest["resume_fingerprint"], "eval_id": "wrong"}))
        self.assertNotEqual(self.run_score().returncode, 0)

    def test_mixed_execution_strata_preserve_corpus_quality_and_signatures(self):
        baseline = score_flores.score_run(self.root, bootstrap_samples=40)
        self.manifest["generation"] = {"batch_size": 256, "concurrency": 256}
        self.manifest["timeouts"] = {"call_seconds": 900}
        for index, job in enumerate(self.manifest["jobs"]):
            extra = {"execution_concurrency": 256, "execution_call_timeout": 900,
                     "duration_seconds": 10 + index}
            if index == 0:
                extra = {"execution_concurrency": 1, "execution_call_timeout": 120,
                         "measurement_epoch": "sequential", "duration_seconds": 2,
                         "imported_from": {"source_resume_fingerprint": "PRIVATE OLD FINGERPRINT",
                                           "source_record_sha256": "PRIVATE HASH"}}
            self.write_record(job, "success", "The small cat sat on the warm mat .", **extra)
        self.write_manifest()
        result = self.run_score("--bootstrap-samples", "40")
        self.assertEqual(result.returncode, 0, result.stderr)
        report, path = self.report()
        policy = report["execution_policy"]
        self.assertTrue(policy["mixed"])
        self.assertEqual(policy["imported_count"], 1)
        self.assertEqual([s["record_count"] for s in policy["strata"]], [1, 8])
        arms = report["targets"]["en"]["conditions"]
        strata = arms["0"]["execution_policy"]["strata"]
        self.assertEqual([s["execution_concurrency"] for s in strata], [1, 256])
        self.assertEqual([s["execution_call_timeout"] for s in strata], [120, 900])
        self.assertEqual([s["record_count"] for s in strata], [1, 2])
        self.assertEqual([s["duration_seconds"]["count"] for s in strata], [1, 2])
        self.assertEqual([s["duration_seconds"]["p50"] for s in strata], [2, 11])
        self.assertEqual([s["duration_seconds"]["p99"] for s in strata], [2, 12])
        self.assertEqual(arms["0"]["duration_seconds"]["count"], 3)
        for shots, arm in arms.items():
            self.assertEqual(arm["metrics"], baseline["targets"]["en"]["conditions"][shots]["metrics"])
            self.assertEqual(arm["completed"], 3)
        self.assertEqual(report["targets"]["en"]["comparisons"], baseline["targets"]["en"]["comparisons"])
        self.assertFalse(arms["5"]["execution_policy"]["mixed"])
        self.assertIn("not a comparable performance measurement", path.with_suffix(".md").read_text())
        self.assertNotIn("PRIVATE", path.read_text())

    def test_legacy_execution_fallbacks_and_record_override(self):
        for generation, expected in (({}, 1), ({"batch_size": 32}, 32),
                                     ({"batch_size": 1, "concurrency": 512}, 512)):
            with self.subTest(generation=generation):
                self.manifest["generation"] = generation
                self.write_manifest()
                before = {p: p.read_bytes() for p in self.root.rglob("*.json")}
                policy = score_flores.score_run(self.root, 0)["execution_policy"]
                self.assertEqual(policy["strata"][0]["execution_concurrency"], expected)
                self.assertEqual(policy["strata"][0]["record_count"], 9)
                self.assertEqual(policy["imported_count"], 0)
                self.assertEqual(before, {p: p.read_bytes() for p in before})
        self.write_record(self.manifest["jobs"][0], "success", "The small cat sat on the warm mat .",
                          execution_concurrency=1, imported_from={}, duration_seconds=None)
        self.manifest["timeouts"] = {"call_seconds": 900}
        self.write_manifest()
        strata = score_flores.score_run(self.root, 0)["execution_policy"]["strata"]
        self.assertEqual(strata[0]["execution_concurrency"], 1)
        self.assertIsNone(strata[0]["execution_call_timeout"])
        self.assertEqual(strata[0]["duration_seconds"]["missing"], 1)
        self.assertIsNone(strata[0]["duration_seconds"]["p95"])
        self.assertEqual(strata[1]["execution_call_timeout"], 900)

    def test_timeout_changes_split_same_concurrency(self):
        for job, timeout in zip(self.manifest["jobs"][:3], (120, 900, None)):
            self.write_record(job, "success", "The small cat sat on the warm mat .",
                              execution_concurrency=256, execution_call_timeout=timeout)
        arm = score_flores.score_run(self.root, 0)["targets"]["en"]["conditions"]["0"]
        self.assertTrue(arm["execution_policy"]["mixed"])
        strata = arm["execution_policy"]["strata"]
        self.assertEqual([s["execution_call_timeout"] for s in strata], [None, 120, 900])
        self.assertEqual([s["record_count"] for s in strata], [1, 1, 1])

    def test_invalid_execution_metadata_is_rejected(self):
        job = self.manifest["jobs"][0]
        for value in (-1, 0, True, False, 513, 1.5, "256", None):
            with self.subTest(concurrency=value):
                self.write_record(job, "success", "valid output", execution_concurrency=value)
                with self.assertRaisesRegex(ValueError, "invalid execution concurrency"):
                    score_flores.score_run(self.root, 0)
                self.write_record(job, "success", "valid output")
                self.manifest["generation"] = {"batch_size": value}
                self.write_manifest()
                with self.assertRaisesRegex(ValueError, "invalid execution concurrency"):
                    score_flores.score_run(self.root, 0)
                self.manifest.pop("generation")
                self.write_manifest()
        for value in (-1, 0, True, 3601, float("nan"), float("inf"), "900"):
            with self.subTest(timeout=value):
                self.write_record(job, "success", "valid output", execution_call_timeout=value)
                with self.assertRaisesRegex(ValueError, "invalid execution call timeout"):
                    score_flores.score_run(self.root, 0)

    def test_paired_bootstrap_is_seeded_and_equal_scores_are_not_significant(self):
        result = self.run_score("--bootstrap-samples", "200")
        self.assertEqual(result.returncode, 0, result.stderr)
        report, _ = self.report()
        pairs = report["targets"]["en"]["comparisons"]
        for pair in pairs:
            for metric in pair["paired_bootstrap"].values():
                self.assertIn("seed:1729", metric["signature"])
                self.assertIn("bs:200", metric["signature"])
                self.assertFalse(metric["significant_holm_0_05"])
                self.assertEqual(metric["p_value"], 1.0)
        self.assertEqual(self.run_score("--bootstrap-samples", "200").returncode, 0)
        repeated, _ = self.report()
        self.assertEqual(report["targets"], repeated["targets"])

    def test_failures_are_blank_not_source_or_partial_output_and_keep_all_rows(self):
        for status in ("truncated", "http_error", "timeout", "invalid_response", "error"):
            with self.subTest(status=status):
                for job in self.manifest["jobs"]:
                    if job["shots"] == 5:
                        self.write_record(job, status, "The small cat sat on the warm mat .",
                                          finish_reason="length" if status == "truncated" else None)
                result = self.run_score()
                self.assertEqual(result.returncode, 0, result.stderr)
                report, _ = self.report()
                arms = report["targets"]["en"]["conditions"]
                self.assertEqual(arms["5"]["metrics"]["BLEU"]["score"], 0)
                self.assertEqual(arms["5"]["metrics"]["chrF++"]["score"], 0)
                self.assertEqual(arms["5"]["blank_hypotheses"], 3)
                self.assertEqual(arms["5"]["status_counts"][status], 3)
                self.assertEqual(arms["5"]["truncation_rate"], int(status == "truncated"))
                self.assertEqual(arms["5"]["duration_seconds"]["p95"], 1.5)
                self.assertEqual([arm["completed"] for arm in arms.values()], [3, 3, 3])
                self.assertAlmostEqual(report["targets"]["en"]["comparisons"][0]["deltas"]["BLEU"], -100)

    def test_incomplete_interrupted_and_mismatched_records_refused_without_text(self):
        job = self.manifest["jobs"][0]
        path = self.root / "records" / (job["job_id"] + ".json")
        original = path.read_bytes()
        path.unlink()
        self.assertNotEqual(self.run_score().returncode, 0)
        for update in ({"eval_id": "PRIVATE SOURCE"}, {"shots": False},
                       {"job_id": "other"}, {"target": "de"}, {"status": "pending"},
                       {"output": None}, {"finish_reason": "length"},
                       {"duration_seconds": -1}, {"duration_seconds": float("nan")}):
            with self.subTest(update=update):
                path.write_text(json.dumps(json.loads(original) | update))
                result = self.run_score()
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("PRIVATE SOURCE", result.stdout + result.stderr)
                self.assertFalse((self.root / "reports").exists())
        path.write_text('{"output":"PRIVATE SOURCE", invalid}')
        result = self.run_score()
        self.assertNotEqual(result.returncode, 0)
        self.assertNotIn("PRIVATE SOURCE", result.stdout + result.stderr)

    def test_duplicate_or_incomplete_manifest_coverage_refused(self):
        original = json.dumps(self.manifest)
        mutations = [
            lambda m: m["jobs"].pop(),
            lambda m: m["jobs"].append(m["jobs"][0]),
            lambda m: m["jobs"][0].update(job_id="../unsafe"),
            lambda m: m["jobs"][0].update(eval_id="unknown"),
            lambda m: m["dataset"]["evaluation"].append(m["dataset"]["evaluation"][0]),
            lambda m: m["dataset"]["evaluation"][0]["references"].clear(),
            lambda m: m["dataset"]["evaluation"][0]["references"].update(de="unplanned"),
            lambda m: m["dataset"]["evaluation"][0].update(id=True),
            lambda m: m.update(conditions=[0, 5]),
            lambda m: m.update(schema_version=True),
        ]
        for index, mutation in enumerate(mutations):
            with self.subTest(mutation=index):
                self.manifest = json.loads(original)
                mutation(self.manifest)
                self.write_manifest()
                self.assertNotEqual(self.run_score().returncode, 0)
                self.assertFalse((self.root / "reports").exists())

    def test_cli_bounds_bootstrap_and_refuses_public_report_directory(self):
        for arguments in (("--bootstrap-samples", "10001"), ("--bootstrap-samples", "-1"),
                          ("--seed", "0")):
            with self.subTest(arguments=arguments):
                self.assertNotEqual(self.run_score(*arguments).returncode, 0)
        (self.root / "reports").mkdir(mode=0o755)
        (self.root / "reports").chmod(0o755)
        self.assertNotEqual(self.run_score().returncode, 0)
        self.assertEqual(list((self.root / "reports").iterdir()), [])

    def test_nonzero_bootstrap_deltas_preserve_seeded_signatures_and_holm(self):
        for job in self.manifest["jobs"]:
            if job["shots"] == 0:
                self.write_record(job, "success", "A completely different sentence with other words .")
        result = self.run_score("--bootstrap-samples", "40", "--seed", "27")
        self.assertEqual(result.returncode, 0, result.stderr)
        report, _ = self.report()
        pair = report["targets"]["en"]["comparisons"][0]
        self.assertGreater(pair["deltas"]["BLEU"], 0)
        for metric in pair["paired_bootstrap"].values():
            self.assertIn("seed:27", metric["signature"])
            self.assertGreaterEqual(metric["p_value_holm"], metric["p_value"])
            self.assertLessEqual(metric["p_value_holm"], 1)
            self.assertIn("ci95_half_width", metric["candidate"])

    def test_all_targets_use_identical_ordered_ids_for_each_arm(self):
        for target in ("de", "uk"):
            self.manifest["models"][target] = {"provider": "PRIVATE METADATA"}
            for row in self.manifest["dataset"]["evaluation"]:
                row["references"][target] = f"A distinct reference for sentence number {row['id']} ."
                for shots in (0, 5, 10):
                    job = {"job_id": f"{target}-{shots}-{row['id']}", "target": target,
                           "shots": shots, "eval_id": row["id"]}
                    self.manifest["jobs"].append(job)
                    self.write_record(job, "success", row["references"][target])
        self.manifest["jobs"].reverse()
        self.manifest["dataset"]["evaluation"].reverse()
        self.write_manifest()
        result = self.run_score()
        self.assertEqual(result.returncode, 0, result.stderr)
        report, path = self.report()
        self.assertEqual(set(report["targets"]), {"en", "de", "uk"})
        for target in report["targets"].values():
            for arm in target["conditions"].values():
                self.assertAlmostEqual(arm["metrics"]["BLEU"]["score"], 100)
        self.assertNotIn("PRIVATE METADATA", path.read_text())

    def test_run_binding_rejects_cross_run_record_and_claim_substitution(self):
        other = json.loads(json.dumps(self.manifest))
        other["models"]["en"] = {"revision": "different-model-run"}
        other_fingerprint = fingerprint(other)
        for directory in ("records", "claims"):
            path = self.root / directory / "en-0-0.json"
            original = path.read_text()
            for value in (other_fingerprint, None):
                with self.subTest(directory=directory, fingerprint=value):
                    record = json.loads(original)
                    if value is None:
                        del record["resume_fingerprint"]
                    else:
                        record["resume_fingerprint"] = value
                    path.write_text(json.dumps(record))
                    self.assertNotEqual(self.run_score().returncode, 0)
            path.write_text(original)

    def test_reference_manifest_tampering_and_missing_fingerprint_refused(self):
        path = self.root / "manifest.json"
        original = path.read_text()
        for rehash in (False, True):
            with self.subTest(rehash=rehash):
                changed = json.loads(original)
                changed["dataset"]["evaluation"][0]["references"]["en"] = "PRIVATE changed reference"
                if rehash:
                    changed["resume_fingerprint"] = fingerprint(changed)
                path.write_text(json.dumps(changed))
                result = self.run_score()
                self.assertNotEqual(result.returncode, 0)
                self.assertNotIn("PRIVATE", result.stderr)
        changed = json.loads(original)
        del changed["resume_fingerprint"]
        path.write_text(json.dumps(changed))
        self.assertNotEqual(self.run_score().returncode, 0)

    def test_terminal_record_requires_durable_claim(self):
        (self.root / "claims" / "en-0-0.json").unlink()
        self.assertNotEqual(self.run_score().returncode, 0)

    def test_report_write_failure_leaves_no_partial_generation_or_owned_temp(self):
        self.assertEqual(self.run_score().returncode, 0)
        directory = self.root / "reports"
        original = {p.relative_to(directory): p.read_bytes() for p in directory.rglob("*") if p.is_file()}
        original_entries = set(directory.iterdir())
        real_fdopen = os.fdopen
        calls = 0

        @contextmanager
        def fail_second_write(*args, **kwargs):
            nonlocal calls
            calls += 1
            with real_fdopen(*args, **kwargs) as stream:
                if calls == 2:
                    class FailingWriter:
                        def write(self, content):
                            stream.write(content[:20])
                            raise OSError("injected private write failure")
                    yield FailingWriter()
                else:
                    yield stream

        with patch.object(sys, "argv", [str(SCRIPT), str(self.root), "--bootstrap-samples", "0"]), \
                patch.object(os, "fdopen", side_effect=fail_second_write), \
                redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            self.assertEqual(score_flores.main(), 1)
        self.assertEqual(calls, 2)
        self.assertEqual(set(directory.iterdir()), original_entries)
        self.assertEqual({p.relative_to(directory): p.read_bytes() for p in directory.rglob("*") if p.is_file()}, original)

    def test_existing_generation_is_never_overwritten_even_if_empty(self):
        directory = self.root / "reports"
        directory.mkdir(mode=0o700)
        stamp = "20260101T000000.000000Z-fixed"
        final = directory / stamp
        final.mkdir(mode=0o700)
        with patch.object(sys, "argv", [str(SCRIPT), str(self.root), "--bootstrap-samples", "0"]), \
                patch.object(score_flores, "datetime") as clock, \
                patch.object(score_flores.uuid, "uuid4") as identifier, \
                redirect_stdout(StringIO()), redirect_stderr(StringIO()):
            clock.now.return_value = datetime(2026, 1, 1, tzinfo=timezone.utc)
            identifier.return_value.hex = "fixed"
            self.assertEqual(score_flores.main(), 1)
        self.assertEqual(list(directory.iterdir()), [final])
        self.assertEqual(list(final.iterdir()), [])

    def test_symlink_report_and_run_paths_are_refused(self):
        outside = self.root / "outside"
        outside.mkdir(mode=0o700)
        (self.root / "reports").symlink_to(outside, target_is_directory=True)
        self.assertNotEqual(self.run_score().returncode, 0)
        self.assertEqual(list(outside.iterdir()), [])
        (self.root / "reports").unlink()
        alias = self.root / "alias"
        alias.symlink_to(self.root, target_is_directory=True)
        result = subprocess.run([sys.executable, str(SCRIPT), str(alias), "--bootstrap-samples", "0"],
                                capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "reports").exists())
