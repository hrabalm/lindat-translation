"""Synthetic-only safety tests; no Hub access or corpus fixtures."""

import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts import prepare_flores_data as prepare


class PrepareFloresTests(unittest.TestCase):
    def fixture(self, integer_ids=False):
        previous_umask = os.umask(0o077)
        self.addCleanup(os.umask, previous_umask)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        data = Path(temporary.name)
        output = data / prepare.PIN
        output.mkdir(mode=0o700)
        manifest = {
            "repository": prepare.REPO, "requested_revision": prepare.PIN,
            "resolved_commit": prepare.PIN, "license": "CC-BY-SA-4.0",
            "license_verified_from_pinned_card_metadata": True,
            "dataset_card": {"revision": prepare.PIN, "path": "metadata/README.md"},
            "validation": {"status": "blocked", "exception_class": "ValueError"}, "files": {},
        }
        for split, count in prepare.COUNTS.items():
            for language, code in prepare.CODES.items():
                rows = [{"id": i if integer_ids else str(i), "split": split,
                         "text": f"  synthetic {split} {code} {i}  ",
                         "iso_639_3": code.split("_")[0], "iso_15924": code.split("_")[1],
                         "glottocode": prepare.GLOTTO[language], "variant": "",
                         "url": f"https://example.invalid/{split}/{i}"} for i in range(count)]
                relative = f"raw/{split}/{code}.jsonl"
                manifest["files"][relative] = {
                    "hub_path": f"{split}/{code}.jsonl",
                    **prepare.store(output / relative, b"".join(prepare.encoded(row) for row in reversed(rows))),
                }
        manifest["files"]["metadata/README.md"] = {
            "hub_path": "README.md", **prepare.store(output / "metadata/README.md",
                                                      b"---\nlicense:\n- cc-by-sa-4.0\n---\nSynthetic card\n")}
        prepare.store(output / "download_manifest.json", prepare.encoded(manifest))
        for patcher in (
            patch.object(prepare, "DATA", data),
            patch.object(prepare.subprocess, "run", return_value=SimpleNamespace(returncode=0)),
            patch.dict("sys.modules", {"huggingface_hub": None}),
            patch("socket.socket", side_effect=AssertionError("network forbidden")),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        return output

    def test_normalizes_integer_ids_without_changing_raw(self):
        output = self.fixture(integer_ids=True)
        raw = output / "raw/dev/ces_Latn.jsonl"
        before = raw.read_bytes()
        corpus, _ = prepare.validate(output / "raw")
        self.assertEqual(set(corpus["dev"]["cs"]), {str(i) for i in range(997)})
        self.assertEqual(corpus["dev"]["cs"]["0"]["text"], "  synthetic dev ces_Latn 0  ")
        self.assertEqual(raw.read_bytes(), before)
        rows = [json.loads(line) for line in before.splitlines()]
        for invalid in (True, -1, 1.5, None, "", "  ", str(rows[1]["id"])):
            rows[0]["id"] = invalid
            raw.chmod(0o600)
            raw.write_bytes(b"".join(prepare.encoded(row) for row in rows))
            raw.chmod(0o400)
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                prepare.validate(output / "raw")

    def test_offline_completion_preserves_evidence_and_refuses_repeat(self):
        output = self.fixture(integer_ids=True)
        before = {path: path.read_bytes() for path in output.rglob("*") if path.is_file()}
        with contextlib.redirect_stdout(io.StringIO()), patch("sys.argv", [
                "prepare", "--revision", prepare.PIN, "--output-dir", str(output), "--prepare-existing"]), \
                patch.object(prepare, "prepare", side_effect=AssertionError("download forbidden")):
            self.assertEqual(prepare.main(), 0)
            with self.assertRaises(FileExistsError):
                prepare.prepare_existing(output)
        for path, content in before.items():
            self.assertEqual(path.read_bytes(), content)
        for split, count in prepare.COUNTS.items():
            path = output / "aligned" / f"{split}.jsonl"
            rows = [json.loads(line) for line in path.read_bytes().splitlines()]
            self.assertEqual(len(rows), count)
            self.assertEqual(rows[0], {"id": "0", "split": split, "source_language": "cs",
                                      "source": f"  synthetic {split} ces_Latn 0  ",
                                      "references": {lang: f"  synthetic {split} {prepare.CODES[lang]} 0  "
                                                     for lang in ("en", "de", "uk")}})
            self.assertEqual(path.stat().st_nlink, 1)
            self.assertEqual(path.stat().st_mode & 0o777, 0o400)
        self.assertTrue((output / "preparation_manifest.json").is_file())

    def test_partial_output_is_never_overwritten(self):
        output = self.fixture()
        partial = output / "aligned/dev.jsonl"
        prepare.store(partial, b"synthetic partial\n")
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(FileExistsError):
            prepare.prepare_existing(output)
        self.assertEqual(partial.read_bytes(), b"synthetic partial\n")
        self.assertFalse((output / "preparation_manifest.json").exists())

    def test_corrupt_evidence_blocks_before_any_output(self):
        output = self.fixture()
        path = output / "download_manifest.json"
        original = json.loads(path.read_bytes())
        for field, value in (("resolved_commit", "0" * 40), ("license", "unknown"),
                             ("files", {})):
            manifest = original | {field: value}
            path.chmod(0o600)
            path.write_bytes(prepare.encoded(manifest))
            path.chmod(0o400)
            with self.subTest(field=field), self.assertRaises(ValueError):
                prepare.prepare_existing(output)
            self.assertFalse((output / "aligned").exists())
        path.chmod(0o600)
        path.write_bytes(prepare.encoded(original))
        path.chmod(0o400)
        raw = output / "raw/dev/ces_Latn.jsonl"
        raw.chmod(0o600)
        raw.write_bytes(b"synthetic corrupted data\n")
        raw.chmod(0o400)
        with self.assertRaises(ValueError):
            prepare.prepare_existing(output)
        self.assertFalse((output / "aligned").exists())

    def test_preview_never_downloads(self):
        with patch("sys.argv", ["prepare", "--revision", prepare.PIN,
                                "--output-dir", str(prepare.DATA / prepare.PIN)]), \
                patch.object(prepare, "prepare") as download, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(prepare.main(), 0)
        download.assert_not_called()
        self.assertIn("no network calls", output.getvalue())

    def test_exclusive_write_preserves_original(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "synthetic.jsonl"
            result = prepare.store(path, b"synthetic\n")
            self.assertEqual(result["bytes"], 10)
            self.assertEqual(path.stat().st_mode & 0o777, 0o400)
            with self.assertRaises(FileExistsError):
                prepare.store(path, b"replacement")
            self.assertEqual(path.read_bytes(), b"synthetic\n")

    def test_error_output_exposes_only_exception_class(self):
        with patch("sys.argv", ["prepare", "--revision", prepare.PIN, "--download",
                                "--output-dir", str(prepare.DATA / prepare.PIN)]), \
                patch.object(prepare, "prepare", side_effect=ValueError("secret response")), \
                patch.object(prepare.os, "umask"), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(prepare.main(), 1)
        self.assertEqual(output.getvalue(), "Preparation blocked: ValueError\n")


if __name__ == "__main__":
    unittest.main()
