"""Synthetic-only tests of the public FLORES loader."""

import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any
import unittest

from scripts.flores_data import load_flores


PIN = "5fec6c13f9e5a4db2f745d4ec0d7c9721ddc4f06"
LANGUAGES = {"ces_Latn": "czec1258", "eng_Latn": "stan1293",
             "deu_Latn": "stan1295", "ukr_Cyrl": "ukra1253"}


class FloresDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for split, count in (("dev", 997), ("devtest", 1012)):
            (self.root / split).mkdir()
            for code, glotto in LANGUAGES.items():
                language, script = code.split("_")
                rows = [{"id": f"{i:04}", "text": f"{split} {code} sentence {i}",
                         "iso_639_3": language, "iso_15924": script,
                         "glottocode": glotto, "variant": "", "split": split,
                         "url": f"https://example.invalid/{split}/{i // 5}"}
                        for i in range(count)]
                self.write(split, code, rows if code == "ces_Latn" else rows[::-1])

    def write(self, split, code, rows):
        (self.root / split / f"{code}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def test_aligns_ids_and_freezes_shared_demonstrations_and_pilot(self):
        data = load_flores(self.root, ["en", "de", "uk"], PIN, limit=3)
        self.assertEqual(data, load_flores(self.root, ["en", "de", "uk"], PIN, limit=3))
        self.assertEqual([r["id"] for r in data["evaluation"]], ["0000", "0001", "0002"])
        self.assertEqual(data["evaluation"][1]["references"]["en"],
                         "devtest eng_Latn sentence 1")
        self.assertEqual(len(data["files"]), 8)
        self.assertEqual(data["split_counts"]["devtest"]["ces_Latn"], 1012)
        self.assertTrue(data["selection"]["pilot"])
        self.assertEqual(len(data["demonstrations"]["en"]), 10)
        self.assertEqual([r["id"] for r in data["demonstrations"]["en"]],
                         [r["id"] for r in data["demonstrations"]["uk"]])
        full = load_flores(self.root, ["en", "de", "uk"], PIN)
        self.assertEqual(data["files"], full["files"])
        self.assertEqual(data["demonstrations"], full["demonstrations"])
        self.assertEqual(data["files"]["dev/ces_Latn.jsonl"], hashlib.sha256(
            (self.root / "dev/ces_Latn.jsonl").read_bytes()).hexdigest())
        json.dumps(data)

    def test_integer_ids_normalize_before_alignment_without_changing_text_or_files(self):
        original = {}
        exact_text = "  A &amp; B &#39;quoted&#39; <tag>\t\u010d  "
        for split in ("dev", "devtest"):
            for code in LANGUAGES:
                path = self.root / split / f"{code}.jsonl"
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                for row in rows:
                    number = int(row["id"])
                    row["id"] = str(number) if code == "eng_Latn" else number
                    if split == "devtest" and number == 0:
                        row["text"] = exact_text
                self.write(split, code, rows)
                original[path] = path.read_bytes()
        result = load_flores(self.root, ["en", "de", "uk"], PIN)
        self.assertEqual([r["id"] for r in result["evaluation"][:4]], ["0", "1", "10", "100"])
        self.assertEqual(result["evaluation"][0]["source"], exact_text)
        self.assertEqual(result["evaluation"][0]["references"], dict.fromkeys(("en", "de", "uk"), exact_text))
        for demos in result["demonstrations"].values():
            self.assertEqual(len(demos), 10)
            self.assertTrue(all(type(row["id"]) is str for row in demos))
        self.assertEqual(result, load_flores(self.root, ["en", "de", "uk"], PIN))
        for path, raw in original.items():
            self.assertEqual(path.read_bytes(), raw)
            self.assertEqual(result["files"][path.relative_to(self.root).as_posix()], hashlib.sha256(raw).hexdigest())

    def test_rejects_integer_string_id_collision_after_normalization(self):
        path = self.root / "dev" / "ces_Latn.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["id"], rows[1]["id"] = 1, "1"
        self.write("dev", "ces_Latn", rows)
        with self.assertRaisesRegex(ValueError, "duplicate ID"):
            load_flores(self.root, ["en"], PIN)

    def test_rejects_invalid_schema_counts_alignment_and_encoding(self):
        path = self.root / "dev" / "eng_Latn.jsonl"
        original = path.read_bytes()
        mutations = [
            lambda rows: rows.pop(),
            lambda rows: rows[0].update(id=True),
            lambda rows: rows[0].update(id=False),
            lambda rows: rows[0].update(id=1.0),
            lambda rows: rows[0].update(id=-1),
            lambda rows: rows[0].update(id="  \t"),
            lambda rows: rows[0].update(id=rows[1]["id"]),
            lambda rows: rows[0].update(id="not-aligned"),
            lambda rows: rows[0].update(text="  "),
            lambda rows: rows[0].update(text="\ud800"),
            lambda rows: rows[0].update(iso_639_3="ces"),
            lambda rows: rows[0].update(iso_15924="Cyrl"),
            lambda rows: rows[0].update(glottocode="wrong"),
            lambda rows: rows[0].update(variant="different"),
            lambda rows: rows[0].update(split="devtest"),
            lambda rows: rows[0].update(url=None),
        ]
        for index, mutation in enumerate(mutations):
            with self.subTest(mutation=index):
                rows = [json.loads(line) for line in original.splitlines()]
                mutation(rows)
                self.write("dev", "eng_Latn", rows)
                with self.assertRaises(ValueError):
                    load_flores(self.root, ["en"], PIN)
        for invalid in (b"\xff", b'{"secret": INVALID JSON}'):
            path.write_bytes(invalid)
            with self.assertRaises(ValueError) as raised:
                load_flores(self.root, ["en"], PIN)
            self.assertNotIn("secret", str(raised.exception))

    def test_excludes_full_test_source_reference_and_article_overlap_even_for_pilot(self):
        baseline = load_flores(self.root, ["en"], PIN, limit=1)
        excluded = [row["id"] for row in baseline["demonstrations"]["en"][:3]]
        for code in ("ces_Latn", "eng_Latn"):
            path = self.root / "dev" / f"{code}.jsonl"
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            indexed = {row["id"]: row for row in rows}
            if code == "ces_Latn":
                indexed[excluded[0]]["text"] = "devtest ces_Latn sentence 500"
                indexed[excluded[2]]["url"] = "https://example.invalid/devtest/100"
            else:
                indexed[excluded[1]]["text"] = "devtest eng_Latn sentence 500"
            self.write("dev", code, rows)
        result = load_flores(self.root, ["en"], PIN, limit=1)
        self.assertFalse(set(excluded) & {row["id"] for row in result["demonstrations"]["en"]})
        self.assertEqual(result["selection"]["eligible_dev_count"], 994)

    def test_no_article_overlap_fallback_when_fewer_than_ten_candidates(self):
        path = self.root / "dev" / "ces_Latn.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows[9:]:
            row["url"] = "https://example.invalid/devtest/0"
        self.write("dev", "ces_Latn", rows)
        with self.assertRaisesRegex(ValueError, "fewer than ten"):
            load_flores(self.root, ["en"], PIN)

    def test_revision_targets_and_pilot_are_explicit_and_bounded(self):
        invalid_options: list[dict[str, Any]] = [{"revision": "main"}, {"revision": "a" * 39},
                        {"targets": ["en", "eng_Latn"]}, {"targets": []},
                        {"targets": ["fr"]}, {"limit": 0}, {"limit": True},
                        {"limit": 1013}, {"seed": True}]
        for options in invalid_options:
            with self.subTest(options=options), self.assertRaises(ValueError):
                load_flores(self.root, **({"targets": ["en"], "revision": PIN} | options))
        result = load_flores(self.root, ["eng_Latn"], PIN, seed=42)
        self.assertEqual(set(result["demonstrations"]), {"eng_Latn"})
        self.assertFalse(result["selection"]["pilot"])
