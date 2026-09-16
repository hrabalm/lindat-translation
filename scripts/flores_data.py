"""Offline FLORES+ preflight; no downloads, gate acceptance, or inference.

Targets are en/de/uk or official configuration names; returned keys preserve
the caller's spelling. The runner must use demonstrations[target][:shots].
Revision is a caller-supplied provenance claim, not authenticated by this module.
Returned corpus-bearing data must remain private.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any


CONFIGS = {"ces_Latn": "czec1258", "eng_Latn": "stan1293",
           "deu_Latn": "stan1295", "ukr_Cyrl": "ukra1253"}
TARGETS = {"en": "eng_Latn", "de": "deu_Latn", "uk": "ukr_Cyrl",
           **{code: code for code in CONFIGS if code != "ces_Latn"}}


def load_flores(data_dir: Path, targets: list[str], revision: str,
                limit: int | None = None, seed: int = 1729) -> dict[str, Any]:
    """Validate full local files, join by ID, and freeze leak-filtered examples.

    IDs accept nonnegative integers (not bool) or nonblank strings and normalize
    to strings before uniqueness/alignment checks. String spelling and all text
    remain unchanged. IDs sort lexicographically after normalization.
    A pilot is the first ``limit`` devtest IDs.
    Sampling uses Random(seed).sample over sorted eligible dev IDs, with no
    fallback if fewer than ten survive full-devtest text/article exclusions.
    """
    if not isinstance(revision, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
        raise ValueError("revision must be a full 40-hex commit claim")
    if (not targets or any(t not in TARGETS for t in targets)
            or len({TARGETS[t] for t in targets}) != len(targets)):
        raise ValueError("targets must be distinct supported FLORES varieties")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if limit is not None and (type(limit) is not int or not 1 <= limit <= 1012):
        raise ValueError("limit must be between 1 and 1012")
    codes = ["ces_Latn", *(TARGETS[t] for t in targets)]
    files: dict[str, str] = {}
    counts: dict[str, dict[str, int]] = {}
    corpus: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    variants: dict[str, str] = {}
    for split, expected in (("dev", 997), ("devtest", 1012)):
        corpus[split], counts[split] = {}, {}
        for code in codes:
            relative = f"{split}/{code}.jsonl"
            try:
                raw = (data_dir / relative).read_bytes()
                rows = [json.loads(line) for line in raw.decode("utf-8").splitlines()]
            except (OSError, UnicodeError, ValueError):
                raise ValueError(f"cannot read valid UTF-8 JSONL: {relative}") from None
            files[relative] = hashlib.sha256(raw).hexdigest()
            if len(rows) != expected:
                raise ValueError(f"unexpected full split count: {relative}")
            indexed: dict[str, dict[str, Any]] = {}
            language, script = code.split("_")
            for row in rows:
                try:
                    # JSON escapes can encode lone surrogates despite valid file bytes.
                    json.dumps(row, ensure_ascii=False).encode("utf-8")
                except UnicodeError:
                    raise ValueError(f"invalid Unicode row: {relative}") from None
                if (not isinstance(row, dict)
                        or not ((type(row.get("id")) is int and row["id"] >= 0)
                                or (isinstance(row.get("id"), str) and row["id"].strip()))
                        or not isinstance(row.get("text"), str) or not row["text"].strip()
                        or row.get("split") != split
                        or row.get("iso_639_3") != language
                        or row.get("iso_15924") != script
                        or row.get("glottocode") != CONFIGS[code]
                        or not isinstance(row.get("variant"), str)
                        or not isinstance(row.get("url"), str) or not row["url"].strip()):
                    raise ValueError(f"invalid row schema or languoid: {relative}")
                row["id"] = str(row["id"])
                if row["id"] in indexed:
                    raise ValueError(f"duplicate ID: {relative}")
                if variants.setdefault(code, row["variant"]) != row["variant"]:
                    raise ValueError(f"inconsistent variety: {relative}")
                indexed[row["id"]] = row
            corpus[split][code] = indexed
            counts[split][code] = len(indexed)
        if any(set(corpus[split][code]) != set(corpus[split]["ces_Latn"]) for code in codes):
            raise ValueError(f"unaligned ID sets: {split}")

    evaluation_ids = sorted(corpus["devtest"]["ces_Latn"])
    selected_ids = evaluation_ids if limit is None else evaluation_ids[:limit]
    # Filter against the entire split, not only the pilot, in every language.
    test_texts = {row["text"] for code in codes for row in corpus["devtest"][code].values()}
    test_urls = {row["url"] for code in codes for row in corpus["devtest"][code].values()}
    eligible = [identifier for identifier in sorted(corpus["dev"]["ces_Latn"])
                if all(corpus["dev"][code][identifier]["text"] not in test_texts
                       and corpus["dev"][code][identifier]["url"] not in test_urls
                       for code in codes)]
    if len(eligible) < 10:
        raise ValueError("fewer than ten dev IDs survive text/article leakage exclusions")
    demonstration_ids = random.Random(seed).sample(eligible, 10)
    return {
        "repository": "openlanguagedata/flores_plus", "revision": revision,
        "files": files, "split_counts": counts, "seed": seed,
        "demonstrations": {target: [
            {"id": identifier, "source": corpus["dev"]["ces_Latn"][identifier]["text"],
             "target": corpus["dev"][TARGETS[target]][identifier]["text"]}
            for identifier in demonstration_ids] for target in targets},
        "evaluation": [
            {"id": identifier, "source": corpus["devtest"]["ces_Latn"][identifier]["text"],
             "references": {target: corpus["devtest"][TARGETS[target]][identifier]["text"]
                            for target in targets}} for identifier in selected_ids],
        "selection": {
            "pilot": len(selected_ids) < len(evaluation_ids), "limit": limit,
            "evaluation": "lexicographically sorted IDs, first limit if provided",
            "demonstrations": "Random(seed).sample(sorted eligible dev IDs, 10)",
            "five_shot": "first five of the ordered ten", "eligible_dev_count": len(eligible),
            "leakage_policy": "exclude exact text and URL overlap with full devtest across all selected languages; no fallback",
            "revision_provenance": "caller supplied; local hashes do not authenticate a Hub revision",
            "languoids": {code: {"iso_639_3": code.split("_")[0],
                                  "iso_15924": code.split("_")[1],
                                  "glottocode": CONFIGS[code], "variant": variants[code]}
                          for code in codes},
        },
    }
