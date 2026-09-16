"""Download only the pinned official FLORES+ subset; never run inference.

Default mode is a local preview with no Hub calls. --prepare-existing completes
a verified download offline, preserving its historical download manifest.
"""

import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import re
import stat
import subprocess
import tempfile


REPO = "openlanguagedata/flores_plus"
PIN = "5fec6c13f9e5a4db2f745d4ec0d7c9721ddc4f06"
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "experiments/flores_plus/data"
CODES = {"cs": "ces_Latn", "en": "eng_Latn", "de": "deu_Latn", "uk": "ukr_Cyrl"}
GLOTTO = {"cs": "czec1258", "en": "stan1293", "de": "stan1295", "uk": "ukra1253"}
COUNTS = {"dev": 997, "devtest": 1012}
FILES = [f"{split}/{code}.jsonl" for split in COUNTS for code in CODES.values()]


def store(path, content):
    """Publish a complete file atomically without replacing any existing path."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".prepare-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fchmod(stream.fileno(), 0o400)
            os.fsync(stream.fileno())
        # link() is an atomic no-replace publication; unlink the temporary name
        # immediately, leaving a private regular file with exactly one link.
        os.link(temporary, path)
    finally:
        os.unlink(temporary)
    return {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def verify_evidence(output):
    """Verify local evidence, not a fresh network authentication of the commit."""
    for path in [DATA, output, *output.rglob("*")]:
        mode = path.lstat()
        directory = stat.S_ISDIR(mode.st_mode)
        if (not directory and not stat.S_ISREG(mode.st_mode)
                or stat.S_IMODE(mode.st_mode) != (0o700 if directory else 0o400)
                or (not directory and mode.st_nlink != 1)):
            raise PermissionError()
        if subprocess.run(["git", "check-ignore", "-q", str(path)], cwd=ROOT, capture_output=True).returncode:
            raise PermissionError()
    evidence = (output / "download_manifest.json").read_bytes()
    manifest = json.loads(evidence)
    if (manifest.get("repository") != REPO or manifest.get("requested_revision") != PIN
            or manifest.get("resolved_commit") != PIN
            or manifest.get("license") != "CC-BY-SA-4.0"
            or manifest.get("license_verified_from_pinned_card_metadata") is not True
            or manifest.get("dataset_card") != {"revision": PIN, "path": "metadata/README.md"}):
        raise ValueError("revision or license evidence")
    required = {f"raw/{name}" for name in FILES} | {"metadata/README.md"}
    allowed = required | {f"metadata/{name}" for name in (
        "LICENSE", "LICENSE.md", "LICENSE.txt", "LICENSE_CODE", "LICENSE_DATA")}
    allowed |= {f"aligned/{split}.jsonl" for split in COUNTS}
    if not required <= set(manifest["files"]) <= allowed:
        raise ValueError("manifest inventory")
    for relative, expected in manifest["files"].items():
        path = output / relative
        content = path.read_bytes()
        if (len(content) != expected["bytes"]
                or hashlib.sha256(content).hexdigest() != expected["sha256"]):
            raise ValueError("file integrity")
        if relative in required and expected.get("hub_path") != relative.split("/", 1)[1]:
            raise ValueError("Hub path evidence")
    card = (output / "metadata/README.md").read_text(encoding="utf-8")
    frontmatter = card.split("---", 2)
    if (len(frontmatter) != 3 or frontmatter[0].strip()
            or not re.search(r"(?m)^license:\s*\n- cc-by-sa-4\.0\s*$", frontmatter[1])):
        raise ValueError("pinned card license")
    preparation_path = output / "preparation_manifest.json"
    if preparation_path.exists():
        preparation = json.loads(preparation_path.read_bytes())
        if (preparation.get("repository") != REPO or preparation.get("resolved_commit") != PIN
                or preparation.get("license") != manifest["license"]
                or preparation.get("download_manifest") != {
                    "path": "download_manifest.json", "bytes": len(evidence),
                    "sha256": hashlib.sha256(evidence).hexdigest()}
                or set(preparation["files"]) != {f"aligned/{split}.jsonl" for split in COUNTS}):
            raise ValueError("preparation evidence")
        for relative, expected in preparation["files"].items():
            content = (output / relative).read_bytes()
            if len(content) != expected["bytes"] or hashlib.sha256(content).hexdigest() != expected["sha256"]:
                raise ValueError("aligned file integrity")
    return manifest, evidence


def inspect_local(output):
    """Read-only, corpus-free diagnostics, including partially prepared downloads."""
    report, corpus = {}, {}
    verify_evidence(output)
    for split in COUNTS:
        corpus[split] = {}
        for language, code in CODES.items():
            relative = f"{split}/{code}.jsonl"
            rows = [json.loads(line) for line in (output / "raw" / relative).read_bytes().splitlines()]
            corpus[split][language] = rows
            report[relative] = {
                "count": len(rows),
                "field_types": {key: sorted({type(row.get(key)).__name__ for row in rows})
                                for key in ("id", "text", "split", "iso_639_3", "iso_15924", "glottocode", "variant", "url")},
                "unique_ids": len({row["id"] for row in rows}),
            }
    eligible = {}
    for target in ("en", "de", "uk", "all"):
        languages = list(CODES) if target == "all" else ["cs", target]
        texts = {r["text"] for lang in languages for r in corpus["devtest"][lang]}
        urls = {r["url"] for lang in languages for r in corpus["devtest"][lang]}
        candidates = [{r["id"] for r in corpus["dev"][lang]
                       if r["text"] not in texts and r["url"] not in urls} for lang in languages]
        eligible[target] = len(set.intersection(*candidates))
    print(json.dumps({"files": report, "eligible_dev_counts": eligible,
                      "id_sets_aligned": {split: all(
                          {r["id"] for r in rows} == {r["id"] for r in corpus[split]["cs"]}
                          for rows in corpus[split].values()) for split in COUNTS},
                      "integrity_permissions_gitignore": "passed",
                      "aligned_exports_exist": all((output / "aligned" / f"{s}.jsonl").exists() for s in COUNTS)}, sort_keys=True))


def validate(raw):
    corpus, schema = {}, {}
    for split, expected in COUNTS.items():
        corpus[split] = {}
        for language, code in CODES.items():
            rows = [json.loads(line) for line in (raw / split / f"{code}.jsonl").read_bytes().splitlines()]
            if len(rows) != expected:
                raise ValueError("split count")
            indexed = {}
            variants = set()
            types = {}
            for row in rows:
                if not isinstance(row, dict):
                    raise ValueError("row type")
                for key in ("id", "text", "split", "iso_639_3", "iso_15924", "glottocode", "variant", "url"):
                    types.setdefault(key, set()).add(type(row.get(key)).__name__)
                identifier = row.get("id")
                if type(identifier) is int and identifier >= 0:
                    identifier = str(identifier)
                elif not isinstance(identifier, str) or not identifier.strip():
                    raise ValueError("invalid ID")
                if (not isinstance(row.get("text"), str) or not row["text"].strip()
                        or row.get("split") != split
                        or row.get("iso_639_3") != code.split("_")[0]
                        or row.get("iso_15924") != code.split("_")[1]
                        or row.get("glottocode") != GLOTTO[language]
                        or not isinstance(row.get("variant"), str)
                        or not isinstance(row.get("url"), str) or not row["url"].strip()
                        or identifier in indexed):
                    raise ValueError("schema or duplicate ID")
                encoded(row)
                variants.add(row["variant"])
                indexed[identifier] = row
            if len(variants) != 1:
                raise ValueError("mixed variants")
            corpus[split][language] = indexed
            schema[f"{split}/{code}.jsonl"] = {key: sorted(value) for key, value in types.items()}
        if any(set(rows) != set(corpus[split]["cs"]) for rows in corpus[split].values()):
            raise ValueError("unaligned IDs")
    for language in CODES:
        if {r["variant"] for r in corpus["dev"][language].values()} != {
                r["variant"] for r in corpus["devtest"][language].values()}:
            raise ValueError("inconsistent variants")
    return corpus, schema


def prepare_existing(output):
    # Any previous publication attempt is ambiguous, even an empty directory.
    if any(path.exists() or path.is_symlink() for path in (
            output / "aligned", output / "preparation_manifest.json")):
        print("Existing or partial preparation detected; refusing to overwrite.")
        raise FileExistsError()
    original, evidence = verify_evidence(output)
    expected_files = {"download_manifest.json", *original["files"]}
    if {str(path.relative_to(output)) for path in output.rglob("*") if path.is_file()} != expected_files:
        raise ValueError("unexpected files in download")
    corpus, schema = validate(output / "raw")
    if __package__:
        from .flores_data import load_flores
    else:
        from flores_data import load_flores
    loaded = load_flores(output / "raw", ["en", "de", "uk"], PIN)
    if loaded["files"] != {name: original["files"][f"raw/{name}"]["sha256"] for name in FILES}:
        raise ValueError("loader evidence mismatch")
    payloads = {}
    for split in COUNTS:
        records = [{
            "id": identifier, "split": split, "source_language": "cs",
            "source": corpus[split]["cs"][identifier]["text"],
            "references": {lang: corpus[split][lang][identifier]["text"] for lang in ("en", "de", "uk")},
        } for identifier in sorted(corpus[split]["cs"])]
        if split == "devtest" and loaded["evaluation"] != [
                {key: row[key] for key in ("id", "source", "references")} for row in records]:
            raise ValueError("loader alignment mismatch")
        payloads[f"aligned/{split}.jsonl"] = b"".join(encoded(row) for row in records)
    manifest = {
        "repository": REPO, "resolved_commit": PIN, "license": original["license"],
        "dataset_card": original["dataset_card"],
        "download_manifest": {"path": "download_manifest.json", "bytes": len(evidence),
                              "sha256": hashlib.sha256(evidence).hexdigest()},
        "verification": "Offline verification of pinned download evidence, all raw hashes, card license, and main loader; no Hub calls.",
        "raw_files": {name: original["files"][f"raw/{name}"] for name in FILES},
        "split_counts": loaded["split_counts"], "schema_types": schema,
        "selection": loaded["selection"], "seed": loaded["seed"],
        "demonstration_ids": [row["id"] for row in loaded["demonstrations"]["en"]],
        "loader_sha256": hashlib.sha256(Path(load_flores.__code__.co_filename).read_bytes()).hexdigest(),
        "model_provenance": "unknown; user attestation required separately before evaluation",
        "inference_authorized": False, "inference_performed": False,
        "language_pairs": ["cs->en", "cs->de", "cs->uk"],
        "modifications": "Nonnegative integer IDs normalized to strings before exact joins; lexicographic ID order. Text unchanged; raw files unchanged.",
        "files": {name: {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
                  for name, content in payloads.items()},
    }
    payloads["preparation_manifest.json"] = encoded(manifest)
    for relative in payloads:
        if subprocess.run(["git", "check-ignore", "-q", str(output / relative)],
                          cwd=ROOT, capture_output=True).returncode:
            raise PermissionError()
    # Recheck evidence immediately before publishing. The exclusive directory
    # claim also prevents concurrent preparers from mixing their outputs.
    if verify_evidence(output)[1] != evidence:
        raise ValueError("changed evidence")
    (output / "aligned").mkdir(mode=0o700)
    for relative, content in payloads.items():
        store(output / relative, content)
    inspect_local(output)
    for relative, content in payloads.items():
        if (output / relative).read_bytes() != content:
            raise ValueError("published content mismatch")
    print("Offline preparation ready; original download evidence unchanged.")
    return 0


def prepare(output):
    # Suppress third-party response/URL logging; exceptions are reported by class only.
    logging.disable(logging.CRITICAL)
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    import huggingface_hub
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils.tqdm import disable_progress_bars

    disable_progress_bars()
    if output.exists() or output.is_symlink():
        raise FileExistsError()
    if DATA.is_symlink() or (DATA.exists() and stat.S_IMODE(DATA.stat().st_mode) != 0o700):
        raise PermissionError()
    if subprocess.run(["git", "check-ignore", "-q", str(output / "raw/dev/ces_Latn.jsonl")],
                      cwd=ROOT, capture_output=True).returncode != 0:
        raise PermissionError()
    info = HfApi(endpoint="https://huggingface.co", token=True).dataset_info(REPO, revision=PIN)
    if info.sha != PIN:
        raise ValueError("revision mismatch")
    license_value = info.card_data.get("license") if info.card_data else None
    if license_value not in ("cc-by-sa-4.0", ["cc-by-sa-4.0"]):
        raise ValueError("license mismatch")
    if info.siblings is None:
        raise ValueError("missing repository file inventory")
    available = {entry.rfilename for entry in info.siblings}
    if not set(FILES) <= available:
        raise FileNotFoundError()
    metadata = [name for name in ("README.md", "LICENSE", "LICENSE.md", "LICENSE.txt", "LICENSE_CODE", "LICENSE_DATA")
                if name in available]
    DATA.mkdir(mode=0o700, exist_ok=True)
    output.mkdir(mode=0o700)
    manifest = {
        "repository": REPO, "requested_revision": PIN, "resolved_commit": info.sha,
        "license": "CC-BY-SA-4.0", "license_verified_from_pinned_card_metadata": True,
        "dataset_card": {"revision": info.sha, "path": "metadata/README.md" if "README.md" in metadata else None},
        "huggingface_hub_version": huggingface_hub.__version__,
        "model_provenance": "unknown; user attestation required separately before evaluation",
        "inference_authorized": False, "inference_performed": False,
        "language_pairs": ["cs->en", "cs->de", "cs->uk"],
        "modifications": "Raw bytes unchanged; aligned exports join exact IDs and project text fields without normalization.",
        "files": {},
    }
    # A private temporary Hub cache may contain symlinks. Only byte copies survive.
    with tempfile.TemporaryDirectory(prefix=".hf-cache-", dir=output) as cache:
        for name in FILES + metadata:
            cached = hf_hub_download(REPO, name, repo_type="dataset", revision=info.sha,
                                     token=True, endpoint="https://huggingface.co", cache_dir=cache)
            relative = f"raw/{name}" if name in FILES else f"metadata/{name}"
            manifest["files"][relative] = {"hub_path": name, **store(output / relative, Path(cached).read_bytes())}
    try:
        corpus, schema = validate(output / "raw")
    except Exception as exc:
        manifest["validation"] = {"status": "blocked", "exception_class": type(exc).__name__}
        store(output / "download_manifest.json", encoded(manifest))
        raise
    manifest["schema_types"] = schema
    manifest["split_counts"] = {split: {lang: len(rows) for lang, rows in languages.items()}
                                for split, languages in corpus.items()}
    eligible = {}
    for target in ("en", "de", "uk", "all"):
        languages = list(CODES) if target == "all" else ["cs", target]
        texts = {r["text"] for lang in languages for r in corpus["devtest"][lang].values()}
        urls = {r["url"] for lang in languages for r in corpus["devtest"][lang].values()}
        eligible[target] = sum(all(corpus["dev"][lang][identifier]["text"] not in texts
                                  and corpus["dev"][lang][identifier]["url"] not in urls
                                  for lang in languages) for identifier in corpus["dev"]["cs"])
    manifest["eligible_dev_counts"] = eligible
    manifest["demonstration_policy"] = "Exact text and article URL exclusion against full devtest; no fallback or selection here."
    for split in COUNTS:
        content = b"".join(encoded({
            "id": identifier, "split": split, "source_language": "cs",
            "source": corpus[split]["cs"][identifier]["text"],
            "references": {lang: corpus[split][lang][identifier]["text"] for lang in ("en", "de", "uk")},
        }) for identifier in sorted(corpus[split]["cs"]))
        relative = f"aligned/{split}.jsonl"
        manifest["files"][relative] = store(output / relative, content)
    try:
        if __package__:
            from .flores_data import load_flores
        else:
            from flores_data import load_flores
        load_flores(output / "raw", ["en", "de", "uk"], PIN)
        manifest["loader_validation"] = {"status": "passed"}
    except Exception as exc:
        manifest["loader_validation"] = {"status": "blocked", "exception_class": type(exc).__name__}
    manifest["validation"] = {"status": "aligned exports validated"}
    store(output / "download_manifest.json", encoded(manifest))
    for path in [output, *output.rglob("*")]:
        mode = path.lstat()
        if (path.is_symlink() or stat.S_IMODE(mode.st_mode) != (0o700 if path.is_dir() else 0o400)
                or (path.is_file() and mode.st_nlink != 1)):
            raise PermissionError()
        if subprocess.run(["git", "check-ignore", "-q", str(path)], cwd=ROOT, capture_output=True).returncode:
            raise PermissionError()
    print(json.dumps({key: manifest[key] for key in (
        "resolved_commit", "license", "split_counts", "eligible_dev_counts", "loader_validation", "schema_types")}, sort_keys=True))
    print(f"Private prepared data: {output.relative_to(ROOT)}")
    return 0 if manifest["loader_validation"]["status"] == "passed" else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True, choices=[PIN])
    parser.add_argument("--output-dir", required=True, type=Path, help="Must be experiments/flores_plus/data/<revision>")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--download", action="store_true", help="Authorize official Hub download, not gate acceptance or inference")
    mode.add_argument("--verify-local", action="store_true", help="Read-only integrity and schema diagnostics; no Hub calls")
    mode.add_argument("--prepare-existing", action="store_true", help="Complete verified existing raw download offline without overwriting evidence")
    args = parser.parse_args()
    try:
        output = args.output_dir.absolute()
        if output != DATA / args.revision or output.resolve() != output:
            raise ValueError("output location")
        if args.verify_local:
            inspect_local(output)
            return 0
        if args.prepare_existing:
            os.umask(0o077)
            return prepare_existing(output)
        if not args.download:
            print(json.dumps({"mode": "preview; no network calls", "repository": REPO,
                              "revision": args.revision, "files": FILES, "output": str(output)}))
            return 0
        os.umask(0o077)
        return prepare(output)
    except Exception as exc:
        print(f"Preparation blocked: {type(exc).__name__}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
