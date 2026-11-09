#!/usr/bin/env python3
"""Prepare question-ready arXiv LaTeX, including exact frozen gold inputs."""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import http.client
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
import zlib


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "gold-manifest.json"
MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024
USER_AGENT = "MathErrataBench-source-reconstruction/1.0"
ID_RE = re.compile(
    r"(?P<paper>(?:\d{4}\.\d{4,5}|[a-z][a-z0-9.-]*/\d{7}))"
    r"(?:v(?P<version>[1-9]\d*))?", re.IGNORECASE
)


class ReconstructionError(RuntimeError):
    pass


class IncompleteDownload(OSError):
    pass


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def fingerprint(value) -> str:
    return sha256(json_bytes(value))


def load_vendor(name: str):
    spec = importlib.util.spec_from_file_location("reconstruction_" + name, ROOT / "vendor" / (name + ".py"))
    if spec is None or spec.loader is None:
        raise ReconstructionError(f"Cannot load preparation module: {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


COMBINER = load_vendor("source_combiner")
ANNOTATIONS = load_vendor("source_annotations")
INVENTORY = load_vendor("source_inventory")


def implementation_hash() -> str:
    files = [Path(__file__).resolve(), *sorted((ROOT / "vendor").glob("*.py"))]
    return fingerprint({str(p.relative_to(ROOT)): sha256(p.read_bytes()) for p in files})


def load_gold(path: Path = MANIFEST) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    papers = data.get("papers", [])
    if data.get("schema_version") != 2 or data.get("paper_count") != len(papers):
        raise ReconstructionError("Invalid gold manifest schema/count")
    identifiers = [p["arxiv_id"] for p in papers]
    if len(set(identifiers)) != len(identifiers):
        raise ReconstructionError("Duplicate identifiers in gold manifest")
    paper_ids = [p["paper_id"] for p in papers]
    if paper_ids != sorted(paper_ids):
        raise ReconstructionError("Gold manifest must use ascending arXiv ID order")
    for name, record in data["implementation"].items():
        if sha256((ROOT / name).read_bytes()) != record["sha256"]:
            raise ReconstructionError(f"Frozen preparation module has changed: {name}")
    return data


def parse_identifier(value: str) -> tuple[str, int | None]:
    value = value.strip()
    if value.lower().startswith(("http://", "https://")):
        url = urllib.parse.urlsplit(value)
        if url.hostname not in {"arxiv.org", "www.arxiv.org", "export.arxiv.org"}:
            raise ReconstructionError("Expected an arxiv.org URL or an arXiv identifier")
        value = re.sub(r"^/(?:abs|pdf|src)/", "", url.path).rstrip("/")
        if value.endswith(".pdf"):
            value = value[:-4]
    value = re.sub(r"^arxiv:\s*", "", value, flags=re.IGNORECASE)
    match = ID_RE.fullmatch(value)
    if match is None:
        raise ReconstructionError(f"Invalid arXiv identifier: {value!r}")
    paper = match["paper"].lower()
    version = int(match["version"]) if match["version"] else None
    return paper, version


def identifier_order_key(item: tuple[str, int | None]) -> tuple[str, bool, int]:
    paper, version = item
    return paper, version is None, version or 0


def paper_directory(paper: str) -> str:
    return paper.replace("/", "_")


class Downloader:
    def __init__(self, delay: float = 3.0):
        self.delay = delay
        self.last_request = 0.0
        self.last_successful_url = None

    def fetch(self, url: str, allow_fallback: bool = True) -> bytes:
        source_request = url.startswith("https://export.arxiv.org/src/")
        def fallback():
            primary = url.replace("https://export.arxiv.org/src/", "https://arxiv.org/src/", 1)
            print("Export source download failed repeatedly; retrying the same version at arxiv.org.", file=sys.stderr)
            return self.fetch(primary, allow_fallback=False)
        for attempt in range(4):
            pause = self.delay - (time.monotonic() - self.last_request)
            if pause > 0:
                time.sleep(pause)
            self.last_request = time.monotonic()
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    content = response.read(MAX_DOWNLOAD_BYTES + 1)
                    length = response.headers.get("Content-Length", "")
                if len(content) > MAX_DOWNLOAD_BYTES:
                    raise ReconstructionError("Source download exceeds the 100 MiB limit")
                if length.isdigit() and len(content) != int(length):
                    raise IncompleteDownload("Incomplete download (Content-Length mismatch)")
                if content.startswith(b"\x1f\x8b"):
                    # Validate the stream before extraction or caching. A
                    # truncated HTTP response can otherwise look successful.
                    gzip.decompress(content)
                self.last_successful_url = url
                return content
            except urllib.error.HTTPError as error:
                if error.code not in {429, 500, 502, 503, 504} or attempt == 3:
                    raise ReconstructionError(f"Download failed for {url}: HTTP {error.code}") from error
                retry_after = error.headers.get("Retry-After", "") if error.headers else ""
                wait = int(retry_after) if retry_after.isdigit() else 5 * 2**attempt
            except (urllib.error.URLError, TimeoutError, OSError, EOFError, zlib.error, http.client.HTTPException) as error:
                incomplete = isinstance(error, (IncompleteDownload, EOFError, zlib.error, http.client.IncompleteRead))
                if source_request and allow_fallback and (attempt == 3 or (attempt >= 1 and incomplete)):
                    return fallback()
                if attempt == 3:
                    raise ReconstructionError(f"Download failed for {url}: {error}") from error
                wait = 5 * 2**attempt
            time.sleep(min(wait, 120))
        raise AssertionError("Unreachable download state")

    def resolve_latest(self, paper: str) -> int:
        query = urllib.parse.urlencode({"id_list": paper})
        raw = self.fetch("https://export.arxiv.org/api/query?" + query)
        try:
            tree = ET.fromstring(raw)
            for entry in tree.findall("{http://www.w3.org/2005/Atom}entry"):
                value = entry.findtext("{http://www.w3.org/2005/Atom}id", "")
                found_paper, version = parse_identifier(value)
                if found_paper == paper and version is not None:
                    return version
        except (ET.ParseError, ReconstructionError) as error:
            raise ReconstructionError(f"Could not resolve the latest version of {paper}; supply an explicit vN") from error
        raise ReconstructionError(f"No versioned arXiv record found for {paper}")


def annotate(combined: bytes) -> tuple[bytes, dict]:
    source = combined.decode("utf-8")
    inventory = INVENTORY.build_source_inventory(source)
    transformed, _mapping, details = ANNOTATIONS.transform_source(source, inventory, INVENTORY)
    normalized, spacing = ANNOTATIONS.normalize_blank_lines(transformed)
    validation = ANNOTATIONS.verify_paper(normalized, {}, details, INVENTORY)
    return normalized.encode("utf-8"), {
        "formal_result_count": len(inventory.formal_blocks),
        "algorithm_count": len(inventory.algorithm_blocks),
        "annotation_edit_count": details["edit_count"],
        "spacing": spacing,
        "validation": validation,
    }


def combine(raw: bytes, main_tex: str | None) -> tuple[bytes, dict]:
    with tempfile.TemporaryDirectory(prefix="arxiv-reconstruct-") as temp:
        extracted = Path(temp) / "extracted"
        try:
            archive_format, warnings = COMBINER.extract_source(raw, extracted)
        except (EOFError, tarfile.TarError, OSError, zlib.error) as error:
            raise ReconstructionError(f"Incomplete or unreadable source archive: {error}") from error
        main, selection = COMBINER.choose_main_file(extracted, main_tex)
        assembler = COMBINER.Assembler(extracted)
        result = assembler.assemble(main).encode("utf-8")
        return result, {
            "archive_format": archive_format,
            "main_tex": COMBINER.relative_name(extracted, main),
            "main_selection": selection,
            "included_files": assembler.included_files,
            "warnings": COMBINER.unique_in_order(warnings + assembler.warnings),
        }


def require_hash(data: bytes, expected: str, label: str):
    actual = sha256(data)
    if actual != expected:
        raise ReconstructionError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")


def immutable_write(path: Path, data: bytes):
    """Publish a complete file without replacing an existing target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + "-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
        try:
            os.link(temporary, path)
        except FileExistsError:
            raise ReconstructionError(f"Refusing to overwrite {path}") from None
    finally:
        os.unlink(temporary)


def obtain_archive(paper: str, version: int, args, downloader: Downloader, gold: dict | None) -> tuple[bytes, str]:
    url = f"https://export.arxiv.org/src/{paper}v{version}"
    cache = args.cache_dir / paper_directory(paper) / f"v{version}" / "raw-source.bin"
    if getattr(args, "source_archive", None):
        raw = args.source_archive.read_bytes()
        origin = "provided_archive"
    elif cache.exists():
        raw = cache.read_bytes()
        origin = "local_cache"
    elif args.offline:
        raise ReconstructionError(f"No cached source for {paper}v{version}: {cache}")
    else:
        print(f"Downloading {paper}v{version}", flush=True)
        raw = downloader.fetch(url)
        origin = "arxiv_download"
    if not raw or raw.lstrip().lower().startswith((b"<!doctype html", b"<html")):
        raise ReconstructionError(f"arXiv returned an HTML/empty response for {paper}v{version}")
    return raw, origin


def artifact_path(output: Path, paper: str, version: int) -> Path:
    return output / paper_directory(paper) / f"v{version}"


def prepare_one(paper: str, version: int, args, downloader: Downloader, gold: dict | None, context: dict,
                *, target_dir: Path | None = None, issues: bytes | None = None) -> dict:
    target = target_dir if target_dir is not None else artifact_path(args.output, paper, version)
    requested_main = getattr(args, "main_tex", None)
    if gold is not None and requested_main is not None and requested_main != gold["main_tex"]:
        raise ReconstructionError(f"Gold input requires --main-tex {gold['main_tex']}")
    main_tex = gold["main_tex"] if gold is not None else requested_main
    config = {
        "paper_id": paper, "version": f"v{version}", "main_tex": main_tex,
        "implementation_sha256": implementation_hash(), "context": context,
        "gold": gold, "issues_sha256": sha256(issues) if issues is not None else None,
        "provided_archive_sha256": sha256(args.source_archive.read_bytes()) if getattr(args, "source_archive", None) else None,
    }
    config_hash = fingerprint(config)
    if target.exists():
        if not args.resume and not getattr(args, "reuse_existing", False):
            raise ReconstructionError(f"Output already exists: {target}; choose a new --output or use --resume")
        try:
            record = json.loads((target / "source-manifest.json").read_text(encoding="utf-8"))
            if record["config_sha256"] != config_hash:
                raise ReconstructionError(f"Saved settings differ for {paper}v{version}; use a new --output")
            source = (target / "source.tex").read_bytes()
            require_hash(source, record["source_sha256"], f"{paper}v{version} saved input")
            if gold is not None:
                require_hash(source, gold["source_sha256"], f"{paper}v{version} gold input")
            if issues is not None:
                require_hash((target / "issues.json").read_bytes(), sha256(issues), f"{paper} gold annotations")
        except (OSError, ValueError, KeyError) as error:
            raise ReconstructionError(f"Incomplete/corrupt output directory: {target}") from error
        print(f"Verified existing {paper}v{version}", flush=True)
        return record
    raw, origin = obtain_archive(paper, version, args, downloader, gold)
    combined, combination = combine(raw, main_tex)
    if gold is not None:
        require_hash(combined, gold["combined_source_sha256"], f"{paper}v{version} combined source")
        if combination["included_files"] != gold["included_files"]:
            raise ReconstructionError(f"Included file provenance changed for {paper}v{version}")
    source, annotation = annotate(combined)
    if gold is not None:
        require_hash(source, gold["source_sha256"], f"{paper}v{version} gold input")
    # arXiv may repackage identical source files. Both derived hashes must
    # still match exactly; retain both the reference and downloaded archive hashes.
    raw_match = sha256(raw) == gold["raw_source_sha256"] if gold else None
    if gold is not None and not raw_match:
        print(f"NOTE {paper}v{version}: archive bytes differ from the reference package; combined and prepared source hashes both match exactly.", file=sys.stderr)
    if origin == "arxiv_download":
        cache = args.cache_dir / paper_directory(paper) / f"v{version}" / "raw-source.bin"
        immutable_write(cache, raw)
    record = {
        "schema_version": 1, "paper_id": paper, "version": f"v{version}",
        "source_url": f"https://export.arxiv.org/src/{paper}v{version}",
        "download_url": downloader.last_successful_url if origin == "arxiv_download" else None,
        "origin": origin, "raw_source_sha256": sha256(raw),
        "reference_raw_source_sha256": gold["raw_source_sha256"] if gold else None,
        "raw_source_matches_reference": raw_match,
        "combined_source_sha256": sha256(combined), "source_sha256": sha256(source),
        "gold_match": gold is not None, "gold_snapshot": context["gold_snapshot"] if gold else None,
        "licence_url": gold["licence_url"] if gold else None,
        "implementation_sha256": config["implementation_sha256"],
        "config_sha256": config_hash, "combination": combination,
        "issues_sha256": config["issues_sha256"],
        "annotation": annotation,
        "prepared_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".preparing-", dir=target.parent) as temporary:
        staging = Path(temporary) / "artifact"
        staging.mkdir()
        (staging / "source.tex").write_bytes(source)
        if issues is not None:
            (staging / "issues.json").write_bytes(issues)
        (staging / "source-manifest.json").write_bytes(json_bytes(record))
        if target.exists():
            raise ReconstructionError(f"Refusing to overwrite {target}")
        os.rename(staging, target)
    print(f"Prepared {paper}v{version}" + (" [gold hash verified]" if gold else "") + f": {target / 'source.tex'}", flush=True)
    for warning in combination["warnings"]:
        print(f"WARNING {paper}v{version}: {warning}", file=sys.stderr)
    return record


def write_or_verify(path: Path, data: bytes):
    if path.exists():
        if path.read_bytes() != data:
            raise ReconstructionError(f"Existing artifact does not match: {path}")
    else:
        immutable_write(path, data)


def load_annotations(gold: dict) -> tuple[bytes, dict[str, bytes]]:
    """Verify issue files against the single root manifest and build the output index."""
    documents = {}
    issue_count = 0
    records = []
    for entry in gold["papers"]:
        paper = entry["paper_id"]
        raw = (ROOT / "annotations" / f"{paper}.json").read_bytes()
        require_hash(raw, entry["issues_sha256"], f"{paper} bundled annotations")
        document = json.loads(raw)
        issues = document.get("issues")
        if document.get("paper_id") != paper or not isinstance(issues, list):
            raise ReconstructionError(f"Invalid gold annotations: {paper}")
        if not all(isinstance(issue, dict) and isinstance(issue.get("issue_id"), str) for issue in issues):
            raise ReconstructionError(f"Invalid gold issue schema: {paper}")
        ids = [issue["issue_id"] for issue in issues]
        if len(ids) != len(set(ids)) or len(issues) != entry["issue_count"]:
            raise ReconstructionError(f"Invalid gold issue count/identities: {paper}")
        documents[paper] = raw
        issue_count += len(issues)
        record = {
            "paper_id": paper, "reviewed_version": entry["version"],
            "version_pair": entry.get("version_pair"),
            "source_annotated_sha256": entry["source_sha256"],
            "issues_annotated_sha256": entry["issues_sha256"],
            "source_original_sha256": entry["combined_source_sha256"],
            "issue_count": entry["issue_count"], "classifications": entry["classifications"],
        }
        if "review_method" in entry:
            record["review_method"] = entry["review_method"]
        records.append(record)
    if issue_count != gold["totals"]["issue_count"]:
        raise ReconstructionError("Gold issue total does not match the root manifest")
    output_manifest = {
        "schema_version": 1, "release": gold["snapshot"], "paper_count": gold["paper_count"],
        "contract": gold["contract"], "totals": gold["totals"], "papers": records,
        "source_manifest_sha256": sha256(MANIFEST.read_bytes()),
    }
    return json_bytes(output_manifest), documents


def prepare_gold(args, gold: dict, context: dict, downloader: Downloader):
    manifest_bytes, annotations = load_annotations(gold)
    context = {**context, "annotation_manifest_sha256": sha256(manifest_bytes)}
    plan = {
        "schema_version": 2, "mode": "gold", "gold_snapshot": gold["snapshot"],
        "gold_manifest_sha256": context["gold_manifest_sha256"],
        "annotation_manifest_sha256": sha256(manifest_bytes),
        "implementation_sha256": implementation_hash(),
        "count": len(gold["papers"]), "arxiv_ids": [p["arxiv_id"] for p in gold["papers"]],
    }
    print(f"Gold benchmark: {plan['count']} version-pinned papers, with fixed annotations.", flush=True)
    if args.dry_run:
        print("Identifiers: " + ", ".join(plan["arxiv_ids"]))
        return
    run_path = args.output / "run-manifest.json"
    if args.output.exists():
        if not args.resume:
            raise ReconstructionError(f"Output directory already exists: {args.output}; choose a new --output or use --resume")
        try:
            saved = json.loads(run_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise ReconstructionError("Cannot resume without a valid saved run manifest") from error
        if saved != plan:
            raise ReconstructionError("Saved run manifest does not match this benchmark/code; use a new --output")
    else:
        if args.resume:
            raise ReconstructionError("--resume requires an existing matching run manifest")
        args.output.mkdir(parents=True)
        immutable_write(run_path, json_bytes(plan))
    write_or_verify(args.output / "manifest.json", manifest_bytes)
    results = []
    for entry in gold["papers"]:
        paper = entry["paper_id"]
        record = prepare_one(paper, int(entry["version"][1:]), args, downloader, entry, context,
                             target_dir=args.output / "papers" / paper, issues=annotations[paper])
        results.append({"arxiv_id": entry["arxiv_id"], "source_sha256": record["source_sha256"],
                        "issues_sha256": record["issues_sha256"], "gold_match": record["gold_match"]})
    indexed = ["manifest.json", "run-manifest.json"]
    for entry in gold["papers"]:
        indexed.extend(f"papers/{entry['paper_id']}/{name}" for name in ("source.tex", "issues.json", "source-manifest.json"))
    checksums = "".join(f"{sha256((args.output / name).read_bytes())}  {name}\n" for name in sorted(indexed)).encode("utf-8")
    write_or_verify(args.output / "CHECKSUMS.sha256", checksums)
    report = {
        "schema_version": 2, "run_manifest_sha256": fingerprint(plan),
        "benchmark_manifest_sha256": sha256(manifest_bytes),
        "benchmark_checksums_sha256": sha256(checksums), "count": len(results),
        "gold_matches": len(results), "annotations_identical": True,
        "issue_count": json.loads(manifest_bytes)["totals"]["issue_count"], "papers": results,
    }
    write_or_verify(args.output / "verification.json", json_bytes(report))
    print(f"Verified {len(results)}/{len(results)} source/annotation pairs. Benchmark: {args.output}")


def requested_identifiers(args) -> list[tuple[str, int | None]]:
    values = list(args.identifiers)
    if args.ids_file is not None:
        for line in args.ids_file.read_text(encoding="utf-8").splitlines():
            value = line.split("#", 1)[0].strip()
            if value:
                values.append(value)
    if not values:
        raise ReconstructionError("Supply at least one arXiv identifier or --ids-file")
    # Validate the whole list before preparing any source. Equivalent repeated
    # identifiers are processed once in canonical ID/version order.
    return sorted(set(parse_identifier(value) for value in values), key=identifier_order_key)


def prepare_papers(args, gold: dict, context: dict, downloader: Downloader):
    requested = requested_identifiers(args)
    if (args.source_archive is not None or args.main_tex is not None) and len(requested) != 1:
        raise ReconstructionError("--source-archive and --main-tex require exactly one paper; prepare that paper separately")
    if any(version is None for _paper, version in requested) and (args.offline or args.source_archive):
        raise ReconstructionError("Offline/local-archive preparation requires explicit versions, e.g. 2502.02843v1")
    resolved = []
    for paper, version in requested:
        if version is None:
            version = downloader.resolve_latest(paper)
            print(f"Resolved: {paper}v{version}", flush=True)
        if (paper, version) not in resolved:
            resolved.append((paper, version))
    resolved.sort(key=identifier_order_key)
    plan = {
        "schema_version": 1, "mode": "paper", "gold_snapshot": gold["snapshot"],
        "gold_manifest_sha256": context["gold_manifest_sha256"],
        "implementation_sha256": implementation_hash(),
        "requested": [{"paper_id": paper, "version": f"v{version}" if version else None} for paper, version in requested],
        "arxiv_ids": [f"{paper}v{version}" for paper, version in resolved],
        "main_tex": args.main_tex,
        "provided_archive_sha256": sha256(args.source_archive.read_bytes()) if args.source_archive else None,
    }
    print(f"Papers: {len(resolved)}; " + ", ".join(plan["arxiv_ids"]), flush=True)
    if args.dry_run:
        return
    plan_hash = fingerprint(plan)
    batch_root = args.output / "batches"
    write_or_verify(batch_root / f"{plan_hash}.json", json_bytes(plan))
    entries = {p["arxiv_id"]: p for p in gold["papers"]}
    results = []
    for paper, version in resolved:
        entry = entries.get(f"{paper}v{version}")
        record = prepare_one(paper, version, args, downloader, entry, context)
        results.append({"arxiv_id": f"{paper}v{version}", "source_sha256": record["source_sha256"], "gold_match": record["gold_match"]})
    report = {"schema_version": 1, "batch_manifest_sha256": plan_hash, "count": len(results), "papers": results}
    report_path = batch_root / f"{plan_hash}-verification.json"
    write_or_verify(report_path, json_bytes(report))
    print(f"Verified {len(results)} prepared inputs. Report: {report_path}")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest="command", required=True)
    paper = sub.add_parser("paper", help="Prepare one or several arXiv identifiers or URLs")
    paper.add_argument("identifiers", nargs="*", help="e.g. 2502.02843v1; no version means latest")
    paper.add_argument("--ids-file", type=Path, help="One identifier or URL per line; blank lines and # comments are ignored")
    paper.add_argument("--source-archive", type=Path, help="Local exact-version archive (one paper with explicit vN)")
    paper.add_argument("--main-tex", help="Archive-relative main TeX path (one paper)")
    paper.set_defaults(reuse_existing=True)
    gold = sub.add_parser("gold", help="Reconstruct all frozen benchmark inputs and package their fixed annotations")
    gold.set_defaults(reuse_existing=False)
    for command, default in [(paper, "generated/papers"), (gold, "generated/gold")]:
        command.add_argument("--output", type=Path, default=Path(default), help=f"Output directory (default: {default})")
        command.add_argument("--cache-dir", type=Path, default=Path(".cache/arxiv"), help="Local archive cache (default: .cache/arxiv)")
        command.add_argument("--offline", action="store_true", help="Read local archives/cache only")
        command.add_argument("--resume", action="store_true", help="Continue only with matching saved code/settings and intact artifacts")
        command.add_argument("--dry-run", action="store_true", help="Print resolved identifiers without source downloads or output writes")
        command.add_argument("--download-delay", type=float, default=3.0, help="Seconds between requests (default: 3)")
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if args.download_delay < 0:
            raise ReconstructionError("--download-delay must be nonnegative")
        gold = load_gold()
        context = {"gold_snapshot": gold["snapshot"], "gold_manifest_sha256": sha256(MANIFEST.read_bytes())}
        downloader = Downloader(args.download_delay)
        if args.command == "gold":
            prepare_gold(args, gold, context, downloader)
        else:
            prepare_papers(args, gold, context, downloader)
        return 0
    except (ReconstructionError, COMBINER.PreparationError, ValueError, KeyError, OSError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted. Completed artifacts are preserved; repeat the paper command or use gold --resume.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
