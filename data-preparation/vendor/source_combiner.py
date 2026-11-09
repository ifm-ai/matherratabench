#!/usr/bin/env python3
"""Create one provenance-frozen combined LaTeX source for an arXiv version.

The command has deliberately narrow scope: one explicit arXiv id and one
explicit version produce one immutable artifact directory.  It never replaces
an existing version directory.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import sys
import tarfile
import tempfile
import time
import urllib.error
import urllib.request


SCHEMA_VERSION = 1
COMBINER_NAME = "arxiv-source-combiner"
COMBINER_VERSION = "2"
RAW_SOURCE_NAME = "raw-source.bin"
COMBINED_SOURCE_NAME = "combined-source.tex"
MANIFEST_NAME = "source-manifest.json"
MAX_INCLUDE_DEPTH = 30
MAX_FETCH_RETRIES = 5
FETCH_RETRY_BASE_SECONDS = 5
LATEX_EXTENSIONS = {".tex", ".sty", ".cls", ".ltx", ".def", ".cfg", ".clo"}
MAIN_EXTENSIONS = {".tex", ".ltx"}
PREFERRED_MAIN_NAMES = {"main.tex", "paper.tex", "article.tex", "ms.tex"}
INPUT_PATTERN = re.compile(r"\\(?:input|include)\s*\{([^}]+)\}")


class PreparationError(RuntimeError):
    """A source package cannot be converted under the frozen contract."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_arxiv_id(value: str) -> str:
    raw = str(value).strip()
    raw = re.sub(r"^https?://arxiv\.org/(?:abs|pdf|src)/", "", raw)
    raw = raw.removesuffix(".pdf").strip("/")
    raw = re.sub(r"v\d+$", "", raw)
    if not raw or any(part in {"", ".", ".."} for part in raw.split("/")):
        raise PreparationError(f"Invalid arXiv id: {value!r}")
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", raw):
        raise PreparationError(f"Invalid arXiv id: {value!r}")
    return raw


def normalize_version(value: str) -> int:
    match = re.fullmatch(r"v?(\d+)", str(value).strip(), flags=re.IGNORECASE)
    if not match or int(match.group(1)) < 1:
        raise PreparationError(f"Version must be an explicit positive value such as v1: {value!r}")
    return int(match.group(1))


def safe_paper_directory(arxiv_id: str) -> str:
    return arxiv_id.replace("/", "_").replace(":", "_")


def source_url(arxiv_id: str, version: int) -> str:
    return f"https://arxiv.org/src/{arxiv_id}v{version}"


def fetch_source(url: str) -> bytes:
    retry_statuses = {429, 500, 502, 503, 504}
    last_error: Exception | None = None
    for attempt in range(MAX_FETCH_RETRIES + 1):
        request = urllib.request.Request(
            url,
            headers={"User-Agent": f"{COMBINER_NAME}/{COMBINER_VERSION}"},
        )
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                return response.read()
        except urllib.error.HTTPError as exc:
            last_error = exc
            if exc.code not in retry_statuses or attempt >= MAX_FETCH_RETRIES:
                break
            retry_after = exc.headers.get("Retry-After") if exc.headers else None
            delay = (
                int(retry_after)
                if retry_after and retry_after.isdigit()
                else FETCH_RETRY_BASE_SECONDS * (2**attempt)
            )
        except urllib.error.URLError as exc:
            last_error = exc
            if attempt >= MAX_FETCH_RETRIES:
                break
            delay = FETCH_RETRY_BASE_SECONDS * (2**attempt)
        time.sleep(delay)
    raise PreparationError(f"Could not download {url}: {last_error}")


def is_safe_archive_name(name: str) -> bool:
    path = Path(name)
    return bool(name) and not path.is_absolute() and ".." not in path.parts


def is_latex_file(path: Path) -> bool:
    return path.suffix.lower() in LATEX_EXTENSIONS


def extract_source(raw: bytes, destination: Path) -> tuple[str, list[str]]:
    """Extract supported textual LaTeX files and return format plus warnings."""

    if raw.startswith(b"%PDF"):
        raise PreparationError("The arXiv source endpoint returned a PDF, not a LaTeX source package")

    destination.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as archive:
            extracted = 0
            for member in archive.getmembers():
                if not member.isfile() or not is_latex_file(Path(member.name)):
                    continue
                if not is_safe_archive_name(member.name):
                    warnings.append(f"unsafe_archive_member_skipped:{member.name}")
                    continue
                source = archive.extractfile(member)
                if source is None:
                    warnings.append(f"unreadable_archive_member_skipped:{member.name}")
                    continue
                target = destination / member.name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(source.read())
                extracted += 1
            if extracted == 0:
                raise PreparationError("The source archive contains no supported LaTeX files")
        archive_format = "tar_gzip" if raw.startswith(b"\x1f\x8b") else "tar"
        return archive_format, warnings
    except tarfile.ReadError:
        pass

    if raw.startswith(b"\x1f\x8b"):
        try:
            unpacked = gzip.decompress(raw)
        except OSError as exc:
            raise PreparationError(f"Unreadable gzip source: {exc}") from exc
        if b"\\documentclass" not in unpacked and b"\\begin{document}" not in unpacked:
            raise PreparationError("The gzip payload is not a recognizable single-file LaTeX source")
        (destination / "source.tex").write_bytes(unpacked)
        return "gzip_single_tex", warnings

    if b"\\documentclass" in raw or b"\\begin{document}" in raw:
        (destination / "source.tex").write_bytes(raw)
        return "plain_tex", warnings

    raise PreparationError("Source is not a supported tar, gzip LaTeX file, or plain LaTeX file")


def strip_latex_comments(text: str) -> str:
    output: list[str] = []
    for line in text.splitlines():
        cut = len(line)
        cursor = 0
        while True:
            marker = line.find("%", cursor)
            if marker == -1:
                break
            backslashes = 0
            index = marker - 1
            while index >= 0 and line[index] == "\\":
                backslashes += 1
                index -= 1
            if backslashes % 2 == 0:
                cut = marker
                break
            cursor = marker + 1
        output.append(line[:cut])
    return "\n".join(output)


def relative_name(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def main_file_score(path: Path, text: str) -> int:
    score = 0
    if "\\documentclass" in text:
        score += 10
    if "\\begin{document}" in text:
        score += 10
    if path.name.lower() in PREFERRED_MAIN_NAMES:
        score += 3
    return score


def choose_main_file(source_root: Path, explicit_main: str | None) -> tuple[Path, str]:
    if explicit_main:
        relative = Path(explicit_main)
        if relative.is_absolute() or ".." in relative.parts:
            raise PreparationError("--main-tex must be a safe path relative to the source archive root")
        candidate = (source_root / relative).resolve()
        try:
            candidate.relative_to(source_root.resolve())
        except ValueError as exc:
            raise PreparationError("--main-tex resolves outside the source archive root") from exc
        if not candidate.is_file():
            raise PreparationError(f"Explicit main TeX file does not exist: {explicit_main}")
        return candidate, "explicit"

    candidates: list[tuple[int, str, Path]] = []
    for path in sorted(source_root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in MAIN_EXTENSIONS:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        candidates.append((main_file_score(path, text), relative_name(source_root, path), path))

    if not candidates:
        raise PreparationError("No .tex or .ltx file is available as a possible main source")
    top_score = max(score for score, _, _ in candidates)
    if top_score < 10:
        raise PreparationError("No recognizable main TeX file; supply --main-tex explicitly")
    leaders = [(name, path) for score, name, path in candidates if score == top_score]
    if len(leaders) != 1:
        names = ", ".join(name for name, _ in leaders)
        raise PreparationError(
            "Ambiguous main TeX file; supply --main-tex explicitly. "
            f"Equal top candidates: {names}"
        )
    return leaders[0][1], "automatic_unique"


class Assembler:
    def __init__(self, source_root: Path) -> None:
        self.source_root = source_root.resolve()
        self.seen: set[Path] = set()
        self.included_files: list[str] = []
        self.warnings: list[str] = []

    def resolve_input(self, current_dir: Path, name: str) -> Path | None:
        cleaned = name.strip().strip("{}")
        bases = [current_dir]
        if current_dir.resolve() != self.source_root:
            # TeX projects commonly write root-relative paths such as
            # ``\input{sections/proof}`` inside ``sections/results.tex``.
            # Preserve the prior same-directory lookup first, then fall back
            # to the archive root instead of incorrectly looking only for
            # ``sections/sections/proof.tex``.
            bases.append(self.source_root)
        candidates: list[Path] = []
        for base in bases:
            raw = base / cleaned
            candidates.append(raw)
            if raw.suffix == "":
                candidates.append(base / f"{cleaned}.tex")
        for candidate in candidates:
            resolved = candidate.resolve()
            try:
                resolved.relative_to(self.source_root)
            except ValueError:
                continue
            if resolved.is_file():
                return resolved
        return None

    def assemble(self, path: Path, depth: int = 0) -> str:
        resolved = path.resolve()
        name = relative_name(self.source_root, resolved)
        if depth > MAX_INCLUDE_DEPTH:
            self.warnings.append(f"maximum_include_depth_skipped:{name}")
            return f"\n% [Maximum include depth skipped: {name}]\n"
        if resolved in self.seen:
            self.warnings.append(f"repeated_or_recursive_input_skipped:{name}")
            return f"\n% [Repeated or recursive input skipped: {name}]\n"

        self.seen.add(resolved)
        self.included_files.append(name)
        text = resolved.read_text(encoding="utf-8", errors="replace")
        text = strip_latex_comments(text)

        def replace(match: re.Match[str]) -> str:
            requested = match.group(1).strip()
            child = self.resolve_input(resolved.parent, requested)
            if child is None:
                self.warnings.append(f"missing_input:{name}:{requested}")
                return f"\n% [Missing input: {requested}]\n"
            return "\n" + self.assemble(child, depth + 1) + "\n"

        return INPUT_PATTERN.sub(replace, text)


def unique_in_order(values: list[str]) -> list[str]:
    return list(dict.fromkeys(values))


def prepare_source(
    *,
    arxiv_id: str,
    version: int,
    output_root: Path,
    source_archive: Path | None,
    explicit_main: str | None,
) -> Path:
    arxiv_id = normalize_arxiv_id(arxiv_id)
    version = normalize_version(str(version))
    version_name = f"v{version}"
    paper_dir = output_root.resolve() / safe_paper_directory(arxiv_id)
    target_dir = paper_dir / version_name
    if target_dir.exists():
        raise PreparationError(
            f"Refusing to overwrite existing artifact directory: {target_dir}"
        )

    url = source_url(arxiv_id, version)
    if source_archive is None:
        raw = fetch_source(url)
        origin = "arxiv_download"
        provided_archive_name = None
    else:
        archive_path = source_archive.resolve()
        if not archive_path.is_file():
            raise PreparationError(f"Source archive does not exist: {archive_path}")
        raw = archive_path.read_bytes()
        origin = "provided_arxiv_archive"
        provided_archive_name = archive_path.name

    paper_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{version_name}-prepare-", dir=paper_dir) as temp:
        temp_root = Path(temp)
        extraction_root = temp_root / "extracted"
        staging = temp_root / "artifact"
        staging.mkdir()

        archive_format, extraction_warnings = extract_source(raw, extraction_root)
        main_file, main_selection = choose_main_file(extraction_root, explicit_main)
        assembler = Assembler(extraction_root)
        combined = assembler.assemble(main_file)

        raw_path = staging / RAW_SOURCE_NAME
        combined_path = staging / COMBINED_SOURCE_NAME
        raw_path.write_bytes(raw)
        with combined_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(combined)

        script_path = Path(__file__).resolve()
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "paper_id": arxiv_id,
            "version": version_name,
            "origin": origin,
            "source_url": url,
            "provided_archive_name": provided_archive_name,
            "prepared_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "raw_source_file": RAW_SOURCE_NAME,
            "raw_source_format": archive_format,
            "raw_source_size_bytes": len(raw),
            "raw_source_sha256": sha256_bytes(raw),
            "combiner": {
                "name": COMBINER_NAME,
                "version": COMBINER_VERSION,
                "script_sha256": sha256_file(script_path),
                "main_tex": relative_name(extraction_root, main_file),
                "main_tex_selection": main_selection,
                "included_files": assembler.included_files,
            },
            "combined_source_file": COMBINED_SOURCE_NAME,
            "combined_source_sha256": sha256_file(combined_path),
            "warnings": unique_in_order(extraction_warnings + assembler.warnings),
        }
        manifest_path = staging / MANIFEST_NAME
        with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

        if target_dir.exists():
            raise PreparationError(
                f"Refusing to overwrite existing artifact directory: {target_dir}"
            )
        os.rename(staging, target_dir)

    return target_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare one immutable combined LaTeX source for an explicit arXiv version."
    )
    parser.add_argument("--arxiv-id", required=True, help="Unversioned arXiv id")
    parser.add_argument("--version", required=True, help="Explicit version, for example v1")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("arxiv-data"),
        help="Artifact root (default: arxiv-data)",
    )
    parser.add_argument(
        "--source-archive",
        type=Path,
        help="Use already-downloaded exact arXiv source bytes instead of the network",
    )
    parser.add_argument(
        "--main-tex",
        help="Explicit archive-relative main TeX path, required when automatic selection is ambiguous",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        target = prepare_source(
            arxiv_id=args.arxiv_id,
            version=normalize_version(args.version),
            output_root=args.output_root,
            source_archive=args.source_archive,
            explicit_main=args.main_tex,
        )
    except PreparationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"Prepared: {target}")
    print(f"Combined source: {target / COMBINED_SOURCE_NAME}")
    print(f"Manifest: {target / MANIFEST_NAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
