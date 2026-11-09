#!/usr/bin/env python3
"""Prepare questions, recover raw agent answers, and validate structured predictions.

The parser is independent of optional_runner.py and never calls a model or
reads gold issue annotations. Use --help for prepare, parse and check commands.
"""

from __future__ import annotations

import argparse
from collections import Counter
import copy
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import tempfile
from typing import Any


HERE = Path(__file__).resolve().parent
PAPER_ID = re.compile(r"[0-9]{4}\.[0-9]{4,5}")
RUN_NAME = re.compile(r"[A-Za-z0-9._-]+")
HASH = re.compile(r"[0-9a-f]{64}")
BUNDLE_SCHEMA = "matherratabench-agent-inputs-v1"
RECOVERY_ORIGIN_SHA256 = "2f601e38e5aa280f900f2c8cff2189465d03d93d4224ccc6c444dfc738f821a9"


class InputError(ValueError):
    pass


def sha256(raw):
    return hashlib.sha256(raw).hexdigest()


def unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise InputError(f"Duplicate JSON key: {key}")
        obj[key] = value
    return obj


def reject_constant(value):
    raise InputError(f"Invalid JSON constant: {value}")


def read_json(raw):
    return json.loads(raw, object_pairs_hook=unique_object, parse_constant=reject_constant)


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(obj, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")


def prepare(args):
    if not RUN_NAME.fullmatch(args.run_name):
        raise InputError("Run name may contain only letters, digits, dots, underscores and hyphens")
    if args.output.exists():
        raise InputError("Output already exists; choose a new input directory")
    papers_root = args.papers.resolve()
    paths = sorted(papers_root.rglob("source-manifest.json"))
    requested = set(args.paper_id or [])
    if any(not PAPER_ID.fullmatch(paper) for paper in requested):
        raise InputError("--paper-id expects a bare arXiv ID, e.g. 2012.03503")
    entries = {}
    for path in paths:
        record = read_json(path.read_bytes())
        if not isinstance(record, dict):
            raise InputError(f"Invalid source manifest: {path}")
        paper = record.get("paper_id")
        if requested and paper not in requested:
            continue
        if not isinstance(paper, str) or not PAPER_ID.fullmatch(paper):
            raise InputError("The question schema supports modern arXiv IDs only")
        if paper in entries:
            raise InputError(f"Multiple versions of {paper}; prepare each version in a separate run")
        version = record.get("version", "")
        if not isinstance(version, str) or not re.fullmatch(r"v[1-9][0-9]*", version):
            raise InputError(f"Missing or invalid source version: {paper}")
        # Read only source files and their provenance, never issue annotations.
        raw = path.with_name("source.tex").read_bytes()
        if sha256(raw) != record.get("source_sha256"):
            raise InputError(f"Source checksum mismatch: {paper}")
        entries[paper] = (record, raw)
    if requested - entries.keys():
        raise InputError("Papers not found: " + ", ".join(sorted(requested - entries.keys())))
    if not entries:
        raise InputError("No prepared sources found; run data-preparation/prepare.py first")
    question = (HERE / "question.txt").read_bytes()
    schema = (HERE / "output_schema.json").read_bytes()
    template = question.decode("utf-8").rstrip("\n")
    for name in ("run_name", "paper_id", "source_sha256", "source"):
        if template.count("{" + name + "}") != 1:
            raise InputError(f"Question must contain exactly one {{{name}}} placeholder")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".agent-inputs-", dir=args.output.parent) as temp:
        staging = Path(temp) / "inputs"
        (staging / "prompts").mkdir(parents=True)
        rows = []
        for paper, (record, raw) in sorted(entries.items()):
            source_hash = sha256(raw)
            prompt = template.replace("{run_name}", args.run_name)
            prompt = prompt.replace("{paper_id}", paper).replace("{source_sha256}", source_hash)
            prompt = prompt.replace("{source}", raw.decode("utf-8"))
            prompt_bytes = prompt.encode("utf-8")
            if not prompt_bytes.endswith(raw):
                raise InputError(f"Source was changed while preparing prompt: {paper}")
            filename = f"prompts/{paper}.txt"
            (staging / filename).write_bytes(prompt_bytes)
            rows.append({"paper_id": paper, "version": record["version"],
                         "source_sha256": source_hash, "source_bytes": len(raw),
                         "prompt_file": filename, "prompt_sha256": sha256(prompt_bytes)})
        (staging / "output_schema.json").write_bytes(schema)
        write_json(staging / "inputs.json", {
            "schema_version": BUNDLE_SCHEMA, "question_id": "Q1", "run_name": args.run_name,
            "question_sha256": sha256(question), "output_schema_sha256": sha256(schema),
            "paper_count": len(rows), "papers": rows,
        })
        staging.rename(args.output)
    print(f"Prepared {len(rows)} questions in {args.output / 'prompts'}")


def load_inputs(folder):
    bundle = read_json((folder / "inputs.json").read_bytes())
    if not isinstance(bundle, dict):
        raise InputError("Agent input manifest must be a JSON object")
    rows = bundle.get("papers")
    if bundle.get("schema_version") != BUNDLE_SCHEMA or not isinstance(rows, list) or not rows:
        raise InputError("Invalid or empty agent input manifest")
    if not isinstance(bundle.get("run_name"), str) or not RUN_NAME.fullmatch(bundle["run_name"]):
        raise InputError("Invalid run name in input manifest")
    ids = [row.get("paper_id") for row in rows]
    if any(not isinstance(p, str) or not PAPER_ID.fullmatch(p) for p in ids):
        raise InputError("Invalid paper ID in input manifest")
    if ids != sorted(set(ids)) or bundle.get("paper_count") != len(rows):
        raise InputError("Input manifest must contain unique papers in ascending arXiv ID order")
    schema_raw = (folder / "output_schema.json").read_bytes()
    if sha256(schema_raw) != bundle.get("output_schema_sha256"):
        raise InputError("Answer schema checksum mismatch")
    for row in rows:
        paper = row["paper_id"]
        if (not isinstance(row.get("source_sha256"), str) or not HASH.fullmatch(row["source_sha256"])
                or not isinstance(row.get("version"), str) or not re.fullmatch(r"v[1-9][0-9]*", row["version"])):
            raise InputError(f"Invalid source provenance: {paper}")
        if row.get("prompt_file") != f"prompts/{paper}.txt":
            raise InputError(f"Invalid prompt filename: {paper}")
        raw = (folder / row["prompt_file"]).read_bytes()
        size = row.get("source_bytes")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0 or size > len(raw):
            raise InputError(f"Invalid source length: {paper}")
        if sha256(raw) != row.get("prompt_sha256") or sha256(raw[-size:]) != row.get("source_sha256"):
            raise InputError(f"Prompt/source checksum mismatch: {paper}")
        raw.decode("utf-8")
    return bundle, read_json(schema_raw)


def validator_for(schema):
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise InputError("Install validation support: python3 -m pip install 'jsonschema>=4.18,<5'") from exc
    Draft202012Validator.check_schema(schema)
    return Draft202012Validator(schema)


def validate_answer(answer, row, run_name, validator):
    errors = list(validator.iter_errors(answer))
    if errors:
        error = errors[0]
        raise InputError(f"Answer schema error at {'/'.join(map(str, error.absolute_path)) or 'root'}: {error.message}")
    for name, expected in (("paper_id", row["paper_id"]), ("run_name", run_name),
                           ("source_sha256", row["source_sha256"])):
        if answer[name] != expected:
            raise InputError(f"Answer {name} differs from its prepared input")
    issues = answer["issues"]
    if answer["issues_found"] != len(issues):
        raise InputError("issues_found must equal the number of entries in issues")
    if answer["status"] == "input_error":
        if issues or not answer["error"]:
            raise InputError("input_error requires an empty issue list and an explanation")
    elif answer["error"] is not None:
        raise InputError("A completed review must have error: null")
    for index, issue in enumerate(issues, 1):
        if issue["issue_id"] != f"ISSUE-{index:03d}":
            raise InputError("Issue IDs must be numbered ISSUE-001, ISSUE-002, ...")
        if issue["target_type"] == "formal_result":
            if not issue["primary_affected_tag"] or issue["failure_mechanism"] == "not_applicable":
                raise InputError("A formal-result issue requires a primary tag and a formal failure mechanism")
        elif issue["failure_mechanism"] != "not_applicable":
            raise InputError("Non-formal issues must use failure_mechanism: not_applicable")
        if issue["target_type"] == "prose_claim" and not issue["diagnostic_evidence"]:
            raise InputError("A prose-claim issue requires diagnostic evidence")


def check_answers(folder, answers):
    bundle, schema = load_inputs(folder)
    validator = validator_for(schema)
    rows = []
    for row in bundle["papers"]:
        paper = row["paper_id"]
        path = answers / f"{paper}.json"
        try:
            answer = read_json(path.read_bytes())
            validate_answer(answer, row, bundle["run_name"], validator)
            rows.append({"paper_id": paper, "status": answer["status"]})
        except (ValueError, OSError) as exc:
            rows.append({"paper_id": paper, "status": "missing" if not path.exists() else "invalid", "error": str(exc)})
    expected = {f"{row['paper_id']}.json" for row in bundle["papers"]}
    extra = sorted(p.name for p in answers.iterdir() if not p.name.startswith(".") and p.name not in expected) if answers.is_dir() else []
    counts = dict(Counter(row["status"] for row in rows))
    valid = not extra and all(row["status"] in ("ok", "input_error") for row in rows)
    return {"valid": valid, "run_name": bundle["run_name"], "paper_count": len(rows),
            "counts": counts, "unexpected_files": extra, "papers": rows}


# JSON recovery algorithms used for the post.
# TODO: When another question is added, share the JSON recovery helpers while
# keeping question-specific response recognition and validation separate.
PARSER_SCHEMA = "benchmark-q1-tolerant-parser-v3"
COMPLETE_FENCE_RE = re.compile(
    r"\A\s*```[A-Za-z0-9_-]*[ \t]*\r?\n(.*)\r?\n[ \t]*```\s*\Z",
    re.DOTALL,
)
OPENING_FENCE_RE = re.compile(
    r"\A\s*```[A-Za-z0-9_-]*[ \t]*\r?\n",
)
CLOSING_FENCE_RE = re.compile(r"\r?\n[ \t]*```\s*\Z")
HEX = frozenset("0123456789abcdefABCDEF")
VALID_SIMPLE_ESCAPES = frozenset('"\\/bfnrt')
Q1_SIGNATURE_KEYS = frozenset(
    {"schema_version", "question_id", "status", "paper_id", "issues"}
)
ISSUE_SIGNATURE_KEYS = frozenset(
    {
        "issue_id",
        "title",
        "target_type",
        "failure_location",
        "failure_explanation",
        "classification",
    }
)
ISSUES_ARRAY_RE = re.compile(r'"issues"\s*:\s*\[')
PLAIN_QUOTED_PHRASE_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _'/-]{0,79}\Z")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def escape_surrogates_for_audit(value: object) -> object:
    """Recursively make diagnostic-only values safe for UTF-8 JSON output."""
    # TODO: Consider json.dump(..., ensure_ascii=True) at the audit write boundary
    # instead of recursive sanitization. First confirm whether decoded audit strings
    # must also be free of surrogates; escaping alone preserves them on JSON reload.
    if isinstance(value, str):
        return "".join(
            f"\\u{ord(character):04x}"
            if 0xD800 <= ord(character) <= 0xDFFF
            else character
            for character in value
        )
    if isinstance(value, list):
        return [escape_surrogates_for_audit(item) for item in value]
    if isinstance(value, tuple):
        return tuple(escape_surrogates_for_audit(item) for item in value)
    if isinstance(value, dict):
        return {
            escape_surrogates_for_audit(key): escape_surrogates_for_audit(item)
            for key, item in value.items()
        }
    return value


def canonical_json_sha256(value: object) -> str:
    try:
        rendered = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except UnicodeEncodeError:
        # Keep malformed model output auditable without changing hashes for valid JSON.
        rendered = json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("ascii")
    return sha256_bytes(rendered)


@dataclass(frozen=True)
class Candidate:
    start: int
    end: int
    document: dict[str, Any]

    def audit_record(self, selected: dict[str, Any]) -> dict[str, object]:
        record = {
            "start": self.start,
            "end": self.end,
            "canonical_json_sha256": canonical_json_sha256(self.document),
            "paper_id": self.document.get("paper_id"),
            "issues_found": self.document.get("issues_found"),
            "identical_to_selected": self.document == selected,
        }
        return escape_surrogates_for_audit(record)  # type: ignore[return-value]


@dataclass(frozen=True)
class IssueCandidate:
    start: int
    end: int
    document: dict[str, Any]

    def audit_record(self) -> dict[str, object]:
        record = {
            "working_text_start": self.start,
            "working_text_end": self.end,
            "canonical_json_sha256": canonical_json_sha256(self.document),
            "issue_id": self.document.get("issue_id"),
            "title": self.document.get("title"),
        }
        return escape_surrogates_for_audit(record)  # type: ignore[return-value]


@dataclass(frozen=True)
class ParseResult:
    strict_parse: str
    strict_error: str | None
    transformations: tuple[dict[str, object], ...]
    candidates: tuple[Candidate, ...]
    issue_candidates: tuple[IssueCandidate, ...]
    selected_candidate_index: int | None
    selected_document: dict[str, Any] | None
    working_text_sha256: str

    def audit_record(self) -> dict[str, object]:
        selected = self.selected_document
        record = {
            "parser_schema": PARSER_SCHEMA,
            "strict_parse": self.strict_parse,
            "strict_error": self.strict_error,
            "transformations": list(self.transformations),
            "working_text_sha256": self.working_text_sha256,
            "candidate_count": len(self.candidates),
            "selected_candidate_index": self.selected_candidate_index,
            "multiple_distinct_candidates": bool(
                selected is not None
                and any(candidate.document != selected for candidate in self.candidates[1:])
            ),
            "candidates": (
                [candidate.audit_record(selected) for candidate in self.candidates]
                if selected is not None
                else []
            ),
            "complete_issue_candidate_count": len(self.issue_candidates),
            "complete_issue_candidates": [
                candidate.audit_record() for candidate in self.issue_candidates
            ],
        }
        return escape_surrogates_for_audit(record)  # type: ignore[return-value]


def _strict_parse(raw: str) -> tuple[dict[str, Any] | None, str, str | None, str]:
    try:
        document = json.loads(raw)
        if isinstance(document, dict):
            return document, "clean_json", None, raw
        return None, "invalid", "top-level JSON value is not an object", raw
    except json.JSONDecodeError as clean_error:
        match = COMPLETE_FENCE_RE.match(raw)
        if match is None:
            return None, "invalid", str(clean_error), raw
        inner = match.group(1)
        try:
            document = json.loads(inner)
        except json.JSONDecodeError as fenced_error:
            return None, "invalid", str(fenced_error), inner
        if not isinstance(document, dict):
            return None, "invalid", "top-level fenced JSON value is not an object", inner
        return document, "enclosing_code_fence", None, inner


def _repair_json_lexemes(text: str) -> tuple[str, list[dict[str, object]]]:
    output: list[str] = []
    invalid_escapes = 0
    control_characters = 0
    in_string = False
    index = 0
    while index < len(text):
        char = text[index]
        if not in_string:
            output.append(char)
            if char == '"':
                in_string = True
            index += 1
            continue

        if char == '"':
            output.append(char)
            in_string = False
            index += 1
            continue

        if char == "\\":
            if index + 1 >= len(text):
                output.append(char)
                index += 1
                continue
            following = text[index + 1]
            if following in VALID_SIMPLE_ESCAPES:
                output.extend((char, following))
                index += 2
                continue
            if following == "u" and (
                index + 5 < len(text)
                and all(value in HEX for value in text[index + 2 : index + 6])
            ):
                output.append(text[index : index + 6])
                index += 6
                continue
            # Turn an illegal JSON escape into a literal backslash.  Decoding
            # the repaired JSON therefore preserves the model's visible text.
            output.append("\\\\")
            invalid_escapes += 1
            index += 1
            continue

        if ord(char) < 0x20:
            replacements = {"\b": "\\b", "\f": "\\f", "\n": "\\n", "\r": "\\r", "\t": "\\t"}
            output.append(replacements.get(char, f"\\u{ord(char):04x}"))
            control_characters += 1
            index += 1
            continue

        output.append(char)
        index += 1

    repairs: list[dict[str, object]] = []
    if invalid_escapes:
        repairs.append({"operation": "escape_invalid_json_backslashes", "count": invalid_escapes})
    if control_characters:
        repairs.append({"operation": "escape_literal_control_characters", "count": control_characters})
    return "".join(output), repairs


def _remove_trailing_commas(text: str) -> tuple[str, int]:
    output: list[str] = []
    in_string = False
    escaped = False
    removed = 0
    for index, char in enumerate(text):
        if in_string:
            output.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            output.append(char)
            continue
        if char == ",":
            cursor = index + 1
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
            if cursor < len(text) and text[cursor] in "}]":
                removed += 1
                continue
        output.append(char)
    return "".join(output), removed


def _is_unescaped_quote(text: str, index: int) -> bool:
    if index < 0 or index >= len(text) or text[index] != '"':
        return False
    backslashes = 0
    cursor = index - 1
    while cursor >= 0 and text[cursor] == "\\":
        backslashes += 1
        cursor -= 1
    return backslashes % 2 == 0


def _inside_json_string_before(text: str, stop: int) -> bool:
    in_string = False
    escaped = False
    for char in text[:stop]:
        if not in_string:
            if char == '"':
                in_string = True
            continue
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif char == '"':
            in_string = False
    return in_string


def _repair_unique_unescaped_quote_pair(
    text: str,
) -> tuple[str, dict[str, object] | None]:
    """Escape one uniquely resolving quoted phrase inside a JSON string.

    The rule is deliberately narrow. It starts only at an ``Expecting ','``
    parse error immediately after an unescaped quote inside a string, considers
    short plain-text closing quotes on the same line, and applies a repair only
    when exactly one pair makes the entire text a Q1-shaped JSON object.
    """
    try:
        json.loads(text)
        return text, None
    except json.JSONDecodeError as error:
        error_message = error.msg
        error_position = error.pos
        if error_message != "Expecting ',' delimiter" or error_position <= 0:
            return text, None

    opening = error_position - 1
    if not _is_unescaped_quote(text, opening):
        return text, None
    if not _inside_json_string_before(text, opening):
        return text, None
    if error_position >= len(text) or not text[error_position].isalnum():
        return text, None

    line_end = text.find("\n", error_position)
    if line_end < 0:
        line_end = len(text)
    successes: list[tuple[int, str, dict[str, Any]]] = []
    for closing in range(error_position + 1, line_end):
        if not _is_unescaped_quote(text, closing):
            continue
        phrase = text[error_position:closing]
        if PLAIN_QUOTED_PHRASE_RE.fullmatch(phrase) is None:
            continue
        if closing + 1 >= len(text) or not text[closing + 1].isspace():
            continue
        cursor = closing + 1
        while cursor < len(text) and text[cursor] in " \t":
            cursor += 1
        if cursor >= line_end or not text[cursor].isalpha():
            continue
        candidate = (
            text[:opening]
            + '\\"'
            + text[opening + 1 : closing]
            + '\\"'
            + text[closing + 1 :]
        )
        try:
            document = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(document, dict) and _looks_like_q1(document):
            successes.append((closing, candidate, document))
    if len(successes) != 1:
        return text, None
    closing, repaired, _document = successes[0]
    return repaired, {
        "operation": "escape_unique_unescaped_quote_pair",
        "count": 1,
        "opening_offset": opening,
        "closing_offset": closing,
        "quoted_text_sha256": sha256_bytes(
            text[error_position:closing].encode("utf-8")
        ),
    }


def _looks_like_q1(document: dict[str, Any]) -> bool:
    keys = set(document)
    return isinstance(document.get("issues"), list) or len(keys & Q1_SIGNATURE_KEYS) >= 3


def _find_candidates(text: str) -> tuple[Candidate, ...]:
    decoder = json.JSONDecoder()
    candidates: list[Candidate] = []
    index = 0
    while index < len(text):
        if text[index] != "{":
            index += 1
            continue
        try:
            document, end = decoder.raw_decode(text, index)
        except json.JSONDecodeError:
            index += 1
            continue
        if isinstance(document, dict) and _looks_like_q1(document):
            candidates.append(Candidate(index, end, document))
            index = end
        else:
            index += 1
    return tuple(candidates)


def _looks_like_issue(document: dict[str, Any]) -> bool:
    keys = set(document)
    return isinstance(document.get("issue_id"), str) and len(
        keys & ISSUE_SIGNATURE_KEYS
    ) >= 3


def _find_issue_candidates(text: str) -> tuple[IssueCandidate, ...]:
    """Find complete objects in the first visible ``issues`` array.

    The offsets refer to the deterministically repaired working text. This is
    deliberately narrower than scanning every object in the response: prose
    after an unfinished answer and later attempted answers are not merged.
    """
    match = ISSUES_ARRAY_RE.search(text)
    if match is None:
        return ()
    decoder = json.JSONDecoder()
    candidates: list[IssueCandidate] = []
    array_depth = 1
    index = match.end()
    in_string = False
    escaped = False
    while index < len(text) and array_depth > 0:
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
            index += 1
            continue
        if char == "[":
            array_depth += 1
            index += 1
            continue
        if char == "]":
            array_depth -= 1
            index += 1
            continue
        if char == "{" and array_depth == 1:
            try:
                document, end = decoder.raw_decode(text, index)
            except json.JSONDecodeError:
                index += 1
                continue
            if isinstance(document, dict) and _looks_like_issue(document):
                candidates.append(IssueCandidate(index, end, document))
            index = end
            continue
        index += 1
    return tuple(candidates)


def parse_q1_visible_output(raw: str) -> ParseResult:
    """Parse one visible response using a fixed first-Q1-object policy."""
    strict_document, strict_parse, strict_error, strict_working = _strict_parse(raw)
    if strict_document is not None:
        transformations: list[dict[str, object]] = []
        if strict_parse == "enclosing_code_fence":
            transformations.append({"operation": "remove_enclosing_code_fence", "count": 1})
        candidate = Candidate(0, len(strict_working), strict_document)
        return ParseResult(
            strict_parse=strict_parse,
            strict_error=None,
            transformations=tuple(transformations),
            candidates=(candidate,),
            issue_candidates=(),
            selected_candidate_index=0,
            selected_document=strict_document,
            working_text_sha256=sha256_bytes(strict_working.encode("utf-8")),
        )

    working = raw
    transformations = []
    if working.startswith("\ufeff"):
        working = working[1:]
        transformations.append({"operation": "remove_utf8_bom", "count": 1})
    opening = OPENING_FENCE_RE.match(working)
    if opening is not None:
        working = working[opening.end() :]
        transformations.append({"operation": "remove_opening_code_fence", "count": 1})
    closing = CLOSING_FENCE_RE.search(working)
    if closing is not None:
        working = working[: closing.start()]
        transformations.append({"operation": "remove_closing_code_fence", "count": 1})

    working, lexical_repairs = _repair_json_lexemes(working)
    transformations.extend(lexical_repairs)
    working, removed_commas = _remove_trailing_commas(working)
    if removed_commas:
        transformations.append({"operation": "remove_trailing_commas", "count": removed_commas})

    working, quote_repair = _repair_unique_unescaped_quote_pair(working)
    if quote_repair is not None:
        transformations.append(quote_repair)

    candidates = _find_candidates(working)
    issue_candidates = _find_issue_candidates(working)
    selected = candidates[0].document if candidates else None
    return ParseResult(
        strict_parse=strict_parse,
        strict_error=strict_error,
        transformations=tuple(transformations),
        candidates=candidates,
        issue_candidates=issue_candidates,
        selected_candidate_index=0 if candidates else None,
        selected_document=selected,
        working_text_sha256=sha256_bytes(working.encode("utf-8")),
    )


def diagnostic_json(value):
    """Keep malformed model values representable in the audit, never in predictions."""
    if isinstance(value, float) and not math.isfinite(value):
        return repr(value)
    if isinstance(value, list):
        return [diagnostic_json(item) for item in value]
    if isinstance(value, dict):
        return {key: diagnostic_json(item) for key, item in value.items()}
    return value




# Request canonicalization, validation, and scoreability rules used for the post.
# Extracted without changing function bodies from the pinned upstream snapshot.
from types import ModuleType

Q1RunError = InputError
QUESTION_ID = "Q1"

PREDICTION_SCHEMA = "benchmark-q1-issue-inventory-prediction-v1"

ISSUE_ID_RE = re.compile(r"^ISSUE-\d{3}$")

TARGET_TYPES = frozenset({"formal_result", "algorithm", "prose_claim", "other"})

FORMAL_MECHANISMS = frozenset(
    {
        "statement_support_mismatch",
        "invalid_input_or_dependency",
        "invalid_or_incomplete_reasoning",
        "other",
    }
)

SEVERITIES = frozenset({"minor_issue", "moderate_issue", "serious_issue"})

def _nonempty_string(value: object, field: str, errors: list[str]) -> None:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{field} must be a nonempty string")

def _validate_location(value: object, field: str, errors: list[str]) -> None:
    if not isinstance(value, dict):
        errors.append(f"{field} must be an object")
        return
    if set(value) != {"tag", "quote"}:
        errors.append(f"{field} must contain exactly tag and quote")
    tag = value.get("tag")
    if tag is not None and (not isinstance(tag, str) or not tag.strip()):
        errors.append(f"{field}.tag must be null or a nonempty string")
    _nonempty_string(value.get("quote"), f"{field}.quote", errors)

def validate_prediction(
    document: object,
    *,
    args: argparse.Namespace,
    paper: dict[str, object],
) -> dict[str, object]:
    if not isinstance(document, dict):
        raise Q1RunError("prediction must be a JSON object")
    errors: list[str] = []
    status = document.get("status")
    expected_keys = {
        "schema_version",
        "question_id",
        "status",
        "run_name",
        "paper_id",
        "source_sha256",
        "issues_found",
        "issues",
        "error",
    }
    if set(document) != expected_keys:
        errors.append(
            "top-level keys differ: "
            f"missing={sorted(expected_keys - set(document))}, "
            f"extra={sorted(set(document) - expected_keys)}"
        )
    if document.get("schema_version") != PREDICTION_SCHEMA:
        errors.append("schema_version mismatch")
    if document.get("question_id") != QUESTION_ID:
        errors.append("question_id mismatch")
    if document.get("run_name") != args.run_name:
        errors.append("run_name mismatch")
    if document.get("paper_id") != paper["paper_id"]:
        errors.append("paper_id mismatch")
    if document.get("source_sha256") != paper["source_sha256"]:
        errors.append("source_sha256 mismatch")
    issues = document.get("issues")
    if not isinstance(issues, list):
        errors.append("issues must be an array")
        issues = []
    issues_found = document.get("issues_found")
    if isinstance(issues_found, bool) or not isinstance(issues_found, int):
        errors.append("issues_found must be a nonnegative integer")
    elif issues_found != len(issues):
        errors.append("issues_found must equal issues.length")
    if status == "input_error":
        if issues or document.get("issues_found") != 0:
            errors.append("input_error must contain no issues")
        _nonempty_string(document.get("error"), "error", errors)
    elif status == "ok":
        if document.get("error") is not None:
            errors.append("ok status requires error to be null")
    else:
        errors.append("status must be ok or input_error")

    seen_ids: set[str] = set()
    for index, issue in enumerate(issues):
        prefix = f"issues[{index}]"
        if not isinstance(issue, dict):
            errors.append(f"{prefix} must be an object")
            continue
        required = {
            "issue_id",
            "title",
            "target_type",
            "primary_affected_tag",
            "failure_mechanism",
            "failure_location",
            "diagnostic_evidence",
            "failure_explanation",
            "dependent_affected_formal_result_tags",
            "overall_effect",
            "least_adequate_overall_repair",
            "classification",
            "why_classification",
        }
        if set(issue) != required:
            errors.append(f"{prefix} keys differ")
        issue_id = issue.get("issue_id")
        expected_id = f"ISSUE-{index + 1:03d}"
        if not isinstance(issue_id, str) or not ISSUE_ID_RE.fullmatch(issue_id):
            errors.append(f"{prefix}.issue_id is invalid")
        elif issue_id != expected_id:
            errors.append(f"{prefix}.issue_id must be {expected_id}")
        elif issue_id in seen_ids:
            errors.append(f"duplicate issue_id: {issue_id}")
        else:
            seen_ids.add(issue_id)
        _nonempty_string(issue.get("title"), f"{prefix}.title", errors)
        target = issue.get("target_type")
        if not isinstance(target, str) or target not in TARGET_TYPES:
            errors.append(f"{prefix}.target_type is invalid")
        primary = issue.get("primary_affected_tag")
        if primary is not None and (
            not isinstance(primary, str) or not primary.strip()
        ):
            errors.append(f"{prefix}.primary_affected_tag is invalid")
        mechanism = issue.get("failure_mechanism")
        if isinstance(target, str) and target in {"formal_result", "other"} and (
            not isinstance(primary, str) or not primary.strip()
        ):
            errors.append(f"{prefix}.{target} requires a primary tag")
        if target == "formal_result":
            if not isinstance(mechanism, str) or mechanism not in FORMAL_MECHANISMS:
                errors.append(f"{prefix}.formal_result mechanism is invalid")
        elif isinstance(target, str) and target in TARGET_TYPES and mechanism != "not_applicable":
            errors.append(f"{prefix}.non-formal mechanism must be not_applicable")
        _validate_location(issue.get("failure_location"), f"{prefix}.failure_location", errors)
        evidence = issue.get("diagnostic_evidence")
        if not isinstance(evidence, list):
            errors.append(f"{prefix}.diagnostic_evidence must be an array")
        else:
            if target == "prose_claim" and not evidence:
                errors.append(
                    f"{prefix}.prose_claim requires diagnostic evidence"
                )
            for location_index, location in enumerate(evidence):
                _validate_location(
                    location,
                    f"{prefix}.diagnostic_evidence[{location_index}]",
                    errors,
                )
        dependent = issue.get("dependent_affected_formal_result_tags")
        if not isinstance(dependent, list) or not all(
            isinstance(value, str) and value.strip() for value in dependent
        ):
            errors.append(f"{prefix}.dependent tags must be nonempty strings")
        elif len(dependent) != len(set(dependent)):
            errors.append(f"{prefix}.dependent tags must be unique")
        for field in (
            "failure_explanation",
            "overall_effect",
            "least_adequate_overall_repair",
            "why_classification",
        ):
            _nonempty_string(issue.get(field), f"{prefix}.{field}", errors)
        classification = issue.get("classification")
        if not isinstance(classification, str) or classification not in SEVERITIES:
            errors.append(f"{prefix}.classification is invalid")
    if errors:
        raise Q1RunError("; ".join(errors))
    return document

def canonicalize_prediction(
    document: object,
    *,
    args: argparse.Namespace,
    paper: dict[str, object],
) -> tuple[object, list[dict[str, object]]]:
    """Normalize only fields determined mechanically by the request or array order."""
    if not isinstance(document, dict):
        return document, []
    canonical = copy.deepcopy(document)
    changes: list[dict[str, object]] = []

    def assign(container: dict[str, object], field: str, value: object, path: str) -> None:
        if field not in container or container.get(field) != value:
            changes.append(
                {
                    "field": path,
                    "before": container.get(field),
                    "after": value,
                }
            )
            container[field] = value

    for field, value in (
        ("schema_version", PREDICTION_SCHEMA),
        ("question_id", QUESTION_ID),
        ("run_name", args.run_name),
        ("paper_id", paper["paper_id"]),
        ("source_sha256", paper["source_sha256"]),
    ):
        assign(canonical, field, value, field)

    issues = canonical.get("issues")
    if isinstance(issues, list):
        assign(canonical, "issues_found", len(issues), "issues_found")
        for index, issue in enumerate(issues):
            if isinstance(issue, dict):
                assign(
                    issue,
                    "issue_id",
                    f"ISSUE-{index + 1:03d}",
                    f"issues[{index}].issue_id",
                )
                dependent = issue.get("dependent_affected_formal_result_tags")
                if isinstance(dependent, list):
                    deduplicated: list[object] = []
                    for value in dependent:
                        if value not in deduplicated:
                            deduplicated.append(value)
                    assign(
                        issue,
                        "dependent_affected_formal_result_tags",
                        deduplicated,
                        f"issues[{index}].dependent_affected_formal_result_tags",
                    )
                if isinstance(issue.get("target_type"), str) and issue.get(
                    "target_type"
                ) in TARGET_TYPES - {"formal_result"}:
                    assign(
                        issue,
                        "failure_mechanism",
                        "not_applicable",
                        f"issues[{index}].failure_mechanism",
                    )
    if canonical.get("status") == "ok":
        assign(canonical, "error", None, "error")
    return canonical, changes

MISSING_VALUE = "NA"

ISSUE_FIELD_DEFAULTS: dict[str, object] = {
    "issue_id": MISSING_VALUE,
    "title": MISSING_VALUE,
    "target_type": MISSING_VALUE,
    "primary_affected_tag": MISSING_VALUE,
    "failure_mechanism": MISSING_VALUE,
    "failure_location": {"tag": MISSING_VALUE, "quote": MISSING_VALUE},
    "diagnostic_evidence": [{"tag": MISSING_VALUE, "quote": MISSING_VALUE}],
    "failure_explanation": MISSING_VALUE,
    "dependent_affected_formal_result_tags": [MISSING_VALUE],
    "overall_effect": MISSING_VALUE,
    "least_adequate_overall_repair": MISSING_VALUE,
    "classification": MISSING_VALUE,
    "why_classification": MISSING_VALUE,
}

def contains_unpaired_surrogate(value: object) -> bool:
    if isinstance(value, str):
        return any(0xD800 <= ord(character) <= 0xDFFF for character in value)
    if isinstance(value, list):
        return any(contains_unpaired_surrogate(item) for item in value)
    if isinstance(value, dict):
        return any(
            contains_unpaired_surrogate(key) or contains_unpaired_surrogate(item)
            for key, item in value.items()
        )
    return False

def normalize_scoreable_issue_shape(
    document: object,
) -> tuple[list[dict[str, object]], tuple[str, ...]]:
    """Mark omitted issue fields as NA and drop unknown issue-level fields.

    These are scoreability adjustments, not claims about the missing values.
    The exact paths and hashes of removed values are returned for the audit
    record. Required Q1 issue fields and their values are never removed here.
    """
    if not isinstance(document, dict) or not isinstance(document.get("issues"), list):
        return [], ()
    adjustments: list[dict[str, object]] = []
    missing_paths: list[str] = []
    required = set(ISSUE_FIELD_DEFAULTS)
    for index, issue in enumerate(document["issues"]):
        if not isinstance(issue, dict):
            continue
        for key in sorted(required - set(issue)):
            path = f"issues[{index}].{key}"
            value = copy.deepcopy(ISSUE_FIELD_DEFAULTS[key])
            issue[key] = value
            missing_paths.append(path)
            adjustments.append(
                {
                    "operation": "fill_missing_field_with_na",
                    "field": path,
                    "value": value,
                }
            )
        for key in sorted(set(issue) - required):
            value = issue[key]
            path = f"issues[{index}].{key}"
            del issue[key]
            adjustments.append(
                {
                    "operation": "remove_unknown_issue_field",
                    "field": path,
                    "removed_value_sha256": canonical_json_sha256(value),
                }
            )
    return adjustments, tuple(missing_paths)

def scoreable_schema_warning_kinds(
    document: object,
    *,
    runner: ModuleType,
    args: argparse.Namespace,
    paper: dict[str, object],
    missing_fields: tuple[str, ...] = (),
) -> tuple[str, ...]:
    """Return narrowly scoreable warning kinds, or an empty tuple.

    A temporary validation probe supplies only schema placeholders. The actual
    parsed document is never changed.
    """
    if not isinstance(document, dict) or not isinstance(document.get("issues"), list):
        return ()
    if contains_unpaired_surrogate(document):
        return ()
    probe = copy.deepcopy(document)
    warning_kinds: set[str] = set()
    for issue in probe["issues"]:
        if not isinstance(issue, dict):
            continue
        target = issue.get("target_type")
        target_missing = target == MISSING_VALUE
        if target_missing:
            mechanism = issue.get("failure_mechanism")
            formal_mechanisms = set(getattr(runner, "FORMAL_MECHANISMS", ()))
            if isinstance(mechanism, str) and mechanism in formal_mechanisms:
                issue["target_type"] = "formal_result"
            else:
                issue["target_type"] = "other"
                if mechanism == MISSING_VALUE:
                    issue["failure_mechanism"] = "not_applicable"
        elif issue.get("failure_mechanism") == MISSING_VALUE:
            issue["failure_mechanism"] = (
                "other"
                if issue.get("target_type") == "formal_result"
                else "not_applicable"
            )
        if issue.get("classification") == MISSING_VALUE:
            issue["classification"] = "minor_issue"
        target = issue.get("target_type")
        if target == "prose_claim" and issue.get(
            "diagnostic_evidence"
        ) == []:
            location = issue.get("failure_location")
            if not isinstance(location, dict):
                return ()
            issue["diagnostic_evidence"] = [copy.deepcopy(location)]
            warning_kinds.add("empty_prose_diagnostic_evidence")
        if isinstance(target, str) and target in {"formal_result", "other"} and issue.get(
            "primary_affected_tag"
        ) is None:
            issue["primary_affected_tag"] = "__VALIDATION_PROBE_MISSING_PRIMARY_TAG__"
            warning_kinds.add("null_primary_affected_tag")
    if missing_fields:
        warning_kinds.add("missing_fields_filled_with_na")
    if not warning_kinds:
        return ()
    try:
        runner.validate_prediction(probe, args=args, paper=paper)
    except runner.Q1RunError:
        return ()
    return tuple(sorted(warning_kinds))

def build_mechanical_prediction(
    issues: list[dict[str, object]],
    *,
    runner: ModuleType,
    args: argparse.Namespace,
    paper: dict[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Build only the request-bound envelope around supplied issue objects."""
    document: dict[str, object] = {
        "schema_version": getattr(runner, "PREDICTION_SCHEMA"),
        "question_id": getattr(runner, "QUESTION_ID"),
        "status": "ok",
        "run_name": args.run_name,
        "paper_id": paper["paper_id"],
        "source_sha256": paper["source_sha256"],
        "issues_found": len(issues),
        "issues": copy.deepcopy(issues),
        "error": None,
    }
    canonical, normalizations = runner.canonicalize_prediction(
        document,
        args=args,
        paper=paper,
    )
    if not isinstance(canonical, dict):
        raise ValueError("generating runner returned a non-object prediction")
    return canonical, normalizations

def salvage_fully_described_issues(
    parsed: ParseResult,
    *,
    runner: ModuleType,
    args: argparse.Namespace,
    paper: dict[str, object],
    pre_salvage_status: str,
    pre_salvage_validation_error: str | None,
) -> tuple[
    dict[str, object] | None,
    list[dict[str, object]],
    dict[str, object],
]:
    """Retain complete issues under the same scoreability policy as siblings.

    This is a format-only fallback for a response that otherwise cannot be
    scored. Missing fields receive explicit NA sentinels; semantic values are
    never inferred or repaired.
    """
    candidate_rows: list[dict[str, object]] = []
    selected = parsed.selected_document
    if isinstance(selected, dict) and isinstance(selected.get("issues"), list):
        source = "selected_q1_object_issues"
        for index, issue in enumerate(selected["issues"]):
            if not isinstance(issue, dict):
                candidate_rows.append(
                    {
                        "source_index": index,
                        "document": None,
                        "candidate_audit": {
                            "source_index": index,
                            "value_type": type(issue).__name__,
                        },
                    }
                )
                continue
            candidate_rows.append(
                {
                    "source_index": index,
                    "document": issue,
                    "candidate_audit": {
                        "source_index": index,
                        "canonical_json_sha256": canonical_json_sha256(issue),
                        "issue_id": escape_surrogates_for_audit(issue.get("issue_id")),
                        "title": escape_surrogates_for_audit(issue.get("title")),
                    },
                }
            )
    else:
        source = "first_visible_issues_array_complete_objects"
        for index, candidate in enumerate(parsed.issue_candidates):
            candidate_rows.append(
                {
                    "source_index": index,
                    "document": candidate.document,
                    "candidate_audit": {
                        "source_index": index,
                        **candidate.audit_record(),
                    },
                }
            )

    retained: list[dict[str, object]] = []
    candidate_audits: list[dict[str, object]] = []
    for row in candidate_rows:
        issue = row["document"]
        audit = dict(row["candidate_audit"])
        if not isinstance(issue, dict):
            audit.update(
                {
                    "decision": "discarded",
                    "validation_error": "issue array value is not an object",
                }
            )
            candidate_audits.append(audit)
            continue
        candidate_issue = copy.deepcopy(issue)
        if contains_unpaired_surrogate(candidate_issue):
            audit.update(
                {
                    "decision": "discarded",
                    "validation_error": "issue contains an unpaired Unicode surrogate",
                    "scoreability_adjustments": [],
                    "scoreable_schema_warning_kinds": [],
                }
            )
            candidate_audits.append(audit)
            continue
        candidate_document = {"issues": [candidate_issue]}
        adjustments, missing_fields = normalize_scoreable_issue_shape(
            candidate_document
        )
        probe, _probe_normalizations = build_mechanical_prediction(
            [candidate_issue],
            runner=runner,
            args=args,
            paper=paper,
        )
        warning_kinds: tuple[str, ...] = ()
        try:
            runner.validate_prediction(probe, args=args, paper=paper)
        except runner.Q1RunError as exc:
            warning_kinds = scoreable_schema_warning_kinds(
                probe,
                runner=runner,
                args=args,
                paper=paper,
                missing_fields=missing_fields,
            )
            if not warning_kinds:
                audit.update(
                    {
                        "decision": "discarded",
                        "validation_error": str(exc),
                        "scoreability_adjustments": adjustments,
                        "scoreable_schema_warning_kinds": [],
                    }
                )
            else:
                audit.update(
                    {
                        "decision": "retained",
                        "validation_error": str(exc),
                        "scoreability_adjustments": adjustments,
                        "scoreable_schema_warning_kinds": list(warning_kinds),
                    }
                )
                retained.append(candidate_issue)
        else:
            audit.update(
                {
                    "decision": "retained",
                    "validation_error": None,
                    "scoreability_adjustments": adjustments,
                    "scoreable_schema_warning_kinds": [],
                }
            )
            retained.append(candidate_issue)
        candidate_audits.append(audit)

    salvage_audit: dict[str, object] = {
        "attempted": True,
        "policy": "retain_complete_issues_under_standard_scoreability_policy",
        "source": source,
        "pre_salvage_status": pre_salvage_status,
        "pre_salvage_validation_error": pre_salvage_validation_error,
        "complete_candidate_count": len(candidate_rows),
        "retained_issue_count": len(retained),
        "discarded_complete_candidate_count": len(candidate_rows) - len(retained),
        "candidates": candidate_audits,
    }
    if not retained:
        return None, [], salvage_audit

    canonical, normalizations = build_mechanical_prediction(
        retained,
        runner=runner,
        args=args,
        paper=paper,
    )
    try:
        runner.validate_prediction(canonical, args=args, paper=paper)
    except runner.Q1RunError as exc:
        has_missing_fields = any(
            adjustment.get("operation") == "fill_missing_field_with_na"
            for candidate in candidate_audits
            if candidate.get("decision") == "retained"
            for adjustment in candidate.get("scoreability_adjustments", [])
            if isinstance(adjustment, dict)
        )
        warning_kinds = scoreable_schema_warning_kinds(
            canonical,
            runner=runner,
            args=args,
            paper=paper,
            missing_fields=("salvaged_issue_missing_field",) if has_missing_fields else (),
        )
        if not warning_kinds:
            salvage_audit["composition_validation_error"] = str(exc)
            return None, normalizations, salvage_audit
        salvage_audit["composition_scoreable_schema_warning_kinds"] = list(
            warning_kinds
        )
    salvage_audit["composition_validation_error"] = None
    return canonical, normalizations, salvage_audit

POSTPROCESSING_ORIGIN = {'upstream_commit': 'c2f72344442e7c8234a36af6cf14d3a4854c9a84', 'postprocessor_sha256': 'ae7955e3cfb3de6e38982aaacbe322cc1ea7db4baab545fff6d758a1796611c3', 'generating_runner_sha256': '85591e1a04ac1ff90c4cf3504b3568d41156de65456884f618593b0a891e2c82'}


def postprocess_visible(raw, row, run_name, *, incomplete=False):
    """Apply postprocessing while retaining a complete audit trail."""
    parsed = parse_q1_visible_output(raw)
    runner = sys.modules[__name__]
    args = argparse.Namespace(run_name=run_name)
    canonical = None
    status = "invalid_json"
    error = "No complete Q1 JSON object was recoverable"
    normalizations, adjustments, missing = [], [], ()
    warnings = ()
    salvage = {"attempted": False}
    scoreable = False
    if parsed.selected_document is not None:
        canonical, normalizations = canonicalize_prediction(parsed.selected_document, args=args, paper=row)
        adjustments, missing = normalize_scoreable_issue_shape(canonical)
        try:
            if contains_unpaired_surrogate(canonical):
                raise InputError("Answer contains an unpaired Unicode surrogate")
            validate_prediction(canonical, args=args, paper=row)
            status, error, scoreable = "valid", None, True
        except InputError as exc:
            error = str(exc)
            warnings = scoreable_schema_warning_kinds(canonical, runner=runner, args=args, paper=row, missing_fields=missing)
            status = "scoreable_schema_warning" if warnings else "invalid_schema"
            scoreable = bool(warnings)
    if not scoreable:
        canonical, normalizations, salvage = salvage_fully_described_issues(
            parsed, runner=runner, args=args, paper=row,
            pre_salvage_status="incomplete_response" if incomplete else status,
            pre_salvage_validation_error=error)
        if canonical is not None:
            status, scoreable, error = "scoreable_partial_issue_salvage", True, None
            adjustments, missing, warnings = [], (), ()
    if scoreable and canonical.get("status") == "input_error":
        status, scoreable = "input_error", False
    if scoreable:
        json.dumps(canonical, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return canonical if scoreable else None, {
        "parse": parsed.audit_record(), "final_status": status,
        "final_validation_error": error, "mechanical_normalizations": normalizations,
        "scoreability_adjustments": adjustments, "missing_field_paths": list(missing),
        "scoreable_schema_warning_kinds": list(warnings), "issue_level_salvage": salvage,
        "issue_inventory_filtered": bool(salvage.get("attempted") and scoreable),
        "semantic_fields_modified": bool(adjustments or any(
            ".failure_mechanism" in str(change.get("field", "")) for change in normalizations)),
        "response_incomplete": incomplete,
    }


def parse_answers(args):
    if args.output.exists():
        raise InputError("Output already exists; choose a new parsed-output directory")
    if not args.raw.is_dir():
        raise InputError(f"Raw answer directory not found: {args.raw}")
    bundle, schema = load_inputs(args.inputs)
    validator = validator_for(schema)
    inputs_bytes = (args.inputs / "inputs.json").read_bytes()
    example_record = args.raw.parent / "run.json"
    example_run = read_json(example_record.read_bytes()) if example_record.is_file() else {}
    is_example = isinstance(example_run, dict) and example_run.get("runner") == "single-user-message-chat-completions-example-v1"
    if is_example and example_run.get("inputs_sha256") != sha256(inputs_bytes):
        raise InputError("The API example run used different prepared inputs")
    parser_hash = sha256(Path(__file__).read_bytes())
    wrapper_hash = sha256(Path(__file__).read_bytes())
    expected_names = {row["paper_id"] + suffix for row in bundle["papers"] for suffix in (".txt", ".json")}
    extra = sorted(path.name for path in args.raw.iterdir()
                   if not path.name.startswith(".") and path.name not in expected_names)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".parse-answers-", dir=args.output.parent) as temp:
        staging = Path(temp) / "parsed"
        for folder in ("raw", "parsed_predictions", "parse_audit"):
            (staging / folder).mkdir(parents=True, exist_ok=True)
        (staging / "inputs.json").write_bytes(inputs_bytes)
        (staging / "output_schema.json").write_bytes((args.inputs / "output_schema.json").read_bytes())
        predictions = []
        audits = []
        for row in bundle["papers"]:
            paper = row["paper_id"]
            candidates = [args.raw / (paper + suffix) for suffix in (".txt", ".json")]
            paths = [path for path in candidates if path.is_file()]
            audit = {
                "schema_version": "benchmark-q1-output-parse-audit-v1",
                "paper_id": paper, "version": row["version"],
                "source_sha256": row["source_sha256"],
                "parser_sha256": parser_hash, "wrapper_sha256": wrapper_hash,
                "recovery_origin_sha256": RECOVERY_ORIGIN_SHA256,
                "postprocessing_origin": POSTPROCESSING_ORIGIN,
                "provider_called": False, "gold_read": False, "raw_response_modified": False,
                "semantic_fields_modified": False, "semantic_values_invented": False,
                "issue_inventory_filtered": False, "scoreable_prediction_written": False,
                "mechanical_normalizations": [], "final_validation_error": None,
                "derived_prediction_sha256": None,
            }
            if not paths:
                audit["final_status"] = "missing_response"
            elif len(paths) > 1:
                audit["final_status"] = "ambiguous_raw_files"
                audit["final_validation_error"] = "Both .txt and .json exist; supply one raw answer per paper"
            else:
                path = paths[0]
                raw_bytes = path.read_bytes()
                audit["raw_response_filename"] = path.name
                audit["raw_response_sha256"] = sha256(raw_bytes)
                (staging / "raw" / path.name).write_bytes(raw_bytes)
                audit["final_status"] = "invalid_json"
                try:
                    if is_example:
                        audit["final_status"] = "incomplete_response"
                        response_raw = (args.raw.parent / "responses" / f"{paper}.json").read_bytes()
                        audit["api_response_sha256"] = sha256(response_raw)
                        response = read_json(response_raw)
                        choice = response["choices"][0]
                        if choice.get("finish_reason") not in ("stop", "length"):
                            raise InputError("The API example response did not finish with a text answer")
                        if choice["message"].get("content") != raw_bytes.decode("utf-8"):
                            raise InputError("Raw answer differs from the saved API response")
                    document, processing = postprocess_visible(
                        raw_bytes.decode("utf-8"), row, bundle["run_name"],
                        incomplete=is_example and choice.get("finish_reason") != "stop")
                    audit.update(processing)
                    if document is not None:
                        destination = staging / "parsed_predictions" / f"{paper}.json"
                        write_json(destination, document)
                        audit.update(scoreable_prediction_written=True,
                                     derived_prediction_sha256=sha256(destination.read_bytes()))
                        predictions.append(document)
                except (ValueError, OSError, KeyError, IndexError, TypeError) as exc:
                    audit["final_validation_error"] = str(exc)
            write_json(staging / "parse_audit" / f"{paper}.json",
                                diagnostic_json(escape_surrogates_for_audit(audit)))
            audits.append(audit)
        counts = dict(sorted(Counter(audit["final_status"] for audit in audits).items()))
        summary = {
            "schema_version": "benchmark-q1-postprocessed-run-v1",
            "implementation": "matherratabench-public-postprocessing-v2",
            "postprocessing_origin": POSTPROCESSING_ORIGIN,
            "source_run_name": bundle["run_name"],
            "source_run_manifest_sha256": sha256(inputs_bytes),
            "source_manifest_kind": "matherratabench-agent-inputs-v1",
            "selected_paper_count": len(audits), "scoreable_prediction_count": len(predictions),
            "final_status_counts": counts, "unexpected_files": extra,
            "provider_called": False, "gold_read": False,
        }
        write_json(staging / "predictions.parsed.json", {
            "schema_version": "benchmark-q1-postprocessed-prediction-set-v1",
            "run_name": bundle["run_name"], "record_count": len(predictions), "records": predictions,
        })
        write_json(staging / "parse_summary.json", summary)
        staging.rename(args.output)
    print(f"Parsed {len(audits)} papers: {json.dumps(counts, sort_keys=True)}")
    print(f"Predictions: {args.output / 'parsed_predictions'}")
    print(f"Audit: {args.output / 'parse_audit'}")
    if extra:
        print("Unexpected raw files: " + ", ".join(extra), file=sys.stderr)
    return 0 if not extra and all(audit["scoreable_prediction_written"] for audit in audits) else 1

def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare", help="Create complete questions from prepared paper sources")
    prep.add_argument("--papers", type=Path, required=True)
    prep.add_argument("--paper-id", action="append", help="Bare arXiv ID; repeat for a subset (default: all found)")
    prep.add_argument("--run-name", required=True)
    prep.add_argument("--output", type=Path, required=True)
    parse = commands.add_parser("parse", help="Recover and validate predictions from raw agent answers")
    parse.add_argument("--inputs", type=Path, required=True)
    parse.add_argument("--raw", type=Path, required=True, help="One <paper_id>.txt or .json raw answer per paper")
    parse.add_argument("--output", type=Path, required=True)
    check = commands.add_parser("check", help="Check already structured JSON answers")
    check.add_argument("--inputs", type=Path, required=True)
    check.add_argument("--answers", type=Path, required=True)
    return result


def main(argv=None):
    command = parser()
    args = command.parse_args(argv)
    try:
        if args.command == "prepare":
            prepare(args)
            return 0
        if args.command == "parse":
            return parse_answers(args)
        report = check_answers(args.inputs, args.answers)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["valid"] else 1
    except (ValueError, OSError) as exc:
        command.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
