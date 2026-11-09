#!/usr/bin/env python3
"""Luna judges: coherent diagnosis and bounded combinations of distinct gold defects."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, NoReturn


SEMANTIC_DIR = Path(__file__).resolve().parent
REPO_ROOT = SEMANTIC_DIR.parent.parent

import score_q1 as mechanical


INPUT_SCHEMA = "benchmark-q1-semantic-issue-coverage-input-v3"
INPUT_MANIFEST_SCHEMA = "benchmark-q1-semantic-issue-coverage-input-manifest-v3"
# All Luna policies use the Luna.5 response contract. These identifiers stay
# fixed so existing judgments and score reports remain valid.
JUDGMENT_SCHEMA = "benchmark-q1-semantic-issue-coverage-judgment-luna5"
POLICY_REVISION = "20260924-restored-strict-luna5"
MAX_SUPPORTING_ENTRIES = 3
SCORE_SCHEMA = "benchmark-q1-semantic-issue-coverage-score-luna5"
QUESTION_ID = "Q1-semantic-issue-coverage"
EVIDENCE_FIELDS = ("failure_explanation", "overall_effect", "least_adequate_overall_repair")
GOLD_WEIGHTS = {"minor_issue": 0.5, "moderate_issue": 1.0, "serious_issue": 1.0}
EXACT_RELATION = "same_causal_defect"
BROADER_RELATION = "same_underlying_weakness"
AUDIT_FIELDS = (
    "all_gold_ids_decided", "each_gold_credited_at_most_once",
    "prediction_reuse_requires_explicit_issue_support", "all_ranked_candidates_considered",
    "target_type_not_used_for_identity", "severity_not_used_for_identity",
    "combined_entries_are_self_contained_diagnoses", "support_limit_three_respected",
)
SUBSTANTIVE_SEVERITIES = ("minor_issue", "moderate_issue", "serious_issue")
SEVERITY_RANK = {value: index for index, value in enumerate(SUBSTANTIVE_SEVERITIES)}
TARGET_TYPES = ("formal_result", "algorithm", "prose_claim", "other")
FORMAL_MECHANISMS = (
    "statement_support_mismatch",
    "invalid_input_or_dependency",
    "invalid_or_incomplete_reasoning",
    "other",
)
SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9._-]+$")
PAPER_ID_RE = re.compile(r"^[0-9]{4}\.[0-9]{4,5}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SENTENCE_BOUNDARY_RE = re.compile(
    r"(?P<end>[.!?](?:[\"'\u2019\u201d)\]}]*))(?P<gap>\s+)(?=\S)"
)
TOKEN_SUFFIXES = ("ing", "edly", "ed", "es", "s")
STOP_WORDS = frozenset(
    """
    the a an and or of to in for with from by as at is are be been being this
    that these those it its their then than on into under over when where which
    while does do not no any all each every same only can may must should would
    could if so such between through because hence thus therefore moreover
    however although but also both either neither we they paper source result
    theorem lemma proposition equation proof claim issue defect failure
    statement formal algorithm mathematical displayed stated asserts asserted
    defined definition given using use used applied applies applying requires
    required requiring follows follow following shows show shown gives give
    yields yield invalid incorrect wrong missing omitted omits fails failed
    supported unsupported established establish
    """.split()
)


class SemanticScoreError(ValueError):
    """A Q1 semantic input, judgment, or score artifact is inconsistent."""


def fail(message: str) -> NoReturn:
    raise SystemExit(f"Error: {message}")


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SemanticScoreError(f"cannot read JSON {path}: {exc}") from exc


def portable_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def resolve_path(value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise SemanticScoreError(f"{label} must be a nonempty path")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def write_json_exclusive(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(document, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as exc:
        raise SemanticScoreError(f"refusing to overwrite existing file: {path}") from exc


def percent(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return round(100.0 * numerator / denominator, 6)


def _unique_quotes(issue: dict[str, Any]) -> list[str]:
    values: list[str] = []
    location = issue.get("failure_location")
    if isinstance(location, dict) and isinstance(location.get("quote"), str):
        values.append(str(location["quote"]))
    evidence = issue.get("diagnostic_evidence")
    if isinstance(evidence, list):
        for item in evidence:
            if isinstance(item, dict) and isinstance(item.get("quote"), str):
                values.append(str(item["quote"]))
    return list(dict.fromkeys(value for value in values if value.strip()))


def split_failure_units(issue_id: str, text: str) -> list[dict[str, object]]:
    if not text.strip():
        raise SemanticScoreError(f"{issue_id} failure_explanation is empty")
    spans: list[tuple[int, int]] = []
    start = 0
    for match in _SENTENCE_BOUNDARY_RE.finditer(text):
        end = match.end("end")
        if text[start:end].strip():
            spans.append((start, end))
        start = match.end("gap")
    if text[start:].strip():
        spans.append((start, len(text)))
    if not spans:
        spans = [(0, len(text))]
    return [
        {
            "unit_id": f"{issue_id}:U{index:03d}",
            "start_char": start_char,
            "end_char": end_char,
            "text": text[start_char:end_char],
        }
        for index, (start_char, end_char) in enumerate(spans, start=1)
    ]



def evidence_issue(issue: dict[str, Any]) -> dict[str, object]:
    issue_id = issue.get("issue_id")
    if not isinstance(issue_id, str) or not issue_id:
        raise SemanticScoreError("issue_id must be nonempty")
    explanation = issue.get("failure_explanation")
    if not isinstance(explanation, str) or not explanation.strip():
        raise SemanticScoreError(f"{issue_id}.failure_explanation must be nonempty")
    units: list[dict[str, object]] = []
    for field in EVIDENCE_FIELDS:
        value = issue.get(field)
        if value is None or value == "":
            continue
        if not isinstance(value, str):
            raise SemanticScoreError(f"{issue_id}.{field} must be text")
        if not value.strip():
            continue
        for unit in split_failure_units(issue_id, value):
            units.append({**unit, "unit_id": f"{issue_id}:U{len(units)+1:03d}",
                          "source_field": field})
    location = issue.get("failure_location")
    locators = {
        "primary_affected_tag": issue.get("primary_affected_tag"),
        "failure_location_tag": location.get("tag") if isinstance(location, dict) else None,
        "dependent_affected_formal_result_tags": issue.get("dependent_affected_formal_result_tags") or [],
    }
    return validate_evidence_issue({
        "issue_id": issue_id,
        "source_quotes": _unique_quotes(issue),
        "evidence_units": units,
        "locators": locators,
    }, issue_id)


def description_tokens(issue: dict[str, Any]) -> list[str]:
    text = issue.get("failure_explanation")
    if not isinstance(text, str):
        return []
    raw_tokens = re.findall(r"[a-z0-9]+", text.lower().replace("\\", " "))
    normalized: list[str] = []
    for token in raw_tokens:
        if len(token) < 2 or token in STOP_WORDS:
            continue
        for suffix in TOKEN_SUFFIXES:
            if len(token) > 5 and token.endswith(suffix):
                token = token[: -len(suffix)]
                break
        normalized.append(token)
    return normalized


def _paper_idf(
    gold_issues: list[dict[str, Any]], predicted_issues: list[dict[str, Any]]
) -> dict[str, float]:
    documents = [
        set(description_tokens(issue))
        for issue in [*gold_issues, *predicted_issues]
    ]
    document_frequency: Counter[str] = Counter(
        token for document in documents for token in document
    )
    return {
        token: math.log((len(documents) + 1) / (frequency + 1)) + 1
        for token, frequency in document_frequency.items()
    }


def description_similarity(
    gold_issue: dict[str, Any],
    predicted_issue: dict[str, Any],
    idf: dict[str, float],
) -> float:
    gold_counts = Counter(description_tokens(gold_issue))
    predicted_counts = Counter(description_tokens(predicted_issue))
    if not gold_counts or not predicted_counts:
        return 0.0
    numerator = sum(
        gold_counts[token]
        * idf.get(token, 1.0)
        * predicted_counts[token]
        * idf.get(token, 1.0)
        for token in set(gold_counts) & set(predicted_counts)
    )
    gold_norm = math.sqrt(
        sum((count * idf.get(token, 1.0)) ** 2 for token, count in gold_counts.items())
    )
    predicted_norm = math.sqrt(
        sum(
            (count * idf.get(token, 1.0)) ** 2
            for token, count in predicted_counts.items()
        )
    )
    return 0.0 if not gold_norm or not predicted_norm else numerator / (
        gold_norm * predicted_norm
    )


def rank_all_candidates(
    gold_issues: list[dict[str, Any]], predicted_issues: list[dict[str, Any]]
) -> list[dict[str, object]]:
    idf = _paper_idf(gold_issues, predicted_issues)
    rankings: list[dict[str, object]] = []
    for gold_issue in gold_issues:
        scored = [
            (
                description_similarity(gold_issue, predicted_issue, idf),
                str(predicted_issue["issue_id"]),
            )
            for predicted_issue in predicted_issues
        ]
        rankings.append(
            {
                "gold_issue_id": str(gold_issue["issue_id"]),
                "prediction_issue_ids_ranked": [
                    issue_id
                    for _, issue_id in sorted(scored, key=lambda item: (-item[0], item[1]))
                ],
            }
        )
    return rankings



def validate_evidence_issue(issue: object, label: str) -> dict[str, Any]:
    if not isinstance(issue, dict) or set(issue) != {
        "issue_id", "source_quotes", "evidence_units", "locators"
    }:
        raise SemanticScoreError(f"{label} has unexpected fields")
    issue_id = issue.get("issue_id")
    if not isinstance(issue_id, str) or not issue_id:
        raise SemanticScoreError(f"{label}.issue_id is invalid")
    quotes = issue.get("source_quotes")
    if (not isinstance(quotes, list)
        or not all(isinstance(value, str) and value.strip() for value in quotes)
        or len(quotes) != len(set(quotes))):
        raise SemanticScoreError(f"{label}.source_quotes are invalid")
    locators = issue.get("locators")
    if not isinstance(locators, dict) or set(locators) != {
        "primary_affected_tag", "failure_location_tag", "dependent_affected_formal_result_tags"
    }:
        raise SemanticScoreError(f"{label}.locators are invalid")
    for key in ("primary_affected_tag", "failure_location_tag"):
        if locators[key] is not None and not isinstance(locators[key], str):
            raise SemanticScoreError(f"{label}.{key} is invalid")
    tags = locators["dependent_affected_formal_result_tags"]
    if not isinstance(tags, list) or not all(isinstance(tag, str) and tag.strip() for tag in tags):
        raise SemanticScoreError(f"{label}.dependent tags are invalid")
    units = issue.get("evidence_units")
    if not isinstance(units, list) or not units:
        raise SemanticScoreError(f"{label}.evidence_units must be nonempty")
    previous_ends: dict[str, int] = {}
    previous_field_index = -1
    for index, unit in enumerate(units, start=1):
        if not isinstance(unit, dict) or set(unit) != {
            "unit_id", "source_field", "start_char", "end_char", "text"
        }:
            raise SemanticScoreError(f"{label}.evidence_units[{index-1}] is invalid")
        field = unit.get("source_field")
        if field not in EVIDENCE_FIELDS:
            raise SemanticScoreError(f"{label} evidence field is invalid")
        field_index = EVIDENCE_FIELDS.index(field)
        if field_index < previous_field_index:
            raise SemanticScoreError(f"{label} evidence field order is invalid")
        previous_field_index = field_index
        expected_id = f"{issue_id}:U{index:03d}"
        if unit.get("unit_id") != expected_id:
            raise SemanticScoreError(f"{label} unit ID must be {expected_id}")
        start, end, text = unit.get("start_char"), unit.get("end_char"), unit.get("text")
        if (type(start) is not int or type(end) is not int or start < 0
            or end <= start or start <= previous_ends.get(field, -1)
            or not isinstance(text, str) or not text.strip() or len(text) != end-start):
            raise SemanticScoreError(f"{label}.{expected_id} has invalid span data")
        previous_ends[field] = end
    if "failure_explanation" not in previous_ends:
        raise SemanticScoreError(f"{label} lacks failure explanation evidence")
    return issue


def validate_input(document: object) -> dict[str, Any]:
    expected = {
        "schema_version",
        "question_id",
        "prediction_set_name",
        "paper_id",
        "source_sha256",
        "gold_issues",
        "predicted_issues",
        "candidate_rankings",
    }
    if not isinstance(document, dict) or set(document) != expected:
        raise SemanticScoreError("judge input has unexpected top-level fields")
    if document.get("schema_version") != INPUT_SCHEMA:
        raise SemanticScoreError("judge input schema mismatch")
    if document.get("question_id") != QUESTION_ID:
        raise SemanticScoreError("judge input question ID mismatch")
    prediction_set_name = document.get("prediction_set_name")
    paper_id = document.get("paper_id")
    source_sha256 = document.get("source_sha256")
    if not isinstance(prediction_set_name, str) or not SAFE_NAME_RE.fullmatch(
        prediction_set_name
    ):
        raise SemanticScoreError("judge input prediction_set_name is invalid")
    if not isinstance(paper_id, str) or not PAPER_ID_RE.fullmatch(paper_id):
        raise SemanticScoreError("judge input paper_id is invalid")
    if not isinstance(source_sha256, str) or not SHA256_RE.fullmatch(source_sha256):
        raise SemanticScoreError("judge input source_sha256 is invalid")
    identifiers: dict[str, list[str]] = {}
    for side in ("gold_issues", "predicted_issues"):
        issues = document.get(side)
        if not isinstance(issues, list) or not issues:
            raise SemanticScoreError(f"judge input {side} must be nonempty")
        ids: list[str] = []
        for index, issue in enumerate(issues):
            validated = validate_evidence_issue(issue, f"{side}[{index}]")
            issue_id = str(validated["issue_id"])
            if issue_id in ids:
                raise SemanticScoreError(f"judge input repeats {side} ID {issue_id}")
            ids.append(issue_id)
        identifiers[side] = ids
    rankings = document.get("candidate_rankings")
    if not isinstance(rankings, list) or len(rankings) != len(identifiers["gold_issues"]):
        raise SemanticScoreError("judge input needs one ranking per gold issue")
    for index, ranking in enumerate(rankings):
        if not isinstance(ranking, dict) or set(ranking) != {
            "gold_issue_id",
            "prediction_issue_ids_ranked",
        }:
            raise SemanticScoreError(f"candidate_rankings[{index}] is invalid")
        if ranking.get("gold_issue_id") != identifiers["gold_issues"][index]:
            raise SemanticScoreError(f"candidate_rankings[{index}] gold order mismatch")
        ranked = ranking.get("prediction_issue_ids_ranked")
        if (
            not isinstance(ranked, list)
            or len(ranked) != len(set(ranked))
            or set(ranked) != set(identifiers["predicted_issues"])
        ):
            raise SemanticScoreError(f"candidate_rankings[{index}] is not exhaustive")
    return document


def build_input_document(
    prediction_set_name: str,
    paper_id: str,
    source_sha256: str,
    gold_issues: list[dict[str, Any]],
    predicted_issues: list[dict[str, Any]],
) -> dict[str, Any]:
    return validate_input(
        {
            "schema_version": INPUT_SCHEMA,
            "question_id": QUESTION_ID,
            "prediction_set_name": prediction_set_name,
            "paper_id": paper_id,
            "source_sha256": source_sha256,
            "gold_issues": [evidence_issue(issue) for issue in gold_issues],
            "predicted_issues": [evidence_issue(issue) for issue in predicted_issues],
            "candidate_rankings": rank_all_candidates(gold_issues, predicted_issues),
        }
    )


def _validated_gold_issue(issue: dict[str, Any]) -> bool:
    classification = issue.get("classification")
    if classification == "typo":
        return False
    if classification not in SUBSTANTIVE_SEVERITIES:
        raise SemanticScoreError(f"invalid gold severity: {issue.get('issue_id')}")
    if issue.get("target_type") not in TARGET_TYPES:
        raise SemanticScoreError(f"invalid gold target type: {issue.get('issue_id')}")
    evidence_issue(issue)
    return True


def _prediction_path(postprocessed_dir: Path, paper_id: str) -> Path:
    path = postprocessed_dir / "parsed_predictions" / f"{paper_id}.json"
    if not path.is_file():
        raise SemanticScoreError(f"parsed prediction file is missing: {path}")
    return path


def build_inputs(gold_dir: Path, postprocessed_dir: Path, output_dir: Path, *,
                 paper_ids: list[str] | None = None, allow_empty_predictions: bool = False) -> dict[str, Any]:
    gold_dir = gold_dir.expanduser().resolve()
    postprocessed_dir = postprocessed_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        raise SemanticScoreError(f"refusing to replace output directory: {output_dir}")
    gold_papers, gold_authorities = mechanical.load_gold(gold_dir)
    predictions, prediction_provenance = mechanical.load_predictions(
        postprocessed_dir, gold_papers, gold_authorities, allow_empty=allow_empty_predictions
    )
    if paper_ids is not None:
        if len(set(paper_ids)) != len(paper_ids) or set(paper_ids) - set(gold_papers):
            raise SemanticScoreError("evaluation paper selection is invalid")
        gold_papers = {pid: gold_papers[pid] for pid in sorted(paper_ids)}
        predictions = {pid: value for pid, value in predictions.items() if pid in gold_papers}
    prediction_set_name = prediction_provenance.get("run_name")
    if not isinstance(prediction_set_name, str) or not SAFE_NAME_RE.fullmatch(
        prediction_set_name
    ):
        raise SemanticScoreError("prediction source run name is invalid")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.partial-", dir=output_dir.parent)
    )
    rows: list[dict[str, Any]] = []
    status_counts: Counter[str] = Counter()
    target_counts: Counter[str] = Counter()
    severity_counts: Counter[str] = Counter()
    try:
        for paper_id, paper in sorted(gold_papers.items()):
            gold_issues = [issue for issue in paper.issues if _validated_gold_issue(issue)]
            prediction = predictions.get(paper_id)
            predicted_issues = list(prediction.get("issues", [])) if prediction else []
            for issue in gold_issues:
                target_counts[str(issue["target_type"])] += 1
                severity_counts[str(issue["classification"])] += 1
            if not gold_issues:
                status = "empty_gold"
            elif prediction is None:
                status = "missing_prediction"
            elif not predicted_issues:
                status = "empty_prediction"
            else:
                status = "ready"
            status_counts[status] += 1
            prediction_path = (
                _prediction_path(postprocessed_dir, paper_id)
                if prediction is not None
                else None
            )
            row: dict[str, Any] = {
                "paper_id": paper_id,
                "status": status,
                "source_sha256": paper.source_sha256,
                "gold_issues_path": portable_path(paper.issues_path),
                "gold_issues_sha256": sha256_path(paper.issues_path),
                "prediction_path": portable_path(prediction_path) if prediction_path else None,
                "prediction_sha256": sha256_path(prediction_path) if prediction_path else None,
                "gold_issue_count": len(gold_issues),
                "prediction_issue_count": len(predicted_issues),
            }
            if status == "ready":
                document = build_input_document(
                    prediction_set_name,
                    paper_id,
                    paper.source_sha256,
                    gold_issues,
                    predicted_issues,
                )
                input_path = temporary / "papers" / f"{paper_id}.json"
                write_json_exclusive(input_path, document)
                row["input_path"] = portable_path(
                    output_dir / "papers" / f"{paper_id}.json"
                )
                row["input_sha256"] = sha256_path(input_path)
            else:
                row["input_path"] = None
                row["input_sha256"] = None
            rows.append(row)

        manifest = {
            "schema_version": INPUT_MANIFEST_SCHEMA,
            "question_id": QUESTION_ID,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "prediction_set_name": prediction_set_name,
            "gold_directory": portable_path(gold_dir),
            "postprocessed_directory": portable_path(postprocessed_dir),
            "gold_manifest_sha256": gold_authorities["benchmark_manifest"]["sha256"],
            "prediction_provenance": prediction_provenance,
            "selected_paper_count": len(gold_papers),
            "scoreable_prediction_paper_count": len(predictions),
            "substantive_gold_issue_count": sum(target_counts.values()),
            "gold_target_type_counts": dict(sorted(target_counts.items())),
            "gold_severity_counts": dict(sorted(severity_counts.items())),
            "status_counts": dict(sorted(status_counts.items())),
            "identity_view_fields": ["issue_id", "source_quotes", "evidence_units", "locators"],
            "evidence_text_fields": list(EVIDENCE_FIELDS),
            "candidate_ranking": {
                "method": "paper-local TF-IDF cosine over failure_explanation",
                "all_predictions_ranked_for_every_gold": True,
                "similarity_scores_shown_to_judge": False,
                "acceptance_threshold": None,
                "candidate_cutoff": None,
                "ranking_is_match_evidence": False,
            },
            "matching_contract": {
                "each_gold_credited_at_most_once": True,
                "maximum_supporting_predictions_per_gold": MAX_SUPPORTING_ENTRIES,
                "combination_requires_distinct_self_contained_diagnoses": True,
                "prediction_reuse_requires_explicit_issue_support": True,
                "accepted_relations": [EXACT_RELATION, BROADER_RELATION],
                "target_type_hidden_during_identity": True,
                "severity_hidden_during_identity": True,
                "unmatched_predictions_unpenalized": True,
            },
            "implementation": {
                "builder_path": portable_path(Path(__file__)),
                "builder_sha256": sha256_path(Path(__file__)),
                "mechanical_loader_path": portable_path(Path(mechanical.__file__)),
                "mechanical_loader_sha256": sha256_path(Path(mechanical.__file__)),
            },
            "papers": rows,
        }
        write_json_exclusive(temporary / "manifest.json", manifest)
        os.replace(temporary, output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return manifest


def load_input_manifest(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    path = path.expanduser().resolve()
    manifest = load_json(path)
    if not isinstance(manifest, dict) or manifest.get("schema_version") != INPUT_MANIFEST_SCHEMA:
        raise SemanticScoreError("semantic input manifest schema mismatch")
    if manifest.get("question_id") != QUESTION_ID:
        raise SemanticScoreError("semantic input manifest question mismatch")
    rows = manifest.get("papers")
    if not isinstance(rows, list) or len(rows) != manifest.get("selected_paper_count"):
        raise SemanticScoreError("semantic input manifest paper inventory mismatch")
    seen: set[str] = set()
    validated_rows: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise SemanticScoreError("semantic input manifest row is invalid")
        paper_id = row.get("paper_id")
        if (
            not isinstance(paper_id, str)
            or not PAPER_ID_RE.fullmatch(paper_id)
            or paper_id in seen
        ):
            raise SemanticScoreError("semantic input manifest paper ID is invalid")
        seen.add(paper_id)
        if row.get("status") == "ready":
            input_path = resolve_path(row.get("input_path"), "judge input path")
            if not input_path.is_file() or sha256_path(input_path) != row.get("input_sha256"):
                raise SemanticScoreError(f"judge input hash mismatch for {paper_id}")
            document = validate_input(load_json(input_path))
            if document.get("paper_id") != paper_id:
                raise SemanticScoreError(f"judge input identity mismatch for {paper_id}")
            row = dict(row)
            row["input_path_resolved"] = input_path
            row["input_document"] = document
        validated_rows.append(row)
    return manifest, sorted(validated_rows, key=lambda row: row["paper_id"])


def specialized_judgment_schema(
    base_schema: dict[str, Any],
    *,
    run_name: str,
    prediction_set_name: str,
    input_document: dict[str, Any],
    input_sha256: str,
) -> dict[str, Any]:
    schema = json.loads(json.dumps(base_schema))
    properties = schema["properties"]
    fixed = {
        "judge_run_name": run_name,
        "prediction_set_name": prediction_set_name,
        "paper_id": input_document["paper_id"],
        "judge_input_sha256": input_sha256,
    }
    for field, value in fixed.items():
        properties[field]["enum"] = [value]
    gold_ids = [issue["issue_id"] for issue in input_document["gold_issues"]]
    prediction_ids = [
        issue["issue_id"] for issue in input_document["predicted_issues"]
    ]
    decisions = properties["gold_decisions"]
    decisions["minItems"] = len(gold_ids)
    decisions["maxItems"] = len(gold_ids)
    decision_properties = decisions["items"]["properties"]
    decision_properties["gold_issue_id"]["enum"] = gold_ids
    decision_properties["focus_prediction_issue_id"]["enum"] = prediction_ids
    decision_properties["supporting_prediction_issue_ids"]["items"]["enum"] = prediction_ids
    component = decision_properties["component_coverage"]["items"]["properties"]
    component["prediction_issue_id"]["enum"] = prediction_ids
    return schema


def _cited_units(value: object, issue: dict[str, Any], label: str) -> set[str]:
    valid = {unit["unit_id"] for unit in issue["evidence_units"]}
    if (not isinstance(value, list) or not value
            or not all(isinstance(item, str) for item in value)
            or len(value) != len(set(value)) or not set(value).issubset(valid)):
        raise SemanticScoreError(f"invalid {label}")
    return set(value)


def validate_judgment(
    document: object,
    *,
    input_document: dict[str, Any],
    input_sha256: str,
    judge_run_name: str,
    prediction_set_name: str,
) -> dict[str, Any]:
    expected_top = {"schema_version", "question_id", "status", "judge_run_name",
                    "prediction_set_name", "paper_id", "judge_input_sha256",
                    "gold_decisions", "audit"}
    if not isinstance(document, dict) or set(document) != expected_top:
        raise SemanticScoreError("judgment has unexpected top-level fields")
    expected = {"schema_version": JUDGMENT_SCHEMA, "question_id": QUESTION_ID,
                "status": "ok", "judge_run_name": judge_run_name,
                "prediction_set_name": prediction_set_name,
                "paper_id": input_document["paper_id"], "judge_input_sha256": input_sha256}
    for field, value in expected.items():
        if document.get(field) != value:
            raise SemanticScoreError(f"judgment {field} mismatch")
    audit = document["audit"]
    if (not isinstance(audit, dict) or set(audit) != set(AUDIT_FIELDS)
            or not all(audit[field] is True for field in AUDIT_FIELDS)):
        raise SemanticScoreError("judgment audit is invalid")
    gold = {i["issue_id"]: i for i in input_document["gold_issues"]}
    predictions = {i["issue_id"]: i for i in input_document["predicted_issues"]}
    decisions = document["gold_decisions"]
    if (not isinstance(decisions, list) or not all(isinstance(d, dict) for d in decisions)
            or [d.get("gold_issue_id") for d in decisions] != list(gold)):
        raise SemanticScoreError("judgment must decide every gold issue in input order")
    fields = {"gold_issue_id", "coverage_status", "supporting_prediction_issue_ids",
              "focus_prediction_issue_id", "focus_relation", "gold_causal_unit_ids",
              "prediction_causal_unit_ids", "same_causal_subject", "same_failed_obligation",
              "same_failure_operation", "identity_explanation", "shared_mathematical_weakness",
              "uncovered_gold_detail", "combination_explanation", "component_coverage"}
    rejected_relations = {"shared_topic_or_location_only", "different_causal_subject",
                          "different_obligation_or_hypothesis", "different_operation_step_or_formula",
                          "upstream_downstream_only", "no_substantive_overlap"}
    component_fields = {"gold_component", "gold_causal_unit_ids", "prediction_issue_id",
                        "prediction_causal_unit_ids", "recognition_explanation", "self_contained_diagnosis"}
    for decision in decisions:
        if set(decision) != fields:
            raise SemanticScoreError("judgment decision has unexpected fields")
        gid = decision["gold_issue_id"]
        focus = decision["focus_prediction_issue_id"]
        if not isinstance(focus, str) or focus not in predictions:
            raise SemanticScoreError(f"invalid focus prediction for {gid}")
        support = decision["supporting_prediction_issue_ids"]
        if (not isinstance(support, list) or not all(isinstance(pid, str) for pid in support)
                or len(support) != len(set(support)) or len(support) > MAX_SUPPORTING_ENTRIES
                or not set(support).issubset(predictions)):
            raise SemanticScoreError(f"invalid supporting predictions (maximum three) for {gid}")
        components = decision["component_coverage"]
        if not isinstance(components, list):
            raise SemanticScoreError(f"invalid component coverage for {gid}")
        cited_gold = _cited_units(decision["gold_causal_unit_ids"], gold[gid], f"gold citations for {gid}")
        _cited_units(decision["prediction_causal_unit_ids"], predictions[focus], f"focus citations for {gid}")
        gates = tuple(decision[k] for k in ("same_causal_subject", "same_failed_obligation", "same_failure_operation"))
        if not all(isinstance(v, bool) for v in gates):
            raise SemanticScoreError(f"invalid identity gates for {gid}")
        for field in ("identity_explanation", "shared_mathematical_weakness", "uncovered_gold_detail", "combination_explanation"):
            if not isinstance(decision[field], str):
                raise SemanticScoreError(f"invalid {field} for {gid}")
        if not decision["identity_explanation"].strip():
            raise SemanticScoreError(f"empty explanation for {gid}")
        relation = decision["focus_relation"]
        shared, uncovered = decision["shared_mathematical_weakness"].strip(), decision["uncovered_gold_detail"].strip()
        if decision["coverage_status"] == "not_matched":
            if (support or components or decision["combination_explanation"].strip()
                    or relation not in rejected_relations or gates == (True, True, True)
                    or shared or uncovered):
                raise SemanticScoreError(f"inconsistent non-match for {gid}")
            continue
        if decision["coverage_status"] != "matched" or not support or focus not in support or not components:
            raise SemanticScoreError(f"inconsistent match for {gid}")
        if not shared or relation not in {EXACT_RELATION, BROADER_RELATION}:
            raise SemanticScoreError(f"unsupported accepted relation for {gid}")
        if relation == EXACT_RELATION and (gates != (True, True, True) or uncovered):
            raise SemanticScoreError(f"inconsistent exact match for {gid}")
        if relation == BROADER_RELATION and (not gates[0] or gates == (True, True, True) or not uncovered):
            raise SemanticScoreError(f"inconsistent broader match for {gid}")
        if bool(decision["combination_explanation"].strip()) != (len(support) > 1):
            raise SemanticScoreError(f"combination needs a distinct-defect explanation for {gid}")
        used, covered_units, names = set(), set(), set()
        for component in components:
            if not isinstance(component, dict) or set(component) != component_fields:
                raise SemanticScoreError(f"invalid component fields for {gid}")
            for field in ("gold_component", "recognition_explanation"):
                if not isinstance(component[field], str) or not component[field].strip():
                    raise SemanticScoreError(f"empty component {field} for {gid}")
            name = component["gold_component"].strip().casefold()
            if name in names:
                raise SemanticScoreError(f"duplicate gold component for {gid}")
            names.add(name)
            pid = component["prediction_issue_id"]
            if not isinstance(pid, str) or pid not in support or component["self_contained_diagnosis"] is not True:
                raise SemanticScoreError(f"component lacks its own selected diagnosis for {gid}")
            units = _cited_units(component["gold_causal_unit_ids"], gold[gid], f"component gold citations for {gid}")
            _cited_units(component["prediction_causal_unit_ids"], predictions[pid], f"component prediction citations for {gid}")
            used.add(pid)
            covered_units.update(units)
        if used != set(support) or covered_units != cited_gold:
            raise SemanticScoreError(f"component map does not account for selected support and gold citations for {gid}")
    return document


def combined_severity(prediction_ids: list[str], predictions: dict[str, dict[str, Any]]) -> str:
    """Highest severity among necessary entries; incomplete severity data stays NA."""
    labels = [predictions[pid].get("classification") for pid in prediction_ids]
    if not labels or any(label not in SEVERITY_RANK for label in labels):
        return "NA"
    return str(max(labels, key=lambda label: SEVERITY_RANK[label]))


def _symmetric_severity_credit(gold: object, predicted: object) -> float:
    if gold not in SEVERITY_RANK or predicted not in SEVERITY_RANK:
        return 0.0
    return max(0.0, 1.0 - 0.5 * abs(SEVERITY_RANK[str(gold)] - SEVERITY_RANK[str(predicted)]))


def _matrix(counter: Counter[tuple[str, str]]) -> dict[str, dict[str, int]]:
    rows: dict[str, dict[str, int]] = {}
    for gold, predicted in sorted(counter):
        rows.setdefault(gold, {})[predicted] = counter[(gold, predicted)]
    return rows


def score_judgments(
    input_manifest_path: Path,
    judgments_dir: Path,
    run_manifest_path: Path,
    *,
    allow_partial: bool,
) -> dict[str, Any]:
    input_manifest, rows = load_input_manifest(input_manifest_path)
    run_manifest = load_json(run_manifest_path.expanduser().resolve())
    if not isinstance(run_manifest, dict):
        raise SemanticScoreError("judge run manifest is invalid")
    judge_run_name = run_manifest.get("judge_run_name")
    if not isinstance(judge_run_name, str):
        raise SemanticScoreError("judge run name is invalid")
    if run_manifest.get("input_manifest", {}).get("sha256") != sha256_path(
        input_manifest_path.expanduser().resolve()
    ):
        raise SemanticScoreError("judge run/input manifest hash mismatch")
    prediction_set_name = str(input_manifest["prediction_set_name"])
    judgments_dir = judgments_dir.expanduser().resolve()

    target_denominators: Counter[str] = Counter()
    target_matches: Counter[str] = Counter()
    severity_denominators: Counter[str] = Counter()
    severity_matches: Counter[str] = Counter()
    relation_counts: Counter[str] = Counter()
    weighted_matches = 0.0
    weighted_exact_matches = 0.0
    target_confusion: Counter[tuple[str, str]] = Counter()
    severity_confusion: Counter[tuple[str, str]] = Counter()
    mechanism_confusion: Counter[tuple[str, str]] = Counter()
    matched_count = 0
    target_exact = 0
    severity_exact = 0
    severity_points = 0.0
    mechanism_exact = 0
    mechanism_denominator = 0
    total_predictions = 0
    prediction_match_counts: Counter[tuple[str, str]] = Counter()
    judgment_schema_versions: set[str] = set()
    required = 0
    valid = 0
    missing: list[str] = []
    per_paper: list[dict[str, Any]] = []

    for row in rows:
        paper_id = str(row["paper_id"])
        gold_path = resolve_path(row["gold_issues_path"], "gold issues path")
        if sha256_path(gold_path) != row.get("gold_issues_sha256"):
            raise SemanticScoreError(f"gold issues changed for {paper_id}")
        gold_document = load_json(gold_path)
        gold_issues = [
            issue
            for issue in gold_document["issues"]
            if issue.get("classification") in SUBSTANTIVE_SEVERITIES
        ]
        gold_by_id = {str(issue["issue_id"]): issue for issue in gold_issues}
        prediction_by_id: dict[str, dict[str, Any]] = {}
        if row.get("prediction_path") is not None:
            prediction_path = resolve_path(row["prediction_path"], "prediction path")
            if sha256_path(prediction_path) != row.get("prediction_sha256"):
                raise SemanticScoreError(f"prediction changed for {paper_id}")
            prediction_document = load_json(prediction_path)
            prediction_by_id = {
                str(issue["issue_id"]): issue for issue in prediction_document["issues"]
            }
        total_predictions += len(prediction_by_id)
        for issue in gold_issues:
            target = str(issue["target_type"])
            target_denominators[target] += 1
            severity_denominators[str(issue["classification"])] += 1

        matched: dict[str, list[str]] = {}
        focus_predictions: dict[str, str] = {}
        matched_relations: dict[str, str] = {}
        status = str(row["status"])
        if status == "ready":
            expected_input = build_input_document(
                prediction_set_name, paper_id, str(row["source_sha256"]),
                gold_issues, list(prediction_by_id.values()),
            )
            if expected_input != row["input_document"]:
                raise SemanticScoreError(f"judge evidence does not match frozen records for {paper_id}")
            required += 1
            judgment_path = judgments_dir / f"{paper_id}.json"
            if not judgment_path.is_file():
                missing.append(paper_id)
            else:
                judgment = validate_judgment(
                    load_json(judgment_path),
                    input_document=row["input_document"],
                    input_sha256=str(row["input_sha256"]),
                    judge_run_name=judge_run_name,
                    prediction_set_name=prediction_set_name,
                )
                judgment_schema_versions.add(str(judgment["schema_version"]))
                valid += 1
                matched = {
                    decision["gold_issue_id"]: decision["supporting_prediction_issue_ids"]
                    for decision in judgment["gold_decisions"]
                    if decision["coverage_status"] == "matched"
                }
                matched_relations = {
                    decision["gold_issue_id"]: decision["focus_relation"]
                    for decision in judgment["gold_decisions"]
                    if decision["coverage_status"] == "matched"
                }
                focus_predictions = {
                    decision["gold_issue_id"]: decision["focus_prediction_issue_id"]
                    for decision in judgment["gold_decisions"]
                    if decision["coverage_status"] == "matched"
                }
        elif status not in {"empty_gold", "empty_prediction", "missing_prediction"}:
            raise SemanticScoreError(f"unsupported input status for {paper_id}: {status}")

        for gold_id, prediction_ids in matched.items():
            prediction_id = focus_predictions[gold_id]
            gold_issue = gold_by_id[gold_id]
            predicted_issue = prediction_by_id[prediction_id]
            gold_target = str(gold_issue["target_type"])
            predicted_target = str(predicted_issue.get("target_type") or "NA")
            gold_severity = str(gold_issue["classification"])
            predicted_severity = combined_severity(prediction_ids, prediction_by_id)
            matched_count += 1
            severity_matches[gold_severity] += 1
            relation_counts[matched_relations[gold_id]] += 1
            weighted_matches += GOLD_WEIGHTS[gold_severity]
            if matched_relations[gold_id] == EXACT_RELATION:
                weighted_exact_matches += GOLD_WEIGHTS[gold_severity]
            for supporting_id in prediction_ids:
                prediction_match_counts[(paper_id, supporting_id)] += 1
            target_matches[gold_target] += 1
            target_confusion[(gold_target, predicted_target)] += 1
            severity_confusion[(gold_severity, predicted_severity)] += 1
            target_exact += int(gold_target == predicted_target)
            severity_exact += int(gold_severity == predicted_severity)
            severity_points += _symmetric_severity_credit(
                gold_severity, predicted_severity
            )
            if gold_target == "formal_result":
                gold_mechanism = str(gold_issue.get("failure_mechanism") or "NA")
                predicted_mechanism = str(
                    predicted_issue.get("failure_mechanism") or "NA"
                )
                mechanism_denominator += 1
                mechanism_exact += int(gold_mechanism == predicted_mechanism)
                mechanism_confusion[(gold_mechanism, predicted_mechanism)] += 1
        per_paper.append(
            {
                "paper_id": paper_id,
                "input_status": status,
                "gold_substantive_issue_count": len(gold_issues),
                "prediction_issue_count": len(prediction_by_id),
                "matched_issue_count": len(matched),
                "matched_pairs": [
                    {
                        "gold_issue_id": gold_id,
                        "supporting_prediction_issue_ids": prediction_ids,
                        "focus_prediction_issue_id": focus_predictions[gold_id],
                        "match_relation": matched_relations[gold_id],
                        "discovery_weight": GOLD_WEIGHTS[gold_by_id[gold_id]["classification"]],
                        "gold_target_type": gold_by_id[gold_id]["target_type"],
                        "predicted_target_type": prediction_by_id[focus_predictions[gold_id]].get("target_type"),
                        "gold_severity": gold_by_id[gold_id]["classification"],
                        "predicted_severity": combined_severity(prediction_ids, prediction_by_id),
                        "supporting_severities": {
                            pid: prediction_by_id[pid].get("classification") or "NA" for pid in prediction_ids
                        },
                    }
                    for gold_id, prediction_ids in matched.items()
                ],
            }
        )

    complete = valid == required
    if not complete and not allow_partial:
        raise SemanticScoreError(
            f"judgments incomplete: valid={valid} required={required}; use --allow-partial only for an explicitly partial score"
        )
    total_gold = sum(target_denominators.values())
    if len(judgment_schema_versions) > 1:
        raise SemanticScoreError("judge run mixes judgment schema versions")
    judgment_schema_version = next(iter(judgment_schema_versions), None)
    by_type = {
        target: {
            "matched": target_matches[target],
            "gold_denominator": target_denominators[target],
            "percent": percent(target_matches[target], target_denominators[target]),
        }
        for target in TARGET_TYPES
    }
    weighted_denominator = sum(GOLD_WEIGHTS[k] * n for k, n in severity_denominators.items())
    report = {
        "schema_version": SCORE_SCHEMA,
        "question_id": QUESTION_ID,
        "scoring_contract": {
            "primary_metric": "gold-severity-weighted semantic issue coverage; exact and broader matches",
            "discovery_weights_by_gold_severity": GOLD_WEIGHTS,
            "accepted_relations": [EXACT_RELATION, BROADER_RELATION],
            "severity_credit_multiplied_into_discovery": False,
            "recall_bucket": "gold target_type",
            "target_type_mismatch_does_not_cancel_recovery": True,
            "conditional_target_type": "exact on covered gold issues",
            "conditional_target_type_aggregation": "one corpus-level aggregate across all target types",
            "conditional_severity": "symmetric ordinal distance on covered gold issues",
            "supporting_severity_aggregation": "highest severity among necessary supporting entries; NA if any required entry is ungraded",
            "ancillary_label_source": "focus prediction selected from mathematical evidence; labels remain hidden during matching",
            "policy_revision": run_manifest.get("judge_policy_revision", POLICY_REVISION),
            "conditional_severity_aggregation": "one corpus-level aggregate across all target types",
            "severity_credit": {"exact": 1.0, "one_level_off": 0.5, "two_levels_off": 0.0},
            "unmatched_predictions": "diagnostic and unpenalized",
            "nonexhaustive_gold": True,
        },
        "authorities": {
            "input_manifest": {
                "path": portable_path(input_manifest_path),
                "sha256": sha256_path(input_manifest_path.expanduser().resolve()),
            },
            "judge_run_manifest": {
                "path": portable_path(run_manifest_path),
                "sha256": sha256_path(run_manifest_path.expanduser().resolve()),
            },
            "judgments_directory": portable_path(judgments_dir),
        },
        "judge": {
            "run_name": judge_run_name,
            "model": run_manifest.get("model"),
            "reasoning_effort": run_manifest.get("reasoning_effort"),
            "backend": run_manifest.get("backend"),
            "policy_version": run_manifest.get("judge_policy_version"),
        },
        "prediction_set_name": prediction_set_name,
        "summary": {
            "scoring_status": "complete" if complete else "partial",
            "selected_paper_count": len(rows),
            "scoreable_prediction_paper_count": input_manifest.get(
                "scoreable_prediction_paper_count"
            ),
            "required_judgment_count": required,
            "valid_judgment_count": valid,
            "missing_judgment_paper_ids": missing,
            "gold_substantive_issue_count": total_gold,
            "matched_issue_count": matched_count,
            "issue_coverage": {
                "matched": matched_count,
                "gold_denominator": total_gold,
                "percent": percent(matched_count, total_gold),
            },
            "issue_coverage_by_gold_target_type": by_type,
            "match_relation_counts": {key: relation_counts[key] for key in (EXACT_RELATION, BROADER_RELATION)},
            "exact_root_recall_diagnostic": {
                "matched": relation_counts[EXACT_RELATION], "gold_denominator": total_gold,
                "percent": percent(relation_counts[EXACT_RELATION], total_gold),
            },
            "weighted_issue_coverage": {
                "earned": weighted_matches, "possible": weighted_denominator,
                "percent": percent(weighted_matches, weighted_denominator),
                "exact_earned": weighted_exact_matches,
                "broader_earned": weighted_matches-weighted_exact_matches,
            },
            "issue_coverage_by_gold_severity": {
                key: {"matched": severity_matches[key], "gold_denominator": severity_denominators[key],
                      "percent": percent(severity_matches[key], severity_denominators[key]),
                      "discovery_weight": GOLD_WEIGHTS[key]}
                for key in SUBSTANTIVE_SEVERITIES
            },
            "conditional_target_type": {
                "exact_count": target_exact,
                "matched_gold_issue_denominator": matched_count,
                "exact_percent": percent(target_exact, matched_count),
                "confusion_gold_rows_prediction_columns": _matrix(target_confusion),
            },
            "conditional_severity": {
                "exact_count": severity_exact,
                "matched_gold_issue_denominator": matched_count,
                "exact_percent": percent(severity_exact, matched_count),
                "symmetric_ordinal_points": severity_points,
                "possible_points": matched_count,
                "symmetric_ordinal_percent": percent(severity_points, matched_count),
                "confusion_gold_rows_prediction_columns": _matrix(severity_confusion),
            },
            "conditional_formal_failure_mechanism_diagnostic": {
                "exact_count": mechanism_exact,
                "matched_gold_formal_denominator": mechanism_denominator,
                "exact_percent": percent(mechanism_exact, mechanism_denominator),
                "confusion_gold_rows_prediction_columns": _matrix(mechanism_confusion),
            },
            "total_prediction_issue_count": total_predictions,
            "matched_prediction_issue_count": len(prediction_match_counts),
            "unmatched_prediction_issue_count_diagnostic": total_predictions
            - len(prediction_match_counts),
        },
        "papers": per_paper,
    }
    if judgment_schema_version == JUDGMENT_SCHEMA:
        reused = {
            key: count for key, count in prediction_match_counts.items() if count > 1
        }
        report["scoring_contract"].update(
            {
                "each_gold_credited_at_most_once": True,
                "maximum_supporting_predictions_per_gold": MAX_SUPPORTING_ENTRIES,
                "combination_requires_distinct_self_contained_diagnoses": True,
                "prediction_may_cover_multiple_gold_issues": True,
                "prediction_reuse_requires_explicit_issue_support": True,
            }
        )
        report["summary"].update(
            {
                "reused_prediction_issue_count": len(reused),
                "gold_issues_covered_by_reused_predictions": sum(reused.values()),
            }
        )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--gold-dir", type=Path, required=True)
    build_parser.add_argument("--postprocessed-dir", type=Path, required=True)
    build_parser.add_argument("--output-dir", type=Path, required=True)
    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--input-manifest", type=Path, required=True)
    score_parser.add_argument("--run-manifest", type=Path, required=True)
    score_parser.add_argument("--judgments-dir", type=Path, required=True)
    score_parser.add_argument("--output", type=Path, required=True)
    score_parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "build":
            manifest = build_inputs(args.gold_dir, args.postprocessed_dir, args.output_dir)
            print(f"Input manifest: {args.output_dir.resolve() / 'manifest.json'}")
            print(f"Papers: {manifest['selected_paper_count']}")
            print(f"Substantive gold issues: {manifest['substantive_gold_issue_count']}")
            print(f"Statuses: {manifest['status_counts']}")
        else:
            output = args.output.expanduser().resolve()
            if output.exists():
                raise SemanticScoreError(f"refusing to overwrite existing score: {output}")
            report = score_judgments(
                args.input_manifest,
                args.judgments_dir,
                args.run_manifest,
                allow_partial=args.allow_partial,
            )
            write_json_exclusive(output, report)
            summary = report["summary"]
            print(f"Score: {output}")
            print(
                f"Issue coverage: {summary['matched_issue_count']}/"
                f"{summary['gold_substantive_issue_count']} "
                f"({summary['issue_coverage']['percent']}%)"
            )
        return 0
    except (SemanticScoreError, mechanical.ScoreError) as exc:
        fail(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
