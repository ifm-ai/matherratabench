#!/usr/bin/env python3
"""Deterministic recall-oriented scorer for parsed Q1 issue inventories."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from typing import Any


REPORT_SCHEMA = "benchmark-q1-formal-tag-recall-score-v4"
POSTPROCESS_SCHEMA = "benchmark-q1-postprocessed-run-v1"
POSTPROCESS_AGGREGATE_SCHEMA = "benchmark-q1-postprocessed-aggregate-v1"
PARSE_AUDIT_SCHEMA = "benchmark-q1-output-parse-audit-v1"
SEVERITIES = ("minor_issue", "moderate_issue", "serious_issue")
SEVERITY_RANK = {value: index for index, value in enumerate(SEVERITIES)}
MISSING_VALUES = {"", "NA"}


class ScoreError(ValueError):
    """The gold package, parsed predictions, or requested output is invalid."""


@dataclass(frozen=True)
class GoldPaper:
    paper_id: str
    source: str
    source_sha256: str
    issues_path: Path
    issues: tuple[dict[str, Any], ...]


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
        raise ScoreError(f"cannot read JSON {path}: {exc}") from exc


def write_json_exclusive(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
    except FileExistsError as exc:
        raise ScoreError(f"refusing to overwrite existing output: {path}") from exc


def percent(numerator: int | float, denominator: int | float) -> float | None:
    if denominator == 0:
        return None
    return round(100.0 * numerator / denominator, 6)


def normalize_tag(value: object) -> str | None:
    """Apply only surrounding-whitespace normalization to source tags."""
    if not isinstance(value, str):
        return None
    value = value.strip()
    return None if value in MISSING_VALUES else value


def load_checksum_index(path: Path) -> dict[str, str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ScoreError(f"cannot read checksum index {path}: {exc}") from exc
    records: dict[str, str] = {}
    for line_number, line in enumerate(lines, start=1):
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if match is None:
            raise ScoreError(f"malformed checksum line {line_number}: {path}")
        digest, relative = match.groups()
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ScoreError(f"unsafe checksum path on line {line_number}: {path}")
        if relative in records:
            raise ScoreError(f"duplicate checksum path {relative}: {path}")
        records[relative] = digest
    return records


def _manifest_hash(record: dict[str, Any], kind: str) -> str | None:
    direct = record.get(f"{kind}_annotated_sha256")
    if isinstance(direct, str):
        return direct
    nested = record.get(kind)
    if isinstance(nested, dict) and isinstance(nested.get("sha256"), str):
        return str(nested["sha256"])
    fallback = record.get(f"{kind}_sha256")
    return fallback if isinstance(fallback, str) else None


def load_gold(gold_dir: Path) -> tuple[dict[str, GoldPaper], dict[str, Any]]:
    gold_dir = gold_dir.expanduser().resolve()
    manifest_path = gold_dir / "manifest.json"
    checksums_path = gold_dir / "CHECKSUMS.sha256"
    manifest = load_json(manifest_path)
    if not isinstance(manifest, dict):
        raise ScoreError("gold manifest must be a JSON object")
    records = manifest.get("papers")
    paper_count = manifest.get("paper_count")
    if (
        not isinstance(records, list)
        or not isinstance(paper_count, int)
        or isinstance(paper_count, bool)
        or len(records) != paper_count
    ):
        raise ScoreError("gold manifest paper inventory is inconsistent")

    checksums = load_checksum_index(checksums_path)
    manifest_sha256 = sha256_path(manifest_path)
    if checksums.get("manifest.json") != manifest_sha256:
        raise ScoreError("gold manifest checksum is missing or mismatched")

    papers: dict[str, GoldPaper] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ScoreError("gold manifest paper record is invalid")
        paper_id = record.get("paper_id")
        if not isinstance(paper_id, str) or not paper_id or paper_id in papers:
            raise ScoreError("gold manifest paper ID is invalid or duplicated")
        source_path = gold_dir / "papers" / paper_id / "source.tex"
        issues_path = gold_dir / "papers" / paper_id / "issues.json"
        source_sha256 = _manifest_hash(record, "source")
        issues_sha256 = _manifest_hash(record, "issues")
        source_relative = f"papers/{paper_id}/source.tex"
        issues_relative = f"papers/{paper_id}/issues.json"
        if (
            not isinstance(source_sha256, str)
            or not source_path.is_file()
            or checksums.get(source_relative) != source_sha256
            or sha256_path(source_path) != source_sha256
        ):
            raise ScoreError(f"gold source hash mismatch for {paper_id}")
        if (
            not isinstance(issues_sha256, str)
            or not issues_path.is_file()
            or checksums.get(issues_relative) != issues_sha256
            or sha256_path(issues_path) != issues_sha256
        ):
            raise ScoreError(f"gold issues hash mismatch for {paper_id}")
        document = load_json(issues_path)
        if not isinstance(document, dict) or document.get("paper_id") != paper_id:
            raise ScoreError(f"gold paper identity mismatch for {paper_id}")
        issues = document.get("issues")
        if not isinstance(issues, list) or not all(
            isinstance(issue, dict) for issue in issues
        ):
            raise ScoreError(f"gold issue array is invalid for {paper_id}")
        issue_ids = [issue.get("issue_id") for issue in issues]
        if any(not isinstance(issue_id, str) for issue_id in issue_ids) or len(
            issue_ids
        ) != len(set(issue_ids)):
            raise ScoreError(f"gold issue IDs are invalid or duplicated for {paper_id}")
        papers[paper_id] = GoldPaper(
            paper_id=paper_id,
            source=source_path.read_text(encoding="utf-8"),
            source_sha256=source_sha256,
            issues_path=issues_path,
            issues=tuple(issues),
        )

    return papers, {
        "gold_directory": str(gold_dir),
        "benchmark_release": manifest.get("release") or manifest.get("release_id"),
        "benchmark_manifest": {
            "path": str(manifest_path),
            "sha256": manifest_sha256,
        },
        "benchmark_checksums": {
            "path": str(checksums_path),
            "sha256": sha256_path(checksums_path),
        },
        "paper_count": len(papers),
    }


def validate_prediction(
    document: object, *, paper: GoldPaper, path: Path
) -> dict[str, Any]:
    if not isinstance(document, dict):
        raise ScoreError(f"prediction must be a JSON object: {path}")
    if document.get("paper_id") != paper.paper_id:
        raise ScoreError(f"prediction paper ID mismatch: {path}")
    if document.get("source_sha256") != paper.source_sha256:
        raise ScoreError(f"prediction source hash mismatch: {path}")
    issues = document.get("issues")
    issues_found = document.get("issues_found")
    if not isinstance(issues, list) or not all(isinstance(issue, dict) for issue in issues):
        raise ScoreError(f"prediction issue array is invalid: {path}")
    if (
        not isinstance(issues_found, int)
        or isinstance(issues_found, bool)
        or issues_found != len(issues)
    ):
        raise ScoreError(f"prediction issues_found is inconsistent: {path}")
    if document.get("status") != "ok" or document.get("error") is not None:
        raise ScoreError(f"parsed scoreable prediction must have status=ok: {path}")
    issue_ids = [issue.get("issue_id") for issue in issues]
    if any(not isinstance(issue_id, str) for issue_id in issue_ids) or len(
        issue_ids
    ) != len(set(issue_ids)):
        raise ScoreError(f"prediction issue IDs are invalid or duplicated: {path}")
    for issue in issues:
        classification = issue.get("classification")
        if classification not in {*SEVERITIES, "NA"}:
            raise ScoreError(f"prediction severity is invalid: {path}")
        dependencies = issue.get("dependent_affected_formal_result_tags")
        if not isinstance(dependencies, list) or not all(
            isinstance(tag, str) for tag in dependencies
        ):
            raise ScoreError(f"prediction dependent tags are invalid: {path}")
        location = issue.get("failure_location")
        if not isinstance(location, dict):
            raise ScoreError(f"prediction failure location is invalid: {path}")
    return document


def _require_dict(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ScoreError(f"{label} must be an object")
    return value


def _require_count(value: object, label: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ScoreError(f"{label} must be a nonnegative integer")
    return value


def _load_run_predictions(
    postprocessed_dir: Path, gold_papers: dict[str, GoldPaper], *, allow_empty: bool = False
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    summary_path = postprocessed_dir / "parse_summary.json"
    aggregate_path = postprocessed_dir / "predictions.parsed.json"
    predictions_dir = postprocessed_dir / "parsed_predictions"
    audit_dir = postprocessed_dir / "parse_audit"
    summary = load_json(summary_path)
    aggregate = load_json(aggregate_path)
    if not isinstance(summary, dict) or summary.get("schema_version") != POSTPROCESS_SCHEMA:
        raise ScoreError("post-processing summary schema mismatch")
    if not isinstance(aggregate, dict):
        raise ScoreError("parsed prediction aggregate must be an object")
    selected_paper_count = summary.get("selected_paper_count")
    if (
        not isinstance(selected_paper_count, int)
        or isinstance(selected_paper_count, bool)
        or selected_paper_count < 0
        or selected_paper_count > len(gold_papers)
    ):
        raise ScoreError("post-processing selected paper count is invalid")
    if summary.get("provider_called") is not False or summary.get("gold_read") is not False:
        raise ScoreError("post-processing provenance flags are invalid")

    aggregate_records = aggregate.get("records")
    if not isinstance(aggregate_records, list):
        raise ScoreError("parsed prediction aggregate records are invalid")
    aggregate_by_id: dict[str, dict[str, Any]] = {}
    for record in aggregate_records:
        paper_id = record.get("paper_id") if isinstance(record, dict) else None
        if not isinstance(paper_id, str) or paper_id in aggregate_by_id:
            raise ScoreError("aggregate prediction paper IDs are invalid or duplicated")
        aggregate_by_id[paper_id] = record
    if aggregate.get("record_count") != len(aggregate_by_id):
        raise ScoreError("aggregate prediction record count is inconsistent")

    audit_paths = sorted(audit_dir.glob("*.json"))
    if len(audit_paths) != selected_paper_count:
        raise ScoreError("one parse audit is required for every selected paper")
    predictions: dict[str, dict[str, Any]] = {}
    audit_statuses: Counter[str] = Counter()
    audited_paper_ids: set[str] = set()
    for audit_path in audit_paths:
        paper_id = audit_path.stem
        if paper_id not in gold_papers or paper_id in audited_paper_ids:
            raise ScoreError(f"parse audit paper ID is invalid: {audit_path}")
        audited_paper_ids.add(paper_id)
        paper = gold_papers[paper_id]
        audit = load_json(audit_path)
        if (
            not isinstance(audit, dict)
            or audit.get("schema_version") != PARSE_AUDIT_SCHEMA
            or audit.get("paper_id") != paper_id
        ):
            raise ScoreError(f"parse audit identity mismatch for {paper_id}")
        status = audit.get("final_status")
        if not isinstance(status, str):
            raise ScoreError(f"parse audit status is invalid for {paper_id}")
        audit_statuses[status] += 1
        prediction_path = predictions_dir / f"{paper_id}.json"
        written = audit.get("scoreable_prediction_written") is True
        if not written:
            if prediction_path.exists() or paper_id in aggregate_by_id:
                raise ScoreError(f"unscoreable paper has a parsed prediction: {paper_id}")
            continue
        if (
            not prediction_path.is_file()
            or sha256_path(prediction_path) != audit.get("derived_prediction_sha256")
        ):
            raise ScoreError(f"parsed prediction hash mismatch for {paper_id}")
        prediction = validate_prediction(
            load_json(prediction_path), paper=paper, path=prediction_path
        )
        if aggregate_by_id.get(paper_id) != prediction:
            raise ScoreError(f"aggregate prediction mismatch for {paper_id}")
        predictions[paper_id] = prediction

    prediction_paths = sorted(predictions_dir.glob("*.json"))
    if {path.stem for path in prediction_paths} != set(predictions):
        raise ScoreError("parsed prediction files do not match scoreable audits")
    if set(aggregate_by_id) != set(predictions):
        raise ScoreError("parsed prediction files do not match the aggregate")
    if summary.get("scoreable_prediction_count") != len(predictions):
        raise ScoreError("scoreable prediction count does not match parsed files")
    if dict(sorted(audit_statuses.items())) != dict(
        sorted((summary.get("final_status_counts") or {}).items())
    ):
        raise ScoreError("parse audit status counts do not match the summary")
    if not predictions and not allow_empty:
        raise ScoreError("no scoreable predictions; prediction set excluded")
    run_names = ({prediction.get("run_name") for prediction in predictions.values()}
                 if predictions else {summary.get("source_run_name")})
    if len(run_names) != 1 or not all(isinstance(name, str) and name for name in run_names):
        raise ScoreError("parsed predictions do not share one valid run name")
    run_name = next(iter(run_names))
    if summary.get("source_run_name") != run_name:
        raise ScoreError("parsed prediction run name does not match the summary")

    return predictions, {
        "postprocessed_directory": str(postprocessed_dir),
        "run_name": run_name,
        "parse_summary": {
            "path": str(summary_path),
            "sha256": sha256_path(summary_path),
        },
        "parsed_prediction_aggregate": {
            "path": str(aggregate_path),
            "sha256": sha256_path(aggregate_path),
        },
        "prediction_manifest": {
            "kind": "source_run_manifest",
            "sha256": summary.get("source_run_manifest_sha256"),
        },
        "source_run_manifest_sha256": summary.get("source_run_manifest_sha256"),
        "selected_paper_count": selected_paper_count,
        "scoreable_prediction_count": len(predictions),
        "final_status_counts": dict(sorted(audit_statuses.items())),
    }


def _load_aggregate_predictions(
    postprocessed_dir: Path,
    gold_papers: dict[str, GoldPaper],
    gold_authorities: dict[str, Any] | None,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    manifest_path = postprocessed_dir / "aggregate_manifest.json"
    aggregate_path = postprocessed_dir / "predictions.aggregate.json"
    predictions_dir = postprocessed_dir / "parsed_predictions"
    provenance_dir = postprocessed_dir / "provenance"
    manifest = _require_dict(load_json(manifest_path), "aggregate manifest")
    aggregate = _require_dict(load_json(aggregate_path), "prediction aggregate")
    if manifest.get("schema_version") != POSTPROCESS_AGGREGATE_SCHEMA:
        raise ScoreError("aggregate manifest schema mismatch")
    if aggregate.get("schema_version") != POSTPROCESS_AGGREGATE_SCHEMA:
        raise ScoreError("prediction aggregate schema mismatch")
    if (
        manifest.get("provider_called") is not False
        or manifest.get("gold_read") is not False
        or manifest.get("predictions_copied_byte_for_byte") is not True
        or manifest.get("source_runs_modified") is not False
    ):
        raise ScoreError("aggregate provenance flags are invalid")

    inventory_count = _require_count(
        manifest.get("paper_inventory_count"), "aggregate paper inventory count"
    )
    scoreable_count = _require_count(
        manifest.get("parseable_unique_paper_count"),
        "aggregate scoreable paper count",
    )
    remaining_count = _require_count(
        manifest.get("remaining_paper_count"), "aggregate remaining paper count"
    )
    if (
        inventory_count > len(gold_papers)
        or scoreable_count > inventory_count
        or remaining_count != inventory_count - scoreable_count
    ):
        raise ScoreError("aggregate paper counts are inconsistent")
    if scoreable_count == 0:
        raise ScoreError("no scoreable predictions; prediction set excluded")

    configuration = _require_dict(
        manifest.get("configuration"), "aggregate configuration"
    )
    if aggregate.get("configuration") != configuration:
        raise ScoreError("aggregate configurations do not match")
    if gold_authorities is not None:
        benchmark_manifest = _require_dict(
            gold_authorities.get("benchmark_manifest"),
            "gold benchmark manifest provenance",
        )
        if (
            configuration.get("benchmark_manifest_sha256")
            != benchmark_manifest.get("sha256")
        ):
            raise ScoreError("aggregate gold-manifest hash mismatch")

    aggregate_records = aggregate.get("records")
    if not isinstance(aggregate_records, list):
        raise ScoreError("aggregate prediction records are invalid")
    aggregate_by_id: dict[str, dict[str, Any]] = {}
    for record in aggregate_records:
        paper_id = record.get("paper_id") if isinstance(record, dict) else None
        if (
            not isinstance(paper_id, str)
            or paper_id not in gold_papers
            or paper_id in aggregate_by_id
        ):
            raise ScoreError("aggregate prediction paper IDs are invalid or duplicated")
        aggregate_by_id[paper_id] = record
    aggregate_record_count = _require_count(
        aggregate.get("record_count"), "aggregate prediction record count"
    )
    if aggregate_record_count != len(aggregate_by_id) or len(
        aggregate_by_id
    ) != scoreable_count:
        raise ScoreError("aggregate prediction record count is inconsistent")

    record_sources = manifest.get("record_sources")
    if not isinstance(record_sources, list):
        raise ScoreError("aggregate record-source inventory is invalid")
    source_by_id: dict[str, dict[str, Any]] = {}
    for source in record_sources:
        paper_id = source.get("paper_id") if isinstance(source, dict) else None
        if (
            not isinstance(paper_id, str)
            or paper_id not in aggregate_by_id
            or paper_id in source_by_id
        ):
            raise ScoreError("aggregate record-source paper IDs are invalid or duplicated")
        source_by_id[paper_id] = source
    if set(source_by_id) != set(aggregate_by_id):
        raise ScoreError("aggregate record sources do not match prediction records")

    source_runs = manifest.get("source_runs")
    if not isinstance(source_runs, list) or not source_runs:
        raise ScoreError("aggregate source-run inventory is invalid")
    source_runs_by_priority: dict[int, dict[str, Any]] = {}
    for source_run in source_runs:
        if not isinstance(source_run, dict):
            raise ScoreError("aggregate source-run record is invalid")
        priority = source_run.get("priority")
        run_name = source_run.get("run_name")
        run_hash = source_run.get("run_manifest_sha256")
        if (
            not isinstance(priority, int)
            or isinstance(priority, bool)
            or priority < 1
            or priority in source_runs_by_priority
            or not isinstance(run_name, str)
            or not run_name
            or not isinstance(run_hash, str)
            or re.fullmatch(r"[0-9a-f]{64}", run_hash) is None
        ):
            raise ScoreError("aggregate source-run identity is invalid or duplicated")
        source_runs_by_priority[priority] = source_run

    prediction_paths = sorted(predictions_dir.glob("*.json"))
    provenance_paths = sorted(provenance_dir.glob("*.json"))
    if {path.stem for path in prediction_paths} != set(aggregate_by_id):
        raise ScoreError("aggregate parsed prediction files do not match records")
    if {path.stem for path in provenance_paths} != set(aggregate_by_id):
        raise ScoreError("aggregate provenance files do not match records")

    predictions: dict[str, dict[str, Any]] = {}
    selected_statuses: Counter[str] = Counter()
    for paper_id, record in sorted(aggregate_by_id.items()):
        prediction_path = predictions_dir / f"{paper_id}.json"
        prediction = validate_prediction(
            load_json(prediction_path),
            paper=gold_papers[paper_id],
            path=prediction_path,
        )
        if prediction != record:
            raise ScoreError(f"aggregate record mismatch for {paper_id}")
        prediction_hash = sha256_path(prediction_path)
        record_source = source_by_id[paper_id]
        priority = record_source.get("selected_source_priority")
        if (
            record_source.get("selected_source_prediction_sha256")
            != prediction_hash
            or not isinstance(priority, int)
            or isinstance(priority, bool)
            or priority not in source_runs_by_priority
        ):
            raise ScoreError(f"aggregate record-source mismatch for {paper_id}")

        provenance_path = provenance_dir / f"{paper_id}.json"
        provenance = _require_dict(
            load_json(provenance_path), f"aggregate provenance for {paper_id}"
        )
        if (
            provenance.get("schema_version") != POSTPROCESS_AGGREGATE_SCHEMA
            or provenance.get("paper_id") != paper_id
            or provenance.get("provider_called") is not False
            or provenance.get("gold_read") is not False
            or provenance.get("prediction_copied_byte_for_byte") is not True
            or provenance.get("selection_policy") != manifest.get("selection_policy")
        ):
            raise ScoreError(f"aggregate provenance mismatch for {paper_id}")
        selected = _require_dict(
            provenance.get("selected_candidate"),
            f"selected aggregate candidate for {paper_id}",
        )
        candidates = provenance.get("all_scoreable_candidates")
        source_run = source_runs_by_priority[priority]
        if (
            not isinstance(candidates, list)
            or not candidates
            or not all(isinstance(candidate, dict) for candidate in candidates)
            or selected not in candidates
            or selected.get("source_priority") != priority
            or selected.get("source_prediction_sha256") != prediction_hash
            or selected.get("source_run_name") != source_run.get("run_name")
            or selected.get("source_run_manifest_sha256")
            != source_run.get("run_manifest_sha256")
        ):
            raise ScoreError(f"selected aggregate candidate mismatch for {paper_id}")
        final_status = selected.get("final_status")
        if not isinstance(final_status, str) or not final_status:
            raise ScoreError(f"selected aggregate status is invalid for {paper_id}")
        selected_statuses[final_status] += 1
        predictions[paper_id] = prediction

    manifest_hash = sha256_path(manifest_path)
    return predictions, {
        "postprocessed_directory": str(postprocessed_dir),
        "run_name": postprocessed_dir.name,
        "aggregate_manifest": {
            "path": str(manifest_path),
            "sha256": manifest_hash,
        },
        "prediction_manifest": {
            "kind": "aggregate_manifest",
            "path": str(manifest_path),
            "sha256": manifest_hash,
        },
        "parsed_prediction_aggregate": {
            "path": str(aggregate_path),
            "sha256": sha256_path(aggregate_path),
        },
        "source_run_manifest_sha256": manifest_hash,
        "source_run_manifest_sha256s": [
            source_runs_by_priority[priority]["run_manifest_sha256"]
            for priority in sorted(source_runs_by_priority)
        ],
        "selected_paper_count": inventory_count,
        "scoreable_prediction_count": len(predictions),
        "final_status_counts": dict(sorted(selected_statuses.items())),
    }


def load_predictions(
    postprocessed_dir: Path,
    gold_papers: dict[str, GoldPaper],
    gold_authorities: dict[str, Any] | None = None,
    *, allow_empty: bool = False,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load either one postprocessed run or a provenance-preserving aggregate."""
    postprocessed_dir = postprocessed_dir.expanduser().resolve()
    aggregate_files = (
        postprocessed_dir / "aggregate_manifest.json",
        postprocessed_dir / "predictions.aggregate.json",
    )
    if any(path.exists() for path in aggregate_files):
        if not all(path.is_file() for path in aggregate_files):
            raise ScoreError("aggregate input is incomplete")
        return _load_aggregate_predictions(
            postprocessed_dir, gold_papers, gold_authorities
        )
    return _load_run_predictions(postprocessed_dir, gold_papers, allow_empty=allow_empty)


def _gold_issue_scope(issue: dict[str, Any]) -> str:
    classification = issue.get("classification")
    if classification == "typo":
        return "excluded_typo"
    if classification not in SEVERITIES:
        raise ScoreError(f"gold severity is invalid: {issue.get('issue_id')}")
    if issue.get("target_type") != "formal_result":
        return "excluded_nonformal"
    if normalize_tag(issue.get("primary_affected_tag")) is None:
        return "excluded_untagged_formal"
    return "eligible"


def _dependency_tags(issue: dict[str, Any]) -> set[str]:
    raw = issue.get("dependent_affected_formal_result_tags", [])
    if not isinstance(raw, list):
        raise ScoreError(f"dependent tags are invalid: {issue.get('issue_id')}")
    return {tag for value in raw if (tag := normalize_tag(value)) is not None}


def severity_credit(gold: object, predicted: object) -> tuple[float, float, bool]:
    """Return symmetric distance, asymmetric floor, and at-least-gold credit."""
    if gold not in SEVERITY_RANK or predicted not in SEVERITY_RANK:
        return 0.0, 0.0, False
    gold_rank = SEVERITY_RANK[str(gold)]
    predicted_rank = SEVERITY_RANK[str(predicted)]
    symmetric = max(0.0, 1.0 - 0.5 * abs(predicted_rank - gold_rank))
    deficit = gold_rank - predicted_rank
    floor = 1.0 if deficit <= 0 else 0.5 if deficit == 1 else 0.0
    return symmetric, floor, predicted_rank >= gold_rank


def highest_severity(issues: list[dict[str, Any]]) -> str | None:
    severities = [
        str(issue.get("classification"))
        for issue in issues
        if issue.get("classification") in SEVERITY_RANK
    ]
    if not severities:
        return None
    return max(severities, key=SEVERITY_RANK.__getitem__)


def score_corpus(
    gold_papers: dict[str, GoldPaper],
    predictions: dict[str, dict[str, Any]],
    *,
    authorities: dict[str, Any] | None = None,
    prediction_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    total_gold_records = 0
    excluded_typo = 0
    excluded_nonformal = 0
    excluded_untagged_formal = 0
    eligible_formal_issues = 0
    total_predicted_issues = 0
    gold_primary_total = 0
    predicted_primary_total = 0
    predicted_problematic_total = 0
    primary_hits = 0
    problematic_hits = 0
    primary_matched_tag_count = 0
    severity_adequacy_points = 0.0
    severity_full_credit_tag_count = 0
    severity_half_credit_tag_count = 0
    severity_zero_credit_tag_count = 0
    severity_exact_tag_count = 0
    severity_one_level_above_tag_count = 0
    severity_two_levels_above_tag_count = 0
    severity_one_level_below_tag_count = 0
    severity_two_levels_below_tag_count = 0
    severity_missing_or_na_tag_count = 0
    severity_transitions: Counter[tuple[str, str]] = Counter()
    global_predicted_primary_tags: set[str] = set()
    global_predicted_problematic_tags: set[str] = set()
    unmatched_gold_primary: list[dict[str, str]] = []
    unrecognized_gold_problematic: list[dict[str, str]] = []
    gold_recognized_only_as_dependent: list[dict[str, Any]] = []
    unmatched_predicted_primary: list[dict[str, str]] = []
    unmatched_predicted_problematic: list[dict[str, str]] = []
    prediction_issues_without_gold_primary_tag: list[dict[str, Any]] = []
    paper_rows: list[dict[str, Any]] = []

    for paper_id, paper in sorted(gold_papers.items()):
        scope_counts: Counter[str] = Counter()
        eligible: list[dict[str, Any]] = []
        for issue in paper.issues:
            total_gold_records += 1
            scope = _gold_issue_scope(issue)
            scope_counts[scope] += 1
            if scope == "excluded_typo":
                excluded_typo += 1
            elif scope == "excluded_nonformal":
                excluded_nonformal += 1
            elif scope == "excluded_untagged_formal":
                excluded_untagged_formal += 1
            else:
                eligible.append(issue)
                eligible_formal_issues += 1

        prediction = predictions.get(paper_id)
        predicted_issues = list(prediction.get("issues", [])) if prediction else []
        total_predicted_issues += len(predicted_issues)
        gold_primary = {
            tag
            for issue in eligible
            if (tag := normalize_tag(issue.get("primary_affected_tag"))) is not None
        }
        predicted_primary = {
            tag
            for issue in predicted_issues
            if (tag := normalize_tag(issue.get("primary_affected_tag"))) is not None
        }
        predicted_dependent = set().union(
            *(_dependency_tags(issue) for issue in predicted_issues), set()
        )
        predicted_problematic = predicted_primary | predicted_dependent
        global_predicted_primary_tags.update(predicted_primary)
        global_predicted_problematic_tags.update(predicted_problematic)
        primary_intersection = gold_primary & predicted_primary
        problematic_intersection = gold_primary & predicted_problematic
        dependent_only_intersection = problematic_intersection - predicted_primary
        gold_primary_total += len(gold_primary)
        predicted_primary_total += len(predicted_primary)
        predicted_problematic_total += len(predicted_problematic)
        primary_hits += len(primary_intersection)
        problematic_hits += len(problematic_intersection)
        unmatched_gold_primary.extend(
            {"paper_id": paper_id, "tag": tag}
            for tag in sorted(gold_primary - predicted_primary)
        )
        unrecognized_gold_problematic.extend(
            {"paper_id": paper_id, "tag": tag}
            for tag in sorted(gold_primary - predicted_problematic)
        )
        for tag in sorted(dependent_only_intersection):
            gold_recognized_only_as_dependent.append(
                {
                    "paper_id": paper_id,
                    "tag": tag,
                    "prediction_issue_ids": sorted(
                        str(issue.get("issue_id"))
                        for issue in predicted_issues
                        if tag in _dependency_tags(issue)
                    ),
                }
            )
        unmatched_predicted_primary.extend(
            {"paper_id": paper_id, "tag": tag}
            for tag in sorted(predicted_primary - gold_primary)
        )
        unmatched_predicted_problematic.extend(
            {"paper_id": paper_id, "tag": tag}
            for tag in sorted(predicted_problematic - gold_primary)
        )
        prediction_issues_without_gold_primary_tag.extend(
            {
                "paper_id": paper_id,
                "prediction_issue_id": str(issue.get("issue_id")),
                "primary_affected_tag": normalize_tag(
                    issue.get("primary_affected_tag")
                ),
                "target_type": issue.get("target_type"),
                "classification": issue.get("classification"),
            }
            for issue in predicted_issues
            if normalize_tag(issue.get("primary_affected_tag")) not in gold_primary
        )

        gold_by_tag: dict[str, list[dict[str, Any]]] = defaultdict(list)
        predicted_by_tag: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for issue in eligible:
            tag = normalize_tag(issue.get("primary_affected_tag"))
            assert tag is not None
            gold_by_tag[tag].append(issue)
        for issue in predicted_issues:
            tag = normalize_tag(issue.get("primary_affected_tag"))
            if tag is not None:
                predicted_by_tag[tag].append(issue)

        group_rows: list[dict[str, Any]] = []
        for tag, gold_group in sorted(gold_by_tag.items()):
            predicted_group = predicted_by_tag.get(tag, [])
            dependent_prediction_ids = sorted(
                str(issue.get("issue_id"))
                for issue in predicted_issues
                if tag in _dependency_tags(issue)
            )
            severity_comparison = None
            if predicted_group:
                primary_matched_tag_count += 1
                gold_maximum = highest_severity(gold_group)
                predicted_maximum = highest_severity(predicted_group)
                assert gold_maximum is not None
                symmetric_credit, adequacy_credit, at_least_gold = severity_credit(
                    gold_maximum, predicted_maximum
                )
                severity_adequacy_points += adequacy_credit
                severity_full_credit_tag_count += int(adequacy_credit == 1.0)
                severity_half_credit_tag_count += int(adequacy_credit == 0.5)
                severity_zero_credit_tag_count += int(adequacy_credit == 0.0)
                predicted_transition = predicted_maximum or "missing_or_NA"
                severity_transitions[(gold_maximum, predicted_transition)] += 1
                if predicted_maximum is None:
                    direction = "missing_or_NA"
                    severity_missing_or_na_tag_count += 1
                else:
                    difference = (
                        SEVERITY_RANK[predicted_maximum]
                        - SEVERITY_RANK[gold_maximum]
                    )
                    direction = {
                        -2: "two_levels_below",
                        -1: "one_level_below",
                        0: "exact",
                        1: "one_level_above",
                        2: "two_levels_above",
                    }[difference]
                    severity_exact_tag_count += int(difference == 0)
                    severity_one_level_above_tag_count += int(difference == 1)
                    severity_two_levels_above_tag_count += int(difference == 2)
                    severity_one_level_below_tag_count += int(difference == -1)
                    severity_two_levels_below_tag_count += int(difference == -2)
                severity_comparison = {
                    "gold_highest_severity": gold_maximum,
                    "predicted_highest_severity": predicted_maximum,
                    "asymmetric_adequacy_credit": adequacy_credit,
                    "symmetric_distance_credit_not_score": symmetric_credit,
                    "predicted_at_least_gold": at_least_gold,
                    "direction": direction,
                    "gold_issue_ids_at_highest_severity": sorted(
                        str(issue.get("issue_id"))
                        for issue in gold_group
                        if issue.get("classification") == gold_maximum
                    ),
                    "predicted_issue_ids_at_highest_severity": sorted(
                        str(issue.get("issue_id"))
                        for issue in predicted_group
                        if issue.get("classification") == predicted_maximum
                    ),
                }
            group_rows.append(
                {
                    "primary_affected_tag": tag,
                    "recovered_as_primary": bool(predicted_group),
                    "reported_problematic_anywhere": tag in predicted_problematic,
                    "reported_as_dependent": bool(dependent_prediction_ids),
                    "gold_formal_issue_count": len(gold_group),
                    "gold_issue_ids": sorted(
                        str(issue.get("issue_id")) for issue in gold_group
                    ),
                    "predicted_primary_issue_count": len(predicted_group),
                    "predicted_primary_issue_ids": sorted(
                        str(issue.get("issue_id")) for issue in predicted_group
                    ),
                    "predicted_dependent_issue_ids": dependent_prediction_ids,
                    "severity_within_primary_match": severity_comparison,
                }
            )

        paper_rows.append(
            {
                "paper_id": paper_id,
                "prediction_status": "scoreable" if prediction else "missing_or_unscoreable",
                "gold_record_count": len(paper.issues),
                "gold_scope_counts": dict(sorted(scope_counts.items())),
                "eligible_formal_gold_issue_count": len(eligible),
                "predicted_issue_count": len(predicted_issues),
                "gold_unique_formal_primary_tag_count": len(gold_primary),
                "matched_as_primary_tag_count": len(primary_intersection),
                "matched_as_problematic_anywhere_tag_count": len(
                    problematic_intersection
                ),
                "recognized_only_as_dependent_tag_count": len(
                    dependent_only_intersection
                ),
                "primary_tag_groups": group_rows,
            }
        )

    return {
        "schema_version": REPORT_SCHEMA,
        "question_id": "Q1",
        "scoring_contract": {
            "gold_issue_scope": (
                "issues with target_type=formal_result, substantive severity, and "
                "a non-null primary_affected_tag"
            ),
            "gold_evaluation_view": (
                "a read-only derived selection from the supplied gold package; "
                "the source gold files are not modified"
            ),
            "primary_tag_unit": "unique (paper_id, primary_affected_tag)",
            "primary_tag_recovery": (
                "the gold tag must occur in prediction.primary_affected_tag"
            ),
            "problematic_tag_recovery": (
                "the same gold primary tag may occur in either the predicted "
                "primary_affected_tag or dependent_affected_formal_result_tags; "
                "failure-location and evidence text do not count"
            ),
            "severity_condition": (
                "only gold buckets whose tag was recovered as a predicted primary "
                "tag; dependent-only recovery does not enter severity scoring"
            ),
            "severity_assignment": (
                "within each exact primary-primary tag bucket, compare the highest "
                "gold severity with the highest predicted severity; each recovered "
                "tag contributes one severity unit"
            ),
            "severity_adequacy_credit": {
                "exact_or_higher": 1.0,
                "one_level_below": 0.5,
                "two_levels_below": 0.0,
                "missing_or_NA_predicted_severity": 0.0,
                "asymmetric": True,
            },
            "overseverity": "reported separately as a diagnostic; no score penalty",
            "reason_or_failure_mechanism_scoring": "disabled",
            "unmatched_predictions": "diagnostic only; no score penalty",
            "tag_normalization": "surrounding whitespace only",
        },
        "authorities": authorities or {},
        "prediction_provenance": prediction_provenance or {},
        "summary": {
            "gold_paper_count": len(gold_papers),
            "scoreable_prediction_paper_count": len(predictions),
            "gold_record_count": total_gold_records,
            "excluded_typo_gold_issue_count": excluded_typo,
            "excluded_nonformal_substantive_gold_issue_count": excluded_nonformal,
            "excluded_untagged_formal_gold_issue_count": excluded_untagged_formal,
            "eligible_formal_gold_issue_count": eligible_formal_issues,
            "total_predicted_issue_count": total_predicted_issues,
            "unique_predicted_primary_tag_unit_count": predicted_primary_total,
            "unique_predicted_problematic_tag_unit_count": predicted_problematic_total,
            "globally_unique_predicted_primary_tag_string_count": len(
                global_predicted_primary_tags
            ),
            "globally_unique_predicted_problematic_tag_string_count": len(
                global_predicted_problematic_tags
            ),
            "formal_primary_tag_recall": {
                "matched": primary_hits,
                "gold_denominator": gold_primary_total,
                "percent": percent(primary_hits, gold_primary_total),
            },
            "formal_problematic_tag_recall": {
                "matched": problematic_hits,
                "gold_denominator": gold_primary_total,
                "percent": percent(
                    problematic_hits,
                    gold_primary_total,
                ),
            },
            "formal_tags_recognized_only_as_dependent_count": (
                problematic_hits - primary_hits
            ),
            "severity_within_primary_matched_tags": {
                "points": severity_adequacy_points,
                "primary_matched_tag_denominator": primary_matched_tag_count,
                "percent": percent(
                    severity_adequacy_points,
                    primary_matched_tag_count,
                ),
                "full_credit_tag_count": severity_full_credit_tag_count,
                "half_credit_tag_count": severity_half_credit_tag_count,
                "zero_credit_tag_count": severity_zero_credit_tag_count,
                "serious_reported_below_serious_tag_count": (
                    severity_transitions[("serious_issue", "moderate_issue")]
                    + severity_transitions[("serious_issue", "minor_issue")]
                    + severity_transitions[("serious_issue", "missing_or_NA")]
                ),
                "serious_or_moderate_reported_as_minor_or_missing_tag_count": (
                    severity_transitions[("serious_issue", "minor_issue")]
                    + severity_transitions[("serious_issue", "missing_or_NA")]
                    + severity_transitions[("moderate_issue", "minor_issue")]
                    + severity_transitions[("moderate_issue", "missing_or_NA")]
                ),
            },
            "overseverity_diagnostic_not_accuracy": {
                "overestimated_tag_count": (
                    severity_one_level_above_tag_count
                    + severity_two_levels_above_tag_count
                ),
                "primary_matched_tag_denominator": primary_matched_tag_count,
                "percent": percent(
                    severity_one_level_above_tag_count
                    + severity_two_levels_above_tag_count,
                    primary_matched_tag_count,
                ),
                "one_level_above_tag_count": severity_one_level_above_tag_count,
                "two_levels_above_tag_count": severity_two_levels_above_tag_count,
                "exact_tag_count": severity_exact_tag_count,
            },
            "efficiency_diagnostic_not_accuracy": {
                "primary_gold_tag_hits_per_100_predicted_issues": percent(
                    primary_hits, total_predicted_issues
                ),
                "problematic_gold_tag_hits_per_100_predicted_issues": percent(
                    problematic_hits, total_predicted_issues
                ),
            },
            "unmatched_gold_primary_tag_count": len(unmatched_gold_primary),
            "unrecognized_gold_problematic_tag_count": len(
                unrecognized_gold_problematic
            ),
            "unmatched_predicted_primary_tag_count": len(unmatched_predicted_primary),
            "unmatched_predicted_problematic_tag_count": len(
                unmatched_predicted_problematic
            ),
            "prediction_issues_without_gold_primary_tag_count": len(
                prediction_issues_without_gold_primary_tag
            ),
        },
        "diagnostics": {
            "severity_direction_counts": {
                "exact": severity_exact_tag_count,
                "one_level_above": severity_one_level_above_tag_count,
                "two_levels_above": severity_two_levels_above_tag_count,
                "one_level_below": severity_one_level_below_tag_count,
                "two_levels_below": severity_two_levels_below_tag_count,
                "missing_or_NA": severity_missing_or_na_tag_count,
            },
            "severity_transition_matrix": {
                gold: {
                    predicted: severity_transitions[(gold, predicted)]
                    for predicted in (*SEVERITIES, "missing_or_NA")
                }
                for gold in SEVERITIES
            },
            "unmatched_gold_primary_tags": unmatched_gold_primary,
            "unrecognized_gold_problematic_tags": unrecognized_gold_problematic,
            "gold_tags_recognized_only_as_dependent": (
                gold_recognized_only_as_dependent
            ),
            "unmatched_predicted_primary_tags": unmatched_predicted_primary,
            "unmatched_predicted_problematic_tags": (
                unmatched_predicted_problematic
            ),
            "prediction_issues_without_gold_primary_tag": (
                prediction_issues_without_gold_primary_tag
            ),
        },
        "papers": paper_rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold-dir", type=Path, required=True)
    parser.add_argument("--postprocessed-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        gold_papers, gold_authorities = load_gold(args.gold_dir)
        predictions, prediction_provenance = load_predictions(
            args.postprocessed_dir, gold_papers, gold_authorities
        )
        report = score_corpus(
            gold_papers,
            predictions,
            authorities=gold_authorities,
            prediction_provenance=prediction_provenance,
        )
        output = args.output.expanduser().resolve()
        write_json_exclusive(output, report)
    except ScoreError as exc:
        print(f"ERROR: {exc}")
        return 2
    summary = report["summary"]
    print(f"Score: {output}")
    print(
        "Primary tag recall (formal gold): "
        f"{summary['formal_primary_tag_recall']['matched']}/"
        f"{summary['formal_primary_tag_recall']['gold_denominator']} "
        f"({summary['formal_primary_tag_recall']['percent']}%)"
    )
    print(
        "Formal tags reported problematic anywhere: "
        f"{summary['formal_problematic_tag_recall']['matched']}/"
        f"{summary['formal_problematic_tag_recall']['gold_denominator']} "
        f"({summary['formal_problematic_tag_recall']['percent']}%)"
    )
    print(
        "Asymmetric severity adequacy within primary-primary matched tags: "
        f"{summary['severity_within_primary_matched_tags']['points']}/"
        f"{summary['severity_within_primary_matched_tags']['primary_matched_tag_denominator']} "
        f"({summary['severity_within_primary_matched_tags']['percent']}%)"
    )
    print(
        "Serious/moderate gold reported as minor/missing: "
        f"{summary['severity_within_primary_matched_tags']['serious_or_moderate_reported_as_minor_or_missing_tag_count']}"
    )
    print(
        "Overseverity (diagnostic, not accuracy): "
        f"{summary['overseverity_diagnostic_not_accuracy']['overestimated_tag_count']}/"
        f"{summary['overseverity_diagnostic_not_accuracy']['primary_matched_tag_denominator']} "
        f"({summary['overseverity_diagnostic_not_accuracy']['percent']}%)"
    )
    print(f"Total predicted issues: {summary['total_predicted_issue_count']}")
    print(
        "Unique predicted primary tag units: "
        f"{summary['unique_predicted_primary_tag_unit_count']}"
    )
    print(
        "Unmatched predicted primary tag units/issues: "
        f"{summary['unmatched_predicted_primary_tag_count']}/"
        f"{summary['prediction_issues_without_gold_primary_tag_count']}"
    )
    print(
        "Efficiency (not accuracy), primary gold tag hits per 100 predictions: "
        f"{summary['efficiency_diagnostic_not_accuracy']['primary_gold_tag_hits_per_100_predicted_issues']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
