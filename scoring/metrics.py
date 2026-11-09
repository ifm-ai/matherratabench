"""Deterministic issue- and paper-level scores from saved evaluation reports."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "evaluation" / "helpers"))
import escalation as judge_reporting

SEVERITIES = ("minor_issue", "moderate_issue", "serious_issue")
TARGET_TYPES = ("formal_result", "algorithm", "prose_claim", "other")
RANK = {label: rank for rank, label in enumerate(SEVERITIES)}
REPORT_SCHEMA = "matherratabench-paper-evaluation-v1"
INDEX_SCHEMA = "matherratabench-evaluation-reports-v1"


class ScoreError(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise ScoreError(message)


def read(path):
    def reject(value):
        raise ScoreError(f"Non-JSON constant: {value}")
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(path.read_bytes(), object_pairs_hook=unique, parse_constant=reject)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tag(value):
    if value is None:
        return None
    require(isinstance(value, str), "Primary tag must be a string or null")
    value = value.strip()
    return None if value in ("", "NA", "N/A", "null", "None") else value


def load_reports(path):
    path = path.expanduser().resolve()
    if path.is_dir() and not (path / "index.json").is_file():
        generations = sorted(p for p in (path / "reports").glob("*/index.json")
                             if re.fullmatch(r"[0-9]{4,}", p.parent.name))
        require(bool(generations), "Supply an evaluation output directory, a report generation, or index.json")
        path = max(generations, key=lambda candidate: int(candidate.parent.name))
    elif path.is_dir():
        path = path / "index.json"
    index = read(path)
    require(isinstance(index, dict) and index.get("schema_version") == INDEX_SCHEMA,
            "Scoring requires the index.json produced by evaluate.py")
    entries = index.get("papers")
    require(isinstance(entries, list) and entries, "Evaluation index has no papers")
    ids = [entry.get("paper_id") if isinstance(entry, dict) else None for entry in entries]
    require(all(isinstance(pid, str) and re.fullmatch(r"[0-9]{4}\.[0-9]{4,5}", pid) for pid in ids),
            "Invalid paper ID in evaluation index")
    require(ids == sorted(set(ids)) and index.get("paper_count") == len(ids),
            "Evaluation index must contain unique papers in ascending arXiv ID order")
    reports = []
    for entry in entries:
        require(entry.get("path") == f"papers/{entry['paper_id']}.json", "Invalid report path")
        report_path = path.parent / entry["path"]
        require(sha(report_path) == entry.get("sha256"), f"Evaluation report checksum mismatch: {entry['paper_id']}")
        report = read(report_path)
        require(isinstance(report, dict) and report.get("schema_version") == REPORT_SCHEMA,
                "Unexpected paper report schema")
        require(report.get("paper_id") == entry["paper_id"] and report.get("run_name") == index.get("run_name"),
                "Report identity differs from evaluation index")
        reports.append(report)
    return reports, {"index_path": str(path), "index_sha256": sha(path),
                     "run_name": index.get("run_name"), "report_count": len(reports)}


def assess(report):
    """Recompute label credit from original labels and decisions, not display flags."""
    predicted = report.get("predicted_issues")
    require(isinstance(predicted, list) and all(isinstance(p, dict) for p in predicted), "Invalid prediction inventory")
    ids = [p.get("issue_id") for p in predicted]
    require(all(isinstance(pid, str) and pid for pid in ids) and len(ids) == len(set(ids)), "Duplicate or invalid prediction ID")
    predictions = dict(zip(ids, predicted))
    records = report.get("gold_issues")
    require(isinstance(records, list), "Invalid gold issue inventory")
    rows, seen = [], set()
    for record in records:
        require(isinstance(record, dict) and isinstance(record.get("gold"), dict), "Invalid gold issue record")
        gold = record["gold"]
        gid = record.get("gold_issue_id")
        require(isinstance(gid, str) and gid and gid == gold.get("issue_id") and gid not in seen,
                "Duplicate or inconsistent gold issue ID")
        seen.add(gid)
        severity = gold.get("classification")
        require(severity in (*SEVERITIES, "typo") and gold.get("target_type") in TARGET_TYPES,
                "Invalid gold severity or target type")
        status, reason = record.get("match_status"), record.get("reason")
        require(status in ("matched", "unmatched", "unresolved", "not_evaluated"), "Invalid match status")
        if severity == "typo":
            require(status == "not_evaluated", "The current question excludes gold typos")
            continue
        require(status != "not_evaluated", "Substantive issues must be assessed or explicitly unresolved")
        decision = record.get("judge_decision")
        matched = status == "matched"
        unresolved = status == "unresolved" and reason == "judge_unavailable"
        if status == "unresolved":
            require(reason in ("judge_unavailable", "prediction_unavailable"), "Unknown unresolved reason")
        support, focus = [], {}
        if decision is not None:
            require(isinstance(decision, dict) and decision.get("gold_issue_id") == gid, "Judge decision identity mismatch")
            require(decision.get("coverage_status") in ("matched", "not_matched"), "Unknown judge coverage status")
            require((decision.get("coverage_status") == "matched") == matched, "Match status contradicts judge decision")
        if matched:
            require(decision is not None, "A match requires a saved judge decision")
            support = decision.get("supporting_prediction_issue_ids")
            require(isinstance(support, list) and 1 <= len(support) <= 3 and len(set(support)) == len(support)
                    and all(isinstance(pid, str) and pid in predictions for pid in support), "Invalid matching prediction IDs")
            focus_id = decision.get("focus_prediction_issue_id")
            require(focus_id in support, "Focus prediction must be one of the necessary supports")
            require(record.get("matched_prediction_ids") == support and record.get("focus_prediction_id") == focus_id,
                    "Public match attribution differs from the saved judge decision")
            focus = predictions[focus_id]
        else:
            require(not record.get("matched_prediction_ids"), "Unmatched issue contains matching prediction IDs")
        labels = [predictions[pid].get("classification") for pid in support]
        predicted_severity = max(labels, key=RANK.get) if labels and all(label in RANK for label in labels) else None
        severity_exact = matched and predicted_severity == severity
        type_exact = matched and focus.get("target_type") == gold["target_type"]
        formal = gold["target_type"] == "formal_result"
        primary_exact = (matched and tag(gold.get("primary_affected_tag")) is not None
                         and tag(focus.get("primary_affected_tag")) == tag(gold.get("primary_affected_tag"))) if formal else None
        severity_points = (max(0.0, 1 - 0.5 * abs(RANK[severity] - RANK[predicted_severity]))
                           if matched and predicted_severity in RANK else 0.0)
        rows.append({"paper_id": report["paper_id"], "gold_issue_id": gid,
                     "gold_severity": severity, "gold_target_type": gold["target_type"],
                     "recovered": None if unresolved else matched,
                     "strict": None if unresolved else bool(matched and severity_exact and type_exact and (not formal or primary_exact)),
                     "strict_formal": None if unresolved and formal else bool(matched and severity_exact and primary_exact) if formal else None,
                     "severity_exact": None if unresolved else bool(severity_exact),
                     "target_type_exact": None if unresolved else bool(type_exact),
                     "primary_tag_exact": None if unresolved and formal else primary_exact,
                     "predicted_severity": predicted_severity, "severity_points": severity_points,
                     "supporting_prediction_issue_ids": support,
                     "focus_prediction_issue_id": focus.get("issue_id"),
                     "resolution": "unresolved_judge" if unresolved else "prediction_failure" if reason == "prediction_unavailable" else "recovered" if matched else "miss"})
    return rows


def fraction(earned, possible):
    return earned / possible if possible else None


def rate(rows, field, weights=None):
    weights = weights or {label: 1.0 for label in SEVERITIES}
    possible = sum(weights[row["gold_severity"]] for row in rows)
    earned = sum(weights[row["gold_severity"]] for row in rows if row[field] is True)
    unknown = sum(weights[row["gold_severity"]] for row in rows if row[field] is None)
    lower, upper = fraction(earned, possible), fraction(earned + unknown, possible)
    return {"count": sum(row[field] is True for row in rows), "denominator": len(rows),
            "earned": earned, "possible": possible,
            "unresolved": sum(row[field] is None for row in rows),
            "value": lower if unknown == 0 else None, "lower_bound": lower, "upper_bound": upper}


def severity_score(rows):
    found = [row for row in rows if row["recovered"] is True]
    unknown = sum(row["recovered"] is None for row in rows)
    points = sum(row["severity_points"] for row in found)
    observed = fraction(points, len(found))
    return {"points": points, "found_issue_count": len(found), "unresolved": unknown,
            "value": observed if not unknown else None, "observed_value": observed,
            "lower_bound": fraction(points, len(found) + unknown),
            "upper_bound": fraction(points + unknown, len(found) + unknown)}


def issue_scores(rows, formal_weights):
    return {
        "issue_recovery": rate(rows, "recovered"),
        "serious_recovery": rate([row for row in rows if row["gold_severity"] == "serious_issue"], "recovered"),
        "severity_score": severity_score(rows),
        "strict_issue_recovery": rate(rows, "strict"),
        "strict_formal_issue_recovery": rate([row for row in rows if row["gold_target_type"] == "formal_result"],
                                              "strict_formal", formal_weights),
    }


def conjunction(values):
    return False if False in values else None if None in values else True


def paper_scores(reports, rows, zero_policy):
    by_paper = {report["paper_id"]: [] for report in reports}
    for row in rows:
        by_paper[row["paper_id"]].append(row)
    papers = []
    for pid, issues in by_paper.items():
        eligible = bool(issues) or zero_policy == "count-as-satisfied"
        papers.append({"paper_id": pid, "eligible": eligible, "gold_issue_count": len(issues),
                       "issue_recovery": rate(issues, "recovered"),
                       "all_recovered": conjunction([row["recovered"] for row in issues]) if eligible else None})
    selected = [paper for paper in papers if paper["eligible"]]
    metrics = {}
    for field in ("all_recovered",):
        count = sum(paper[field] is True for paper in selected)
        unknown = sum(paper[field] is None for paper in selected)
        metrics[field] = {"count": count, "denominator": len(selected), "unresolved": unknown,
                          "value": fraction(count, len(selected)) if not unknown else None,
                          "lower_bound": fraction(count, len(selected)),
                          "upper_bound": fraction(count + unknown, len(selected))}
    return papers, metrics


def score(args, level):
    reports, source = load_reports(args.evaluations)
    requested = set(args.paper_id or [])
    require(not requested - {report["paper_id"] for report in reports}, "Requested paper is absent from these evaluations")
    reports = [report for report in reports if not requested or report["paper_id"] in requested]
    rows = [row for report in reports for row in assess(report)]
    rows = [row for row in rows if (not args.target_type or row["gold_target_type"] in args.target_type)
            and (not args.gold_severity or row["gold_severity"] in args.gold_severity)]
    unknown = [row for row in rows if row["resolution"] == "unresolved_judge"]
    if args.unresolved_policy == "error":
        require(not unknown, "Unresolved judge decisions exist; use bounds or explicit exclusion")
    if args.unresolved_policy == "exclude":
        rows = [row for row in rows if row["resolution"] != "unresolved_judge"]
    contract = {"eligible_gold": "selected non-typo issues; all denominators computed from the reports",
                "missing_or_unparseable_predictions": "zero recovery, retained in denominator",
                "unmatched_predictions": "unpenalized; gold is nonexhaustive",
                "severity_attribution": "highest among necessary supports; ungraded if any is ungraded",
                "type_and_primary_tag_attribution": "judge-selected focus entry; no best-label selection",
                "strict_issue": "match + exact severity + exact target type + exact nonmissing primary tag for formal results",
                "strict_formal_issue": "gold formal result + match + exact severity + exact nonmissing primary tag; predicted target type is not an additional gate",
                "severity_score": "over matched issues: exact=1, one level off=0.5, otherwise=0; averaged",
                "unresolved_policy": args.unresolved_policy,
                "paper_ids": [report["paper_id"] for report in reports],
                "target_types": args.target_type or list(TARGET_TYPES),
                "gold_severities": args.gold_severity or list(SEVERITIES)}
    summary = {"selected_papers": len(reports), "eligible_issues": len(rows),
               "unresolved_judge_issues_before_policy": len(unknown),
               "prediction_failure_papers": [report["paper_id"] for report in reports
                                             if not report.get("prediction_available", report["input_status"] in
                                                               ("valid", "scoreable_schema_warning", "scoreable_partial_issue_salvage"))]}
    if level == "issues":
        weights = dict(zip(SEVERITIES, args.formal_weights))
        require(all(math.isfinite(value) and value >= 0 for value in weights.values()), "Formal weights must be finite and nonnegative")
        contract["formal_weights"] = weights
        metrics = issue_scores(rows, weights)
        details = {"issues": rows}
    else:
        contract["paper_success"] = "all eligible issues satisfy the selected criterion; not an average of per-paper issue fractions"
        contract["zero_eligible_policy"] = args.zero_eligible
        papers, metrics = paper_scores(reports, rows, args.zero_eligible)
        summary["eligible_papers"] = sum(paper["eligible"] for paper in papers)
        summary["zero_eligible_papers"] = sum(paper["gold_issue_count"] == 0 for paper in papers)
        details = {"papers": papers}
    return {"schema_version": "matherratabench-scores-v1", "level": level,
            "source": source, "scorer_sha256": sha(Path(__file__)),
            "judge_reporting_sha256": sha(Path(judge_reporting.__file__)),
            "judges": judge_reporting.summarize_judges(reports),
            "scoring_status": "partial" if unknown else "complete", "contract": contract,
            "summary": summary, "metrics": metrics, **details}


def main(level, argv=None):
    parser = argparse.ArgumentParser(description=f"Score saved evaluation reports per {level}; no model calls.")
    parser.add_argument("--evaluations", type=Path, required=True, help="Evaluation root, report generation, or its index.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper-id", action="append")
    parser.add_argument("--target-type", choices=TARGET_TYPES, action="append")
    parser.add_argument("--gold-severity", choices=SEVERITIES, action="append")
    parser.add_argument("--unresolved-policy", choices=("bounds", "error", "exclude"), default="error")
    if level == "issues":
        parser.add_argument("--formal-weights", type=float, nargs=3, default=[0.5, 1.0, 1.0], metavar=("MINOR", "MODERATE", "SERIOUS"))
    else:
        parser.add_argument("--zero-eligible", choices=("exclude", "count-as-satisfied"), default="exclude")
    args = parser.parse_args(argv)
    try:
        require(not args.output.exists(), "Score output already exists; choose a new filename")
        report = score(args, level)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        print(json.dumps(report["metrics"], indent=2))
        return 0
    except (ValueError, OSError, TypeError, KeyError) as exc:
        parser.exit(1, f"Error: {exc}\n")
