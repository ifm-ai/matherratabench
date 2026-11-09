#!/usr/bin/env python3
"""Measure exact primary-tag coverage of formal gold results without a judge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "evaluation"))
import evaluate

SEVERITIES = ("minor_issue", "moderate_issue", "serious_issue")


def assess(paper_id, gold, predictions, severities=SEVERITIES):
    """Expose the existing Q1 scorer's formal_primary_tag_recall unchanged."""
    engine = evaluate.semantic.mechanical
    selected = tuple(issue for issue in gold if issue.get("classification") in severities)
    paper = engine.GoldPaper(paper_id, "", "", Path("issues.json"), selected)
    native = engine.score_corpus({paper_id: paper}, {paper_id: {"issues": predictions}})
    metric = native["summary"]["formal_primary_tag_recall"]
    groups = native["papers"][0]["primary_tag_groups"]
    covered, denominator = metric["matched"], metric["gold_denominator"]
    return {"paper_id": paper_id, "covered": covered, "denominator": denominator,
            "value": covered / denominator if denominator else None,
            "covered_tags": [group["primary_affected_tag"] for group in groups if group["recovered_as_primary"]],
            "missed_tags": [group["primary_affected_tag"] for group in groups if not group["recovered_as_primary"]],
            "gold_issue_ids_by_tag": {group["primary_affected_tag"]: group["gold_issue_ids"] for group in groups},
            "legacy_metric": metric}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True, help="Full output directory from parse.py parse")
    parser.add_argument("--gold", type=Path, required=True, help="Prepared gold package from prepare.py gold")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper-id", action="append", help="Bare arXiv ID; repeat for a subset")
    parser.add_argument("--gold-severity", choices=SEVERITIES, action="append")
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError("Score output already exists; choose a new filename")
        predictions = args.predictions.expanduser().resolve()
        gold = args.gold.expanduser().resolve()
        bundle, papers = evaluate.inspect_predictions(predictions, gold)
        selected = set(args.paper_id or [])
        if selected - {row["paper_id"] for row in bundle["papers"]}:
            raise ValueError("Requested paper is absent from these parsed predictions")
        rows, failures = [], []
        for row in bundle["papers"]:
            paper = row["paper_id"]
            if selected and paper not in selected:
                continue
            audit = evaluate.read(predictions / "parse_audit" / f"{paper}.json")
            available = audit.get("scoreable_prediction_written") is True
            issues = evaluate.read(predictions / "parsed_predictions" / f"{paper}.json")["issues"] if available else []
            if not available:
                failures.append(paper)
            result = assess(paper, papers[paper].issues, issues, args.gold_severity or SEVERITIES)
            result["input_status"] = audit["final_status"]
            result["prediction_available"] = available
            rows.append(result)
        covered = sum(row["covered"] for row in rows)
        denominator = sum(row["denominator"] for row in rows)
        report = {"schema_version": "matherratabench-tag-coverage-v1", "run_name": bundle["run_name"],
                  "metric": "formal_primary_tag_recall", "judge_used": False,
                  "covered": covered, "denominator": denominator,
                  "value": covered / denominator if denominator else None,
                  "prediction_failure_papers": failures, "papers": rows,
                  "contract": {"targets": "distinct (paper ID, primary tag) pairs on non-typo formal-result gold issues",
                               "predictions": "primary tags on any predicted issue; no severity or target-type gate",
                               "normalization": "unchanged Q1 normalize_tag and score_corpus rules",
                               "gold_severities": args.gold_severity or list(SEVERITIES),
                               "unmatched_predictions": "unpenalized; gold is nonexhaustive",
                               "meaning": "tag overlap only; neither defect identity nor issue recovery is established"},
                  "provenance": {"inputs_sha256": evaluate.digest(predictions / "inputs.json"),
                                 "parse_summary_sha256": evaluate.digest(predictions / "parse_summary.json"),
                                 "gold_manifest_sha256": evaluate.digest(gold / "manifest.json"),
                                 "gold_checksums_sha256": evaluate.digest(gold / "CHECKSUMS.sha256"),
                                 "implementation": {"scoring/score_tags.py": evaluate.digest(Path(__file__)),
                                                    **evaluate.runtime_hashes()}}}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        print(f"Exact primary-tag coverage: {covered}/{denominator}")
        return 0
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(1, f"Error: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
