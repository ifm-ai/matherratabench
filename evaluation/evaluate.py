#!/usr/bin/env python3
"""Evaluate parsed agent answers with the existing Luna.8 judge; save per-issue reports.

This stage performs matching and deterministic label comparisons. Separate
stage-4 scoring scripts choose how to turn these reports into numerical scores.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
CORE_DIR = HERE / "helpers"
sys.path.insert(0, str(CORE_DIR))
import judge_contract as semantic
import run_judge_luna as judge_runner

REPORT_SCHEMA = "matherratabench-paper-evaluation-v1"
INDEX_SCHEMA = "matherratabench-evaluation-reports-v1"
PLAN_SCHEMA = "matherratabench-evaluation-plan-v1"


class EvaluationError(ValueError):
    pass


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return semantic.load_json(path)


def write(path, value):
    semantic.write_json_exclusive(path, value)


def runtime_hashes():
    paths = [Path(__file__), HERE / "judge_prompt.txt", HERE / "judge_schema.json",
             CORE_DIR / "judge_contract.py", CORE_DIR / "run_judge_luna.py",
             CORE_DIR / "escalation.py", CORE_DIR / "score_q1.py"]
    return {path.relative_to(ROOT).as_posix(): digest(path) for path in paths}


def inspect_predictions(predictions, gold):
    """Select from the original input inventory, including failed/missing answers."""
    bundle = read(predictions / "inputs.json")
    if (not isinstance(bundle, dict) or bundle.get("schema_version") != "matherratabench-agent-inputs-v1"
            or not isinstance(bundle.get("papers"), list) or not bundle["papers"]):
        raise EvaluationError("--predictions must be the full output directory from run-agent/parse.py parse")
    rows = bundle["papers"]
    if not all(isinstance(row, dict) and isinstance(row.get("paper_id"), str) for row in rows):
        raise EvaluationError("Invalid original paper inventory")
    ids = [row["paper_id"] for row in rows]
    if ids != sorted(set(ids)) or bundle.get("paper_count") != len(rows):
        raise EvaluationError("Original paper inventory must be unique and in ascending arXiv ID order")
    gold_papers, authorities = semantic.mechanical.load_gold(gold)
    if set(ids) - set(gold_papers):
        raise EvaluationError("Some selected papers are absent from the supplied gold package")
    checksums = semantic.mechanical.load_checksum_index(gold / "CHECKSUMS.sha256")
    for row in rows:
        paper = row["paper_id"]
        if row.get("source_sha256") != gold_papers[paper].source_sha256:
            raise EvaluationError(f"Predicted source differs from the gold input: {paper}")
        source_manifest = gold / "papers" / paper / "source-manifest.json"
        if checksums.get(f"papers/{paper}/source-manifest.json") != digest(source_manifest):
            raise EvaluationError(f"Gold source provenance checksum mismatch: {paper}")
        if read(source_manifest).get("version") != row.get("version"):
            raise EvaluationError(f"Predicted arXiv version differs from gold: {paper}")
    summary = read(predictions / "parse_summary.json")
    if summary.get("source_run_manifest_sha256") != digest(predictions / "inputs.json"):
        raise EvaluationError("Parsed input-manifest checksum mismatch")
    if {path.stem for path in (predictions / "parse_audit").glob("*.json")} != set(ids):
        raise EvaluationError("Parse audits must cover exactly the original paper inventory")
    # Reuse the existing inventory, audit, aggregate and source-hash validation.
    semantic.mechanical.load_predictions(predictions, gold_papers, authorities, allow_empty=True)
    if not isinstance(bundle.get("run_name"), str) or not semantic.SAFE_NAME_RE.fullmatch(bundle["run_name"]):
        raise EvaluationError("Invalid agent run name")
    return bundle, gold_papers


def plan_for(args, bundle):
    files = {path.relative_to(args.predictions).as_posix(): digest(path)
             for path in sorted(args.predictions.rglob("*"))
             if path.is_file() and not path.name.startswith(".")}
    return {
        "schema_version": PLAN_SCHEMA, "prediction_directory": str(args.predictions),
        "prediction_files": files, "gold_directory": str(args.gold),
        "gold_manifest_sha256": digest(args.gold / "manifest.json"),
        "gold_checksums_sha256": digest(args.gold / "CHECKSUMS.sha256"),
        "papers": [{key: row[key] for key in ("paper_id", "version", "source_sha256")}
                   for row in bundle["papers"]],
        "agent_run_name": bundle["run_name"], "judge_run_name": args.run_name,
        "judge_policy": "Luna.8", "judge_model": args.judge_model,
        "reasoning_effort": args.reasoning_effort, "judge_runner": args.judge_runner,
        "base_url": args.base_url, "codex_bin": str(args.codex_bin) if args.codex_bin else None,
        "retries": args.retries, "max_output_tokens": args.max_output_tokens,
        "timeout_seconds": args.timeout_seconds, "jobs": args.jobs,
        "escalation": {"enabled": not args.no_escalation,
                       "judge_runner": "api", "backend": "openai_responses_api",
                       "extra_luna_attempts": args.extra_luna_attempts,
                       "fallback_model": args.fallback_model,
                       "fallback_attempts": args.fallback_attempts},
        "implementation": runtime_hashes(),
    }


def judge_arguments(args, input_manifest, output, *, dry_run=False, resume=False):
    values = ["--all", "--policy", "Luna.8", "--input-manifest", str(input_manifest),
              "--output-dir", str(output), "--run-name", args.run_name,
              "--judge-runner", args.judge_runner, "--model", args.judge_model,
              "--reasoning-effort", args.reasoning_effort,
              "--extra-luna-attempts", str(args.extra_luna_attempts),
              "--fallback-model", args.fallback_model,
              "--fallback-attempts", str(args.fallback_attempts),
              "--base-url", args.base_url, "--retries", str(args.retries),
              "--max-output-tokens", str(args.max_output_tokens),
              "--timeout-seconds", str(args.timeout_seconds), "--max-concurrency", str(args.jobs)]
    if args.codex_bin:
        values.extend(["--codex-bin", str(args.codex_bin)])
    if args.no_escalation:
        values.append("--no-escalation")
    if dry_run:
        values.append("--dry-run")
    if resume:
        values.append("--resume")
    return values


def create_inputs(args, bundle, output):
    return semantic.build_inputs(args.gold, args.predictions, output,
                                 paper_ids=[row["paper_id"] for row in bundle["papers"]],
                                 allow_empty_predictions=True)


def compare(gold, predicted, *, assessed, applicable=True):
    result = ("not_applicable" if not applicable else "not_assessed" if not assessed
              else "correct" if gold == predicted and gold is not None else "incorrect")
    return {"gold": gold, "predicted": predicted, "result": result}


def label_comparisons(gold, predicted, *, assessed):
    return {
        "severity": compare(gold.get("classification"), predicted.get("classification"), assessed=assessed),
        "target_type": compare(gold.get("target_type"), predicted.get("target_type"), assessed=assessed),
        "primary_affected_tag": compare(semantic.mechanical.normalize_tag(gold.get("primary_affected_tag")),
                                        semantic.mechanical.normalize_tag(predicted.get("primary_affected_tag")),
                                        assessed=assessed, applicable=gold.get("target_type") == "formal_result"),
    }


def build_paper_report(source_row, gold_issues, prediction_document, decisions, *, run_name,
                       input_status, needs_judgment, provenance, input_response_incomplete=False):
    """Build the report consumed by the scorers from a paper's evidence and decisions."""
    predictions = {issue["issue_id"]: issue for issue in prediction_document["issues"]} if prediction_document else {}
    prediction_failed = prediction_document is None
    judge_missing = needs_judgment and decisions is None
    by_gold = {decision["gold_issue_id"]: decision for decision in (decisions or [])}
    records, used = [], set()
    for gold in gold_issues:
        gid = gold["issue_id"]
        decision = by_gold.get(gid)
        typo = gold["classification"] == "typo"
        matched = decision is not None and decision["coverage_status"] == "matched"
        if typo:
            status, reason = "not_evaluated", "typos_excluded_by_question"
        elif prediction_failed:
            status, reason = "unresolved", "prediction_unavailable"
        elif judge_missing:
            status, reason = "unresolved", "judge_unavailable"
        elif matched:
            status, reason = "matched", None
        else:
            status, reason = "unmatched", "no_predictions" if not predictions else "no_accepted_match"
        supports = decision["supporting_prediction_issue_ids"] if matched else []
        focus = decision["focus_prediction_issue_id"] if matched else None
        focus_issue = predictions.get(focus, {})
        attributed = dict(focus_issue)
        if matched:
            attributed["classification"] = semantic.combined_severity(supports, predictions)
        used.update(supports)
        records.append({
            "gold_issue_id": gid, "gold": dict(gold), "match_status": status,
            "reason": reason, "matched_prediction_ids": supports, "focus_prediction_id": focus,
            "judge_explanation": decision["identity_explanation"] if decision else None,
            "judge_decision": decision,
            "prediction_comparisons": [{"prediction_id": pid,
                                        **label_comparisons(gold, predictions[pid], assessed=True)} for pid in supports],
            "attributed_label_comparisons": label_comparisons(gold, attributed, assessed=matched and not typo),
        })
    return {
        "schema_version": REPORT_SCHEMA, "paper_id": source_row["paper_id"], "version": source_row["version"],
        "source_sha256": source_row["source_sha256"], "run_name": run_name,
        "input_status": input_status,
        "input_response_incomplete": input_response_incomplete,
        "prediction_available": prediction_document is not None,
        "judgment_status": "missing" if judge_missing else "complete" if needs_judgment else "not_required",
        "label_attribution": {"severity": "highest among necessary supports; NA if any is ungraded",
                              "target_type_and_primary_tag": "judge-selected focus prediction",
                              "per_prediction_comparisons": "diagnostic; do not select the best labels independently"},
        "predicted_issues": list(predictions.values()), "gold_issues": records,
        "unmatched_prediction_ids": [pid for pid in predictions if pid not in used],
        "counts": dict(Counter(record["match_status"] for record in records)),
        "provenance": {**provenance, "judge_prompt_sha256": digest(HERE / "judge_prompt.txt"),
                       "judge_schema_sha256": digest(HERE / "judge_schema.json")},
    }


def paper_report(args, bundle, source_row, input_row, gold_paper, decisions, *, judgment_path):
    paper = source_row["paper_id"]
    audit_path = args.predictions / "parse_audit" / f"{paper}.json"
    parse_audit = read(audit_path)
    parsed_path = args.predictions / "parsed_predictions" / f"{paper}.json"
    prediction_document = read(parsed_path) if input_row.get("prediction_path") else None
    return build_paper_report(
        source_row, gold_paper.issues, prediction_document, decisions,
        run_name=bundle["run_name"], input_status=parse_audit["final_status"],
        needs_judgment=input_row["status"] == "ready",
        input_response_incomplete=parse_audit.get("response_incomplete", False),
        provenance={"parse_audit_sha256": digest(audit_path),
                    "prediction_sha256": digest(parsed_path) if prediction_document else None,
                    "gold_issues_sha256": digest(gold_paper.issues_path),
                    "judgment_sha256": digest(judgment_path) if judgment_path.is_file() else None},
    )


def save_reports(args, bundle, gold_papers):
    inputs_path = args.output / "judge-inputs" / "manifest.json"
    run_path = args.output / "judge-run" / "run_manifest.json"
    manifest, rows = semantic.load_input_manifest(inputs_path)
    # Keep normal and escalated judgments immutable, validating the accepted
    # decision through the same matching contract before building reports.
    accepted = {}
    with tempfile.TemporaryDirectory(prefix="matherratabench-accepted-") as temp:
        for row in rows:
            paper = row["paper_id"]
            path = args.output / "judge-run" / "judgments" / f"{paper}.json"
            escalation_dir = args.output / "judge-run" / "escalation"
            if path.is_file():
                judge_runner.validate_existing(path, row=row, run_name=args.run_name,
                                               prediction_set_name=manifest["prediction_set_name"])
            if not path.is_file() and (escalation_dir / "judgments" / f"{paper}.json").is_file():
                judge_runner.validate_escalated(
                    escalation_dir, row=row, run_name=args.run_name,
                    prediction_set_name=manifest["prediction_set_name"])
                path = escalation_dir / "judgments" / f"{paper}.json"
            elif not path.is_file() and (escalation_dir / "records" / f"{paper}.json").is_file():
                record = judge_runner.escalation_lib.load_record(escalation_dir, paper)
                if record.get("resolved"):
                    raise EvaluationError(f"Resolved escalation has no saved judgment: {paper}")
            accepted[paper] = path
            if path.is_file():
                shutil.copyfile(path, Path(temp) / f"{paper}.json")
        verified = semantic.score_judgments(inputs_path, Path(temp), run_path, allow_partial=True)
    generations = args.output / "reports"
    generations.mkdir(exist_ok=True)
    number = max([int(path.name) for path in generations.iterdir() if path.name.isdigit()] or [0]) + 1
    destination = generations / f"{number:04d}"
    source_rows = {row["paper_id"]: row for row in bundle["papers"]}
    with tempfile.TemporaryDirectory(prefix=".reports-", dir=generations) as temp:
        staging = Path(temp) / "reports"
        (staging / "papers").mkdir(parents=True)
        files, statuses, reports = [], Counter(), []
        for row in rows:
            paper = row["paper_id"]
            judgment_path = accepted[paper]
            decisions = read(judgment_path)["gold_decisions"] if judgment_path.is_file() else None
            report = paper_report(args, bundle, source_rows[paper], row, gold_papers[paper], decisions,
                                  judgment_path=judgment_path)
            path = staging / "papers" / f"{paper}.json"
            write(path, report)
            statuses.update(report["counts"])
            reports.append(report)
            files.append({"paper_id": paper, "path": f"papers/{paper}.json", "sha256": digest(path)})
        write(staging / "index.json", {
            "schema_version": INDEX_SCHEMA, "run_name": bundle["run_name"],
            "paper_count": len(files), "papers": files, "issue_status_counts": dict(statuses),
            "judges": judge_runner.escalation_lib.summarize_judges(reports),
            "missing_judgment_papers": verified["summary"]["missing_judgment_paper_ids"],
            "prediction_failure_papers": [pid for pid in source_rows if not read(args.predictions / "parse_audit" / f"{pid}.json").get("scoreable_prediction_written")],
            "evaluation_manifest_sha256": digest(args.output / "evaluation_manifest.json"),
            "judge_input_manifest_sha256": digest(inputs_path), "judge_run_manifest_sha256": digest(run_path),
        })
        staging.rename(destination)
    print(f"Evaluation reports: {destination}")
    print(f"Gold issue statuses: {json.dumps(dict(statuses), sort_keys=True)}")
    return destination, statuses


def evaluate(args):
    args.predictions = args.predictions.expanduser().resolve()
    args.gold = args.gold.expanduser().resolve()
    args.output = args.output.expanduser().resolve()
    if any(args.output == source or source in args.output.parents for source in (args.predictions, args.gold)):
        raise EvaluationError("Keep evaluation output outside the parsed-prediction and gold directories")
    if min(args.jobs, args.timeout_seconds, args.max_output_tokens) < 1 or min(args.retries, args.extra_luna_attempts, args.fallback_attempts) < 0:
        raise EvaluationError("Jobs, timeout and token budget must be positive; retries must be nonnegative")
    bundle, gold_papers = inspect_predictions(args.predictions, args.gold)
    args.run_name = args.run_name or bundle["run_name"] + "-luna8"
    if not semantic.SAFE_NAME_RE.fullmatch(args.run_name):
        raise EvaluationError("Invalid judge run name")
    # Validate the URL before recording it; credentials are never plan fields.
    from urllib.parse import urlsplit
    url = urlsplit(args.base_url)
    if url.scheme != "https" or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise EvaluationError("Judge API base URL must be HTTPS without embedded credentials, query or fragment")
    plan = plan_for(args, bundle)
    if args.output.exists():
        if not args.resume:
            raise EvaluationError("Output already exists; choose a new directory or use --resume with the same configuration")
        if read(args.output / "evaluation_manifest.json") != plan:
            raise EvaluationError("Resume configuration, input data, or implementation differs from the saved manifest")
    elif args.resume:
        raise EvaluationError("--resume requires an existing matching output directory")
    with tempfile.TemporaryDirectory(prefix="matherratabench-evaluate-") as temp:
        preflight = Path(temp) / "judge-inputs"
        preview = create_inputs(args, bundle, preflight)
        ready = preview["status_counts"].get("ready", 0)
        if ready:
            code = judge_runner.main(judge_arguments(args, preflight / "manifest.json", Path(temp) / "judge-run", dry_run=True))
            if code:
                raise EvaluationError("Judge configuration preflight failed")
        print(f"Selected {len(bundle['papers'])} papers; {ready} require model judging")
        if args.dry_run:
            print("Dry run: no model calls and no output directory created")
            return 0
    uses_api = args.judge_runner == "api" or (not args.no_escalation and
               (args.extra_luna_attempts > 0 or args.fallback_attempts > 0))
    if ready and uses_api and not os.environ.get("OPENAI_API_KEY"):
        raise EvaluationError("Set OPENAI_API_KEY privately for API judging or Responses API escalation")
    if not args.output.exists():
        args.output.mkdir(parents=True)
        write(args.output / "evaluation_manifest.json", plan)
        create_inputs(args, bundle, args.output / "judge-inputs")
        write(args.output / "prepared.json", {"judge_input_manifest_sha256": digest(args.output / "judge-inputs/manifest.json")})
    else:
        if read(args.output / "prepared.json")["judge_input_manifest_sha256"] != digest(args.output / "judge-inputs/manifest.json"):
            raise EvaluationError("Saved judge inputs changed")
        semantic.load_input_manifest(args.output / "judge-inputs/manifest.json")
    run_path = args.output / "judge-run/run_manifest.json"
    if ready:
        result = judge_runner.main(judge_arguments(args, args.output / "judge-inputs/manifest.json",
                                                  args.output / "judge-run", resume=run_path.exists()))
        if not run_path.exists():
            raise EvaluationError("Judge failed before creating its run manifest")
    elif not run_path.exists():
        (args.output / "judge-run/judgments").mkdir(parents=True)
        write(run_path, {"schema_version": judge_runner.RUN_MANIFEST_SCHEMA,
                         "judge_run_name": args.run_name, "judge_policy_version": "Luna.8",
                         "model": args.judge_model, "reasoning_effort": args.reasoning_effort,
                         "backend": "not_called", "provider_called": False,
                         "input_manifest": {"sha256": digest(args.output / "judge-inputs/manifest.json")}})
    _, statuses = save_reports(args, bundle, gold_papers)
    return 1 if statuses.get("unresolved", 0) else 0


def parser():
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--predictions", type=Path, required=True, help="Full output directory from parse.py parse")
    command.add_argument("--gold", type=Path, required=True, help="Prepared gold package from prepare.py gold")
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--judge-runner", choices=("api", "codex"), default="codex")
    command.add_argument("--judge-model", default="gpt-5.6-luna")
    command.add_argument("--reasoning-effort", default="medium")
    command.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    command.add_argument("--codex-bin", type=Path)
    command.add_argument("--run-name")
    command.add_argument("--jobs", type=int, default=1)
    command.add_argument("--retries", type=int, default=2)
    command.add_argument("--extra-luna-attempts", type=int, default=1)
    command.add_argument("--fallback-model", default="gpt-6.1-sol")
    command.add_argument("--fallback-attempts", type=int, default=3)
    command.add_argument("--no-escalation", action="store_true", help="Disable the judge fallback policy")
    command.add_argument("--max-output-tokens", type=int, default=16384)
    command.add_argument("--timeout-seconds", type=int, default=1800)
    command.add_argument("--dry-run", action="store_true")
    command.add_argument("--resume", action="store_true")
    return command


def main(argv=None):
    command = parser()
    args = command.parse_args(argv)
    try:
        return evaluate(args)
    except (ValueError, OSError, semantic.SemanticScoreError, semantic.mechanical.ScoreError,
            judge_runner.JudgeRunError) as exc:
        command.exit(1, f"Error: {judge_runner.redact(str(exc))}\n")


if __name__ == "__main__":
    raise SystemExit(main())
