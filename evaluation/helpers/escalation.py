"""Escalation for Luna judge runs: further attempts after a paper exhausts its normal attempt budget.

Protocol (shared by every judge runner): after the normal budget is exhausted without a valid judgment, make
``extra_luna_attempts`` more attempts with the same Luna judge model, then up to ``fallback_attempts`` attempts with
the fallback model. Prompt, response schema and validator are unchanged; the first valid judgment wins.

Everything is written under the escalation directory; the run's existing attempts and judgments are never modified:

    escalation/judgments/<paper>.json                     accepted judgment (if any)
    escalation/records/<paper>.json                       attempt chain, accepted_by_model, resolved flag
    escalation/events/<paper>.<stage>.attempt-<n>.jsonl   transport trace of each attempt
    escalation/logs/<paper>.<stage>.attempt-<n>.log       transport log of each attempt
    escalation/attempt_errors/<paper>.<stage>.attempt-<n>.json
    escalation/failed_raw/<paper>.<stage>.attempt-<n>.json.partial
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Awaitable, Callable

RECORD_SCHEMA = "benchmark-q1-semantic-judge-escalation-record-v1"
DIRECTORIES = ("judgments", "records", "events", "logs", "attempt_errors", "failed_raw")


def summarize_judges(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Count papers with accepted, unresolved, or unnecessary judgments."""
    accepted = 0
    unresolved, not_required = [], []
    for report in reports:
        paper = report["paper_id"]
        status = report.get("judgment_status")
        if status == "missing":
            unresolved.append(paper)
        elif status == "not_required":
            not_required.append(paper)
        elif status == "complete":
            accepted += 1
        else:
            raise ValueError(f"Unknown judgment status for {paper}: {status}")
    return {"paper_count": len(reports), "accepted_judgments": accepted,
            "unresolved_papers": sorted(unresolved),
            "unresolved": len(unresolved),
            "no_judgment_required_papers": sorted(not_required),
            "no_judgment_required": len(not_required),
            "scope": "selected paper cohort; accepted per-paper judgments before gold-issue filters"}

# run_attempt(model, event_path, log_path, partial_path) -> (validated response, served model);
# it raises on any transport, trace or validation failure.
RunAttempt = Callable[[str, Path, Path, Path], Awaitable[tuple[dict[str, Any], str]]]


class EscalationError(RuntimeError):
    """Escalation records are inconsistent."""


_SECRETS: set[str] = set()


def register_secret(value: str | None) -> None:
    """Remember a credential so that redact() removes it from any text that is written or printed."""
    if value:
        _SECRETS.add(value)


def redact(text: object) -> str:
    """Return str(text) with every registered secret replaced; use before writing or printing any error."""
    result = str(text)
    for secret in sorted(_SECRETS, key=len, reverse=True):
        result = result.replace(secret, "[REDACTED]")
    return result


def stages(*, luna_model: str, extra_luna_attempts: int, fallback_model: str,
           fallback_attempts: int) -> list[dict[str, Any]]:
    return [
        {"stage": "luna-extra", "model": luna_model, "attempts": extra_luna_attempts},
        {"stage": "fallback", "model": fallback_model, "attempts": fallback_attempts},
    ]


def attempt_paths(escalation_dir: Path, paper_id: str, stage: str, attempt: int) -> dict[str, Path]:
    stem = f"{paper_id}.{stage}.attempt-{attempt}"
    return {
        "event": escalation_dir / "events" / f"{stem}.jsonl",
        "log": escalation_dir / "logs" / f"{stem}.log",
        "error": escalation_dir / "attempt_errors" / f"{stem}.json",
        "failed_raw": escalation_dir / "failed_raw" / f"{stem}.json.partial",
    }


def existing_attempts(escalation_dir: Path, paper_id: str, stage: str) -> int:
    events = sorted((escalation_dir / "events").glob(f"{paper_id}.{stage}.attempt-*.jsonl"))
    logs = sorted((escalation_dir / "logs").glob(f"{paper_id}.{stage}.attempt-*.log"))
    expected = {f"{paper_id}.{stage}.attempt-{n}" for n in range(1, len(events) + 1)}
    if {p.stem for p in events} != {p.stem for p in logs} or {p.stem for p in events} != expected:
        raise EscalationError(f"incomplete escalation attempt records for {paper_id} ({stage})")
    return len(events)


def _write_json_atomic(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp",
                                         delete=False, encoding="utf-8")
    with handle:
        json.dump(document, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    os.replace(handle.name, path)


def _chain_from_disk(escalation_dir: Path, paper_id: str, plan: list[dict[str, Any]]) -> list[dict[str, Any]]:
    chain = []
    for stage in plan:
        for attempt in range(1, existing_attempts(escalation_dir, paper_id, stage["stage"]) + 1):
            error_path = attempt_paths(escalation_dir, paper_id, stage["stage"], attempt)["error"]
            if error_path.exists():
                error = json.loads(error_path.read_text(encoding="utf-8"))
                chain.append({"stage": stage["stage"], "attempt": attempt, "requested_model": stage["model"],
                              "status": "failed", "error_type": error.get("error_type"),
                              "error": redact(error.get("error"))})
            else:
                chain.append({"stage": stage["stage"], "attempt": attempt, "requested_model": stage["model"],
                              "status": "interrupted"})
    return chain


def load_record(escalation_dir: Path, paper_id: str) -> dict[str, Any] | None:
    path = escalation_dir / "records" / f"{paper_id}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


async def escalate_paper(
    *,
    paper_id: str,
    escalation_dir: Path,
    plan: list[dict[str, Any]],
    run_attempt: RunAttempt,
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    log: Callable[[str], None] = lambda message: print(message, flush=True),
) -> dict[str, Any]:
    """Run the escalation stages for one paper; return its record (first valid judgment wins)."""
    judgment_path = escalation_dir / "judgments" / f"{paper_id}.json"
    record_path = escalation_dir / "records" / f"{paper_id}.json"
    if judgment_path.exists():
        record = load_record(escalation_dir, paper_id)
        if not record or not record.get("resolved"):
            raise EscalationError(f"escalated judgment for {paper_id} has no resolved record")
        return record
    for directory in DIRECTORIES:
        (escalation_dir / directory).mkdir(parents=True, exist_ok=True)
    chain = _chain_from_disk(escalation_dir, paper_id, plan)
    record: dict[str, Any] = {"schema_version": RECORD_SCHEMA, "paper_id": paper_id, "plan": plan,
                              "resolved": False, "accepted_by_model": None, "chain": chain}
    remaining = [(stage, attempt) for stage in plan
                 for attempt in range(existing_attempts(escalation_dir, paper_id, stage["stage"]) + 1,
                                      stage["attempts"] + 1)]
    for index, (stage, attempt) in enumerate(remaining):
        paths = attempt_paths(escalation_dir, paper_id, stage["stage"], attempt)
        if paths["event"].exists() or paths["log"].exists():
            raise EscalationError(f"refusing to overwrite escalation attempt records for {paper_id}")
        partial = Path(tempfile.NamedTemporaryFile(
            prefix=f".{paper_id}.{stage['stage']}.attempt-{attempt}.", suffix=".json.partial",
            dir=escalation_dir / "judgments", delete=False).name)
        log(redact(f"ESCALATE {paper_id} {stage['stage']} attempt {attempt}/{stage['attempts']} model={stage['model']}"))
        started = datetime.now(timezone.utc).isoformat()
        try:
            response, served_model = await run_attempt(stage["model"], paths["event"], paths["log"], partial)
            del response  # validated by run_attempt; the bytes on disk are the judgment
            os.replace(partial, judgment_path)
        except asyncio.CancelledError:
            if partial.exists():
                os.replace(partial, paths["failed_raw"])
            raise
        except Exception as exc:  # recorded, then the next attempt or stage runs
            if partial.exists():
                if partial.stat().st_size:
                    os.replace(partial, paths["failed_raw"])
                else:
                    partial.unlink()
            message = redact(exc)
            error = {"paper_id": paper_id, "stage": stage["stage"], "attempt": attempt,
                     "requested_model": stage["model"], "error_type": type(exc).__name__, "error": message,
                     "started_at": started, "finished_at": datetime.now(timezone.utc).isoformat()}
            paths["error"].write_text(json.dumps(error, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            chain.append({"stage": stage["stage"], "attempt": attempt, "requested_model": stage["model"],
                          "status": "failed", "error_type": type(exc).__name__, "error": message})
            _write_json_atomic(record_path, record)
            if index + 1 < len(remaining):  # no pause after the final attempt
                await sleep(min(60, 5 * 2 ** (attempt - 1)))
            continue
        chain.append({"stage": stage["stage"], "attempt": attempt, "requested_model": stage["model"],
                      "served_model": served_model, "status": "accepted"})
        record.update(resolved=True, accepted_by_model=served_model, accepted_requested_model=stage["model"],
                      accepted_stage=stage["stage"], accepted_attempt=attempt,
                      judgment=str(judgment_path.relative_to(escalation_dir)))
        _write_json_atomic(record_path, record)
        log(redact(f"RESOLVED {paper_id} by {served_model} ({stage['stage']} attempt {attempt})"))
        return record
    record["exhausted"] = True
    _write_json_atomic(record_path, record)
    log(f"UNRESOLVED {paper_id} after escalation")
    return record
