#!/usr/bin/env python3
"""Run a Luna Q1 semantic issue-coverage judge policy through the Codex CLI or the OpenAI Responses API.

``--judge-runner`` selects only the transport (``codex``, the default, or ``api``). Policy, prompt, response
schema, validator, attempt budget and escalation are identical for both. When a paper exhausts its attempt
budget without a valid judgment, the shared escalation (``escalation.py``) runs automatically unless
``--no-escalation`` is given. ``--resolve-unresolved`` runs only that escalation on an already finished run.
Escalation always uses the Responses API, including after Codex CLI attempts.
"""

from __future__ import annotations

import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

import escalation as escalation_lib
import judge_contract as semantic


SCRIPT_PATH = Path(__file__).resolve()
SEMANTIC_DIR = SCRIPT_PATH.parent
REPO_ROOT = SEMANTIC_DIR.parent.parent
DEFAULT_SCHEMA = REPO_ROOT / "evaluation" / "judge_schema.json"
# The public release includes the final matching policy only.
POLICIES = {
    "Luna.8": {
        "prompt": REPO_ROOT / "evaluation" / "judge_prompt.txt",
        "revision": "20260930-minimal-luna5-clarification-luna8",
    },
}
JUDGE_RUNNERS = {"codex": "codex_exec_chatgpt_auth", "api": "openai_responses_api"}
CHATGPT_APP_CODEX = Path("/Applications/ChatGPT.app/Contents/Resources/codex")
RUN_MANIFEST_SCHEMA = "benchmark-q1-semantic-judge-run-manifest-v1"
ESCALATION_MANIFEST_SCHEMA = "benchmark-q1-semantic-judge-escalation-manifest-v1"
DEFAULT_API_BASE_URL = "https://api.openai.com/v1"
API_RETRYABLE_HTTP = {408, 409, 429, 500, 502, 503, 504}
API_SCHEMA_NAME = "q1_luna5_judgment"
UNSAFE_KEY_CHARACTERS = re.compile(r"[\s\x00-\x1f\x7f]")
redact = escalation_lib.redact


class JudgeRunError(RuntimeError):
    """The judge run configuration or an individual response is invalid."""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
                                     allow_abbrev=False)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--paper-id",
        action="append",
        help="Judge one paper; repeat for several papers.",
    )
    selection.add_argument(
        "--all",
        action="store_true",
        help="Judge every ready paper in the frozen input manifest.",
    )
    parser.add_argument("--input-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--run-name",
        help="Judge run name (required, except with --resolve-unresolved, which reads it from the run manifest).",
    )
    parser.add_argument(
        "--policy", choices=sorted(POLICIES),
        help="Judge policy; selects the prompt and is recorded in the run manifest (required, except with "
             "--resolve-unresolved, which defaults to the source run's policy).",
    )
    parser.add_argument(
        "--judge-runner", choices=sorted(JUDGE_RUNNERS), default="codex",
        help="Transport: codex (Codex CLI with ChatGPT login, default) or api (OpenAI Responses API, "
             "key from OPENAI_API_KEY). Everything else is identical.",
    )
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--model", help="Judge model (default: gpt-5.6-luna; with --resolve-unresolved, the source run's).")
    parser.add_argument("--reasoning-effort",
                        help="Reasoning effort (default: medium; with --resolve-unresolved, the source run's).")
    parser.add_argument(
        "--codex-bin",
        type=Path,
        help="Codex CLI path (codex runner); otherwise use PATH or the ChatGPT app bundle.",
    )
    parser.add_argument(
        "--base-url", default=os.environ.get("OPENAI_BASE_URL", DEFAULT_API_BASE_URL),
        help="Responses API base URL (api runner).",
    )
    parser.add_argument("--max-output-tokens", type=int, default=16384, help="api runner (default: 16384).")
    parser.add_argument(
        "--api-transport-retries", type=int, default=4,
        help="api runner: retries of HTTP 408/409/429/5xx, timeouts and connection errors within one attempt "
             "(like the Codex CLI's own stream retries); they do not consume the attempt budget (default: 4).",
    )
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument(
        "--retries", type=int, default=2,
        help="Maximum retries after failure per paper, including saved attempts on resume (default: 2).",
    )
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue only when the saved manifest exactly matches this configuration.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the complete run and selection without writing or calling a model.",
    )
    escalation = parser.add_argument_group("escalation after the attempt budget is exhausted (default: on)")
    escalation.add_argument("--no-escalation", action="store_true", help="Disable escalation.")
    escalation.add_argument("--extra-luna-attempts", type=int, default=1,
                            help="Further attempts with --model (default: 1).")
    escalation.add_argument("--fallback-model", default="gpt-6.1-sol", help="Fallback judge model (default: gpt-6.1-sol).")
    escalation.add_argument("--fallback-attempts", type=int, default=3, help="Fallback attempts (default: 3).")
    escalation.add_argument(
        "--resolve-unresolved", action="store_true",
        help="Run only the escalation on an already finished run in --output-dir (this runner's layout or the "
             "2026-09-24 Responses-API layout) for papers that exhausted its attempt budget.",
    )
    escalation.add_argument(
        "--escalation-dir", type=Path,
        help="Escalation output directory (default: <output-dir>/escalation); use for read-only published runs.",
    )
    escalation.add_argument(
        "--path-root", type=Path,
        help="Directory against which relative paths in the input manifest are resolved (default: repository root).",
    )
    args = parser.parse_args(argv)
    if not args.resolve_unresolved:
        if args.policy is None:
            parser.error("--policy is required")
        args.model = args.model or "gpt-5.6-luna"
        args.reasoning_effort = args.reasoning_effort or "medium"
    args.prompt = POLICIES[args.policy]["prompt"] if args.policy else None
    args.escalation = not args.no_escalation
    return args


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def resolve_codex_bin(explicit: Path | None) -> Path:
    candidates: list[Path] = []
    if explicit is not None:
        candidates.append(explicit.expanduser())
    path_match = shutil.which("codex")
    if path_match:
        candidates.append(Path(path_match))
    candidates.append(CHATGPT_APP_CODEX)
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.is_file() and os.access(resolved, os.X_OK):
            return resolved
    raise JudgeRunError(
        "Codex CLI was not found; pass --codex-bin /absolute/path/to/codex"
    )


def codex_version(codex_bin: Path) -> str:
    process = subprocess.run(
        [str(codex_bin), "--version"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    lines = (process.stdout or process.stderr).strip().splitlines()
    if process.returncode != 0 or not lines:
        raise JudgeRunError(f"could not determine Codex version from {codex_bin}")
    return lines[-1]


def load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise JudgeRunError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise JudgeRunError(f"{label} must be a JSON object: {path}")
    return document


def escalation_settings(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "enabled": args.escalation,
        "selection": "first_valid",
        "stages": escalation_lib.stages(
            luna_model=args.model, extra_luna_attempts=args.extra_luna_attempts,
            fallback_model=args.fallback_model, fallback_attempts=args.fallback_attempts,
        ) if args.escalation else [],
        "location": "escalation/",
        "judge_runner": "api",
        "backend": "openai_responses_api",
    }


def transport_settings(args: argparse.Namespace, codex_bin: Path | None, cli_version: str | None) -> dict[str, Any]:
    if args.judge_runner == "codex":
        return {
            "actual_output_token_limit": "Codex CLI managed",
            "codex": {
                "path": str(codex_bin),
                "version": cli_version,
                "isolation": {
                    "ephemeral": True,
                    "ignore_user_config": True,
                    "memories_disabled": True,
                    "multi_agent_disabled": True,
                    "sandbox": "read-only",
                    "empty_working_directory": True,
                },
            },
        }
    return {
        "actual_output_token_limit": args.max_output_tokens,
        "api": {
            "base_url": args.base_url.rstrip("/"),
            "endpoint": "/responses",
            "max_output_tokens": args.max_output_tokens,
            "store": False,
            "tools": [],
            "service_tier": "default",
            "text_format": {"type": "json_schema", "name": API_SCHEMA_NAME, "strict": True},
            "transport_retries": args.api_transport_retries,
            "transport_retry_http_codes": sorted(API_RETRYABLE_HTTP),
            "authorization": "OPENAI_API_KEY from the environment; never written",
        },
    }


def configuration_payload(
    args: argparse.Namespace,
    *,
    input_manifest_path: Path,
    input_manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    prompt_path: Path,
    schema_path: Path,
    codex_bin: Path | None,
    cli_version: str | None,
) -> dict[str, Any]:
    ready_rows = sorted((row for row in rows if row.get("status") == "ready"), key=lambda row: row["paper_id"])
    return {
        "schema_version": RUN_MANIFEST_SCHEMA,
        "judge_run_name": args.run_name,
        "question_id": semantic.QUESTION_ID,
        "judge_policy_version": args.policy,
        "judge_policy_revision": POLICIES[args.policy]["revision"],
        "prediction_set_name": input_manifest["prediction_set_name"],
        "judge_runner": args.judge_runner,
        "backend": JUDGE_RUNNERS[args.judge_runner],
        "model": args.model,
        "reasoning_effort": args.reasoning_effort,
        "service_tier": "default",
        "retry_policy": {
            "version": "first-valid-v1",
            "budget_scope": "saved_run",
            "budget_extendable_on_resume": True,
            "request": "unchanged_original_prompt",
            "selection": "first_valid",
        },
        "escalation": escalation_settings(args),
        "timeout_seconds": args.timeout_seconds,
        "luna_api_max_output_tokens_reference": 12000,
        **transport_settings(args, codex_bin, cli_version),
        "input_manifest": {
            "path": semantic.portable_path(input_manifest_path),
            "sha256": semantic.sha256_path(input_manifest_path),
        },
        "prompt": {
            "path": semantic.portable_path(prompt_path),
            "sha256": semantic.sha256_path(prompt_path),
        },
        "response_schema": {
            "path": semantic.portable_path(schema_path),
            "sha256": semantic.sha256_path(schema_path),
        },
        "runner": {
            "path": semantic.portable_path(SCRIPT_PATH),
            "sha256": semantic.sha256_path(SCRIPT_PATH),
        },
        "escalation_module": {
            "path": semantic.portable_path(Path(escalation_lib.__file__).resolve()),
            "sha256": semantic.sha256_path(Path(escalation_lib.__file__).resolve()),
        },
        "validator": {
            "path": semantic.portable_path(Path(semantic.__file__).resolve()),
            "sha256": semantic.sha256_path(Path(semantic.__file__).resolve()),
        },
        "ready_paper_count": len(ready_rows),
        "ready_inputs": [
            {
                "paper_id": row["paper_id"],
                "input_sha256": row["input_sha256"],
            }
            for row in ready_rows
        ],
    }


def configuration_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return sha256_bytes(encoded)


def validate_saved_configuration(
    saved: dict[str, Any], configuration: dict[str, Any], expected_hash: str
) -> None:
    saved_configuration = {
        key: value
        for key, value in saved.items()
        if key not in {"configuration_sha256", "created_at"}
    }
    if (
        saved_configuration != configuration
        or saved.get("configuration_sha256") != expected_hash
        or configuration_sha256(saved_configuration) != expected_hash
    ):
        raise JudgeRunError("saved run manifest does not match this configuration")


def prepare_run_manifest(
    args: argparse.Namespace,
    output_dir: Path,
    configuration: dict[str, Any],
) -> dict[str, Any]:
    manifest_path = output_dir / "run_manifest.json"
    expected_hash = configuration_sha256(configuration)
    if args.dry_run:
        if manifest_path.is_file():
            saved = load_object(manifest_path, "run manifest")
            validate_saved_configuration(saved, configuration, expected_hash)
        return {**configuration, "configuration_sha256": expected_hash}

    if args.resume:
        if not manifest_path.is_file():
            raise JudgeRunError("--resume requires an existing run_manifest.json")
        saved = load_object(manifest_path, "run manifest")
        validate_saved_configuration(saved, configuration, expected_hash)
        return saved

    if output_dir.exists():
        raise JudgeRunError(
            f"refusing to replace existing run directory without --resume: {output_dir}"
        )
    output_dir.mkdir(parents=True)
    manifest = {
        **configuration,
        "configuration_sha256": expected_hash,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    semantic.write_json_exclusive(manifest_path, manifest)
    return manifest


def selected_rows(
    args: argparse.Namespace, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    ready = {str(row["paper_id"]): row for row in rows if row.get("status") == "ready"}
    if args.all:
        return [ready[paper_id] for paper_id in sorted(ready)]
    requested = list(args.paper_id or [])
    duplicates = sorted(
        paper_id for paper_id in set(requested) if requested.count(paper_id) > 1
    )
    if duplicates:
        raise JudgeRunError(f"duplicate paper selection: {', '.join(duplicates)}")
    missing = [paper_id for paper_id in requested if paper_id not in ready]
    if missing:
        raise JudgeRunError(
            "selected papers are absent or not ready: " + ", ".join(missing)
        )
    return [ready[paper_id] for paper_id in sorted(requested)]


def build_prompt(
    instructions: str,
    *,
    run_name: str,
    prediction_set_name: str,
    row: dict[str, Any],
) -> str:
    input_json = json.dumps(
        row["input_document"], ensure_ascii=False, indent=2, sort_keys=True
    )
    return (
        "The instructions and JSON evidence packet below are the complete procedure. "
        "Use only their contents, do not use tools, and return only the required JSON.\n\n"
        "<BEGIN_Q1_SEMANTIC_JUDGE_INSTRUCTIONS>\n"
        f"{instructions.rstrip()}\n"
        "<END_Q1_SEMANTIC_JUDGE_INSTRUCTIONS>\n\n"
        f"JUDGE_RUN_NAME: {run_name}\n"
        f"PREDICTION_SET_NAME: {prediction_set_name}\n"
        f"PAPER_ID: {row['paper_id']}\n"
        f"JUDGE_INPUT_SHA256: {row['input_sha256']}\n\n"
        "<BEGIN_Q1_SEMANTIC_JUDGE_INPUT>\n"
        f"{input_json}\n"
        "<END_Q1_SEMANTIC_JUDGE_INPUT>\n"
    )


# ---------------------------------------------------------------------------------------------- Codex transport

def codex_command(
    args: argparse.Namespace,
    codex_bin: Path,
    *,
    working_dir: Path,
    schema_path: Path,
    output_path: Path,
    model: str | None = None,
) -> list[str]:
    return [
        str(codex_bin),
        "exec",
        "--ephemeral",
        "--ignore-user-config",
        "--disable",
        "memories",
        "--disable",
        "multi_agent",
        "--sandbox",
        "read-only",
        "--color",
        "never",
        "--skip-git-repo-check",
        "-C",
        str(working_dir),
        "-c",
        "memories.use_memories=false",
        "-c",
        "memories.generate_memories=false",
        "--model",
        model or args.model,
        "-c",
        f'model_reasoning_effort="{args.reasoning_effort}"',
        "--json",
        "--output-schema",
        str(schema_path),
        "--output-last-message",
        str(output_path),
        "-",
    ]


def validate_no_tool_use(event_path: Path, response: dict[str, Any] | None = None) -> None:
    completed = False
    final_text = None
    for line in event_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise JudgeRunError("Codex event stream contains invalid JSON") from exc
        if not isinstance(event, dict):
            raise JudgeRunError("Codex event stream contains a non-object event")
        completed = completed or event.get("type") == "turn.completed"
        item = event.get("item")
        if isinstance(item, dict) and item.get("type") not in {"agent_message", "reasoning"}:
            raise JudgeRunError(f"judge used a tool or unsupported item: {item.get('type')}")
        if isinstance(item, dict) and item.get("type") == "agent_message":
            final_text = item.get("text")
    if not completed or not isinstance(final_text, str):
        raise JudgeRunError("judge trace does not contain a completed response")
    if response is not None:
        try:
            event_response = json.loads(final_text)
        except json.JSONDecodeError as exc:
            raise JudgeRunError("judge trace final response is not JSON") from exc
        if event_response != response:
            raise JudgeRunError("saved judgment differs from its completed event trace")


async def terminate_process(process: asyncio.subprocess.Process) -> None:
    """Reap a running judge before removing its files or retrying its request."""
    if process.returncode is not None:
        return
    try:
        process.terminate()
    except ProcessLookupError:
        await process.wait()
        return
    try:
        await asyncio.wait_for(process.wait(), timeout=10)
    except asyncio.TimeoutError:
        try:
            process.kill()
        except ProcessLookupError:
            # The child may exit between the timeout and kill.
            pass
        await process.wait()


async def codex_attempt(
    args: argparse.Namespace,
    codex_bin: Path,
    *,
    model: str,
    prompt: str,
    schema: dict[str, Any],
    event_path: Path,
    log_path: Path,
    partial_path: Path,
) -> tuple[dict[str, Any], str]:
    isolated_dir = Path(tempfile.mkdtemp(prefix="q1-semantic-"))
    schema_path = isolated_dir / "response.schema.json"
    schema_path.write_text(json.dumps(schema, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    process = None
    try:
        with event_path.open("wb") as event_handle, log_path.open("wb") as log_handle:
            process = await asyncio.create_subprocess_exec(
                *codex_command(args, codex_bin, working_dir=isolated_dir, schema_path=schema_path,
                               output_path=partial_path, model=model),
                cwd=isolated_dir,
                # The API key is for escalation; primary Codex keeps its own login.
                env={key: value for key, value in os.environ.items() if key != "OPENAI_API_KEY"},
                stdin=asyncio.subprocess.PIPE,
                stdout=event_handle,
                stderr=log_handle,
            )
            try:
                await asyncio.wait_for(process.communicate(prompt.encode("utf-8")), timeout=args.timeout_seconds)
            except asyncio.TimeoutError:
                await terminate_process(process)
                raise JudgeRunError(f"Codex timed out after {args.timeout_seconds}s; see {log_path}")
        if process.returncode != 0:
            raise JudgeRunError(f"Codex exited with status {process.returncode}; see {log_path}")
        response = load_object(partial_path, "Codex final response")
        validate_no_tool_use(event_path, response)
        return response, model
    except asyncio.CancelledError:
        if process is not None:
            await terminate_process(process)
        raise
    finally:
        shutil.rmtree(isolated_dir, ignore_errors=True)


# ---------------------------------------------------------------------------------------------- API transport

def api_schema(schema: Any) -> Any:
    """Equivalent API schema: const -> singleton enum; metadata removed (as in the 2026-09-24 API re-judge)."""
    if isinstance(schema, list):
        return [api_schema(value) for value in schema]
    if not isinstance(schema, dict):
        return schema
    result = {key: api_schema(value) for key, value in schema.items() if key not in ("$schema", "$id", "const")}
    if "const" in schema:
        result["enum"] = [schema["const"]]
    return result


def api_request_body(args: argparse.Namespace, *, model: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": model,
        "reasoning": {"effort": args.reasoning_effort},
        "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        "text": {"format": {"type": "json_schema", "name": API_SCHEMA_NAME, "strict": True,
                            "schema": api_schema(schema)}},
        "max_output_tokens": args.max_output_tokens,
        "store": False,
        "tools": [],
        "service_tier": "default",
    }


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: ANN001
        raise urllib.error.URLError("API redirect refused; use the final API base URL directly")


def post_responses(body: dict[str, Any], base_url: str, api_key: str, timeout: int) -> dict[str, Any]:
    """POST one Responses API request. Headers (including the key) are never returned or written."""
    request = urllib.request.Request(
        base_url.rstrip("/") + "/responses", data=json.dumps(body).encode("utf-8"),
        headers={"Authorization": "Bearer " + api_key, "Content-Type": "application/json"}, method="POST",
    )
    with urllib.request.build_opener(_NoRedirect()).open(request, timeout=timeout) as result:
        return json.load(result)


def extract_api_text(response: dict[str, Any], requested_model: str) -> str:
    if response.get("status") != "completed":
        raise JudgeRunError(f"incomplete API response: {response.get('status')}; {response.get('incomplete_details')}")
    served = str(response.get("model", ""))
    if served != requested_model and not served.startswith(requested_model + "-"):
        raise JudgeRunError(f"unexpected served model: {served}")
    texts = []
    for item in response.get("output", []):
        if item.get("type") == "reasoning":
            continue
        if item.get("type") != "message" or item.get("role") != "assistant" or item.get("status") != "completed":
            raise JudgeRunError(f"judge used a tool or unsupported item: {item.get('type')}")
        for part in item.get("content", []):
            if part.get("type") != "output_text":
                raise JudgeRunError("API response contains a refusal or unexpected message content")
            texts.append(part["text"])
    if not texts:
        raise JudgeRunError("API response has no final text")
    return "".join(texts)


def validate_api_trace(event_path: Path, response: dict[str, Any] | None = None) -> None:
    events = [json.loads(line) for line in event_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    final = [event for event in events if event.get("type") == "api.response"]
    if not final:
        raise JudgeRunError("judge trace does not contain a completed response")
    request = next((event["request"] for event in events if event.get("type") == "api.request"), None)
    if not isinstance(request, dict) or request.get("tools") != [] or request.get("store") is not False:
        raise JudgeRunError("API request in the trace is missing or allows tools/storage")
    text = extract_api_text(final[-1]["response"], str(request.get("model")))
    if response is not None:
        try:
            if json.loads(text) != response:
                raise JudgeRunError("saved judgment differs from its completed API trace")
        except json.JSONDecodeError as exc:
            raise JudgeRunError("judge trace final response is not JSON") from exc


def validate_trace(event_path: Path, response: dict[str, Any] | None = None) -> None:
    first = next((line for line in event_path.read_text(encoding="utf-8").splitlines() if line.strip()), "")
    try:
        kind = str(json.loads(first).get("type", "")) if first else ""
    except (json.JSONDecodeError, AttributeError):
        kind = ""
    (validate_api_trace if kind.startswith("api.") else validate_no_tool_use)(event_path, response)


async def api_attempt(
    args: argparse.Namespace,
    api_key: str,
    *,
    model: str,
    prompt: str,
    schema: dict[str, Any],
    event_path: Path,
    log_path: Path,
    partial_path: Path,
) -> tuple[dict[str, Any], str]:
    body = api_request_body(args, model=model, prompt=prompt, schema=schema)
    with event_path.open("w", encoding="utf-8") as events, log_path.open("w", encoding="utf-8") as log:
        events.write(json.dumps({"type": "api.request", "request": body}, ensure_ascii=False) + "\n")
        events.flush()
        response = None
        for transport_try in range(1, args.api_transport_retries + 2):
            started = time.monotonic()
            try:
                response = await asyncio.to_thread(post_responses, body, args.base_url, api_key, args.timeout_seconds)
                log.write(f"try {transport_try}: HTTP 200 in {time.monotonic() - started:.1f}s\n")
                break
            except urllib.error.HTTPError as exc:
                detail = redact(exc.read().decode("utf-8", errors="replace"))
                log.write(f"try {transport_try}: HTTP {exc.code} in {time.monotonic() - started:.1f}s: {detail[:2000]}\n")
                log.flush()
                if exc.code == 429:
                    print(f"THROTTLE HTTP 429 model={model} try={transport_try}: {detail[:200]}", flush=True)
                if exc.code not in API_RETRYABLE_HTTP or transport_try > args.api_transport_retries:
                    raise JudgeRunError(f"Responses API HTTP {exc.code}: {detail[:500]}") from None
            except (urllib.error.URLError, TimeoutError, socket.timeout, ConnectionError) as exc:
                message = redact(exc)
                log.write(f"try {transport_try}: {type(exc).__name__} after {time.monotonic() - started:.1f}s: {message}\n")
                log.flush()
                if transport_try > args.api_transport_retries:
                    raise JudgeRunError(f"Responses API transport failure: {type(exc).__name__}: {message}") from None
            await asyncio.sleep(min(60, 5 * 2 ** (transport_try - 1)))
        events.write(json.dumps({"type": "api.response", "response": response}, ensure_ascii=False) + "\n")
    text = extract_api_text(response, model)
    partial_path.write_text(text, encoding="utf-8")
    document = load_object(partial_path, "API final response")
    return document, str(response.get("model"))


async def judge_attempt(
    args: argparse.Namespace,
    *,
    codex_bin: Path | None,
    api_key: str | None,
    model: str,
    prompt: str,
    schema: dict[str, Any],
    event_path: Path,
    log_path: Path,
    partial_path: Path,
) -> tuple[dict[str, Any], str]:
    """One judge attempt through the selected transport; returns the (trace-checked) response and served model."""
    if args.judge_runner == "api":
        if not api_key:
            raise JudgeRunError("OPENAI_API_KEY is not set")
        escalation_lib.register_secret(api_key)
        return await api_attempt(args, api_key, model=model, prompt=prompt, schema=schema,
                                 event_path=event_path, log_path=log_path, partial_path=partial_path)
    return await codex_attempt(args, codex_bin, model=model, prompt=prompt, schema=schema,
                               event_path=event_path, log_path=log_path, partial_path=partial_path)


# ---------------------------------------------------------------------------------------------- judging

def validate_existing(
    path: Path,
    *,
    row: dict[str, Any],
    run_name: str,
    prediction_set_name: str,
) -> None:
    response = load_object(path, "existing judgment")
    event_paths = list((path.parent.parent / "events").glob(f"{row['paper_id']}.attempt-*.jsonl"))
    if not event_paths:
        raise JudgeRunError("existing judgment has no event trace")
    latest = max(event_paths, key=lambda value: int(value.stem.rsplit("attempt-", 1)[1]))
    validate_trace(latest, response)
    semantic.validate_judgment(
        response,
        input_document=row["input_document"],
        input_sha256=str(row["input_sha256"]),
        judge_run_name=run_name,
        prediction_set_name=prediction_set_name,
    )


def validate_escalated(
    escalation_dir: Path,
    *,
    row: dict[str, Any],
    run_name: str,
    prediction_set_name: str,
) -> dict[str, Any]:
    paper_id = str(row["paper_id"])
    record = escalation_lib.load_record(escalation_dir, paper_id)
    if not record or not record.get("resolved"):
        raise JudgeRunError(f"escalated judgment for {paper_id} has no resolved record")
    response = load_object(escalation_dir / "judgments" / f"{paper_id}.json", "escalated judgment")
    paths = escalation_lib.attempt_paths(escalation_dir, paper_id, record["accepted_stage"], record["accepted_attempt"])
    validate_trace(paths["event"], response)
    semantic.validate_judgment(response, input_document=row["input_document"], input_sha256=str(row["input_sha256"]),
                               judge_run_name=run_name, prediction_set_name=prediction_set_name)
    return record


def escalation_directory(args: argparse.Namespace, output_dir: Path) -> Path:
    return (args.escalation_dir.expanduser().resolve() if getattr(args, "escalation_dir", None)
            else output_dir / "escalation")


async def escalate_one(
    args: argparse.Namespace,
    *,
    row: dict[str, Any],
    output_dir: Path,
    codex_bin: Path | None,
    api_key: str | None,
    prompt: str,
    schema: dict[str, Any],
    run_name: str,
    prediction_set_name: str,
) -> str:
    # The post used Responses for both extra Luna
    # and fallback attempts, including when the primary judge used Codex.
    api_args = argparse.Namespace(**vars(args))
    api_args.judge_runner = "api"
    api_key = api_key or os.environ.get("OPENAI_API_KEY") or None
    has_attempts = args.extra_luna_attempts > 0 or args.fallback_attempts > 0
    if has_attempts and not api_key:
        raise JudgeRunError("Set OPENAI_API_KEY privately for Responses API escalation")
    if api_key and UNSAFE_KEY_CHARACTERS.search(api_key):
        raise JudgeRunError("OPENAI_API_KEY contains whitespace or control characters")
    escalation_lib.register_secret(api_key)
    escalation_dir = escalation_directory(args, output_dir)
    ensure = getattr(args, "_ensure_escalation_manifest", None)
    if ensure is not None:
        await ensure()

    async def run_attempt(model: str, event_path: Path, log_path: Path, partial_path: Path):
        response, served = await judge_attempt(api_args, codex_bin=None, api_key=api_key, model=model, prompt=prompt,
                                               schema=schema, event_path=event_path, log_path=log_path,
                                               partial_path=partial_path)
        semantic.validate_judgment(response, input_document=row["input_document"],
                                   input_sha256=str(row["input_sha256"]), judge_run_name=run_name,
                                   prediction_set_name=prediction_set_name)
        return response, served

    try:
        record = await escalation_lib.escalate_paper(
            paper_id=str(row["paper_id"]), escalation_dir=escalation_dir,
            plan=escalation_settings(args)["stages"], run_attempt=run_attempt, sleep=asyncio.sleep,
        )
    except escalation_lib.EscalationError as exc:
        raise JudgeRunError(redact(exc)) from exc
    if record.get("resolved"):
        return "escalated"
    raise JudgeRunError(f"no valid judgment for {row['paper_id']} after the attempt budget and escalation")


async def run_one(
    args: argparse.Namespace,
    *,
    row: dict[str, Any],
    output_dir: Path,
    codex_bin: Path | None,
    instructions: str,
    base_schema: dict[str, Any],
    prediction_set_name: str,
    api_key: str | None = None,
) -> str:
    paper_id = str(row["paper_id"])
    final_path = output_dir / "judgments" / f"{paper_id}.json"
    if final_path.exists():
        validate_existing(
            final_path,
            row=row,
            run_name=args.run_name,
            prediction_set_name=prediction_set_name,
        )
        print(f"SKIP {paper_id}: valid judgment already exists", flush=True)
        return "skipped"
    escalation_dir = escalation_directory(args, output_dir)
    if args.escalation and (escalation_dir / "judgments" / f"{paper_id}.json").exists():
        validate_escalated(escalation_dir, row=row, run_name=args.run_name, prediction_set_name=prediction_set_name)
        print(f"SKIP {paper_id}: valid escalated judgment already exists", flush=True)
        return "skipped"
    if args.dry_run:
        print(f"PLAN {paper_id}: {final_path}", flush=True)
        return "planned"

    for directory in ("judgments", "events", "logs", "failed_raw", "attempt_errors"):
        (output_dir / directory).mkdir(parents=True, exist_ok=True)
    prompt = build_prompt(
        instructions,
        run_name=args.run_name,
        prediction_set_name=prediction_set_name,
        row=row,
    )
    schema = semantic.specialized_judgment_schema(
        base_schema,
        run_name=args.run_name,
        prediction_set_name=prediction_set_name,
        input_document=row["input_document"],
        input_sha256=str(row["input_sha256"]),
    )
    escalation_kwargs = dict(row=row, output_dir=output_dir, codex_bin=codex_bin, api_key=api_key, prompt=prompt,
                             schema=schema, run_name=args.run_name, prediction_set_name=prediction_set_name)

    existing_events = sorted((output_dir / "events").glob(f"{paper_id}.attempt-*.jsonl"))
    existing_logs = sorted((output_dir / "logs").glob(f"{paper_id}.attempt-*.log"))
    event_ids = {path.stem for path in existing_events}
    log_ids = {path.stem for path in existing_logs}
    expected_ids = {f"{paper_id}.attempt-{n}" for n in range(1, len(existing_events) + 1)}
    if event_ids != log_ids or event_ids != expected_ids:
        raise JudgeRunError(f"incomplete prior attempt records for {paper_id}")
    prior_attempt_count = len(existing_events)
    max_attempts = args.retries + 1
    if prior_attempt_count >= max_attempts:
        if args.escalation:
            return await escalate_one(args, **escalation_kwargs)
        raise JudgeRunError(
            f"retry budget exhausted for {paper_id}: {prior_attempt_count}/{max_attempts} attempts; "
            "no valid judgment exists; rerun with --resume and a larger --retries to extend it"
        )
    for attempt in range(prior_attempt_count + 1, max_attempts + 1):
        event_path = output_dir / "events" / f"{paper_id}.attempt-{attempt}.jsonl"
        log_path = output_dir / "logs" / f"{paper_id}.attempt-{attempt}.log"
        if event_path.exists() or log_path.exists():
            raise JudgeRunError(f"refusing to overwrite attempt records for {paper_id}")
        partial_handle = tempfile.NamedTemporaryFile(
            prefix=f".{paper_id}.attempt-{attempt}.",
            suffix=".json.partial",
            dir=output_dir / "judgments",
            delete=False,
        )
        partial_path = Path(partial_handle.name)
        partial_handle.close()
        print(f"RUN  {paper_id} attempt {attempt}", flush=True)
        try:
            response, _served = await judge_attempt(
                args, codex_bin=codex_bin, api_key=api_key, model=args.model, prompt=prompt, schema=schema,
                event_path=event_path, log_path=log_path, partial_path=partial_path,
            )
            semantic.validate_judgment(
                response,
                input_document=row["input_document"],
                input_sha256=str(row["input_sha256"]),
                judge_run_name=args.run_name,
                prediction_set_name=prediction_set_name,
            )
            os.replace(partial_path, final_path)
        except asyncio.CancelledError:
            if partial_path.exists():
                os.replace(
                    partial_path,
                    output_dir / "failed_raw" / f"{paper_id}.attempt-{attempt}.json.partial",
                )
            raise
        except Exception as exc:
            failed_path = (
                output_dir
                / "failed_raw"
                / f"{paper_id}.attempt-{attempt}.json.partial"
            )
            if partial_path.exists():
                os.replace(partial_path, failed_path)
            semantic.write_json_exclusive(
                output_dir / "attempt_errors" / f"{paper_id}.attempt-{attempt}.json",
                {"paper_id": paper_id, "attempt": attempt, "max_attempts": max_attempts,
                 "error_type": type(exc).__name__, "error": redact(exc)},
            )
            if attempt >= max_attempts:
                if args.escalation:
                    print(redact(f"EXHAUSTED {paper_id}: {max_attempts} attempts failed ({exc}); escalating"), flush=True)
                    return await escalate_one(args, **escalation_kwargs)
                raise
            delay = min(60, 5 * 2 ** (attempt - 1))
            print(redact(f"RETRY {paper_id}: attempt {attempt}/{max_attempts} failed ({exc}); waiting {delay}s"), flush=True)
            await asyncio.sleep(delay)
            continue
        print(f"DONE {paper_id}: {final_path}", flush=True)
        return "completed"
    raise AssertionError("unreachable")


async def run_many(
    args: argparse.Namespace,
    *,
    rows: list[dict[str, Any]],
    output_dir: Path,
    codex_bin: Path | None,
    instructions: str,
    base_schema: dict[str, Any],
    prediction_set_name: str,
    api_key: str | None = None,
    worker=None,
) -> dict[str, int]:
    semaphore = asyncio.Semaphore(args.max_concurrency)
    worker = worker or run_one

    async def bounded(row: dict[str, Any]) -> tuple[str, str | Exception]:
        async with semaphore:
            try:
                result = await worker(
                    args,
                    row=row,
                    output_dir=output_dir,
                    codex_bin=codex_bin,
                    instructions=instructions,
                    base_schema=base_schema,
                    prediction_set_name=prediction_set_name,
                    api_key=api_key,
                )
                return str(row["paper_id"]), result
            except Exception as exc:
                return str(row["paper_id"]), exc

    tasks = [asyncio.create_task(bounded(row)) for row in rows]
    counts = {"completed": 0, "skipped": 0, "planned": 0, "failed": 0}
    if args.escalation:
        counts["escalated"] = 0
    for task in asyncio.as_completed(tasks):
        paper_id, result = await task
        if isinstance(result, Exception):
            counts["failed"] += 1
            print(redact(f"FAIL {paper_id}: {result}"), file=sys.stderr, flush=True)
        else:
            counts[result] += 1
    return counts


# ---------------------------------------------------------------------------------------------- resolve mode

def finished_run_state(output_dir: Path) -> dict[str, Any]:
    """Describe a finished judge run from its own records: layout, settings and per-paper attempt budgets."""
    manifest = load_object(output_dir / "run_manifest.json", "run manifest")
    if (output_dir / "attempts").is_dir():
        # 2026-09-24 Responses-API layout: attempts/<paper>/<NNN>/{request,response,outcome}.json.
        configuration_path = output_dir.parent / "configuration.json"
        configuration = load_object(configuration_path, "API configuration") if configuration_path.is_file() else {}
        runtime = configuration.get("runtime_sha256") or {}
        prompts = [v for k, v in runtime.items() if Path(k).name.startswith("prompt_luna")]
        schemas = [v for k, v in runtime.items() if Path(k).name.startswith("judgment_luna")]
        policy = manifest.get("judge_policy_version") or (
            "Luna.5" if manifest.get("judge_policy_revision") == semantic.POLICY_REVISION else None)
        recorded = (configuration.get("attempt_budget") or {}).get("max_attempts_per_paper")

        def attempts(paper_id: str) -> int:
            folder = output_dir / "attempts" / paper_id
            return sum(1 for path in folder.glob("[0-9]*") if (path / "response.json").exists()) if folder.is_dir() else 0

        def budget(paper_id: str) -> int | None:
            return recorded
        layout = "responses-api-20260924"
    else:
        prompts = [(manifest.get("prompt") or {}).get("sha256")]
        schemas = [(manifest.get("response_schema") or {}).get("sha256")]
        policy = manifest.get("judge_policy_version")
        recorded = (manifest.get("retry_policy") or {}).get("max_attempts_per_paper")

        def attempts(paper_id: str) -> int:
            return len(list((output_dir / "events").glob(f"{paper_id}.attempt-*.jsonl")))

        def budget(paper_id: str) -> int | None:
            # The runner records each failed attempt's budget; otherwise use the manifest's, if any.
            values = [load_object(path, "attempt error").get("max_attempts")
                      for path in (output_dir / "attempt_errors").glob(f"{paper_id}.attempt-*.json")]
            values = [value for value in values if isinstance(value, int)]
            return max(values) if values else recorded
        layout = "luna-runner"
    settings = {"policy": policy, "prompt_sha256": prompts[0] if len(prompts) == 1 else None,
                "schema_sha256": schemas[0] if len(schemas) == 1 else None,
                "model": manifest.get("model"), "reasoning_effort": manifest.get("reasoning_effort")}
    return {"manifest": manifest, "layout": layout, "settings": settings, "attempts": attempts, "budget": budget}


def apply_source_settings(args: argparse.Namespace, state: dict[str, Any]) -> None:
    """Default to the source run's recorded judge settings and refuse any mismatch."""
    recorded = state["settings"]
    missing = [name for name, value in recorded.items() if not value]
    if missing:
        raise JudgeRunError("source run does not record its " + ", ".join(missing) + "; cannot check settings")
    args.policy = args.policy or recorded["policy"]
    args.model = args.model or recorded["model"]
    args.reasoning_effort = args.reasoning_effort or recorded["reasoning_effort"]
    if args.policy not in POLICIES:
        raise JudgeRunError(f"source run policy {args.policy!r} is not a known Luna policy")
    args.prompt = POLICIES[args.policy]["prompt"]
    requested = {"policy": args.policy, "prompt_sha256": semantic.sha256_path(args.prompt),
                 "schema_sha256": semantic.sha256_path(args.schema.expanduser().resolve()),
                 "model": args.model, "reasoning_effort": args.reasoning_effort}
    mismatched = [f"{name}: run {recorded[name]!r} vs requested {requested[name]!r}"
                  for name in requested if requested[name] != recorded[name]]
    if mismatched:
        raise JudgeRunError("requested settings differ from the source run; refusing to escalate (" +
                            "; ".join(mismatched) + ")")


async def resolve_unresolved(
    args: argparse.Namespace,
    *,
    output_dir: Path,
    input_manifest_path: Path,
    input_manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    instructions: str,
    base_schema: dict[str, Any],
    codex_bin: Path | None,
    cli_version: str | None,
    api_key: str | None,
) -> int:
    state = finished_run_state(output_dir)
    manifest = state["manifest"]
    run_name = str(manifest["judge_run_name"])
    if args.run_name and args.run_name != run_name:
        raise JudgeRunError(f"--run-name {args.run_name} differs from the run manifest ({run_name})")
    args.run_name = run_name
    prediction_set_name = str(input_manifest["prediction_set_name"])
    if manifest.get("prediction_set_name") != prediction_set_name:
        raise JudgeRunError("input manifest prediction set differs from the judge run")
    recorded_inputs = {item["paper_id"]: item["input_sha256"] for item in manifest.get("ready_inputs", [])}
    if not recorded_inputs and (manifest.get("input_manifest") or {}).get("sha256") != semantic.sha256_path(input_manifest_path):
        raise JudgeRunError("input manifest differs from the one recorded in the judge run")
    selection = selected_rows(args, rows)
    unresolved, not_exhausted = [], []
    for row in selection:
        paper_id = str(row["paper_id"])
        if recorded_inputs and recorded_inputs.get(paper_id) != row["input_sha256"]:
            raise JudgeRunError(f"judge input for {paper_id} differs from the judge run")
        if (output_dir / "judgments" / f"{paper_id}.json").exists():
            continue
        budget = state["budget"](paper_id)
        (unresolved if budget is not None and state["attempts"](paper_id) >= budget else not_exhausted).append(row)
    escalation_dir = escalation_directory(args, output_dir)
    print(f"RESOLVE layout={state['layout']} policy={args.policy} model={args.model} effort={args.reasoning_effort} "
          f"unresolved={len(unresolved)} "
          f"not_exhausted={len(not_exhausted)} escalation_dir={escalation_dir}", flush=True)
    if not_exhausted:
        print("NOT EXHAUSTED (left alone): " + ", ".join(str(row["paper_id"]) for row in not_exhausted), flush=True)
    payload = escalation_manifest_payload(args, mode="resolve-unresolved", output_dir=output_dir,
                                          input_manifest_path=input_manifest_path, source_layout=state["layout"],
                                          source_budget="recorded per paper by the source run", codex_bin=codex_bin,
                                          cli_version=cli_version)
    if args.dry_run:
        for row in unresolved:
            print(f"PLAN escalation {row['paper_id']}", flush=True)
        return 0
    prepare_escalation_manifest(escalation_dir, payload)

    async def worker(args, *, row, output_dir, codex_bin, instructions, base_schema, prediction_set_name, api_key):
        judgment = escalation_dir / "judgments" / f"{row['paper_id']}.json"
        if judgment.exists():
            validate_escalated(escalation_dir, row=row, run_name=run_name, prediction_set_name=prediction_set_name)
            return "skipped"
        prompt = build_prompt(instructions, run_name=run_name, prediction_set_name=prediction_set_name, row=row)
        schema = semantic.specialized_judgment_schema(base_schema, run_name=run_name,
                                                      prediction_set_name=prediction_set_name,
                                                      input_document=row["input_document"],
                                                      input_sha256=str(row["input_sha256"]))
        return await escalate_one(args, row=row, output_dir=output_dir, codex_bin=codex_bin, api_key=api_key,
                                  prompt=prompt, schema=schema, run_name=run_name,
                                  prediction_set_name=prediction_set_name)

    counts = await run_many(args, rows=unresolved, output_dir=output_dir, codex_bin=codex_bin,
                            instructions=instructions, base_schema=base_schema,
                            prediction_set_name=prediction_set_name, api_key=api_key, worker=worker)
    summary = {"created_at": datetime.now(timezone.utc).isoformat(), "mode": "resolve-unresolved",
               "source_run": str(output_dir), "source_layout": state["layout"],
               "unresolved_before": [str(row["paper_id"]) for row in unresolved],
               "not_exhausted": [str(row["paper_id"]) for row in not_exhausted],
               "records": {str(row["paper_id"]): escalation_lib.load_record(escalation_dir, str(row["paper_id"]))
                           for row in unresolved},
               "counts": counts}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    (escalation_dir / "summaries").mkdir(parents=True, exist_ok=True)
    semantic.write_json_exclusive(escalation_dir / "summaries" / f"resolve-{stamp}.json", summary)
    print(f"SUMMARY {json.dumps(counts, sort_keys=True)}", flush=True)
    return 1 if counts["failed"] else 0


def escalation_manifest_payload(
    args: argparse.Namespace,
    *,
    mode: str,
    output_dir: Path,
    input_manifest_path: Path,
    source_layout: str,
    source_budget: int | str,
    codex_bin: Path | None,
    cli_version: str | None,
) -> dict[str, Any]:
    primary_runner = args.judge_runner
    args = argparse.Namespace(**vars(args))
    args.judge_runner = "api"
    run_manifest_path = output_dir / "run_manifest.json"
    return {
        "schema_version": ESCALATION_MANIFEST_SCHEMA,
        "mode": mode,
        "source_run": {
            "path": str(output_dir),
            "layout": source_layout,
            "attempt_budget": source_budget,
            "run_manifest_sha256": semantic.sha256_path(run_manifest_path) if run_manifest_path.is_file() else None,
        },
        "judge_run_name": args.run_name,
        "judge_policy_version": args.policy,
        "judge_policy_revision": POLICIES[args.policy]["revision"],
        "primary_judge_runner": primary_runner,
        "judge_runner": args.judge_runner,
        "backend": JUDGE_RUNNERS[args.judge_runner],
        "reasoning_effort": args.reasoning_effort,
        "escalation": escalation_settings(args),
        "timeout_seconds": args.timeout_seconds,
        **transport_settings(args, codex_bin, cli_version),
        "input_manifest": {"path": str(input_manifest_path), "sha256": semantic.sha256_path(input_manifest_path)},
        "prompt": {"path": semantic.portable_path(args.prompt), "sha256": semantic.sha256_path(args.prompt)},
        "response_schema": {"path": semantic.portable_path(args.schema.expanduser().resolve()),
                            "sha256": semantic.sha256_path(args.schema.expanduser().resolve())},
        "runner": {"path": semantic.portable_path(SCRIPT_PATH), "sha256": semantic.sha256_path(SCRIPT_PATH)},
        "escalation_module": {"path": semantic.portable_path(Path(escalation_lib.__file__).resolve()),
                              "sha256": semantic.sha256_path(Path(escalation_lib.__file__).resolve())},
        "validator": {"path": semantic.portable_path(Path(semantic.__file__).resolve()),
                      "sha256": semantic.sha256_path(Path(semantic.__file__).resolve())},
    }


def prepare_escalation_manifest(escalation_dir: Path, payload: dict[str, Any]) -> None:
    path = escalation_dir / "escalation_manifest.json"
    if path.is_file():
        saved = {k: v for k, v in load_object(path, "escalation manifest").items() if k != "created_at"}
        if saved != payload:
            raise JudgeRunError("saved escalation manifest does not match this configuration")
        return
    escalation_dir.mkdir(parents=True, exist_ok=True)
    semantic.write_json_exclusive(path, {**payload, "created_at": datetime.now(timezone.utc).isoformat()})


# ---------------------------------------------------------------------------------------------- main

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if not args.resolve_unresolved and not args.run_name:
            raise JudgeRunError("--run-name is required")
        if args.run_name and not semantic.SAFE_NAME_RE.fullmatch(args.run_name):
            raise JudgeRunError("--run-name must contain only letters, digits, ._- characters")
        if args.max_concurrency < 1:
            raise JudgeRunError("--max-concurrency must be at least 1")
        if args.retries < 0:
            raise JudgeRunError("--retries cannot be negative")
        if args.timeout_seconds < 1 or args.max_output_tokens < 1:
            raise JudgeRunError("--timeout-seconds and --max-output-tokens must be at least 1")
        if min(args.extra_luna_attempts, args.fallback_attempts, args.api_transport_retries) < 0:
            raise JudgeRunError("escalation attempt counts and --api-transport-retries cannot be negative")
        if args.resolve_unresolved and not args.escalation:
            raise JudgeRunError("--resolve-unresolved needs escalation; remove --no-escalation")
        codex_bin = cli_version = api_key = None
        uses_api = args.judge_runner == "api" or (args.escalation and
                   (args.extra_luna_attempts > 0 or args.fallback_attempts > 0))
        if uses_api:
            api_key = os.environ.get("OPENAI_API_KEY") or None
            if api_key and UNSAFE_KEY_CHARACTERS.search(api_key):
                raise JudgeRunError("OPENAI_API_KEY contains whitespace or control characters (for example a "
                                    "trailing newline); remove them and retry")
            escalation_lib.register_secret(api_key)
            if not api_key and not args.dry_run:
                raise JudgeRunError("set OPENAI_API_KEY privately for API judging or API escalation")
            url = urllib.parse.urlsplit(args.base_url.rstrip("/"))
            if url.scheme != "https" or not url.hostname or url.username or url.password or url.query or url.fragment:
                raise JudgeRunError("--base-url must be HTTPS, without credentials, query or fragment")
        if args.path_root is not None:
            semantic.REPO_ROOT = args.path_root.expanduser().resolve()
        output_dir = args.output_dir.expanduser().resolve()
        if args.resolve_unresolved:
            if not (output_dir / "run_manifest.json").is_file():
                raise JudgeRunError(f"--resolve-unresolved needs a finished judge run with run_manifest.json: {output_dir}")
            apply_source_settings(args, finished_run_state(output_dir))
        input_manifest_path = args.input_manifest.expanduser().resolve()
        prompt_path = args.prompt.expanduser().resolve()
        schema_path = args.schema.expanduser().resolve()
        for path, label in (
            (input_manifest_path, "input manifest"),
            (prompt_path, "prompt"),
            (schema_path, "response schema"),
        ):
            if not path.is_file():
                raise JudgeRunError(f"missing {label}: {path}")
        input_manifest, rows = semantic.load_input_manifest(input_manifest_path)
        instructions = prompt_path.read_text(encoding="utf-8")
        base_schema = load_object(schema_path, "response schema")
        if args.judge_runner == "codex":
            codex_bin = resolve_codex_bin(args.codex_bin)
            cli_version = codex_version(codex_bin)
            print(f"CODEX {codex_bin} ({cli_version})", flush=True)
        else:
            print(f"API {args.base_url.rstrip('/')}/responses", flush=True)
        if args.resolve_unresolved:
            return asyncio.run(resolve_unresolved(
                args, output_dir=output_dir, input_manifest_path=input_manifest_path, input_manifest=input_manifest,
                rows=rows, instructions=instructions, base_schema=base_schema, codex_bin=codex_bin,
                cli_version=cli_version, api_key=api_key))
        selection = selected_rows(args, rows)
        configuration = configuration_payload(
            args,
            input_manifest_path=input_manifest_path,
            input_manifest=input_manifest,
            rows=rows,
            prompt_path=prompt_path,
            schema_path=schema_path,
            codex_bin=codex_bin,
            cli_version=cli_version,
        )
        prepare_run_manifest(args, output_dir, configuration)
        if args.escalation and not args.dry_run:
            lock = asyncio.Lock
            payload = escalation_manifest_payload(
                args, mode="automatic", output_dir=output_dir, input_manifest_path=input_manifest_path,
                source_layout="luna-runner", source_budget=args.retries + 1, codex_bin=codex_bin,
                cli_version=cli_version)
            state = {"lock": None}

            async def ensure_manifest() -> None:
                state["lock"] = state["lock"] or lock()
                async with state["lock"]:
                    prepare_escalation_manifest(escalation_directory(args, output_dir), payload)
            args._ensure_escalation_manifest = ensure_manifest
        print(
            f"JUDGE {args.policy} {args.model} effort={args.reasoning_effort} runner={args.judge_runner} "
            f"papers={len(selection)} escalation={'on' if args.escalation else 'off'}",
            flush=True,
        )
        counts = asyncio.run(
            run_many(
                args,
                rows=selection,
                output_dir=output_dir,
                codex_bin=codex_bin,
                instructions=instructions,
                base_schema=base_schema,
                prediction_set_name=str(input_manifest["prediction_set_name"]),
                api_key=api_key,
            )
        )
        print(f"SUMMARY {json.dumps(counts, sort_keys=True)}", flush=True)
        return 1 if counts["failed"] else 0
    except (JudgeRunError, semantic.SemanticScoreError) as exc:
        print(f"Error: {redact(exc)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
