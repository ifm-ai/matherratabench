#!/usr/bin/env python3
"""Build a prompt-ready, fully annotated copy of benchmark-papers.

The input release is never modified.  The derived release makes every
theorem-like result and mechanically identifiable algorithm addressable in the
LaTeX source, replaces duplicated labels by occurrence-specific labels, and
normalizes repeated blank rows outside verbatim-like environments.  It rewrites
only old-source tag fields and derived source-hash metadata in issues.json.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Iterable, Optional


RELEASE_NAME = "benchmark-papers-annotated-v1"
BLANK_LINE_NORMALIZATION = "latex-blank-lines-v1"
OLD_SOURCE_TAG_FIELDS = (
    "primary_affected_tag",
    "failure_location.tag",
    "diagnostic_evidence[].tag",
    "dependent_affected_formal_result_tags[]",
)
KIND_PREFIX = {
    "theorem": "thm",
    "lemma": "lem",
    "proposition": "prop",
    "corollary": "cor",
    "claim": "claim",
    "algorithm": "alg",
}
NAME_KIND = {
    "theorem": "theorem",
    "thm": "theorem",
    "thr": "theorem",
    "th": "theorem",
    "mytheorem": "theorem",
    "lemma": "lemma",
    "lem": "lemma",
    "lm": "lemma",
    "proposition": "proposition",
    "prop": "proposition",
    "propo": "proposition",
    "corollary": "corollary",
    "cor": "corollary",
    "corol": "corollary",
    "coro": "corollary",
    "corollaryc": "corollary",
    "claim": "claim",
    "clm": "claim",
}
STANDARD_FORMAL_ENVS = set(KIND_PREFIX) - {"algorithm"}
BEGIN_DOCUMENT_RE = re.compile(r"\\begin\s*\{\s*document\s*\}")
ANY_ENV_RE = re.compile(
    r"\\(?:begin|end)\s*\{\s*([A-Za-z@][A-Za-z0-9@*:_-]*)\s*\}"
)
ADDED_RE_TEMPLATE = r"^{prefix}:added-(\d+)$"
PROTECTED_BLANK_LINE_ENVIRONMENTS = frozenset(
    {
        "verbatim",
        "verbatim*",
        "Verbatim",
        "BVerbatim",
        "LVerbatim",
        "SaveVerbatim",
        "lstlisting",
        "minted",
        "alltt",
        "filecontents",
        "filecontents*",
        "pycode",
        "pythoncode",
        "sageblock",
        "luacode",
        "luacode*",
        "markdown",
    }
)
PROTECTED_BEGIN_RE = re.compile(
    r"\\begin\s*\{\s*("
    + "|".join(
        sorted(
            (re.escape(name) for name in PROTECTED_BLANK_LINE_ENVIRONMENTS),
            key=len,
            reverse=True,
        )
    )
    + r")\s*\}"
)


@dataclass(frozen=True)
class Edit:
    start: int
    end: int
    replacement: str
    reason: str


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def unescaped_comment_start(line: str) -> int:
    """Return the first active TeX comment marker, or the line length."""

    for index, character in enumerate(line):
        if character != "%":
            continue
        backslashes = 0
        cursor = index - 1
        while cursor >= 0 and line[cursor] == "\\":
            backslashes += 1
            cursor -= 1
        if backslashes % 2 == 0:
            return index
    return len(line)


def protected_environment_blocks(source: str) -> list[str]:
    """Extract protected raw-text blocks for byte-preservation checks."""

    blocks: list[str] = []
    cursor = 0
    while cursor < len(source):
        match = PROTECTED_BEGIN_RE.search(source, cursor)
        while match is not None:
            line_start = source.rfind("\n", cursor, match.start()) + 1
            prefix = source[line_start : match.start()]
            if unescaped_comment_start(prefix) == len(prefix):
                break
            match = PROTECTED_BEGIN_RE.search(source, match.end())
        if match is None:
            break
        environment = match.group(1)
        end_re = re.compile(
            rf"\\end\s*\{{\s*{re.escape(environment)}\s*\}}"
        )
        end = end_re.search(source, match.end())
        if end is None:
            raise ValueError(
                f"Unclosed protected environment {environment!r} at offset {match.start()}"
            )
        blocks.append(source[match.start() : end.end()])
        cursor = end.end()
    return blocks


def normalize_blank_lines(source: str) -> tuple[str, dict[str, int]]:
    """Apply the canonical blank-row policy outside raw-text environments."""

    if "\r" in source:
        raise ValueError("Source contains non-LF line endings")
    before_blank_lines = sum(
        not line.rstrip("\n").strip(" \t") for line in source.splitlines(keepends=True)
    )
    output: list[str] = []
    blank_pending = False
    protected_environment: Optional[str] = None
    protected_end_re: Optional[re.Pattern[str]] = None

    for line in source.splitlines(keepends=True):
        if protected_environment is not None:
            output.append(line)
            assert protected_end_re is not None
            if protected_end_re.search(line):
                protected_environment = None
                protected_end_re = None
            continue

        if not line.rstrip("\n").strip(" \t"):
            blank_pending = True
            continue

        if blank_pending and output:
            output.append("\n")
        blank_pending = False
        output.append(line)

        visible = line[: unescaped_comment_start(line)]
        begin = PROTECTED_BEGIN_RE.search(visible)
        if begin is not None:
            environment = begin.group(1)
            end_re = re.compile(
                rf"\\end\s*\{{\s*{re.escape(environment)}\s*\}}"
            )
            if end_re.search(visible, begin.end()) is None:
                protected_environment = environment
                protected_end_re = end_re

    if protected_environment is not None:
        raise ValueError(f"Unclosed protected environment {protected_environment!r}")

    normalized = "".join(output).rstrip("\n") + "\n"
    after_blank_lines = sum(
        not line.rstrip("\n").strip(" \t")
        for line in normalized.splitlines(keepends=True)
    )
    if normalize_blank_lines_once(normalized) != normalized:
        raise ValueError("Blank-line normalization is not idempotent")
    if nonblank_lines(source) != nonblank_lines(normalized):
        raise ValueError("Blank-line normalization changed a nonblank source line")
    if protected_environment_blocks(source) != protected_environment_blocks(normalized):
        raise ValueError("Blank-line normalization changed a protected environment")
    return normalized, {
        "blank_lines_before": before_blank_lines,
        "blank_lines_after": after_blank_lines,
        "blank_lines_removed": before_blank_lines - after_blank_lines,
    }


def normalize_blank_lines_once(source: str) -> str:
    """Internal normalization pass used to verify idempotence."""

    output: list[str] = []
    blank_pending = False
    protected_environment: Optional[str] = None
    protected_end_re: Optional[re.Pattern[str]] = None
    for line in source.splitlines(keepends=True):
        if protected_environment is not None:
            output.append(line)
            assert protected_end_re is not None
            if protected_end_re.search(line):
                protected_environment = None
                protected_end_re = None
            continue
        if not line.rstrip("\n").strip(" \t"):
            blank_pending = True
            continue
        if blank_pending and output:
            output.append("\n")
        blank_pending = False
        output.append(line)
        visible = line[: unescaped_comment_start(line)]
        begin = PROTECTED_BEGIN_RE.search(visible)
        if begin is not None:
            environment = begin.group(1)
            end_re = re.compile(
                rf"\\end\s*\{{\s*{re.escape(environment)}\s*\}}"
            )
            if end_re.search(visible, begin.end()) is None:
                protected_environment = environment
                protected_end_re = end_re
    return "".join(output).rstrip("\n") + "\n"


def nonblank_lines(source: str) -> list[str]:
    return [line for line in source.splitlines() if line.strip(" \t")]


def load_identifier_module(input_root: Path) -> Any:
    module_path = input_root / "build_identifier_addendum.py"
    spec = importlib.util.spec_from_file_location(
        "benchmark_identifier_authority", module_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def declaration_titles(source: str, identifier_module: Any) -> dict[str, list[str]]:
    result: dict[str, list[str]] = defaultdict(list)
    for match in identifier_module.NEWTHEOREM_RE.finditer(source):
        result[match.group("env")].append(match.group("title"))
    for match in identifier_module.NEWMDTHEOREM_RE.finditer(source):
        result[match.group("env")].append(match.group("title"))
    for match in identifier_module.DECLARETHEOREM_RE.finditer(source):
        result[match.group("env")].append(match.group("options") or "")
    return result


def wrapper_bases(source: str, identifier_module: Any) -> dict[str, str]:
    return {
        match.group("name"): match.group("base")
        for match in identifier_module.SIMPLE_ENV_WRAPPER_RE.finditer(source)
    }


def semantic_kind(
    environment: str,
    *,
    titles: dict[str, list[str]],
    wrappers: dict[str, str],
    seen: frozenset[str] = frozenset(),
) -> str:
    normalized = environment.rstrip("*").lower()
    if normalized in NAME_KIND:
        return NAME_KIND[normalized]
    for title in titles.get(environment, []):
        words = set(re.findall(r"[A-Za-z]+", title.lower()))
        for kind in ("corollary", "proposition", "theorem", "lemma", "claim"):
            if kind in words:
                return kind
    if environment in wrappers and environment not in seen:
        return semantic_kind(
            wrappers[environment],
            titles=titles,
            wrappers=wrappers,
            seen=seen | {environment},
        )
    raise ValueError(f"Cannot determine semantic kind for environment {environment!r}")


def occupied_environment_names(source: str, identifier_module: Any) -> set[str]:
    names = set(ANY_ENV_RE.findall(identifier_module.mask_comments(source)))
    for pattern in (
        identifier_module.NEWTHEOREM_RE,
        identifier_module.NEWMDTHEOREM_RE,
        identifier_module.DECLARETHEOREM_RE,
        identifier_module.SIMPLE_ENV_WRAPPER_RE,
    ):
        for match in pattern.finditer(source):
            group = "name" if "name" in match.groupdict() else "env"
            names.add(match.group(group))
    return names


def build_environment_map(
    source: str, inventory: Any, identifier_module: Any
) -> tuple[dict[str, str], dict[str, str]]:
    """Return custom-name -> readable-name and custom-name -> semantic-kind."""

    titles = declaration_titles(source, identifier_module)
    wrappers = wrapper_bases(source, identifier_module)
    first_position: dict[str, int] = {}
    kinds: dict[str, str] = {}
    for block in inventory.formal_blocks:
        first_position.setdefault(block.environment, block.start)
        kinds[block.environment] = semantic_kind(
            block.environment, titles=titles, wrappers=wrappers
        )

    # Uppercase Algorithm is a custom theorem-like algorithm environment in
    # this corpus.  algorithmic, procedure, and lstlisting remain unchanged
    # because they have distinct syntax and semantics.
    for block in inventory.algorithm_blocks:
        if block.environment == "Algorithm":
            first_position.setdefault(block.environment, block.start)
            kinds[block.environment] = "algorithm"

    occupied = occupied_environment_names(source, identifier_module)
    assigned: set[str] = set()
    result: dict[str, str] = {}
    for environment in sorted(first_position, key=lambda item: first_position[item]):
        kind = kinds[environment]
        if environment == kind:
            continue
        if kind not in occupied and kind not in assigned:
            target = kind
        else:
            suffix = 2
            while f"{kind}{suffix}" in occupied or f"{kind}{suffix}" in assigned:
                suffix += 1
            target = f"{kind}{suffix}"
        result[environment] = target
        assigned.add(target)
        occupied.add(target)
    return result, kinds


def alias_block(environment_pairs: list[tuple[str, str]]) -> str:
    if not environment_pairs:
        return ""
    lines: list[str] = []
    for original, annotated in environment_pairs:
        lines.extend(
            [
                rf"\expandafter\let\csname {annotated}\expandafter\endcsname\csname {original}\endcsname",
                rf"\expandafter\let\csname end{annotated}\expandafter\endcsname\csname end{original}\endcsname",
            ]
        )
    return "\n".join(lines) + "\n"


def environment_declaration_ends(
    source: str, environment: str, identifier_module: Any
) -> list[int]:
    ends: list[int] = []
    for pattern in (
        identifier_module.NEWTHEOREM_RE,
        identifier_module.NEWMDTHEOREM_RE,
        identifier_module.DECLARETHEOREM_RE,
        identifier_module.SIMPLE_ENV_WRAPPER_RE,
    ):
        for match in pattern.finditer(source):
            group = "name" if "name" in match.groupdict() else "env"
            if match.group(group) == environment:
                end = match.end()
                if pattern is identifier_module.NEWTHEOREM_RE:
                    # \newtheorem{env}{Title}[within] has a trailing counter
                    # option that is not part of the frozen discovery regex.
                    end = optional_argument_end(source, end)
                ends.append(end)
    return ends


def optional_argument_end(source: str, start: int) -> int:
    cursor = start
    while cursor < len(source) and source[cursor].isspace():
        cursor += 1
    if cursor >= len(source) or source[cursor] != "[":
        return start
    bracket_depth = 0
    brace_depth = 0
    index = cursor
    while index < len(source):
        char = source[index]
        escaped = False
        back = index - 1
        while back >= 0 and source[back] == "\\":
            escaped = not escaped
            back -= 1
        if not escaped:
            if char == "{":
                brace_depth += 1
            elif char == "}" and brace_depth:
                brace_depth -= 1
            elif char == "[" and brace_depth == 0:
                bracket_depth += 1
            elif char == "]" and brace_depth == 0:
                bracket_depth -= 1
                if bracket_depth == 0:
                    return index + 1
        index += 1
    raise ValueError(f"Unclosed optional environment argument at offset {cursor}")


def block_boundary_edits(block: Any, target_environment: str) -> tuple[list[Edit], int]:
    edits: list[Edit] = []
    if block.wrapper_kind == "environment":
        opener = re.match(
            rf"\\begin\s*\{{\s*{re.escape(block.environment)}\s*\}}", block.text
        )
        closer_matches = list(
            re.finditer(
                rf"\\end\s*\{{\s*{re.escape(block.environment)}\s*\}}",
                block.text,
            )
        )
        if opener is None or not closer_matches:
            raise ValueError(
                f"Could not parse boundaries of {block.environment!r} at {block.start}"
            )
        closer = closer_matches[-1]
        if target_environment != block.environment:
            edits.append(
                Edit(
                    block.start + opener.start(),
                    block.start + opener.end(),
                    rf"\begin{{{target_environment}}}",
                    "normalize_environment_begin",
                )
            )
            edits.append(
                Edit(
                    block.start + closer.start(),
                    block.start + closer.end(),
                    rf"\end{{{target_environment}}}",
                    "normalize_environment_end",
                )
            )
        return edits, block.start + opener.end()

    opener = re.match(r"\\[A-Za-z@]+", block.text)
    commands = list(re.finditer(r"\\[A-Za-z@]+", block.text))
    if opener is None or not commands:
        raise ValueError(f"Could not parse command wrapper at {block.start}")
    closer = commands[-1]
    edits.extend(
        [
            Edit(
                block.start + opener.start(),
                block.start + opener.end(),
                rf"\begin{{{target_environment}}}",
                "expand_environment_wrapper_begin",
            ),
            Edit(
                block.start + closer.start(),
                block.start + closer.end(),
                rf"\end{{{target_environment}}}",
                "expand_environment_wrapper_end",
            ),
        ]
    )
    return edits, block.start + opener.end()


def allocate_added_labels(
    source_labels: Iterable[str], blocks: list[tuple[int, str, str]]
) -> dict[str, str]:
    """Map generated canonical ids to natural labels in source order."""

    labels = set(source_labels)
    next_number: dict[str, int] = {}
    for prefix in set(KIND_PREFIX.values()):
        pattern = re.compile(ADDED_RE_TEMPLATE.format(prefix=re.escape(prefix)))
        used = [int(match.group(1)) for label in labels if (match := pattern.match(label))]
        next_number[prefix] = max(used, default=0) + 1

    result: dict[str, str] = {}
    for _position, canonical_id, kind in sorted(blocks):
        prefix = KIND_PREFIX[kind]
        number = next_number[prefix]
        candidate = f"{prefix}:added-{number}"
        while candidate in labels:
            number += 1
            candidate = f"{prefix}:added-{number}"
        result[canonical_id] = candidate
        labels.add(candidate)
        next_number[prefix] = number + 1
    return result


def apply_edits(source: str, edits: list[Edit]) -> str:
    ordered = sorted(edits, key=lambda edit: (edit.start, edit.end))
    previous_end = 0
    for edit in ordered:
        if edit.start < previous_end:
            raise ValueError(
                f"Overlapping edits near {edit.start}: {edit.reason} begins before {previous_end}"
            )
        if edit.start > edit.end:
            raise ValueError(f"Invalid edit span {edit.start}:{edit.end}")
        previous_end = max(previous_end, edit.end)
    transformed = source
    for edit in reversed(ordered):
        transformed = transformed[: edit.start] + edit.replacement + transformed[edit.end :]
    return transformed


def transform_source(
    source: str, inventory: Any, identifier_module: Any
) -> tuple[str, dict[str, str], dict[str, Any]]:
    source_labels = [item.value for item in inventory.labels]
    edits: list[Edit] = []
    identifier_map: dict[str, str] = {}
    label_records: list[dict[str, Any]] = []

    # Every duplicated source label becomes occurrence-specific.  A single
    # compatibility alias at the last occurrence preserves LaTeX's previous
    # last-definition-wins behavior for existing \ref{raw-label} commands.
    duplicate_groups: dict[str, list[Any]] = defaultdict(list)
    for occurrence in inventory.labels:
        if occurrence.occurrence_count > 1:
            duplicate_groups[occurrence.value].append(occurrence)
    for raw_label, occurrences in duplicate_groups.items():
        for occurrence in occurrences:
            new_label = f"{raw_label}-{occurrence.occurrence}"
            identifier_map[occurrence.canonical_id] = new_label
            edits.append(
                Edit(
                    occurrence.start,
                    occurrence.end,
                    rf"\label{{{new_label}}}",
                    "disambiguate_duplicate_label",
                )
            )
            label_records.append(
                {
                    "old_identifier": occurrence.canonical_id,
                    "old_raw_label": raw_label,
                    "new_label": new_label,
                    "reason": "duplicated_author_label",
                    "occurrence": occurrence.occurrence,
                    "occurrence_count": occurrence.occurrence_count,
                    "original_offset": occurrence.start,
                }
            )
        final_occurrence = occurrences[-1]
        edits.append(
            Edit(
                final_occurrence.end,
                final_occurrence.end,
                rf"\label{{{raw_label}}}",
                "preserve_duplicate_label_compatibility_alias",
            )
        )

    environment_map, formal_kinds = build_environment_map(
        source, inventory, identifier_module
    )

    unlabeled: list[tuple[int, str, str]] = []
    for block in inventory.formal_blocks:
        if block.source_label is None:
            unlabeled.append((block.start, block.canonical_id, formal_kinds[block.environment]))
    for block in inventory.algorithm_blocks:
        if block.source_label is None:
            unlabeled.append((block.start, block.canonical_id, "algorithm"))
    added_map = allocate_added_labels(source_labels, unlabeled)
    identifier_map.update(added_map)

    normalized_blocks: list[dict[str, Any]] = []
    all_blocks = [
        (block, formal_kinds[block.environment], "formal_result")
        for block in inventory.formal_blocks
    ] + [(block, "algorithm", "algorithm") for block in inventory.algorithm_blocks]
    for block, kind, block_type in sorted(all_blocks, key=lambda item: item[0].start):
        target_environment = environment_map.get(block.environment, block.environment)
        boundary_edits, opener_end = block_boundary_edits(block, target_environment)
        edits.extend(boundary_edits)
        if target_environment != block.environment or block.wrapper_kind == "command_wrapper":
            normalized_blocks.append(
                {
                    "block_type": block_type,
                    "original_environment": block.environment,
                    "annotated_environment": target_environment,
                    "original_offset": block.start,
                    "wrapper_kind": block.wrapper_kind,
                }
            )

        if block.source_label is None:
            new_label = added_map[block.canonical_id]
            if block.environment == "lstlisting":
                # A \label command inside verbatim listing content is inert.
                # listings' native label option is therefore the meaningful
                # inline annotation.
                option_end = optional_argument_end(source, opener_end)
                if option_end == opener_end:
                    edits.append(
                        Edit(
                            opener_end,
                            opener_end,
                            f"[label={{{new_label}}}]",
                            "add_listing_label_option",
                        )
                    )
                    placement = "environment_option"
                else:
                    close = option_end - 1
                    content = source[opener_end:close]
                    separator = "" if content.rstrip().endswith("[") else ","
                    edits.append(
                        Edit(
                            close,
                            close,
                            f"{separator}label={{{new_label}}}",
                            "add_listing_label_option",
                        )
                    )
                    placement = "environment_option"
            else:
                insertion = optional_argument_end(source, opener_end)
                edits.append(
                    Edit(
                        insertion,
                        insertion,
                        rf"\label{{{new_label}}}",
                        "add_missing_block_label",
                    )
                )
                placement = "after_opening_environment_declaration"
            label_records.append(
                {
                    "old_identifier": block.canonical_id,
                    "old_raw_label": None,
                    "new_label": new_label,
                    "reason": f"unlabelled_{block_type}",
                    "semantic_kind": kind,
                    "original_environment": block.environment,
                    "annotated_environment": target_environment,
                    "placement": placement,
                    "original_offset": block.start,
                }
            )

    document_match = BEGIN_DOCUMENT_RE.search(identifier_module.mask_comments(source))
    if document_match is None:
        raise ValueError("Source has no \\begin{document}")

    # Place each alias only after its original environment has been declared.
    # This matters for the few papers that declare theorem environments inside
    # the document immediately before use.  Aliases with a common insertion
    # point are grouped to keep the annotation compact.
    alias_insertions: dict[int, list[tuple[str, str]]] = defaultdict(list)
    first_block_position: dict[str, int] = {}
    for block, _kind, _block_type in all_blocks:
        if block.environment in environment_map:
            first_block_position.setdefault(block.environment, block.start)
    for original, annotated_name in environment_map.items():
        first_use = first_block_position[original]
        eligible_ends = [
            end
            for end in environment_declaration_ends(
                source, original, identifier_module
            )
            if end <= first_use
        ]
        insertion = max(eligible_ends, default=document_match.start())
        alias_insertions[insertion].append((original, annotated_name))
    for insertion, pairs in alias_insertions.items():
        aliases = alias_block(pairs)
        edits.append(
            Edit(
                insertion,
                insertion,
                "\n" + aliases,
                "insert_environment_aliases",
            )
        )

    annotated = apply_edits(source, edits)
    details = {
        "environment_aliases": [
            {
                "original_environment": original,
                "annotated_environment": annotated_name,
                "semantic_kind": formal_kinds[original],
            }
            for original, annotated_name in environment_map.items()
        ],
        "normalized_blocks": normalized_blocks,
        "identifier_changes": sorted(
            label_records, key=lambda item: (item["original_offset"], item["new_label"])
        ),
        "edit_count": len(edits),
    }
    return annotated, identifier_map, details


def iter_old_source_tag_slots(document: dict[str, Any]) -> Iterable[tuple[Any, Any]]:
    for issue in document.get("issues", []):
        if "primary_affected_tag" in issue:
            yield issue, "primary_affected_tag"
        failure = issue.get("failure_location")
        if isinstance(failure, dict) and "tag" in failure:
            yield failure, "tag"
        for evidence in issue.get("diagnostic_evidence", []):
            if isinstance(evidence, dict) and "tag" in evidence:
                yield evidence, "tag"
        dependent = issue.get("dependent_affected_formal_result_tags")
        if isinstance(dependent, list):
            for index in range(len(dependent)):
                yield dependent, index


def rewrite_issues(
    original: dict[str, Any], identifier_map: dict[str, str], source_sha256: str
) -> tuple[dict[str, Any], int]:
    # JSON round-trip gives us a deep copy while guaranteeing serializability.
    rewritten = json.loads(json.dumps(original))
    replacements = 0
    for container, key in iter_old_source_tag_slots(rewritten):
        value = container[key]
        if isinstance(value, str) and value in identifier_map:
            container[key] = identifier_map[value]
            replacements += 1
    rewritten["benchmark_release"] = RELEASE_NAME
    rewritten["source_annotation"] = {
        "release": RELEASE_NAME,
        "source_sha256": source_sha256,
        "old_source_tag_fields_rewritten": list(OLD_SOURCE_TAG_FIELDS),
        "new_version_repair_locators_rewritten": False,
    }
    return rewritten, replacements


def listing_option_labels(source: str) -> list[str]:
    labels: list[str] = []
    pattern = re.compile(
        r"\\begin\s*\{\s*lstlisting\s*\}\s*\[(?P<options>.*?)\]",
        re.DOTALL,
    )
    for match in pattern.finditer(source):
        for label_match in re.finditer(
            r"(?:^|,)\s*label\s*=\s*(?:\{([^{}]+)\}|([^,\]]+))",
            match.group("options"),
        ):
            labels.append((label_match.group(1) or label_match.group(2)).strip())
    return labels


def old_source_tags(document: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for container, key in iter_old_source_tag_slots(document):
        value = container[key]
        if isinstance(value, str):
            values.append(value)
    return values


def old_source_quotes(document: dict[str, Any]) -> list[str]:
    values: list[str] = []
    for issue in document.get("issues", []):
        locations = [issue.get("failure_location")]
        locations.extend(issue.get("diagnostic_evidence", []))
        for location in locations:
            if isinstance(location, dict):
                quote = location.get("quote")
                if isinstance(quote, str) and quote:
                    values.append(quote)
    return values


def verify_paper(
    annotated_source: str,
    annotated_issues: dict[str, Any],
    details: dict[str, Any],
    identifier_module: Any,
) -> dict[str, int]:
    latex_labels = [value for value, _start, _end in identifier_module.scan_label_arguments(annotated_source)]
    addressable_labels = latex_labels + listing_option_labels(annotated_source)
    counts = Counter(addressable_labels)
    duplicates = sorted(label for label, count in counts.items() if count != 1)
    if duplicates:
        raise ValueError(f"Annotated source labels are not unique: {duplicates[:5]}")

    unresolved = sorted(
        set(tag for tag in old_source_tags(annotated_issues) if counts[tag] != 1)
    )
    if unresolved:
        raise ValueError(f"Old-source JSON tags do not resolve exactly once: {unresolved[:5]}")

    expected_added = {
        item["new_label"]
        for item in details["identifier_changes"]
        if item["old_raw_label"] is None
    }
    missing_added = sorted(expected_added - set(addressable_labels))
    if missing_added:
        raise ValueError(f"Added labels missing from source: {missing_added[:5]}")

    generated_remaining = [
        tag for tag in old_source_tags(annotated_issues) if tag.startswith("__STAGED_AUTOGEN_")
    ]
    if generated_remaining:
        raise ValueError("Generated runtime identifiers remain in old-source JSON fields")
    missing_quotes = [
        quote for quote in old_source_quotes(annotated_issues) if quote not in annotated_source
    ]
    if missing_quotes:
        raise ValueError(
            "Old-source JSON quotes no longer occur exactly in source: "
            f"{missing_quotes[:3]}"
        )
    return {
        "latex_label_count": len(latex_labels),
        "listing_option_label_count": len(listing_option_labels(annotated_source)),
        "old_source_json_tag_reference_count": len(old_source_tags(annotated_issues)),
        "unique_old_source_json_tag_count": len(set(old_source_tags(annotated_issues))),
        "old_source_json_quote_count": len(old_source_quotes(annotated_issues)),
    }


def paper_ids(input_root: Path, selected: list[str]) -> list[str]:
    available = sorted(path.parent.name for path in (input_root / "papers").glob("*/source.tex"))
    if not selected:
        return available
    missing = sorted(set(selected) - set(available))
    if missing:
        raise ValueError(f"Unknown paper ids: {', '.join(missing)}")
    return [paper_id for paper_id in available if paper_id in set(selected)]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def build_release(input_root: Path, output_root: Path, selected: list[str]) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output_root}")
    ids = paper_ids(input_root, selected)
    identifier_module = load_identifier_module(input_root)
    output_root.mkdir(parents=True)
    (output_root / "papers").mkdir()

    totals: Counter[str] = Counter()
    paper_entries: list[dict[str, Any]] = []
    for paper_id in ids:
        input_dir = input_root / "papers" / paper_id
        output_dir = output_root / "papers" / paper_id
        output_dir.mkdir()
        original_source_bytes = (input_dir / "source.tex").read_bytes()
        original_issues_bytes = (input_dir / "issues.json").read_bytes()
        source = original_source_bytes.decode("utf-8")
        original_issues = json.loads(original_issues_bytes)
        inventory = identifier_module.build_source_inventory(source)
        annotated_source_before_spacing, identifier_map, details = transform_source(
            source, inventory, identifier_module
        )
        annotated_source, spacing = normalize_blank_lines(
            annotated_source_before_spacing
        )
        annotated_issues, replacement_count = rewrite_issues(
            original_issues, identifier_map, sha256_text(annotated_source)
        )
        verification = verify_paper(
            annotated_source, annotated_issues, details, identifier_module
        )

        (output_dir / "source.tex").write_text(annotated_source, encoding="utf-8")
        write_json(output_dir / "issues.json", annotated_issues)
        reason_counts = Counter(
            item["reason"] for item in details["identifier_changes"]
        )
        totals.update(reason_counts)
        totals["environment_alias_count"] += len(details["environment_aliases"])
        totals["normalized_block_count"] += len(details["normalized_blocks"])
        totals["json_old_source_tag_replacement_count"] += replacement_count
        totals["blank_lines_before"] += spacing["blank_lines_before"]
        totals["blank_lines_after"] += spacing["blank_lines_after"]
        totals["blank_lines_removed"] += spacing["blank_lines_removed"]
        totals["papers_with_blank_lines_removed"] += int(
            spacing["blank_lines_removed"] > 0
        )
        totals["paper_count"] += 1
        paper_entries.append(
            {
                "paper_id": paper_id,
                "source_original_sha256": sha256_bytes(original_source_bytes),
                "source_annotated_sha256": sha256_text(annotated_source),
                "issues_original_sha256": sha256_bytes(original_issues_bytes),
                "issues_annotated_sha256": sha256_bytes(
                    (output_dir / "issues.json").read_bytes()
                ),
                "identifier_change_count": len(details["identifier_changes"]),
                "environment_alias_count": len(details["environment_aliases"]),
                "json_old_source_tag_replacement_count": replacement_count,
                "blank_lines_removed": spacing["blank_lines_removed"],
            }
        )

    input_manifest = json.loads((input_root / "manifest.json").read_text(encoding="utf-8"))
    manifest = {
        "schema_version": 1,
        "release": RELEASE_NAME,
        "paper_count": len(ids),
        "complete_200_paper_release": len(ids) == 200 and not selected,
        "input_release": {
            "path": str(input_root),
            "manifest_sha256": sha256_bytes((input_root / "manifest.json").read_bytes()),
            "paper_count": input_manifest.get("paper_count"),
        },
        "contract": {
            "prompt_input": "papers/<paper_id>/source.tex only",
            "runtime_source_preprocessing_required": False,
            "all_formal_results_and_identifiable_algorithm_blocks_are_annotated": True,
            "custom_theorem_like_environments_are_normalized_inline": True,
            "duplicate_author_labels_are_occurrence_specific": True,
            "old_source_json_tag_fields_rewritten": list(OLD_SOURCE_TAG_FIELDS),
            "new_version_repair_locators_rewritten": False,
            "unlabelled_equations_annotated": False,
            "blank_line_normalization": {
                "version": BLANK_LINE_NORMALIZATION,
                "line_endings": "LF",
                "maximum_consecutive_empty_lines_outside_protected_environments": 1,
                "leading_empty_lines": 0,
                "final_newline_count": 1,
                "protected_environments": sorted(
                    PROTECTED_BLANK_LINE_ENVIRONMENTS
                ),
            },
        },
        "totals": dict(sorted(totals.items())),
        "papers": paper_entries,
    }
    write_json(output_root / "manifest.json", manifest)
    readme = f"""# Prompt-ready annotated benchmark sources

This is a deterministic, non-overwriting derivative of `benchmark-papers` for
{len(ids)} papers.  Give a Q1-style model only `papers/<paper_id>/source.tex`;
no runtime identifier addendum, environment resolution, marker generation, or
tag transformation is required.

Each paper directory contains:

- `source.tex`: prompt-ready source with all theorem-like results and
  mechanically identifiable algorithms annotated;
- `issues.json`: the corresponding gold with only old-source tag fields
  rewritten to those literal source identifiers.

The unchanged inputs remain in the sibling `benchmark-papers` release; they
are not duplicated inside each annotated paper directory.
Release-level provenance, transformation counts, and hashes are recorded in
`manifest.json` and `CHECKSUMS.sha256`.

Custom theorem-like names are replaced in the document body by readable
semantic names (`theorem`, `lemma`, `proposition`, `corollary`, `claim`, or a
numbered variant when the base name is already occupied).  Preamble aliases
preserve the original environment definitions, counters, styles, and titles.

Missing block labels use natural identifiers such as `thm:added-1` and
`alg:added-1`.  Duplicate author labels become `original-1`, `original-2`, and
so on; a single compatibility alias on the last occurrence preserves existing
LaTeX references to `original`.

Existing equation labels are retained and duplicate equation labels are
disambiguated by the same occurrence rule.  Unlabelled equations, starred
equations, and unnumbered displays are intentionally not assigned artificial
labels.  Row-level labels in multi-line displays therefore keep their meaning.

Outside verbatim-like and code environments, whitespace-only rows are empty,
consecutive empty rows are collapsed to one, leading empty rows are removed,
and every source ends with one newline.  The exact deterministic policy and its
protected-environment list are recorded in `manifest.json` as
`{BLANK_LINE_NORMALIZATION}`.

`new_version_repair_locator` belongs to the later paper version and is kept
unchanged as provenance; it is not an old-source locator and need not resolve
in `source.tex`.
"""
    (output_root / "README.md").write_text(readme, encoding="utf-8")
    write_checksums(output_root)
    return manifest


def write_checksums(output_root: Path) -> None:
    lines = []
    for path in sorted(output_root.rglob("*")):
        if path.is_file() and path.name not in {"CHECKSUMS.sha256", ".DS_Store"}:
            relative = path.relative_to(output_root)
            lines.append(f"{sha256_bytes(path.read_bytes())}  {relative.as_posix()}")
    (output_root / "CHECKSUMS.sha256").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("benchmark-papers"))
    parser.add_argument(
        "--output", type=Path, default=Path(RELEASE_NAME), help="Must not exist"
    )
    parser.add_argument("--paper-id", action="append", default=[])
    args = parser.parse_args()
    manifest = build_release(args.input, args.output, args.paper_id)
    print(json.dumps({"output": str(args.output), **manifest["totals"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
