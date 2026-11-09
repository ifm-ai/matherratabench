#!/usr/bin/env python3
"""Deterministic source identities for source-only issue finding.

The original LaTeX remains unchanged.  This module discovers explicit
theorem-like environments and wrappers, parses ``\\label`` arguments with
balanced braces, and assigns collision-resistant generated identifiers when a
formal result or identifiable algorithm block has no direct label, or when a
source label is duplicated.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import re
from typing import Any


BASE_FORMAL_ENVS = {
    "theorem",
    "thm",
    "lemma",
    "lem",
    "proposition",
    "prop",
    "corollary",
    "cor",
    "claim",
}
FORMAL_TITLES = ("theorem", "lemma", "proposition", "corollary", "claim")
ENV_NAME_RE = r"[A-Za-z@][A-Za-z0-9@*:_-]*"
COMMAND_NAME_RE = r"[A-Za-z@]+"
NEWTHEOREM_RE = re.compile(
    rf"""
    \\newtheorem\*?\s*\{{\s*(?P<env>{ENV_NAME_RE})\s*\}}
    (?:\s*\[[^\]]*\])?
    \s*\{{(?P<title>(?:[^{{}}]|\{{[^{{}}]*\}})*)\}}
    """,
    re.VERBOSE | re.DOTALL,
)
DECLARETHEOREM_RE = re.compile(
    rf"\\declaretheorem\s*(?:\[(?P<options>[^\]]*)\])?\s*"
    rf"\{{\s*(?P<env>{ENV_NAME_RE})\s*\}}",
    re.DOTALL,
)
NEWMDTHEOREM_RE = re.compile(
    rf"\\newmdtheoremenv\s*(?:\[[^\]]*\])?\s*"
    rf"\{{\s*(?P<env>{ENV_NAME_RE})\s*\}}\s*"
    rf"\{{(?P<title>(?:[^{{}}]|\{{[^{{}}]*\}})*)\}}",
    re.DOTALL,
)
SIMPLE_ENV_WRAPPER_RE = re.compile(
    rf"""
    \\(?:newenvironment|renewenvironment)\*?\s*
    \{{\s*(?P<name>{ENV_NAME_RE})\s*\}}
    (?:\s*\[[^\]]*\]){{0,2}}
    \s*\{{\s*\\begin\s*\{{\s*(?P<base>{ENV_NAME_RE})\s*\}}\s*\}}
    \s*\{{\s*\\end\s*\{{\s*(?P=base)\s*\}}\s*\}}
    """,
    re.VERBOSE,
)
SIMPLE_COMMAND_WRAPPER_RE = re.compile(
    rf"""
    \\(?:newcommand|renewcommand|providecommand)\*?\s*
    (?:\{{\s*\\(?P<braced>{COMMAND_NAME_RE})\s*\}}|\\(?P<bare>{COMMAND_NAME_RE}))
    (?:\s*\[\s*0\s*\])?
    \s*\{{\s*\\(?P<kind>begin|end)\s*\{{\s*(?P<env>{ENV_NAME_RE})\s*\}}\s*\}}
    """,
    re.VERBOSE,
)
SIMPLE_DEF_WRAPPER_RE = re.compile(
    rf"""
    \\def\s*\\(?P<command>{COMMAND_NAME_RE})\s*
    \{{\s*\\(?P<kind>begin|end)\s*\{{\s*(?P<env>{ENV_NAME_RE})\s*\}}\s*\}}
    """,
    re.VERBOSE,
)
BEGIN_END_RE = re.compile(
    rf"\\(?P<kind>begin|end)\s*\{{\s*(?P<env>{ENV_NAME_RE})\s*\}}"
)


@dataclass(frozen=True)
class LabelOccurrence:
    value: str
    start: int
    end: int
    ordinal: int
    occurrence: int
    occurrence_count: int
    canonical_id: str


@dataclass(frozen=True)
class FormalBlock:
    start: int
    end: int
    environment: str
    text: str
    ordinal: int
    canonical_id: str
    source_label: str | None
    label_start: int | None
    wrapper_kind: str


@dataclass(frozen=True)
class SourceInventory:
    source_sha256: str
    generated_prefix: str
    labels: tuple[LabelOccurrence, ...]
    formal_blocks: tuple[FormalBlock, ...]

    @property
    def locator_ids(self) -> tuple[str, ...]:
        return tuple(item.canonical_id for item in self.labels)

    @property
    def formal_ids(self) -> tuple[str, ...]:
        return tuple(item.canonical_id for item in self.formal_blocks)

    @property
    def all_target_ids(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.locator_ids, *self.formal_ids)))

    def resolve(self, canonical_id: str) -> tuple[int, int] | None:
        label_matches = [
            item for item in self.labels if item.canonical_id == canonical_id
        ]
        formal_matches = [
            item for item in self.formal_blocks if item.canonical_id == canonical_id
        ]
        matches = [
            *((item.start, item.end) for item in label_matches),
            *((item.start, item.end) for item in formal_matches),
        ]
        if not matches:
            return None
        # A directly labeled formal block intentionally shares its canonical ID
        # with the label occurrence.  Prefer the precise label span.
        if label_matches:
            return label_matches[0].start, label_matches[0].end
        return matches[0]

    def raw_label_occurrences(self, value: str) -> tuple[LabelOccurrence, ...]:
        return tuple(item for item in self.labels if item.value == value)

    def formal_block_for_id(self, canonical_id: str) -> FormalBlock | None:
        matches = [
            item for item in self.formal_blocks if item.canonical_id == canonical_id
        ]
        return matches[0] if len(matches) == 1 else None

    def prompt_exceptions(self, source: str, *, source_name: str) -> str:
        generated_formals = [
            item
            for item in self.formal_blocks
            if item.canonical_id.startswith(self.generated_prefix)
        ]
        duplicated_labels = [item for item in self.labels if item.occurrence_count > 1]
        if not generated_formals and not duplicated_labels:
            return f"{source_name}: no generated source identifiers are required."

        lines = [
            f"{source_name} GENERATED SOURCE IDENTIFIERS",
            "Use each canonical_id exactly as written. The source itself is unchanged.",
        ]
        if generated_formals:
            lines.append("FORMAL RESULTS WITH GENERATED IDS")
            for item in generated_formals:
                lines.append(
                    _prompt_json_line(
                        {
                            "canonical_id": item.canonical_id,
                            "environment": item.environment,
                            "source_label": item.source_label,
                            "source_position": item.start,
                            "opening_quote": _opening_quote(item.text),
                        }
                    )
                )
        if duplicated_labels:
            lines.append("DUPLICATED LABEL OCCURRENCES")
            for item in duplicated_labels:
                lines.append(
                    _prompt_json_line(
                        {
                            "canonical_id": item.canonical_id,
                            "raw_label": item.value,
                            "occurrence": item.occurrence,
                            "source_position": item.start,
                            "local_quote": _local_quote(source, item.start, item.end),
                        }
                    )
                )
        return "\n".join(lines)


def _prompt_json_line(value: dict[str, Any]) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _opening_quote(text: str, limit: int = 320) -> str:
    return text[:limit]


def _local_quote(source: str, start: int, end: int, radius: int = 180) -> str:
    return source[max(0, start - radius) : min(len(source), end + radius)]


def _escaped(source: str, index: int) -> bool:
    backslashes = 0
    index -= 1
    while index >= 0 and source[index] == "\\":
        backslashes += 1
        index -= 1
    return backslashes % 2 == 1


def mask_comments(source: str) -> str:
    characters = list(source)
    index = 0
    while index < len(source):
        if source[index] == "%" and not _escaped(source, index):
            end = source.find("\n", index)
            if end < 0:
                end = len(source)
            for position in range(index, end):
                characters[position] = " "
            index = end
        else:
            index += 1
    return "".join(characters)


def scan_label_arguments(source: str) -> list[tuple[str, int, int]]:
    """Return balanced ``\\label{...}`` values and exact command spans."""

    searchable = mask_comments(source)
    results: list[tuple[str, int, int]] = []
    for match in re.finditer(r"\\label(?![A-Za-z@])", searchable):
        cursor = match.end()
        while cursor < len(searchable) and searchable[cursor].isspace():
            cursor += 1
        if cursor >= len(searchable) or searchable[cursor] != "{":
            continue
        depth = 0
        end = cursor
        while end < len(searchable):
            character = searchable[end]
            if character == "{" and not _escaped(searchable, end):
                depth += 1
            elif character == "}" and not _escaped(searchable, end):
                depth -= 1
                if depth == 0:
                    value = source[cursor + 1 : end].strip()
                    if value:
                        results.append((value, match.start(), end + 1))
                    break
            end += 1
    return results


def discover_formal_envs(source: str) -> set[str]:
    envs = set(BASE_FORMAL_ENVS)
    for match in NEWTHEOREM_RE.finditer(source):
        env = match.group("env")
        title = match.group("title")
        if _is_formal_declaration(env, title):
            envs.add(env)
    for match in DECLARETHEOREM_RE.finditer(source):
        env = match.group("env")
        options = match.group("options") or ""
        name_match = re.search(r"(?:^|,)\s*name\s*=\s*([^,]+)", options)
        displayed_name = name_match.group(1) if name_match else ""
        if _is_formal_declaration(env, displayed_name):
            envs.add(env)
    for match in NEWMDTHEOREM_RE.finditer(source):
        env = match.group("env")
        title = match.group("title")
        if _is_formal_declaration(env, title):
            envs.add(env)

    wrappers = [
        (match.group("name"), match.group("base"))
        for match in SIMPLE_ENV_WRAPPER_RE.finditer(source)
    ]
    changed = True
    while changed:
        changed = False
        for name, base in wrappers:
            if base in envs and name not in envs:
                envs.add(name)
                changed = True
    return envs


def _is_formal_declaration(env: str, displayed_name: str) -> bool:
    if env.rstrip("*").lower() in BASE_FORMAL_ENVS:
        return True
    words = re.findall(r"[A-Za-z]+", displayed_name.lower())
    return any(kind in words for kind in FORMAL_TITLES)


def discover_formal_command_wrappers(
    source: str,
    envs: set[str],
) -> tuple[dict[str, set[str]], dict[str, set[str]], list[tuple[int, int]]]:
    begin_commands: dict[str, set[str]] = {}
    end_commands: dict[str, set[str]] = {}
    definition_spans: list[tuple[int, int]] = []
    matches = list(SIMPLE_COMMAND_WRAPPER_RE.finditer(source))
    matches.extend(SIMPLE_DEF_WRAPPER_RE.finditer(source))
    for match in matches:
        env = match.group("env")
        if env not in envs:
            continue
        command = match.groupdict().get("command")
        if command is None:
            command = match.group("braced") or match.group("bare")
        target = begin_commands if match.group("kind") == "begin" else end_commands
        target.setdefault(env, set()).add(command)
        definition_spans.append((match.start(), match.end()))
    return begin_commands, end_commands, definition_spans


def mask_spans(source: str, spans: list[tuple[int, int]]) -> str:
    characters = list(source)
    for start, end in spans:
        for index in range(start, end):
            if characters[index] not in "\r\n":
                characters[index] = " "
    return "".join(characters)


def discover_formal_block_spans(
    source: str,
) -> list[tuple[int, int, str, str, str]]:
    envs = discover_formal_envs(source)
    begin_commands, end_commands, command_definition_spans = (
        discover_formal_command_wrappers(source, envs)
    )
    environment_definition_spans = [
        (match.start(), match.end()) for match in SIMPLE_ENV_WRAPPER_RE.finditer(source)
    ]
    searchable = mask_comments(
        mask_spans(source, command_definition_spans + environment_definition_spans)
    )
    escaped = "|".join(sorted(map(re.escape, envs), key=len, reverse=True))
    blocks: list[tuple[int, int, str, str, str]] = []
    if escaped:
        pattern = re.compile(
            rf"\\begin\{{({escaped})\}}.*?\\end\{{\1\}}",
            re.DOTALL,
        )
        blocks.extend(
            (
                match.start(),
                match.end(),
                match.group(1),
                source[match.start() : match.end()],
                "environment",
            )
            for match in pattern.finditer(searchable)
        )

    for env, starts in begin_commands.items():
        for start_command in starts:
            for end_command in end_commands.get(env, set()):
                pattern = re.compile(
                    rf"\\{re.escape(start_command)}(?![A-Za-z@]).*?"
                    rf"\\{re.escape(end_command)}(?![A-Za-z@])",
                    re.DOTALL,
                )
                blocks.extend(
                    (
                        match.start(),
                        match.end(),
                        env,
                        source[match.start() : match.end()],
                        "command_wrapper",
                    )
                    for match in pattern.finditer(searchable)
                )
    unique = {
        (start, end, env): (start, end, env, text, wrapper_kind)
        for start, end, env, text, wrapper_kind in blocks
    }
    return sorted(unique.values(), key=lambda item: (item[0], item[1], item[2]))


def _direct_labels_for_block(
    source: str,
    block: tuple[int, int, str, str, str],
    labels: list[tuple[str, int, int]],
) -> list[tuple[str, int, int]]:
    start, end, environment, text, wrapper_kind = block
    candidates = [item for item in labels if start <= item[1] and item[2] <= end]
    if not candidates:
        return []

    tokens = list(BEGIN_END_RE.finditer(text))
    direct: list[tuple[str, int, int]] = []
    for label in candidates:
        relative = label[1] - start
        stack: list[str] = []
        for token in tokens:
            if token.start() >= relative:
                break
            env = token.group("env")
            if token.group("kind") == "begin":
                stack.append(env)
            elif stack:
                if stack[-1] == env:
                    stack.pop()
                elif env in stack:
                    del stack[stack.index(env) :]
        if wrapper_kind == "environment":
            is_direct = stack == [environment]
        else:
            is_direct = not stack
        if is_direct:
            direct.append(label)
    return sorted(direct, key=lambda item: item[1])


def build_source_inventory(source: str) -> SourceInventory:
    source_sha256 = hashlib.sha256(source.encode("utf-8")).hexdigest()
    raw_labels = scan_label_arguments(source)
    grouped: dict[str, list[tuple[str, int, int]]] = {}
    for item in raw_labels:
        grouped.setdefault(item[0], []).append(item)

    prefix_length = 16
    while True:
        prefix = f"__STAGED_AUTOGEN_{source_sha256[:prefix_length].upper()}_"
        if not any(value.startswith(prefix) for value in grouped):
            break
        prefix_length += 4

    labels: list[LabelOccurrence] = []
    occurrence_seen: dict[str, int] = {}
    for ordinal, (value, start, end) in enumerate(raw_labels, start=1):
        occurrence_seen[value] = occurrence_seen.get(value, 0) + 1
        count = len(grouped[value])
        canonical_id = (
            value if count == 1 else f"{prefix}LABEL_{ordinal:06d}__"
        )
        labels.append(
            LabelOccurrence(
                value=value,
                start=start,
                end=end,
                ordinal=ordinal,
                occurrence=occurrence_seen[value],
                occurrence_count=count,
                canonical_id=canonical_id,
            )
        )
    label_by_start = {item.start: item for item in labels}

    raw_blocks = discover_formal_block_spans(source)
    formal_blocks: list[FormalBlock] = []
    for ordinal, block in enumerate(raw_blocks, start=1):
        start, end, environment, text, wrapper_kind = block
        direct_labels = _direct_labels_for_block(source, block, raw_labels)
        direct = label_by_start[direct_labels[0][1]] if direct_labels else None
        canonical_id = (
            direct.canonical_id
            if direct is not None
            else f"{prefix}FORMAL_{ordinal:06d}__"
        )
        formal_blocks.append(
            FormalBlock(
                start=start,
                end=end,
                environment=environment,
                text=text,
                ordinal=ordinal,
                canonical_id=canonical_id,
                source_label=direct.value if direct is not None else None,
                label_start=direct.start if direct is not None else None,
                wrapper_kind=wrapper_kind,
            )
        )
    return SourceInventory(
        source_sha256=source_sha256,
        generated_prefix=prefix,
        labels=tuple(labels),
        formal_blocks=tuple(formal_blocks),
    )


# Q1-only algorithm locator extension.  The inherited formal-result and
# duplicate-label behavior above remains the frozen source of truth.
_FROZEN_SOURCE_INVENTORY = SourceInventory
_FROZEN_BUILD_SOURCE_INVENTORY = build_source_inventory

BASE_ALGORITHM_ENVS = {
    "algorithm",
    "algorithm*",
    "algorithmic",
    "algorithmic*",
    "pseudocode",
    "procedure",
    "program",
    "code",
    "lstlisting",
    "minted",
}


@dataclass(frozen=True)
class AlgorithmBlock:
    start: int
    end: int
    environment: str
    text: str
    ordinal: int
    canonical_id: str
    source_label: str | None
    label_start: int | None
    wrapper_kind: str


@dataclass(frozen=True)
class Q1SourceInventory(_FROZEN_SOURCE_INVENTORY):
    algorithm_blocks: tuple[AlgorithmBlock, ...]

    @property
    def algorithm_ids(self) -> tuple[str, ...]:
        return tuple(item.canonical_id for item in self.algorithm_blocks)

    @property
    def all_target_ids(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (*self.locator_ids, *self.formal_ids, *self.algorithm_ids)
            )
        )

    def resolve(self, canonical_id: str) -> tuple[int, int] | None:
        inherited = _FROZEN_SOURCE_INVENTORY.resolve(self, canonical_id)
        if inherited is not None:
            return inherited
        matches = [
            item
            for item in self.algorithm_blocks
            if item.canonical_id == canonical_id
        ]
        return (matches[0].start, matches[0].end) if len(matches) == 1 else None

    def algorithm_block_for_id(self, canonical_id: str) -> AlgorithmBlock | None:
        matches = [
            item
            for item in self.algorithm_blocks
            if item.canonical_id == canonical_id
        ]
        return matches[0] if len(matches) == 1 else None

    def frozen_prompt_exceptions(self, source: str, *, source_name: str) -> str:
        return _FROZEN_SOURCE_INVENTORY.prompt_exceptions(
            self, source, source_name=source_name
        )

    def prompt_exceptions(self, source: str, *, source_name: str) -> str:
        inherited = self.frozen_prompt_exceptions(source, source_name=source_name)
        generated_algorithms = [
            item
            for item in self.algorithm_blocks
            if item.canonical_id.startswith(self.generated_prefix)
        ]
        if not generated_algorithms:
            return inherited

        no_generated = f"{source_name}: no generated source identifiers are required."
        if inherited == no_generated:
            lines = [
                f"{source_name} GENERATED SOURCE IDENTIFIERS",
                "Use each canonical_id exactly as written. The source itself is unchanged.",
            ]
        else:
            lines = inherited.splitlines()
        lines.append("ALGORITHM BLOCKS WITH GENERATED IDS")
        for item in generated_algorithms:
            lines.append(
                _prompt_json_line(
                    {
                        "canonical_id": item.canonical_id,
                        "environment": item.environment,
                        "source_label": item.source_label,
                        "source_position": item.start,
                        "opening_quote": _opening_quote(item.text),
                    }
                )
            )
        return "\n".join(lines)


def _is_algorithm_environment_name(environment: str) -> bool:
    normalized = environment.rstrip("*").lower()
    if environment in BASE_ALGORITHM_ENVS or normalized in BASE_ALGORITHM_ENVS:
        return True
    return (
        "algorithm" in normalized
        or "pseudocode" in normalized
        or "procedure" in normalized
        or "listing" in normalized
        or normalized in {"code", "program"}
    )


def discover_algorithm_envs(source: str) -> set[str]:
    envs = set(BASE_ALGORITHM_ENVS)
    searchable = mask_comments(source)
    for match in BEGIN_END_RE.finditer(searchable):
        environment = match.group("env")
        if _is_algorithm_environment_name(environment):
            envs.add(environment)

    wrappers = [
        (match.group("name"), match.group("base"))
        for match in SIMPLE_ENV_WRAPPER_RE.finditer(source)
    ]
    changed = True
    while changed:
        changed = False
        for name, base in wrappers:
            if (base in envs or _is_algorithm_environment_name(name)) and name not in envs:
                envs.add(name)
                changed = True
    return envs


def discover_algorithm_block_spans(
    source: str,
) -> list[tuple[int, int, str, str, str]]:
    envs = discover_algorithm_envs(source)
    begin_commands, end_commands, command_definition_spans = (
        discover_formal_command_wrappers(source, envs)
    )
    environment_definition_spans = [
        (match.start(), match.end())
        for match in SIMPLE_ENV_WRAPPER_RE.finditer(source)
    ]
    searchable = mask_comments(
        mask_spans(source, command_definition_spans + environment_definition_spans)
    )
    escaped = "|".join(sorted(map(re.escape, envs), key=len, reverse=True))
    candidates: list[tuple[int, int, str, str, str]] = []
    if escaped:
        pattern = re.compile(
            rf"\\begin\{{({escaped})\}}.*?\\end\{{\1\}}",
            re.DOTALL,
        )
        candidates.extend(
            (
                match.start(),
                match.end(),
                match.group(1),
                source[match.start() : match.end()],
                "environment",
            )
            for match in pattern.finditer(searchable)
        )

    for environment, starts in begin_commands.items():
        for start_command in starts:
            for end_command in end_commands.get(environment, set()):
                pattern = re.compile(
                    rf"\\{re.escape(start_command)}(?![A-Za-z@]).*?"
                    rf"\\{re.escape(end_command)}(?![A-Za-z@])",
                    re.DOTALL,
                )
                candidates.extend(
                    (
                        match.start(),
                        match.end(),
                        environment,
                        source[match.start() : match.end()],
                        "command_wrapper",
                    )
                    for match in pattern.finditer(searchable)
                )

    unique = {
        (start, end, environment): (
            start,
            end,
            environment,
            text,
            wrapper_kind,
        )
        for start, end, environment, text, wrapper_kind in candidates
    }
    document_begin = re.search(r"\\begin\s*\{\s*document\s*\}", searchable)
    document_end = re.search(r"\\end\s*\{\s*document\s*\}", searchable)
    body_start = document_begin.end() if document_begin else 0
    body_end = document_end.start() if document_end else len(source)
    ordered = sorted(
        (
            item
            for item in unique.values()
            if body_start <= item[0] and item[1] <= body_end
        ),
        key=lambda item: (item[0], -item[1], item[2]),
    )
    outermost: list[tuple[int, int, str, str, str]] = []
    for candidate in ordered:
        start, end = candidate[0], candidate[1]
        if any(parent[0] <= start and end <= parent[1] for parent in outermost):
            continue
        outermost.append(candidate)
    return sorted(outermost, key=lambda item: (item[0], item[1], item[2]))


def build_source_inventory(source: str) -> Q1SourceInventory:
    inherited = _FROZEN_BUILD_SOURCE_INVENTORY(source)
    raw_labels = scan_label_arguments(source)
    label_by_start = {item.start: item for item in inherited.labels}
    algorithm_blocks: list[AlgorithmBlock] = []
    for ordinal, block in enumerate(discover_algorithm_block_spans(source), start=1):
        start, end, environment, text, wrapper_kind = block
        direct_labels = _direct_labels_for_block(source, block, raw_labels)
        direct = label_by_start[direct_labels[0][1]] if direct_labels else None
        canonical_id = (
            direct.canonical_id
            if direct is not None
            else f"{inherited.generated_prefix}ALGORITHM_{ordinal:06d}__"
        )
        algorithm_blocks.append(
            AlgorithmBlock(
                start=start,
                end=end,
                environment=environment,
                text=text,
                ordinal=ordinal,
                canonical_id=canonical_id,
                source_label=direct.value if direct is not None else None,
                label_start=direct.start if direct is not None else None,
                wrapper_kind=wrapper_kind,
            )
        )
    return Q1SourceInventory(
        source_sha256=inherited.source_sha256,
        generated_prefix=inherited.generated_prefix,
        labels=inherited.labels,
        formal_blocks=inherited.formal_blocks,
        algorithm_blocks=tuple(algorithm_blocks),
    )



def _main() -> int:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Generate the canonical exceptional-identifier addendum for source.tex."
    )
    parser.add_argument("source", type=Path, help="Path to the exact source.tex file")
    args = parser.parse_args()
    source = args.source.read_text(encoding="utf-8")
    inventory = build_source_inventory(source)
    print(inventory.prompt_exceptions(source, source_name="OLD SOURCE"))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
