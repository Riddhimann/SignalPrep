from __future__ import annotations

import re

_INJECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "instruction_override",
        re.compile(
            r"\b(ignore|disregard|forget)\b.{0,40}\b(previous|prior|system|developer)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "role_override",
        re.compile(
            r"\b(you are now|act as|new system prompt|developer message)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "prompt_exfiltration",
        re.compile(
            r"\b(reveal|print|repeat|show)\b.{0,35}\b(system prompt|hidden instructions)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "tool_override",
        re.compile(
            r"\b(call|execute|run|invoke)\b.{0,30}\b(tool|command|shell|function)\b",
            re.IGNORECASE,
        ),
    ),
)


def prompt_injection_flags(text: str) -> list[str]:
    """Return conservative indicators; the caller still treats all document text as untrusted."""

    return [name for name, pattern in _INJECTION_PATTERNS if pattern.search(text)]


def quarantine_prompt_injection(text: str) -> tuple[str, list[str]]:
    """Remove flagged lines before model prompting while retaining auditable event names."""

    clean_lines: list[str] = []
    observed: set[str] = set()
    for line in text.splitlines():
        flags = prompt_injection_flags(line)
        if flags:
            observed.update(flags)
            clean_lines.append("[QUARANTINED UNTRUSTED INSTRUCTION]")
        else:
            clean_lines.append(line)
    return "\n".join(clean_lines), sorted(observed)
