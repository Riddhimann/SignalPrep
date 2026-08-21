from __future__ import annotations

import re
from typing import Literal

from interview_coach.schemas import DocumentChunk

Source = Literal["resume", "job_description", "rubric"]
HEADING = re.compile(r"^[A-Z][A-Za-z0-9 /&(),+-]{1,60}:?$")


def _sections(text: str) -> list[tuple[str | None, str]]:
    sections: list[tuple[str | None, str]] = []
    heading: str | None = None
    body: list[str] = []
    for line in text.splitlines():
        candidate = line.strip().rstrip(":")
        looks_like_heading = bool(HEADING.fullmatch(line.strip())) and len(line.split()) <= 7
        if looks_like_heading:
            if body:
                sections.append((heading, " ".join(body)))
            heading, body = candidate, []
        elif line.strip():
            body.append(line.strip())
    if body:
        sections.append((heading, " ".join(body)))
    return sections or [(None, text)]


def chunk_document(
    text: str,
    source: Source,
    *,
    target_words: int = 360,
    overlap_words: int = 50,
) -> list[DocumentChunk]:
    if target_words <= overlap_words or overlap_words < 0:
        raise ValueError("target_words must be greater than non-negative overlap_words")
    chunks: list[DocumentChunk] = []
    counter = 0
    for section, body in _sections(text):
        words = body.split()
        start = 0
        while start < len(words):
            counter += 1
            end = min(start + target_words, len(words))
            prefix = source.replace("job_description", "jd")
            chunk_id = f"{prefix}_{counter:03d}"
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    source=source,
                    section=section,
                    text=" ".join(words[start:end]),
                    metadata={"word_start": str(start), "word_end": str(end)},
                )
            )
            if end == len(words):
                break
            start = end - overlap_words
    return chunks
