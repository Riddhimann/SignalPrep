from __future__ import annotations

import re

from interview_coach.schemas import RequirementExtraction

_TOKEN = re.compile(r"[a-zA-Z][a-zA-Z0-9+#.-]{1,}")
_STOPWORDS = {
    "ability",
    "and",
    "for",
    "from",
    "into",
    "of",
    "or",
    "strong",
    "the",
    "to",
    "using",
    "with",
}


def _tokens(text: str) -> set[str]:
    return {token.lower() for token in _TOKEN.findall(text) if token.lower() not in _STOPWORDS}


def _supported(item: str, document: str, *, threshold: float) -> bool:
    normalized_item = " ".join(item.lower().split())
    normalized_document = " ".join(document.lower().split())
    if normalized_item and normalized_item in normalized_document:
        return True
    item_tokens = _tokens(item)
    if not item_tokens:
        return False
    document_tokens = _tokens(document)
    return len(item_tokens & document_tokens) / len(item_tokens) >= threshold


def ground_requirement_extraction(
    extraction: RequirementExtraction,
    job_description: str,
) -> tuple[RequirementExtraction, list[str]]:
    """Fail closed on unsupported extracted items while preserving their audit labels."""

    removed: list[str] = []

    def filtered(items: list[str], threshold: float, label: str) -> list[str]:
        output: list[str] = []
        for item in items:
            if _supported(item, job_description, threshold=threshold):
                output.append(item)
            else:
                removed.append(f"{label}: {item}")
        return output

    grounded = extraction.model_copy(
        update={
            "responsibilities": filtered(extraction.responsibilities, 0.45, "responsibility"),
            "required_skills": filtered(extraction.required_skills, 0.75, "required_skill"),
            "preferred_skills": filtered(extraction.preferred_skills, 0.75, "preferred_skill"),
            "evaluation_topics": filtered(extraction.evaluation_topics, 0.65, "evaluation_topic"),
        }
    )
    return grounded, removed
