from __future__ import annotations

import re

from interview_coach.schemas import GroundingAudit, RetrievedEvidence

_TOKEN = re.compile(r"[a-zA-Z][a-zA-Z0-9+#.-]{1,}")
_STOPWORDS = {
    "about",
    "and",
    "could",
    "describe",
    "did",
    "for",
    "from",
    "have",
    "how",
    "into",
    "me",
    "that",
    "the",
    "their",
    "this",
    "tell",
    "what",
    "when",
    "where",
    "which",
    "with",
    "would",
    "you",
    "your",
}


def _tokens(text: str) -> set[str]:
    return {
        token.lower()
        for token in _TOKEN.findall(text)
        if token.lower() not in _STOPWORDS and len(token) > 2
    }


def audit_grounding(
    text: str,
    cited_ids: list[str],
    evidence: list[RetrievedEvidence],
) -> GroundingAudit:
    available = {item.chunk_id: item for item in evidence}
    invalid = sorted(set(cited_ids) - set(available))
    cited = [available[item_id] for item_id in dict.fromkeys(cited_ids) if item_id in available]
    risk_flags = sorted({flag for item in cited for flag in item.risk_flags})
    if not evidence:
        return GroundingAudit(
            status="no_evidence",
            citation_valid=not invalid,
            support_score=0,
            cited_evidence_ids=[],
            invalid_evidence_ids=invalid,
            risk_flags=risk_flags,
        )
    if invalid or not cited:
        return GroundingAudit(
            status="ungrounded",
            citation_valid=False,
            support_score=0,
            cited_evidence_ids=[item.chunk_id for item in cited],
            invalid_evidence_ids=invalid,
            risk_flags=risk_flags,
        )

    claim_tokens = _tokens(text)
    evidence_tokens = _tokens(" ".join(item.text for item in cited))
    overlap = len(claim_tokens & evidence_tokens)
    lexical_support = min(overlap / max(min(len(claim_tokens), 12), 1), 1.0)
    semantic_values = [item.semantic_score for item in cited if item.semantic_score is not None]
    semantic_support = max(semantic_values) if semantic_values else None
    if lexical_support >= 0.12:
        status = "grounded"
        support_score = lexical_support
        support_method = "lexical"
    elif lexical_support >= 0.05:
        status = "partial"
        support_score = lexical_support
        support_method = "lexical"
    elif semantic_support is not None and semantic_support >= 0.72:
        status = "partial"
        support_score = semantic_support
        support_method = "semantic_indicator"
    else:
        status = "ungrounded"
        support_score = lexical_support
        support_method = "none"
    return GroundingAudit(
        status=status,
        citation_valid=not invalid,
        support_score=support_score,
        lexical_support_score=lexical_support,
        semantic_support_score=semantic_support,
        support_method=support_method,
        cited_evidence_ids=[item.chunk_id for item in cited],
        invalid_evidence_ids=invalid,
        risk_flags=risk_flags,
    )
