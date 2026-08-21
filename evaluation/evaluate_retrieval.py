from __future__ import annotations

import json
from pathlib import Path

from interview_coach.rag.retriever import RagService

ROOT = Path(__file__).parents[1]


def main() -> None:
    resume = (ROOT / "data/samples/sample_resume.txt").read_text(encoding="utf-8")
    jd = (ROOT / "data/samples/sample_jd.txt").read_text(encoding="utf-8")
    cases = json.loads((ROOT / "data/evaluation/retrieval_cases.json").read_text(encoding="utf-8"))
    service = RagService("lexical_demo", "")
    service.index_session("evaluation", resume, jd)
    hits = 0
    details = []
    for case in cases:
        returned = [item.chunk_id for item in service.search("evaluation", case["query"])]
        relevant = set(case["relevant_ids"])
        hit = bool(relevant.intersection(returned))
        hits += int(hit)
        details.append({"query": case["query"], "returned": returned, "hit": hit})
    result = {
        "backend": "lexical_demo",
        "cases": len(cases),
        "recall_case_hit_rate": hits / len(cases) if cases else None,
        "details": details,
        "note": "Small smoke dataset; not a product-quality benchmark.",
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
