from __future__ import annotations

import asyncio
import json
from pathlib import Path

from pydantic import ValidationError

from interview_coach.exceptions import StructuredOutputError
from interview_coach.llm.gateway import DeterministicDemoGateway
from interview_coach.llm.prompts import EVALUATION_SYSTEM
from interview_coach.schemas import AnswerEvaluation

ROOT = Path(__file__).parents[1]


async def evaluate(runs: int = 25) -> dict:
    gateway = DeterministicDemoGateway()
    valid = 0
    payload = json.dumps(
        {
            "transcript": "I built and validated a Python API that reduced latency by 30 percent.",
            "evidence": [{"chunk_id": "resume_001", "score": 0.8}],
        }
    )
    for _ in range(runs):
        try:
            await gateway.generate_structured(EVALUATION_SYSTEM, payload, AnswerEvaluation)
            valid += 1
        except (StructuredOutputError, ValidationError):
            continue
    return {
        "provider": "deterministic_demo",
        "runs": runs,
        "schema_valid": valid,
        "schema_valid_rate": valid / runs,
        "note": "This measures deterministic contract stability, not remote LLM quality.",
    }


if __name__ == "__main__":
    print(json.dumps(asyncio.run(evaluate()), indent=2))
