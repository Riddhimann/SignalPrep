from __future__ import annotations

import asyncio
import json
import re
from typing import Any, Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from interview_coach.config import Settings
from interview_coach.exceptions import StructuredOutputError
from interview_coach.schemas import (
    AnswerEvaluation,
    GeneratedQuestion,
    RequirementExtraction,
    Scorecard,
)

T = TypeVar("T", bound=BaseModel)


class LLMGateway(Protocol):
    provider_name: str
    model_name: str
    structured_output_mode: str

    async def generate_structured(
        self, system_prompt: str, user_prompt: str, output_schema: type[T]
    ) -> T: ...


def _payload(user_prompt: str) -> dict:
    try:
        return json.loads(user_prompt)
    except json.JSONDecodeError as exc:
        raise StructuredOutputError("Internal prompt payload was not valid JSON") from exc


class DeterministicDemoGateway:
    """Auditable offline behavior for demos/tests; deliberately not represented as an LLM."""

    provider_name = "deterministic_demo"
    model_name = "transparent-rules-v1"
    structured_output_mode = "local_pydantic"
    KNOWN_SKILLS = (
        "Python",
        "SQL",
        "machine learning",
        "deep learning",
        "NLP",
        "statistics",
        "data visualization",
        "AWS",
        "Azure",
        "GCP",
        "Spark",
        "Docker",
        "Kubernetes",
        "FastAPI",
        "PyTorch",
        "TensorFlow",
        "scikit-learn",
        "LLM",
        "RAG",
        "MLOps",
        "communication",
        "leadership",
        "stakeholder management",
        "experimentation",
    )

    async def generate_structured(
        self, system_prompt: str, user_prompt: str, output_schema: type[T]
    ) -> T:
        data = _payload(user_prompt)
        if output_schema is RequirementExtraction:
            result = self._requirements(data)
        elif output_schema is GeneratedQuestion:
            result = self._question(data)
        elif output_schema is AnswerEvaluation:
            result = self._evaluation(data)
        else:
            raise StructuredOutputError(
                f"The deterministic demo does not implement {output_schema.__name__}"
            )
        return output_schema.model_validate(result)

    def _requirements(self, data: dict) -> RequirementExtraction:
        jd = str(data.get("job_description", ""))
        lower = jd.lower()
        skills = [skill for skill in self.KNOWN_SKILLS if skill.lower() in lower]
        bullet_lines = [
            re.sub(r"^[\s*\-•]+", "", line).strip()
            for line in jd.splitlines()
            if re.match(r"^[\s]*[*\-•]", line) and len(line.split()) >= 3
        ]
        role = str(data.get("role") or "Practice role")
        return RequirementExtraction(
            target_role=role,
            responsibilities=bullet_lines[:5],
            required_skills=skills[:10],
            preferred_skills=[],
            evaluation_topics=skills[:6] or ["role-specific experience", "problem solving"],
        )

    def _question(self, data: dict) -> GeneratedQuestion:
        action = data.get("action", "change_topic")
        evidence = data.get("evidence", [])
        topic = str(data.get("topic") or "role-specific experience")
        ids = [item["chunk_id"] for item in evidence if "chunk_id" in item][:2]
        last_answer = str(data.get("last_answer", ""))
        if action == "clarify":
            question = (
                f"Could you give a specific example that demonstrates your experience with {topic}?"
            )
        elif action == "probe":
            claim = " ".join(last_answer.split()[:14]).rstrip(".,")
            question = f"You mentioned “{claim}.” What approach and validation evidence supported that result?"
        else:
            question = (
                f"Tell me about a concrete situation where you applied {topic}. "
                "What did you do, why, and what measurable result followed?"
            )
        return GeneratedQuestion(
            question=question,
            topic=topic,
            evidence_ids=ids,
            rationale=f"Grounded practice question for {topic}",
        )

    def _evaluation(self, data: dict) -> AnswerEvaluation:
        transcript = str(data.get("transcript", "")).strip()
        words = transcript.split()
        lower = transcript.lower()
        evidence = data.get("evidence", [])
        evidence_ids = [item["chunk_id"] for item in evidence if item.get("score", 0) > 0][:3]
        length_score = min(9, max(2, 2 + len(words) // 18))
        has_structure = any(
            term in lower for term in ("situation", "task", "action", "result", "first", "then")
        )
        has_metric = bool(
            re.search(
                r"\b\d+(?:\.\d+)?\s*(?:%|percent|seconds?|hours?|days?|x|users?|records?)?\b",
                lower,
            )
        )
        technical_terms = sum(
            term in lower
            for term in (
                "model",
                "api",
                "database",
                "validation",
                "metric",
                "pipeline",
                "algorithm",
                "test",
                "deploy",
            )
        )
        relevance = min(10, length_score + (2 if evidence_ids else 0))
        clarity = min(10, length_score + (1 if len(words) >= 35 else 0))
        structure = min(10, length_score + (2 if has_structure else 0))
        depth = min(10, 3 + technical_terms + (1 if len(words) >= 60 else 0))
        evidence_score = min(10, 3 + (3 if has_metric else 0) + (2 if evidence_ids else 0))
        strengths: list[str] = []
        if len(words) >= 35:
            strengths.append("Provided enough detail to evaluate the approach")
        if has_structure:
            strengths.append("Used a recognizable answer structure")
        if has_metric:
            strengths.append("Included a measurable detail")
        if not strengths:
            strengths.append("Attempted to address the question directly")
        improvements: list[str] = []
        if len(words) < 35:
            improvements.append("Add a specific situation, your actions, and the outcome")
        if not has_structure:
            improvements.append("Use STAR or a problem-approach-validation-result structure")
        if not has_metric:
            improvements.append("Quantify the scale or result when evidence is available")
        if technical_terms < 2:
            improvements.append("Explain key technical decisions and trade-offs")
        suggestion = (
            "clarify"
            if len(words) < 20 or relevance <= 4
            else "probe"
            if depth <= 6
            else "change_topic"
        )
        return AnswerEvaluation(
            scores=Scorecard(
                relevance=relevance,
                clarity=clarity,
                structure=structure,
                technical_depth=depth,
                evidence=evidence_score,
            ),
            strengths=strengths[:3],
            improvements=improvements[:3],
            improved_answer_outline=[
                "Context: state the situation or problem and constraints",
                "Contribution: explain your decisions, actions, and trade-offs",
                "Evidence: give validation, measurable outcome, and learning",
            ],
            evidence_used=evidence_ids,
            next_action_suggestion=suggestion,
            suggested_next_question="Please expand on the most important decision and how you validated it.",
        )


class OpenAICompatibleGateway:
    """Calls a configured OpenAI-compatible chat-completions endpoint."""

    provider_name = "openai_compatible"
    def __init__(self, settings: Settings) -> None:
        self._model = settings.llm_model
        self.model_name = settings.llm_model
        self._url = settings.llm_base_url.rstrip("/") + "/chat/completions"
        self._api_key = settings.llm_api_key
        self._timeout = settings.llm_timeout_seconds
        self._max_tokens = settings.llm_max_tokens
        self._seed = settings.llm_seed
        self._max_retries = settings.llm_max_retries
        self.structured_output_mode = settings.llm_structured_output_mode
        self._reasoning_effort = settings.llm_reasoning_effort
        if not self._reasoning_effort and "qwen" in self._model.lower():
            self._reasoning_effort = "none"

    async def generate_structured(
        self, system_prompt: str, user_prompt: str, output_schema: type[T]
    ) -> T:
        validation_feedback = ""
        for attempt in range(self._max_retries + 1):
            schema_instruction = json.dumps(output_schema.model_json_schema())
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": f"{user_prompt}\nReturn JSON matching this schema: {schema_instruction}{validation_feedback}",
                },
            ]
            headers = {"Content-Type": "application/json"}
            if self._api_key:
                headers["Authorization"] = f"Bearer {self._api_key}"
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response_format: dict[str, Any]
                    if self.structured_output_mode == "json_object":
                        response_format = {"type": "json_object"}
                    else:
                        response_format = {
                            "type": "json_schema",
                            "json_schema": {
                                "name": output_schema.__name__,
                                "strict": True,
                                "schema": output_schema.model_json_schema(),
                            },
                        }
                    payload: dict[str, Any] = {
                        "model": self._model,
                        "messages": messages,
                        "temperature": 0,
                        "seed": self._seed,
                        "max_completion_tokens": self._max_tokens,
                        "response_format": response_format,
                    }
                    if self._reasoning_effort:
                        payload["reasoning_effort"] = self._reasoning_effort
                    response = await client.post(
                        self._url,
                        headers=headers,
                        json=payload,
                    )
                    if (
                        (
                            response.status_code in {408, 429, 500, 502, 503, 504}
                            or (
                                response.status_code == 400
                                and "json_validate_failed" in response.text
                            )
                        )
                        and attempt < self._max_retries
                    ):
                        await asyncio.sleep(0.5 * (2**attempt))
                        continue
                    response.raise_for_status()
                    content = response.json()["choices"][0]["message"]["content"]
                content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
                return output_schema.model_validate_json(content)
            except (KeyError, json.JSONDecodeError, ValidationError) as exc:
                validation_feedback = f"\nPrevious output failed validation: {exc}"
                if attempt == self._max_retries:
                    raise StructuredOutputError(
                        f"Model returned invalid structured output after {attempt + 1} attempts"
                    ) from exc
            except httpx.HTTPStatusError as exc:
                detail = exc.response.text.strip().replace("\n", " ")[:500]
                raise StructuredOutputError(
                    f"LLM provider returned HTTP {exc.response.status_code}"
                    f"{': ' + detail if detail else ''}"
                ) from exc
            except httpx.HTTPError as exc:
                raise StructuredOutputError(f"LLM request failed: {type(exc).__name__}") from exc
        raise StructuredOutputError("Structured generation failed")


class OllamaGateway:
    """Local Ollama adapter using native JSON-schema constrained generation."""

    provider_name = "ollama"
    structured_output_mode = "native_json_schema"

    def __init__(self, settings: Settings) -> None:
        self.model_name = settings.llm_model
        self._url = settings.llm_base_url.rstrip("/") + "/api/chat"
        self._timeout = settings.llm_timeout_seconds
        self._max_tokens = settings.llm_max_tokens
        self._seed = settings.llm_seed
        self._max_retries = settings.llm_max_retries

    async def generate_structured(
        self, system_prompt: str, user_prompt: str, output_schema: type[T]
    ) -> T:
        validation_feedback = ""
        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": user_prompt + validation_feedback,
                },
            ]
            payload = {
                "model": self.model_name,
                "messages": messages,
                "stream": False,
                "format": _ollama_compatible_schema(output_schema.model_json_schema()),
                "options": {
                    "temperature": 0,
                    "seed": self._seed,
                    "num_predict": self._max_tokens,
                },
            }
            try:
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(self._timeout, connect=min(10, self._timeout))
                ) as client:
                    response = await client.post(self._url, json=payload)
                    if (
                        response.status_code in {408, 429, 500, 502, 503, 504}
                        and attempt < self._max_retries
                    ):
                        await asyncio.sleep(0.5 * (2**attempt))
                        continue
                    response.raise_for_status()
                    content = response.json()["message"]["content"]
                if not isinstance(content, str):
                    raise TypeError("Ollama message.content was not text")
                content = _strip_json_fence(content)
                return output_schema.model_validate_json(content)
            except (KeyError, TypeError, json.JSONDecodeError, ValidationError) as exc:
                last_error = exc
                validation_feedback = (
                    "\nYour previous output failed application validation. Return a corrected JSON "
                    f"object only. Validation error: {str(exc)[:600]}"
                )
            except httpx.HTTPStatusError as exc:
                detail = exc.response.text.strip().replace("\n", " ")[:400]
                raise StructuredOutputError(
                    f"Ollama request failed with HTTP {exc.response.status_code}"
                    f"{': ' + detail if detail else ''}"
                ) from exc
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt < self._max_retries:
                    await asyncio.sleep(0.5 * (2**attempt))
                    continue
            if attempt < self._max_retries:
                continue
        raise StructuredOutputError(
            f"Ollama returned no valid {output_schema.__name__} after "
            f"{self._max_retries + 1} attempts: {type(last_error).__name__ if last_error else 'unknown'}"
        ) from last_error


def _strip_json_fence(content: str) -> str:
    return re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())


def _ollama_compatible_schema(value: Any) -> Any:
    """Keep provider grammar compact; Pydantic still enforces the complete schema locally."""

    omitted = {
        "title",
        "description",
        "default",
        "examples",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "minimum",
        "maximum",
        "pattern",
    }
    if isinstance(value, dict):
        return {
            key: _ollama_compatible_schema(item)
            for key, item in value.items()
            if key not in omitted
        }
    if isinstance(value, list):
        return [_ollama_compatible_schema(item) for item in value]
    return value


def create_gateway(settings: Settings) -> LLMGateway:
    if settings.llm_provider == "ollama":
        return OllamaGateway(settings)
    if settings.llm_provider == "openai_compatible":
        return OpenAICompatibleGateway(settings)
    return DeterministicDemoGateway()
