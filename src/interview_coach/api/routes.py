from __future__ import annotations

import asyncio
import os
import tempfile
from pathlib import Path
from typing import Literal

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from interview_coach.exceptions import AudioError
from interview_coach.interview.orchestrator import InterviewOrchestrator
from interview_coach.rag.parser import parse_document, parse_pasted_text


class CreateSessionRequest(BaseModel):
    role: str = Field(min_length=1)
    interview_type: Literal["behavioral", "technical", "mixed"] = "mixed"
    difficulty: Literal["beginner", "intermediate", "advanced"] = "intermediate"
    max_questions: int = Field(default=5, ge=1, le=10)


class DocumentsRequest(BaseModel):
    resume_text: str = Field(min_length=1)
    job_description: str = Field(min_length=1)


class AnswerRequest(BaseModel):
    transcript: str = ""


class TranscriptPatch(BaseModel):
    transcript: str = Field(min_length=1)


def build_router(orchestrator: InterviewOrchestrator) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    async def health() -> dict:
        llm_status = "available"
        limitations: list[str] = []
        if orchestrator.gateway.provider_name == "ollama":
            try:
                async with httpx.AsyncClient(timeout=3) as client:
                    response = await client.get(
                        orchestrator.settings.llm_base_url.rstrip("/") + "/api/tags"
                    )
                    response.raise_for_status()
                    installed = {
                        str(item.get("name") or item.get("model"))
                        for item in response.json().get("models", [])
                        if isinstance(item, dict)
                    }
                if orchestrator.gateway.model_name not in installed:
                    llm_status = "unavailable"
                    limitations.append(
                        f"Configured model {orchestrator.gateway.model_name!r} is not installed."
                    )
            except (httpx.HTTPError, KeyError, TypeError, ValueError):
                llm_status = "unavailable"
                limitations.append(
                    "Ollama is not reachable; real question generation is unavailable."
                )
        elif orchestrator.gateway.provider_name == "deterministic_demo":
            llm_status = "degraded"
            limitations.append("Generation uses transparent deterministic rules, not an LLM.")

        retrieval_status = "degraded" if orchestrator.rag.mode == "lexical_demo" else "available"
        if retrieval_status == "degraded":
            limitations.append("Retrieval is lexical-only; semantic embeddings are disabled.")
        stt_status = "unavailable" if orchestrator.stt.model_id == "unavailable" else "available"
        speech_status = (
            "unavailable" if orchestrator.speech_emotion.model_id == "unavailable" else "available"
        )
        text_status = (
            "unavailable" if orchestrator.text_emotion.model_id == "unavailable" else "available"
        )
        if stt_status == "unavailable":
            limitations.append("Speech-to-text is unavailable; typed transcripts remain supported.")
        if speech_status == "unavailable" or text_status == "unavailable":
            limitations.append(
                "One or more optional communication-cue models are unavailable; content scoring is unaffected."
            )
        return {
            "status": "ready" if llm_status == retrieval_status == "available" else "degraded",
            "components": {
                "llm": f"{orchestrator.gateway.provider_name} / {orchestrator.gateway.model_name}",
                "retrieval": orchestrator.rag.mode,
                "stt": orchestrator.stt.model_id,
                "speech_emotion": orchestrator.speech_emotion.model_id,
                "text_emotion": orchestrator.text_emotion.model_id,
            },
            "component_status": {
                "llm": llm_status,
                "retrieval": retrieval_status,
                "stt": stt_status,
                "speech_emotion": speech_status,
                "text_emotion": text_status,
            },
            "runtime": {
                "provider": orchestrator.gateway.provider_name,
                "model": orchestrator.gateway.model_name,
                "structured_output": orchestrator.gateway.structured_output_mode,
                "retrieval": orchestrator.rag.mode,
                "embedding_model": orchestrator.settings.embedding_model,
            },
            "limitations": list(dict.fromkeys(limitations)),
        }

    @router.post("/sessions", status_code=201)
    async def create_session(payload: CreateSessionRequest):
        return orchestrator.create_session(**payload.model_dump())

    @router.post("/sessions/{session_id}/documents")
    async def documents(session_id: str, request: Request):
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            payload = DocumentsRequest.model_validate(await request.json())
            resume = parse_pasted_text(
                payload.resume_text, "Resume", orchestrator.settings.max_document_bytes
            )
            jd = parse_pasted_text(
                payload.job_description,
                "Job description",
                orchestrator.settings.max_document_bytes,
            )
        else:
            form = await request.form()
            resume_text = str(form.get("resume_text") or "")
            resume_file = form.get("resume_file")
            if resume_file and getattr(resume_file, "filename", None):
                content = await resume_file.read()
                resume = parse_document(
                    filename=resume_file.filename,
                    content=content,
                    max_bytes=orchestrator.settings.max_document_bytes,
                )
            else:
                resume = parse_pasted_text(
                    resume_text, "Resume", orchestrator.settings.max_document_bytes
                )
            jd = parse_pasted_text(
                str(form.get("job_description") or ""),
                "Job description",
                orchestrator.settings.max_document_bytes,
            )
        return await orchestrator.index_documents(session_id, resume, jd)

    @router.post("/sessions/{session_id}/start")
    async def start(session_id: str):
        return await orchestrator.start(session_id)

    @router.post("/sessions/{session_id}/answers")
    async def answer(session_id: str, request: Request):
        content_type = request.headers.get("content-type", "")
        temp_name: str | None = None
        try:
            if "application/json" in content_type:
                payload = AnswerRequest.model_validate(await request.json())
                transcript = payload.transcript
                audio_path = None
            else:
                form = await request.form()
                transcript = str(form.get("transcript") or "")
                audio = form.get("audio")
                audio_path = None
                if audio and getattr(audio, "filename", None):
                    suffix = Path(audio.filename).suffix[:10] or ".wav"
                    handle, temp_name = tempfile.mkstemp(suffix=suffix, prefix="coach-upload-")
                    with os.fdopen(handle, "wb") as output:
                        total = 0
                        while chunk := await audio.read(1024 * 1024):
                            total += len(chunk)
                            if total > orchestrator.settings.max_audio_bytes:
                                raise AudioError("Audio upload exceeds configured size limit")
                            output.write(chunk)
                    audio_path = temp_name
            return await orchestrator.process_answer(session_id, transcript, audio_path)
        finally:
            if temp_name:
                await asyncio.to_thread(Path(temp_name).unlink, missing_ok=True)

    @router.patch("/sessions/{session_id}/turns/{turn_number}/transcript")
    async def patch_transcript(session_id: str, turn_number: int, payload: TranscriptPatch):
        return await orchestrator.correct_transcript(session_id, turn_number, payload.transcript)

    @router.get("/sessions/{session_id}")
    async def get_session(session_id: str):
        return orchestrator.store.get(session_id)

    @router.post("/sessions/{session_id}/complete")
    async def complete(session_id: str):
        return orchestrator.complete(session_id)

    @router.get("/sessions/{session_id}/report")
    async def report(session_id: str, format: Literal["json", "markdown"] = "json"):
        result = orchestrator.report(session_id)
        if format == "markdown":
            return PlainTextResponse(
                result.markdown,
                media_type="text/markdown",
                headers={
                    "Content-Disposition": f'attachment; filename="interview-report-{session_id}.md"'
                },
            )
        return result

    return router
