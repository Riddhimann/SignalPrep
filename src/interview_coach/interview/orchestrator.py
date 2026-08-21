from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from uuid import uuid4

from interview_coach.audio.preprocessing import preprocess_audio
from interview_coach.audio.speech_emotion import SpeechEmotionAnalyzer
from interview_coach.audio.stt import SpeechToText
from interview_coach.config import Settings
from interview_coach.exceptions import (
    InvalidSessionStateError,
    ModelUnavailableError,
    StructuredOutputError,
)
from interview_coach.interview.controller import ControllerAction, choose_action
from interview_coach.interview.report import build_report
from interview_coach.interview.requirements import ground_requirement_extraction
from interview_coach.interview.rubric import RUBRIC
from interview_coach.interview.scoring import calibrate_evaluation
from interview_coach.llm.gateway import LLMGateway
from interview_coach.llm.prompts import (
    EVALUATION_SYSTEM,
    PROMPT_VERSION,
    QUESTION_SYSTEM,
    REQUIREMENT_SYSTEM,
    RUBRIC_VERSION,
)
from interview_coach.nlp.fusion import combine_signals
from interview_coach.nlp.text_emotion import TextEmotionAnalyzer
from interview_coach.rag.grounding import audit_grounding
from interview_coach.rag.retriever import RagService
from interview_coach.rag.safety import quarantine_prompt_injection
from interview_coach.schemas import (
    AnswerEvaluation,
    FinalReport,
    GeneratedQuestion,
    GroundingAudit,
    InterviewSession,
    InterviewTurn,
    RequirementExtraction,
    RetrievedEvidence,
    RuntimeProvenance,
    TurnResult,
)
from interview_coach.storage.session_store import SessionStore

logger = logging.getLogger(__name__)


class InterviewOrchestrator:
    def __init__(
        self,
        settings: Settings,
        store: SessionStore,
        rag: RagService,
        gateway: LLMGateway,
        stt: SpeechToText,
        speech_emotion: SpeechEmotionAnalyzer,
        text_emotion: TextEmotionAnalyzer,
    ) -> None:
        self.settings = settings
        self.store = store
        self.rag = rag
        self.gateway = gateway
        self.stt = stt
        self.speech_emotion = speech_emotion
        self.text_emotion = text_emotion

    def create_session(
        self,
        role: str,
        interview_type: str = "mixed",
        difficulty: str = "intermediate",
        max_questions: int | None = None,
    ) -> InterviewSession:
        session = InterviewSession(
            session_id=str(uuid4()),
            role=role.strip() or "Practice role",
            interview_type=interview_type,
            difficulty=difficulty,
            max_questions=max_questions or self.settings.max_interview_questions,
            runtime=RuntimeProvenance(
                generation_provider=self.gateway.provider_name,
                model_name=self.gateway.model_name,
                structured_output_mode=self.gateway.structured_output_mode,
                retrieval_configured=self.rag.mode,
                retrieval_effective=self.rag.mode,
                embedding_model=self.settings.embedding_model,
                prompt_version=PROMPT_VERSION,
                rubric_version=RUBRIC_VERSION,
            ),
        )
        if self.rag.mode == "lexical_demo":
            session.degraded_modes.append(
                "Retrieval uses transparent lexical TF-IDF demo mode, not semantic embeddings."
            )
        if self.gateway.provider_name == "deterministic_demo":
            session.degraded_modes.append(
                "Generation/evaluation uses deterministic demo rules, not an LLM."
            )
        if self.stt.model_id == "unavailable":
            session.degraded_modes.append(
                "Speech-to-text unavailable; enter or correct the transcript."
            )
        if self.speech_emotion.model_id == "unavailable":
            session.degraded_modes.append("Speech communication-cue model unavailable.")
        if self.text_emotion.model_id == "unavailable":
            session.degraded_modes.append("Text communication-cue model unavailable.")
        return self.store.create(session)

    async def index_documents(
        self, session_id: str, resume: str, job_description: str
    ) -> InterviewSession:
        session = self.store.get(session_id)
        safe_job_description, injection_flags = quarantine_prompt_injection(job_description)
        if injection_flags:
            session.security_events.append(
                "Quarantined untrusted job-description instruction patterns: "
                + ", ".join(injection_flags)
            )
        requirements = await self.gateway.generate_structured(
            REQUIREMENT_SYSTEM,
            json.dumps({"role": session.role, "job_description": safe_job_description}),
            RequirementExtraction,
        )
        requirements, removed_requirements = ground_requirement_extraction(
            requirements, safe_job_description
        )
        if removed_requirements:
            session.security_events.append(
                "Removed unsupported requirement-extraction items: "
                + "; ".join(removed_requirements[:8])
            )
        await asyncio.to_thread(self.rag.index_session, session_id, resume, job_description)
        if session.runtime:
            session.runtime.retrieval_effective = self.rag.effective_mode(session_id)
        if degraded := self.rag.degraded_message(session_id):
            session.degraded_modes.append(degraded)
        session.required_skills = requirements.required_skills or requirements.evaluation_topics
        session.status = "ready"
        return self.store.save(session)

    async def start(self, session_id: str) -> InterviewSession:
        session = self.store.get(session_id)
        if session.status not in {"ready", "active"} or not self.rag.has_index(session_id):
            raise InvalidSessionStateError("Index documents before starting the interview")
        if session.current_question:
            return session
        topic = (
            session.required_skills[0] if session.required_skills else "role-specific experience"
        )
        evidence = await asyncio.to_thread(self.rag.search, session_id, f"{session.role} {topic}")
        generated, grounding = await self._generate_question(
            session, "change_topic", topic, evidence, ""
        )
        session.current_question = generated.question
        session.current_topic = generated.topic
        session.current_question_evidence_ids = generated.evidence_ids
        session.current_question_grounding = grounding
        session.status = "active"
        return self.store.save(session)

    async def process_answer(
        self,
        session_id: str,
        transcript: str = "",
        audio_path: str | Path | None = None,
    ) -> TurnResult:
        session = self.store.get(session_id)
        if session.status != "active" or not session.current_question:
            raise InvalidSessionStateError("The interview is not awaiting an answer")

        speech_prediction = None
        prepared = None
        try:
            if audio_path:
                prepared = preprocess_audio(
                    audio_path,
                    max_bytes=self.settings.max_audio_bytes,
                    max_seconds=self.settings.max_audio_seconds,
                )
                tasks = [asyncio.to_thread(self.speech_emotion.predict, prepared.path)]
                if not transcript.strip():
                    tasks.append(
                        asyncio.to_thread(
                            self.stt.transcribe,
                            prepared.path,
                            prepared.duration_seconds,
                        )
                    )
                results = await asyncio.gather(*tasks, return_exceptions=True)
                speech_prediction = None if isinstance(results[0], Exception) else results[0]
                if not transcript.strip() and len(results) > 1:
                    if isinstance(results[1], Exception):
                        raise ModelUnavailableError(str(results[1]))
                    transcript = results[1].text
            transcript = transcript.strip()
            if not transcript:
                raise InvalidSessionStateError(
                    "Transcript is empty. Record again or enter a corrected transcript."
                )

            try:
                text_prediction = await asyncio.to_thread(self.text_emotion.predict, transcript)
            except Exception:  # noqa: BLE001 - optional cue-model failures must not block content scoring
                text_prediction = None
            communication = combine_signals(speech_prediction, text_prediction)
            evidence = await asyncio.to_thread(self.rag.search, session_id, transcript)
            evaluation = await self._evaluate(
                session,
                session.current_question,
                transcript,
                evidence,
                communication.model_dump(),
            )
            action = choose_action(session, evaluation, transcript)

            if session.current_topic and session.current_topic not in session.covered_skills:
                session.covered_skills.append(session.current_topic)
            next_question: str | None = None
            next_topic: str | None = None
            if action != "finish":
                next_topic = self._choose_topic(session, action)
                next_evidence = await asyncio.to_thread(
                    self.rag.search, session_id, f"{next_topic} {transcript}"
                )
                generated, next_grounding = await self._generate_question(
                    session, action, next_topic, next_evidence, transcript
                )
                next_question = generated.question
            else:
                generated = None
                next_grounding = None

            turn = InterviewTurn(
                turn_number=len(session.turns) + 1,
                question=session.current_question,
                topic=session.current_topic or "general",
                transcript=transcript,
                speech_emotion=speech_prediction,
                text_emotion=text_prediction,
                communication_signal=communication,
                evidence=evidence,
                question_evidence_ids=session.current_question_evidence_ids,
                question_grounding=session.current_question_grounding,
                evaluation=evaluation,
                controller_action=action,
                next_question=next_question,
            )
            session.turns.append(turn)
            session.current_question = next_question
            session.current_topic = next_topic
            session.current_question_evidence_ids = generated.evidence_ids if generated else []
            session.current_question_grounding = next_grounding
            session.status = "completed" if action == "finish" else "active"
            self.store.save(session)
            return TurnResult(turn=turn, session_status=session.status)
        finally:
            if prepared:
                prepared.cleanup()

    async def correct_transcript(
        self, session_id: str, turn_number: int, transcript: str
    ) -> InterviewSession:
        session = self.store.get(session_id)
        if turn_number < 1 or turn_number > len(session.turns):
            raise InvalidSessionStateError("Turn number does not exist")
        corrected = transcript.strip()
        if not corrected:
            raise InvalidSessionStateError("Corrected transcript cannot be empty")
        turn = session.turns[turn_number - 1]
        evidence = await asyncio.to_thread(self.rag.search, session_id, corrected)
        try:
            text_prediction = await asyncio.to_thread(self.text_emotion.predict, corrected)
        except Exception:  # noqa: BLE001 - optional cue-model failures must not block correction
            text_prediction = None
        communication = combine_signals(turn.speech_emotion, text_prediction)
        evaluation = await self._evaluate(
            session,
            turn.question,
            corrected,
            evidence,
            communication.model_dump(),
        )
        turn.transcript = corrected
        turn.text_emotion = text_prediction
        turn.communication_signal = communication
        turn.evidence = evidence
        turn.evaluation = evaluation
        session.turns[turn_number - 1] = turn
        return self.store.save(session)

    def complete(self, session_id: str) -> FinalReport:
        session = self.store.get(session_id)
        if not session.turns:
            raise InvalidSessionStateError("Complete at least one turn before generating a report")
        session.status = "completed"
        session.current_question = None
        self.store.save(session)
        return build_report(session)

    def report(self, session_id: str) -> FinalReport:
        return build_report(self.store.get(session_id))

    async def _generate_question(
        self,
        session: InterviewSession,
        action: ControllerAction,
        topic: str,
        evidence: list[RetrievedEvidence],
        last_answer: str,
    ) -> tuple[GeneratedQuestion, GroundingAudit]:
        payload = {
            "role": session.role,
            "difficulty": session.difficulty,
            "interview_type": session.interview_type,
            "covered_skills": session.covered_skills,
            "previous_questions": [turn.question for turn in session.turns],
            "action": action,
            "topic": topic,
            "last_answer": last_answer,
            "evidence": [item.model_dump() for item in evidence],
        }
        last_error: StructuredOutputError | None = None
        allowed_ids = ", ".join(item.chunk_id for item in evidence) or "none"
        for attempt in range(2):
            repair = (
                "\nThe previous question failed grounding validation. Cite supplied evidence and "
                "make the question substantively overlap the cited role/resume content."
                if attempt
                else ""
            )
            generated = await self.gateway.generate_structured(
                QUESTION_SYSTEM
                + f"\nThe only allowed evidence IDs for this request are: {allowed_ids}."
                + repair,
                json.dumps(payload),
                GeneratedQuestion,
            )
            try:
                self._check_evidence_ids(
                    generated.evidence_ids,
                    evidence,
                    require_citation=bool(evidence),
                )
                grounding = audit_grounding(
                    f"{generated.topic} {generated.question}",
                    generated.evidence_ids,
                    evidence,
                )
                if evidence and grounding.status == "ungrounded":
                    raise StructuredOutputError(
                        "Generated question was not supported by cited evidence"
                    )
                return generated, grounding
            except StructuredOutputError as exc:
                last_error = exc
        raise last_error or StructuredOutputError("Question grounding validation failed")

    async def _evaluate(
        self,
        session: InterviewSession,
        question: str,
        transcript: str,
        evidence: list[RetrievedEvidence],
        communication: dict,
    ) -> AnswerEvaluation:
        payload = {
            "role": session.role,
            "question": question,
            "transcript": transcript,
            "evidence": [item.model_dump() for item in evidence],
            "rubric": RUBRIC,
            "communication_cue_for_separate_feedback_only": communication,
        }
        last_error: StructuredOutputError | None = None
        allowed_ids = ", ".join(item.chunk_id for item in evidence) or "none"
        for attempt in range(2):
            repair = (
                "\nThe previous evaluation omitted or invented evidence citations. Cite at least one "
                "supplied ID when evidence is available and never cite any other ID."
                if attempt
                else ""
            )
            evaluation = await self.gateway.generate_structured(
                EVALUATION_SYSTEM
                + f"\nThe only allowed evidence IDs for this request are: {allowed_ids}."
                + repair,
                json.dumps(payload),
                AnswerEvaluation,
            )
            try:
                self._check_evidence_ids(
                    evaluation.evidence_used,
                    evidence,
                    require_citation=bool(evidence),
                )
                return calibrate_evaluation(evaluation, question, transcript)
            except StructuredOutputError as exc:
                last_error = exc
        raise last_error or StructuredOutputError("Evaluation evidence validation failed")

    @staticmethod
    def _check_evidence_ids(
        ids: list[str],
        evidence: list[RetrievedEvidence],
        *,
        require_citation: bool = False,
    ) -> None:
        allowed = {item.chunk_id for item in evidence}
        unsupported = set(ids) - allowed
        if unsupported:
            raise StructuredOutputError(
                f"Model cited evidence that was not supplied: {sorted(unsupported)}"
            )
        if require_citation and not ids:
            raise StructuredOutputError("Model omitted required evidence citations")

    @staticmethod
    def _choose_topic(session: InterviewSession, action: ControllerAction) -> str:
        if action in {"probe", "clarify"} and session.current_topic:
            return session.current_topic
        return next(
            (skill for skill in session.required_skills if skill not in session.covered_skills),
            "role-specific problem solving",
        )
