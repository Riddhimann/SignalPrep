from __future__ import annotations

import asyncio
import json
import tempfile
from pathlib import Path

from interview_coach.container import create_container
from interview_coach.rag.parser import parse_document, parse_pasted_text


def build_demo():
    try:
        import gradio as gr
    except ImportError as exc:
        raise RuntimeError("The browser UI requires: pip install -e '.[ui]'") from exc

    container = create_container()
    coach = container.orchestrator
    project_root = Path(__file__).resolve().parent
    sample_resume = (project_root / "data/samples/sample_resume.txt").read_text(encoding="utf-8")
    sample_jd = (project_root / "data/samples/sample_jd.txt").read_text(encoding="utf-8")

    def setup(role, interview_type, difficulty, count, resume_file, resume_text, jd):
        try:
            resume = resume_text or ""
            if resume_file:
                path = Path(resume_file)
                resume = parse_document(
                    filename=path.name,
                    content=path.read_bytes(),
                    max_bytes=container.settings.max_document_bytes,
                )
            resume = parse_pasted_text(resume, "Resume", container.settings.max_document_bytes)
            jd = parse_pasted_text(jd, "Job description", container.settings.max_document_bytes)
            session = coach.create_session(role, interview_type, difficulty, int(count))
            asyncio.run(coach.index_documents(session.session_id, resume, jd))
            session = asyncio.run(coach.start(session.session_id))
            modes = "\n".join(f"- {item}" for item in session.degraded_modes)
            status = (
                "### Interview ready\n\n"
                "The Interview tab is now active. Answer this first question:\n\n"
                f"> {session.current_question}\n\n"
                "#### Current model status\n\n"
                f"{modes or '- All configured components available.'}"
            )
            gr.Info("Interview created. Opening the Interview tab now.")
            return (
                session.session_id,
                session.current_question,
                status,
                "",
                {},
                "",
                gr.Tabs(selected="interview"),
            )
        except Exception as exc:
            message = f"{type(exc).__name__}: {exc}"
            gr.Warning(f"Setup failed: {message}")
            return (
                "",
                "Create an interview after correcting the Setup error.",
                f"### Setup error\n\n{message}",
                "",
                {},
                "",
                gr.Tabs(selected="setup"),
            )

    def submit(session_id, audio_path, transcript):
        if not session_id:
            raise gr.Error("Create an interview first.")
        result = asyncio.run(coach.process_answer(session_id, transcript or "", audio_path))
        turn = result.turn
        feedback = {
            "scores": turn.evaluation.scores.model_dump(),
            "strengths": turn.evaluation.strengths,
            "improvements": turn.evaluation.improvements,
            "improved_answer_outline": turn.evaluation.improved_answer_outline,
            "controller_action": turn.controller_action,
            "communication_cue": turn.communication_signal.model_dump(),
        }
        evidence = "\n\n".join(
            f"**{item.chunk_id} ({item.source}, {item.score:.2f})**\n\n{item.text}"
            for item in turn.evidence
        ) or "No useful evidence retrieved; feedback is generic and does not invent context."
        next_question = turn.next_question or "Interview complete — generate your report."
        return turn.transcript, next_question, feedback, evidence

    def make_report(session_id):
        if not session_id:
            raise gr.Error("No interview session exists.")
        result = coach.complete(session_id)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", prefix="interview-report-", encoding="utf-8", delete=False
        ) as output:
            output.write(result.markdown)
        return result.markdown, output.name

    with gr.Blocks(title="AI Emotion-Aware Interview Coach") as demo:
        gr.Markdown(
            "# AI Emotion-Aware Interview Coach\n"
            "Practice-only coaching. Communication cues are uncertain and never affect content scores."
        )
        session_id = gr.State("")
        with gr.Tabs(selected="setup") as tabs:
            with gr.Tab("1. Setup", id="setup"):
                with gr.Row():
                    role = gr.Textbox(label="Target role", value="Data Scientist")
                    interview_type = gr.Dropdown(
                        ["behavioral", "technical", "mixed"], value="mixed", label="Interview type"
                    )
                    difficulty = gr.Dropdown(
                        ["beginner", "intermediate", "advanced"], value="intermediate", label="Difficulty"
                    )
                    count = gr.Slider(1, 10, value=5, step=1, label="Questions")
                model_status = gr.Markdown(
                    "### Ready for setup\n\nSample data is loaded. Click **Create and start interview**."
                )
                resume_file = gr.File(
                    label="Resume (.txt or .pdf)", file_types=[".txt", ".pdf"], type="filepath"
                )
                resume_text = gr.Textbox(
                    label="Or paste resume text (sample loaded for your first test)",
                    lines=8,
                    value=sample_resume,
                )
                jd = gr.Textbox(
                    label="Job description (sample loaded for your first test)",
                    lines=10,
                    value=sample_jd,
                )
                create = gr.Button("Create and start interview", variant="primary")
            with gr.Tab("2. Interview", id="interview"):
                question = gr.Markdown("Create an interview to receive the first question.")
                audio = gr.Audio(
                    sources=["microphone", "upload"], type="filepath", label="Optional spoken answer"
                )
                transcript = gr.Textbox(
                    label="Transcript (editable; required when STT is unavailable)", lines=8
                )
                submit_button = gr.Button("Evaluate answer", variant="primary")
            with gr.Tab("3. Feedback", id="feedback"):
                feedback = gr.JSON(label="Scores and coaching feedback")
                evidence = gr.Markdown()
            with gr.Tab("4. Final report", id="report"):
                report_button = gr.Button("Complete interview and generate report")
                report = gr.Markdown()
                report_file = gr.File(label="Download Markdown report")

        create.click(
            setup,
            [role, interview_type, difficulty, count, resume_file, resume_text, jd],
            [session_id, question, model_status, transcript, feedback, evidence, tabs],
            scroll_to_output=True,
            show_progress="full",
        )
        submit_button.click(
            submit, [session_id, audio, transcript], [transcript, question, feedback, evidence]
        )
        report_button.click(make_report, [session_id], [report, report_file])
    return demo


if __name__ == "__main__":
    # The portfolio application is the React production build served by FastAPI.
    # build_demo() remains importable only as a lightweight adapter/debugging surface.
    import uvicorn

    uvicorn.run("interview_coach.api.app:app", host="127.0.0.1", port=8000, reload=False)
