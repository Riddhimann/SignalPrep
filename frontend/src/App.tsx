import { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  ArrowRight,
  AudioLines,
  BarChart3,
  BrainCircuit,
  BriefcaseBusiness,
  Check,
  CheckCircle2,
  ChevronRight,
  CircleAlert,
  Clock3,
  FileText,
  Gauge,
  Layers3,
  LoaderCircle,
  LockKeyhole,
  MessageSquareText,
  Mic2,
  RotateCcw,
  ShieldCheck,
  Sparkles,
  Target,
  TrendingUp,
  Upload,
} from "lucide-react";
import { api } from "./api";
import { SAMPLE_JD, SAMPLE_RESUME } from "./sample";
import type { FinalReport, Health, Scorecard, Session, Turn } from "./types";

type Phase = "setup" | "interview" | "report";

const scoreLabels: Record<keyof Scorecard, string> = {
  relevance: "Relevance",
  clarity: "Clarity",
  structure: "Structure",
  technical_depth: "Technical depth",
  evidence: "Evidence",
};

function titleCase(value: string) {
  return value.replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function Brand() {
  return (
    <div className="brand" aria-label="SignalPrep AI Interview Lab">
      <div className="brand-mark"><Activity size={19} strokeWidth={2.4} /></div>
      <div>
        <strong>SignalPrep</strong>
        <span>AI INTERVIEW LAB</span>
      </div>
    </div>
  );
}

function StatusDot({ ok }: { ok: boolean }) {
  return <span className={`status-dot ${ok ? "online" : "offline"}`} />;
}

function ModelBadge({ health }: { health: Health | null }) {
  const ready = health?.status === "ready";
  return (
    <div className={`model-badge ${health?.status ?? "connecting"}`}>
      <StatusDot ok={ready} />
      <span>{!health ? "Connecting to API" : ready ? `Ready · ${health.runtime.model}` : "Degraded mode"}</span>
    </div>
  );
}

function App() {
  const [phase, setPhase] = useState<Phase>("setup");
  const [health, setHealth] = useState<Health | null>(null);
  const [session, setSession] = useState<Session | null>(null);
  const [lastTurn, setLastTurn] = useState<Turn | null>(null);
  const [report, setReport] = useState<FinalReport | null>(null);
  const [busy, setBusy] = useState(false);
  const [startStage, setStartStage] = useState("");
  const [startElapsed, setStartElapsed] = useState(0);
  const [error, setError] = useState("");
  const [role, setRole] = useState("Data Scientist");
  const [interviewType, setInterviewType] = useState("mixed");
  const [difficulty, setDifficulty] = useState("intermediate");
  const [questionCount, setQuestionCount] = useState(5);
  const [resumeText, setResumeText] = useState("");
  const [resumeFile, setResumeFile] = useState<File | null>(null);
  const [jobDescription, setJobDescription] = useState("");
  const [transcript, setTranscript] = useState("");
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [elapsed, setElapsed] = useState(0);
  const transcriptRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    api.health().then(setHealth).catch(() => setHealth(null));
  }, []);

  useEffect(() => {
    if (phase !== "interview") return;
    const timer = window.setInterval(() => setElapsed((value) => value + 1), 1000);
    return () => window.clearInterval(timer);
  }, [phase]);

  useEffect(() => {
    if (!busy || phase !== "setup") {
      setStartElapsed(0);
      return;
    }
    const timer = window.setInterval(() => setStartElapsed((value) => value + 1), 1000);
    return () => window.clearInterval(timer);
  }, [busy, phase]);

  const minutes = String(Math.floor(elapsed / 60)).padStart(2, "0");
  const seconds = String(elapsed % 60).padStart(2, "0");

  async function startInterview() {
    setError("");
    if (!role.trim() || (!resumeFile && !resumeText.trim()) || !jobDescription.trim()) {
      setError("Add a target role, resume, and job description before starting.");
      return;
    }
    setBusy(true);
    setStartStage("Creating the interview session");
    try {
      const created = await api.createSession({
        role: role.trim(), interview_type: interviewType, difficulty, max_questions: questionCount,
      });
      setStartStage("Extracting JD requirements and building the hybrid index");
      await api.indexDocuments(created.session_id, resumeFile ?? resumeText, jobDescription);
      setStartStage("Generating and grounding the first question");
      const started = await api.start(created.session_id);
      setSession(started);
      setElapsed(0);
      setPhase("interview");
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not start the interview.");
    } finally {
      setBusy(false);
      setStartStage("");
    }
  }

  async function submitAnswer() {
    if (!session) return;
    setError("");
    if (!transcript.trim() && !audioFile) {
      setError("Write an answer or attach a recording before requesting feedback.");
      transcriptRef.current?.focus();
      return;
    }
    setBusy(true);
    try {
      const result = await api.answer(session.session_id, transcript, audioFile);
      setLastTurn(result.turn);
      const updated = await api.getSession(session.session_id);
      setSession(updated);
      window.scrollTo({ top: 0, behavior: "smooth" });
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Could not evaluate this answer.");
    } finally {
      setBusy(false);
    }
  }

  async function continueInterview() {
    if (!session || !lastTurn) return;
    if (lastTurn.controller_action === "finish" || session.status === "completed") {
      setBusy(true);
      try {
        const completed = await api.complete(session.session_id);
        setReport(completed);
        setPhase("report");
        setLastTurn(null);
        window.scrollTo({ top: 0, behavior: "smooth" });
      } catch (cause) {
        setError(cause instanceof Error ? cause.message : "Could not generate the report.");
      } finally {
        setBusy(false);
      }
      return;
    }
    setTranscript("");
    setAudioFile(null);
    setLastTurn(null);
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  function loadSample() {
    setResumeFile(null);
    setResumeText(SAMPLE_RESUME);
    setJobDescription(SAMPLE_JD);
    setRole("Data Scientist");
    setError("");
  }

  function reset() {
    setPhase("setup");
    setSession(null);
    setLastTurn(null);
    setReport(null);
    setTranscript("");
    setAudioFile(null);
    setElapsed(0);
    setError("");
  }

  if (phase === "setup") {
    return (
      <SetupPage
        health={health} role={role} setRole={setRole}
        interviewType={interviewType} setInterviewType={setInterviewType}
        difficulty={difficulty} setDifficulty={setDifficulty}
        questionCount={questionCount} setQuestionCount={setQuestionCount}
        resumeText={resumeText} setResumeText={setResumeText}
        resumeFile={resumeFile} setResumeFile={setResumeFile}
        jobDescription={jobDescription} setJobDescription={setJobDescription}
        loadSample={loadSample} startInterview={startInterview} busy={busy}
        startStage={startStage} startElapsed={startElapsed} error={error}
      />
    );
  }

  if (phase === "report" && session && report) {
    return <ReportPage session={session} report={report} health={health} reset={reset} />;
  }

  return session ? (
    <InterviewPage
      session={session} health={health} lastTurn={lastTurn} transcript={transcript}
      setTranscript={setTranscript} audioFile={audioFile} setAudioFile={setAudioFile}
      busy={busy} error={error} submitAnswer={submitAnswer}
      continueInterview={continueInterview} time={`${minutes}:${seconds}`} transcriptRef={transcriptRef}
      reset={reset}
    />
  ) : null;
}

type SetupProps = {
  health: Health | null;
  role: string; setRole: (value: string) => void;
  interviewType: string; setInterviewType: (value: string) => void;
  difficulty: string; setDifficulty: (value: string) => void;
  questionCount: number; setQuestionCount: (value: number) => void;
  resumeText: string; setResumeText: (value: string) => void;
  resumeFile: File | null; setResumeFile: (value: File | null) => void;
  jobDescription: string; setJobDescription: (value: string) => void;
  loadSample: () => void; startInterview: () => void; busy: boolean;
  startStage: string; startElapsed: number; error: string;
};

function SetupPage(props: SetupProps) {
  return (
    <div className="app-shell setup-page">
      <header className="topbar">
        <Brand />
        <div className="topbar-actions">
          <span className="privacy-note"><LockKeyhole size={14} /> In-memory session</span>
          <ModelBadge health={props.health} />
        </div>
      </header>

      <main className="setup-main">
        <section className="setup-intro">
          <div className="eyebrow"><Sparkles size={15} /> EVIDENCE-AWARE PRACTICE</div>
          <h1>Train the reasoning behind<br /><em>strong interview answers.</em></h1>
          <p className="hero-copy">
            A structured practice environment that connects your experience to the role, challenges vague
            claims, and turns every response into an actionable improvement plan.
          </p>
          <div className="value-grid">
            <div><Target size={19} /><span><strong>Role grounded</strong>Questions use resume and JD evidence</span></div>
            <div><BrainCircuit size={19} /><span><strong>Adaptive</strong>Follow-ups react to answer depth</span></div>
            <div><BarChart3 size={19} /><span><strong>Measurable</strong>Five-dimensional rubric feedback</span></div>
          </div>
          <div className="architecture-note">
            <ShieldCheck size={18} />
            <p><strong>Responsible by design.</strong> Communication cues are isolated from content scores and never used for hiring recommendations.</p>
          </div>
        </section>

        <section className="setup-card">
          <div className="card-heading">
            <div><span className="step-label">SESSION BLUEPRINT</span><h2>Configure your interview</h2></div>
            <button className="text-button" onClick={props.loadSample}><Sparkles size={15} /> Load sample</button>
          </div>

          {props.error && <div className="error-banner"><CircleAlert size={18} /><span>{props.error}</span></div>}
          {props.health?.limitations?.length ? <div className={`runtime-banner ${props.health.status}`}><CircleAlert size={18} /><div><strong>{props.health.status === "ready" ? "Optional capabilities limited" : "Core runtime degraded"}</strong><ul>{props.health.limitations.map((item) => <li key={item}>{item}</li>)}</ul></div></div> : null}

          <label className="field full-field">
            <span>Target role</span>
            <div className="input-with-icon"><BriefcaseBusiness size={17} /><input value={props.role} onChange={(e) => props.setRole(e.target.value)} placeholder="e.g. Senior Data Scientist" /></div>
          </label>

          <div className="form-grid three-columns">
            <label className="field"><span>Interview type</span><select value={props.interviewType} onChange={(e) => props.setInterviewType(e.target.value)}><option value="mixed">Mixed</option><option value="technical">Technical</option><option value="behavioral">Behavioral</option></select></label>
            <label className="field"><span>Difficulty</span><select value={props.difficulty} onChange={(e) => props.setDifficulty(e.target.value)}><option value="beginner">Beginner</option><option value="intermediate">Intermediate</option><option value="advanced">Advanced</option></select></label>
            <label className="field"><span>Questions</span><select value={props.questionCount} onChange={(e) => props.setQuestionCount(Number(e.target.value))}>{[3, 4, 5, 6, 7].map((n) => <option key={n} value={n}>{n} questions</option>)}</select></label>
          </div>

          <div className="document-grid">
            <div className="field document-field">
              <div className="field-title"><span>Resume context</span><small>TXT / PDF or pasted text</small></div>
              <label className={`file-drop ${props.resumeFile ? "has-file" : ""}`}>
                <input type="file" accept=".txt,.pdf" onChange={(e) => props.setResumeFile(e.target.files?.[0] ?? null)} />
                {props.resumeFile ? <><CheckCircle2 size={20} /><span>{props.resumeFile.name}</span></> : <><Upload size={20} /><span>Upload resume</span></>}
              </label>
              <textarea value={props.resumeText} onChange={(e) => props.setResumeText(e.target.value)} disabled={Boolean(props.resumeFile)} placeholder="Or paste resume text here…" />
            </div>
            <label className="field document-field">
              <div className="field-title"><span>Job description</span><small>Required</small></div>
              <textarea value={props.jobDescription} onChange={(e) => props.setJobDescription(e.target.value)} placeholder="Paste the target job description…" />
            </label>
          </div>

          <button className="primary-button start-button" onClick={props.startInterview} disabled={props.busy}>
            {props.busy ? <><LoaderCircle className="spin" size={18} /> {props.startStage}…</> : <>Start tailored interview <ArrowRight size={18} /></>}
          </button>
          {props.busy ? (
            <div className="build-progress" role="status" aria-live="polite">
              <LoaderCircle className="spin" size={18} />
              <div>
                <strong>{props.startStage}</strong>
                <span>Local Qwen and BGE cold starts can take 60–120 seconds. Keep this page open.</span>
              </div>
              <time>{props.startElapsed}s</time>
            </div>
          ) : null}
          <p className="form-footnote"><LockKeyhole size={13} /> Documents stay in process memory and are not logged by default.</p>
        </section>
      </main>

      <section className="system-strip">
        <div className="strip-label"><Layers3 size={16} /> PIPELINE STATUS</div>
        {Object.entries(props.health?.components ?? { api: "connecting" }).map(([name, value]) => (
          <div className="pipeline-item" key={name}><StatusDot ok={props.health?.component_status?.[name] === "available"} /><span>{titleCase(name)}</span><strong>{titleCase(value)}</strong></div>
        ))}
      </section>
    </div>
  );
}

type InterviewProps = {
  session: Session; health: Health | null; lastTurn: Turn | null;
  transcript: string; setTranscript: (value: string) => void;
  audioFile: File | null; setAudioFile: (value: File | null) => void;
  busy: boolean; error: string; submitAnswer: () => void; continueInterview: () => void;
  time: string; transcriptRef: React.RefObject<HTMLTextAreaElement | null>; reset: () => void;
};

function InterviewPage(props: InterviewProps) {
  const completed = props.session.turns.length;
  const progress = Math.min(100, (completed / props.session.max_questions) * 100);
  return (
    <div className="app-shell workspace-page">
      <header className="topbar workspace-topbar">
        <Brand />
        <div className="session-title"><span>ACTIVE SESSION</span><strong>{props.session.role}</strong></div>
        <div className="topbar-actions"><div className="timer"><Clock3 size={15} /> {props.time}</div><ModelBadge health={props.health} /><button className="icon-button" onClick={props.reset} title="End session"><RotateCcw size={17} /></button></div>
      </header>

      <div className="progress-header">
        <div><span>Question {Math.min(completed + 1, props.session.max_questions)} of {props.session.max_questions}</span><strong>{props.lastTurn ? "Response analysis" : titleCase(props.session.current_topic ?? "Interview")}</strong></div>
        <div className="progress-track"><span style={{ width: `${progress}%` }} /></div>
        <span>{Math.round(progress)}% complete</span>
      </div>

      {props.session.degraded_modes.length ? <div className="session-limitations"><CircleAlert size={16} /><span>{props.session.degraded_modes.join(" ")}</span></div> : null}
      {props.session.security_events.length ? <div className="session-limitations security"><ShieldCheck size={16} /><span>{props.session.security_events.join(" ")}</span></div> : null}

      {props.lastTurn ? (
        <FeedbackView turn={props.lastTurn} busy={props.busy} continueInterview={props.continueInterview} />
      ) : (
        <main className="workspace-grid">
          <section className="interview-column">
            <div className="question-card">
              <div className="question-meta"><span className="question-index">Q{String(completed + 1).padStart(2, "0")}</span><span className="topic-pill">{props.session.current_topic ?? "Role experience"}</span></div>
              <blockquote>{props.session.current_question}</blockquote>
              {props.session.current_question_grounding ? <div className={`grounding-summary ${props.session.current_question_grounding.status}`}><ShieldCheck size={15} /><span>{titleCase(props.session.current_question_grounding.status)} · {Math.round(props.session.current_question_grounding.support_score * 100)}% {titleCase(props.session.current_question_grounding.support_method)} · {props.session.current_question_evidence_ids.join(", ") || "no citation"}</span></div> : null}
              <div className="question-guidance"><MessageSquareText size={16} /><span>Use a specific example. Explain your decisions, trade-offs, and measurable outcome.</span></div>
            </div>

            {props.error && <div className="error-banner"><CircleAlert size={18} /><span>{props.error}</span></div>}

            <div className="answer-card">
              <div className="answer-heading"><div><span className="step-label">YOUR RESPONSE</span><h2>Build a precise answer</h2></div><span className="word-count">{props.transcript.trim() ? props.transcript.trim().split(/\s+/).length : 0} words</span></div>
              <textarea ref={props.transcriptRef} value={props.transcript} onChange={(e) => props.setTranscript(e.target.value)} placeholder="Start with the context, then explain your contribution, key decisions, validation, and result…" />
              <div className="answer-actions">
                <label className="audio-button"><input type="file" accept="audio/*" onChange={(e) => props.setAudioFile(e.target.files?.[0] ?? null)} /><Mic2 size={17} />{props.audioFile ? props.audioFile.name : "Attach recording"}</label>
                <button className="primary-button" onClick={props.submitAnswer} disabled={props.busy}>{props.busy ? <><LoaderCircle className="spin" size={17} /> Analyzing response…</> : <>Analyze response <ArrowRight size={17} /></>}</button>
              </div>
            </div>
          </section>

          <aside className="context-column">
            <div className="side-card">
              <div className="side-heading"><Target size={17} /><span>Role coverage</span></div>
              <div className="skill-list">
                {(props.session.required_skills.length ? props.session.required_skills : ["Role experience"]).map((skill) => {
                  const covered = props.session.covered_skills.includes(skill);
                  return <div className={covered ? "covered" : ""} key={skill}><span>{covered ? <Check size={13} /> : null}</span>{skill}</div>;
                })}
              </div>
            </div>
            <div className="side-card">
              <div className="side-heading"><Activity size={17} /><span>Evaluation rubric</span></div>
              <ul className="rubric-list"><li>Relevance to the question</li><li>Clarity and precision</li><li>Logical answer structure</li><li>Technical decision depth</li><li>Evidence and measurable impact</li></ul>
            </div>
            <div className="side-card subtle-card">
              <div className="side-heading"><ShieldCheck size={17} /><span>Scoring boundary</span></div>
              <p>Communication signals are optional coaching context. They never alter your content scores.</p>
            </div>
            {props.session.runtime ? <div className="side-card runtime-card"><div className="side-heading"><Layers3 size={17} /><span>Runtime provenance</span></div><dl><div><dt>Model</dt><dd>{props.session.runtime.model_name}</dd></div><div><dt>Retrieval</dt><dd>{titleCase(props.session.runtime.retrieval_effective)}</dd></div><div><dt>Output</dt><dd>{titleCase(props.session.runtime.structured_output_mode)}</dd></div><div><dt>Prompt</dt><dd>{props.session.runtime.prompt_version}</dd></div></dl></div> : null}
          </aside>
        </main>
      )}
    </div>
  );
}

function ScoreGrid({ scores }: { scores: Scorecard }) {
  return <div className="score-grid">{(Object.keys(scoreLabels) as (keyof Scorecard)[]).map((key) => <div className="score-card" key={key}><div><span>{scoreLabels[key]}</span><strong>{scores[key]}<small>/10</small></strong></div><div className="score-track"><span style={{ width: `${scores[key] * 10}%` }} /></div></div>)}</div>;
}

function FeedbackView({ turn, busy, continueInterview }: { turn: Turn; busy: boolean; continueInterview: () => void }) {
  const finish = turn.controller_action === "finish";
  return (
    <main className="feedback-main">
      <div className="feedback-header"><div><div className="eyebrow"><CheckCircle2 size={15} /> RESPONSE ANALYZED</div><h1>What landed—and what to sharpen.</h1><p>Feedback is grounded in the retrieved role and resume context shown below.</p></div><div className={`action-chip ${turn.controller_action}`}><Activity size={15} /> Next action: {titleCase(turn.controller_action)}</div></div>
      <ScoreGrid scores={turn.evaluation.scores} />
      {turn.evaluation.calibration ? <div className="score-provenance"><ShieldCheck size={15} /><span>Hybrid anchored scoring · {Math.round(turn.evaluation.calibration.model_weight * 100)}% model judgment · {Math.round((1 - turn.evaluation.calibration.model_weight) * 100)}% observable answer signals</span></div> : null}
      <div className="feedback-grid">
        <section className="insight-card strength-card"><div className="insight-title"><CheckCircle2 size={19} /><div><span>WHAT WORKED</span><h2>Strengths to retain</h2></div></div><ul>{turn.evaluation.strengths.map((item) => <li key={item}>{item}</li>)}</ul></section>
        <section className="insight-card improvement-card"><div className="insight-title"><TrendingUp size={19} /><div><span>NEXT ITERATION</span><h2>Highest-impact improvements</h2></div></div><ul>{turn.evaluation.improvements.map((item) => <li key={item}>{item}</li>)}</ul></section>
      </div>
      <section className="outline-card"><div className="section-heading"><Layers3 size={18} /><div><span>REBUILD THE ANSWER</span><h2>Recommended response architecture</h2></div></div><div className="outline-steps">{turn.evaluation.improved_answer_outline.map((item, index) => <div key={item}><span>{index + 1}</span><p>{item}</p></div>)}</div></section>
      <section className="evidence-card"><div className="section-heading"><FileText size={18} /><div><span>GROUNDING TRACE</span><h2>Evidence retrieved for this evaluation</h2></div>{turn.question_grounding ? <span className={`audit-pill ${turn.question_grounding.status}`}>{titleCase(turn.question_grounding.status)} · {Math.round(turn.question_grounding.support_score * 100)}%</span> : null}</div><div className="evidence-list">{turn.evidence.length ? turn.evidence.map((item) => <details key={item.chunk_id}><summary><div><span className={`source-tag ${item.source}`}>{item.source === "resume" ? "RESUME" : "JOB DESCRIPTION"}</span><strong>{item.chunk_id}</strong></div><div><span>{titleCase(item.retrieval_method)} · {Math.round(item.score * 100)}%</span><ChevronRight size={16} /></div></summary><p>{item.text}</p><div className="evidence-metrics"><span>Dense {item.semantic_score == null ? "n/a" : `${Math.round(item.semantic_score * 100)}%`}</span><span>Lexical {item.lexical_score == null ? "n/a" : `${Math.round(item.lexical_score * 100)}%`}</span>{item.risk_flags.length ? <span className="risk-flag">Untrusted instruction pattern: {item.risk_flags.join(", ")}</span> : null}</div></details>) : <p className="empty-note">No useful context was retrieved; generic feedback was used without invented evidence.</p>}</div></section>
      <section className="communication-card"><AudioLines size={18} /><div><span>COMMUNICATION CUE · {titleCase(turn.communication_signal.status)}</span><p>{turn.communication_signal.explanation}</p></div><strong>{turn.communication_signal.label === "unavailable" ? "Not scored" : titleCase(turn.communication_signal.label)}</strong></section>
      <div className="feedback-footer"><div><span>{finish ? "INTERVIEW COMPLETE" : "UP NEXT"}</span><p>{finish ? "Your coaching report is ready to generate." : turn.next_question}</p></div><button className="primary-button" onClick={continueInterview} disabled={busy}>{busy ? <LoaderCircle className="spin" size={17} /> : finish ? <BarChart3 size={17} /> : null}{finish ? "Generate final report" : "Continue to next question"}<ArrowRight size={17} /></button></div>
    </main>
  );
}

function ReportPage({ session, report, health, reset }: { session: Session; report: FinalReport; health: Health | null; reset: () => void }) {
  const averages = useMemo(() => Object.fromEntries(Object.entries(report.score_trends).map(([key, values]) => [key, values.reduce((a, b) => a + b, 0) / Math.max(values.length, 1)])), [report]);
  return (
    <div className="app-shell report-page">
      <header className="topbar"><Brand /><div className="topbar-actions"><ModelBadge health={health} /><button className="secondary-button" onClick={reset}><RotateCcw size={15} /> New session</button></div></header>
      <main className="report-main">
        <div className="report-hero"><div><div className="eyebrow"><BarChart3 size={15} /> COACHING REPORT</div><h1>Interview performance,<br /><em>translated into practice.</em></h1><p>{session.role} · {report.completed_questions} completed questions · {titleCase(session.difficulty)} difficulty</p></div><a className="primary-button download-button" href={`/sessions/${session.session_id}/report?format=markdown`} download><FileText size={17} /> Download report</a></div>
        <div className="report-layout">
          <section className="report-panel"><div className="section-heading"><Gauge size={18} /><div><span>RUBRIC SUMMARY</span><h2>Average content scores</h2></div></div><div className="trend-list">{Object.entries(averages).map(([key, value]) => <div key={key}><div><span>{titleCase(key)}</span><strong>{value.toFixed(1)} / 10</strong></div><div className="score-track"><span style={{ width: `${value * 10}%` }} /></div><small>Turns: {report.score_trends[key].join(" → ")}</small></div>)}</div></section>
          <section className="report-panel"><div className="section-heading"><Target size={18} /><div><span>FOCUSED PRACTICE</span><h2>Your next three drills</h2></div></div><ol className="practice-list">{report.practice_plan.map((item, index) => <li key={item}><span>{String(index + 1).padStart(2, "0")}</span><p>{item}</p></li>)}</ol></section>
        </div>
        <div className="feedback-grid report-insights"><section className="insight-card strength-card"><div className="insight-title"><CheckCircle2 size={19} /><div><span>CONSISTENT SIGNALS</span><h2>Recurring strengths</h2></div></div><ul>{report.recurring_strengths.map((item) => <li key={item}>{item}</li>)}</ul></section><section className="insight-card improvement-card"><div className="insight-title"><TrendingUp size={19} /><div><span>DEVELOPMENT AREAS</span><h2>Recurring improvements</h2></div></div><ul>{report.recurring_improvements.map((item) => <li key={item}>{item}</li>)}</ul></section></div>
        <div className="report-disclaimer"><ShieldCheck size={18} /><p>{report.disclaimer}</p></div>
      </main>
    </div>
  );
}

export default App;
