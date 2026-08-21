# SignalPrep

SignalPrep is a measured, grounded GenAI interview-practice system. It converts a resume and target
job description into evidence-cited questions, evaluates typed or spoken answers with an anchored
rubric, selects bounded follow-ups, and produces a coaching report without making hiring decisions.

The portfolio runtime uses a real local language model and hybrid retrieval by default. Deterministic
rules remain available only as a clearly labeled offline test/demo path.

## Live demo

[Launch SignalPrep](https://signal-prep.vercel.app/) or inspect its
[runtime health and provenance](https://signal-prep.vercel.app/api/health).

The public Vercel profile uses hosted Qwen inference through Groq, lexical TF-IDF retrieval, and
typed answers. It is intentionally separate from the measured local Qwen 2.5 3B + hybrid-BGE
profile below. The current public deployment keeps active sessions in function memory, so a cold
start can reset an unfinished interview; the repository also includes an optional expiring Upstash
REST backend for cross-instance continuity.

## Measured real-model result

A frozen evaluation used local `qwen2.5:3b`, BGE embeddings, temperature 0, seed 42, and the synthetic
SignalPrep v1 regression suite:

| Metric | Result |
|---|---:|
| Retrieval Recall@3 (20 cases) | **100%** |
| Retrieval MRR | **0.925** |
| Structured-output validity (6 answer cases) | **100%** |
| Citation validity after bounded repair | **100%** |
| Strong-answer-over-weak pair ordering | **100% (3/3)** |
| Expected score-band accuracy | 66.7% |
| Supported generated questions | **100% (4/4)** |
| Question/requirement injection cases blocked | **100% (2/2)** |
| Retrieval median / p95 latency | 418 / 464 ms |
| Generation median / p95 latency | 13.36 / 29.07 s |

These are observed results on a small synthetic benchmark, not evidence of hiring validity or
production readiness. The 66.7% score-band result shows that calibration still needs a larger,
human-rated dataset. Read the
[published report](evaluation/results/published/qwen2.5-3b-v1-2026-08-16/REPORT.md) and
[raw result](evaluation/results/published/qwen2.5-3b-v1-2026-08-16/result.json).

## What is implemented

- Native Ollama JSON-schema generation with Pydantic validation, transient retry/backoff, seed,
  token limit, timeout, and one bounded application repair.
- Hybrid local retrieval: BGE dense embeddings fused with transparent TF-IDF scores, with separate
  resume/JD filtering and explicit lexical fallback.
- Prompt-injection indicators, risky-chunk quarantine for generation, untrusted-data prompt policy,
  allowlisted evidence IDs, and question-level lexical/semantic support audits.
- Anchored content scoring with raw model scores plus observable signals for specificity, structure,
  validation, technical decisions, metrics, and outcomes.
- Deterministic controller for clarify/probe/change-topic/finish decisions and hard question limits.
- Typed answers, optional audio upload, optional faster-whisper transcription, editable transcripts,
  and optional communication-cue adapters that never affect content scores.
- React/TypeScript workspace, FastAPI/OpenAPI service, downloadable reports, runtime provenance,
  visible degraded modes, and retrieval/grounding traces.
- Versioned evaluation corpus, real-model evaluation runner, tests, linting, CI, Dockerfile, and
  Compose configuration.
- A Vercel demo profile with a static React build, serverless FastAPI entry point, hosted-model
  adapter, and optional expiring Upstash state for function-instance changes.

## Quick start on Windows

Prerequisites: Python 3.11+, Node.js 20+, and Ollama.

```powershell
cd SignalPrep
python -m pip install -e ".[dev]"

ollama pull qwen2.5:3b
ollama pull hf.co/CompendiumLabs/bge-base-en-v1.5-gguf:latest

cd frontend
npm install
npm run build
cd ..

python -m uvicorn interview_coach.api.app:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000). Health and exact component provenance are
available at `/health`; interactive API documentation is at `/docs`.

The first embedding/model request may be slow while Ollama loads weights. The UI shows progress and
does not label optional communication models as available when they are not configured.

## Runtime flow

```text
Resume + JD
    |
parse -> section chunk -> injection indicators -> dense + lexical indexes
    |
retrieve resume/JD separately -> quarantine risky chunks -> evidence IDs
    |
Qwen structured question -> citation validation -> support audit -> bounded repair
    |
answer/transcript -> anchored LLM evaluation -> observable score calibration
    |
deterministic controller -> follow-up or report
```

Important boundaries:

- Uploaded documents and transcripts are untrusted data, never instructions.
- Unknown citation IDs are rejected; flagged document chunks are visible to audit but excluded from
  generation by default.
- Semantic similarity is labeled only as a support indicator, not entailment proof.
- Communication cues are optional, uncertain, and excluded from correctness/content scores.
- Sessions are process-local and in memory; production deployment still needs authentication,
  persistent storage, tenant isolation, quotas, encryption, and deletion workflows.

The local runtime above and the current public demo use process memory. The Vercel profile can use
expiring Upstash storage for serverless continuity when its Redis environment variables are
configured; it is still a public portfolio demo rather than a multi-tenant production service. See
the [deployment guide](docs/deployment.md).

See [architecture](docs/architecture.md), [model/system card](MODEL_CARD.md), and
[interview learning guide](docs/interview_learning_guide.md).

## Evaluation

```powershell
python evaluation/run_real_model_evaluation.py --output-name my-run
python -m pytest -q
python -m ruff check src tests evaluation
cd frontend
npm run build
```

The v1 corpus contains 20 labeled retrieval queries, three paired strong/weak answer comparisons,
four grounded-question cases, and adversarial requirement/question cases. Expected labels are used
only by the evaluator. The published result includes raw LLM scores, observable scores, calibrated
scores, citations, per-case retrieval ranks, latency, and failure details.

Before any domain deployment claim, add independently sampled tasks, two human raters, agreement
statistics, repeated stochastic runs, WER for consented audio, prompt-injection coverage, fairness
review, cost, p95 latency, and online failure monitoring.

## Configuration

Copy `.env.example` when overriding defaults. Key variables:

- `LLM_PROVIDER`: `ollama`, `openai_compatible`, or `deterministic_demo`
- `LLM_MODEL`, `LLM_BASE_URL`, `LLM_API_KEY`
- `LLM_TIMEOUT_SECONDS`, `LLM_MAX_TOKENS`, `LLM_SEED`, `LLM_MAX_RETRIES`
- `RETRIEVAL_BACKEND`: `hybrid_ollama`, `semantic`, or `lexical_demo`
- `EMBEDDING_MODEL`
- optional STT and communication-cue model IDs

No API key, document, transcript, or raw audio is deliberately stored in evaluation artifacts.

## Responsible use

SignalPrep is practice and coaching software. It must not rank candidates, recommend hiring or
rejection, infer honesty/personality/mental health, or treat communication-model output as a person's
true emotional state. Optional cue models require separate model-card, licensing, bias, calibration,
accent/noise, and consent evaluation before use.
