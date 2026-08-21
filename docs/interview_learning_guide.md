# Interview learning guide

## The 30-second explanation

“I built a grounded GenAI interview-practice system using a local Qwen model, hybrid BGE and lexical
retrieval, native structured outputs, evidence validation, prompt-injection quarantine, and a
deterministic follow-up controller. I found that the small model repeated score templates, so I retained
its raw judgment and added an auditable calibration layer based on observable answer signals. A versioned
synthetic evaluation measures retrieval, citation validity, score ordering, injection behavior, and
latency. Optional communication cues remain separate and cannot affect content scores.”

## Learn it in this order

### 1. Domain contracts

Start at `schemas.py`. Pydantic objects are trust boundaries: scores are constrained to 1–10, actions are
literals, extra fields are rejected, and every turn records evidence and the controller's final action.
This prevents “JSON-shaped text” from spreading through the program.

Be ready to explain why validation is necessary but insufficient: valid JSON can still hallucinate an
evidence ID, so the orchestrator performs a semantic allow-list check after schema validation.

### 2. Retrieval

`parser.py` validates bytes and extracts text. `chunker.py` keeps sources separate and attaches stable IDs.
`vector_store.py` implements a transparent TF-IDF baseline, a Sentence Transformers/FAISS adapter, and
the measured default: Ollama BGE embeddings fused with lexical scores. Search is run independently for
JD and resume, avoiding one source drowning out the other. If embeddings fail, the service records an
explicit lexical fallback instead of pretending hybrid retrieval ran.

TF-IDF converts a document into weighted terms:

`tfidf(t,d) = (1 + log count(t,d)) * (log((1+N)/(1+df(t))) + 1)`

Cosine similarity ranks chunks by vector direction. Dense retrieval instead encodes semantic meaning and
uses inner-product search over normalized embeddings. A strong interview answer explains the
trade-off: lexical retrieval is cheap/auditable but misses paraphrases; dense retrieval improves recall
but adds model latency, memory, versioning, and evaluation requirements.

### 3. Structured generation and grounding

`gateway.py` exposes one generic `generate_structured` method. The Ollama adapter uses native JSON-schema
generation, validates with the full Pydantic contract, retries transient provider errors with bounded
backoff, and allows a bounded application repair. Temperature, seed, token limit, model, prompt version,
and runtime are recorded. The deterministic adapter remains only for tests/offline demonstrations.

Prompt instructions help, but programmatic guards create the actual invariant. Suspicious document lines
are flagged and quarantined from generation, evidence IDs are allowlisted, and generated questions receive
lexical plus semantic support audits. Semantic similarity is reported as an indicator, not proof of
entailment. A production version still needs sentence-level human-reviewed support labels.

### 3.1 Score calibration

`scoring.py` exists because valid structured output did not make the small model's absolute scores
reliable. The first benchmark showed repeated score templates for strong and weak answers. SignalPrep now
retains raw model scores and combines them with observable specificity, structure, validation, technical,
decision, metric, and outcome signals. The measured v1 result correctly ordered all three strong/weak
pairs but matched only four of six authored absolute score bands. Say exactly that; do not call the scores
human-validated.

### 4. State machine

Read `controller.py` top to bottom. Rule precedence matters:

1. Reaching the limit always finishes.
2. Fewer than 20 words or relevance ≤ 4 clarifies.
3. A technical claim with depth ≤ 6 probes.
4. Two questions on one topic change topic.
5. Only then may the validated model suggestion win.

This is “agentic” because output changes future behavior and state, but bounded because probabilistic text
does not control termination or safety.

### 5. Multimodal processing

Audio is validated for size/duration and normalized to mono 16 kHz PCM. STT is blocking because no answer
can be scored without usable text. Speech/text cue models are optional: one failure uses the other and two
failures produce an explicit unavailable signal. The fusion weights are a configurable heuristic, not a
learned truth. Content evaluation receives the cue only under a separately named field and its prompt
forbids using it for scores.

### 6. Orchestration and failure semantics

`orchestrator.py` coordinates adapters and stores typed turns. Notice the distinction between:

- controlled user errors: empty documents/transcripts, wrong state;
- optional degradation: cue classifier unavailable;
- blocking dependencies: STT when no corrected transcript exists, retrieval index before start;
- integrity errors: unsupported evidence IDs or repeatedly invalid structured output.

That classification is a common system-design interview theme.

## Questions you should be able to answer

1. Why RAG instead of putting the full documents in every prompt?
2. How would you measure Recall@k and grounding separately?
3. Why does schema validation not stop hallucination by itself?
4. How do deterministic controller rules prevent loops and unsafe actions?
5. Why run retrieval separately by source?
6. Why is STT failure blocking while emotion failure is not?
7. How does async concurrency reduce turn latency, and what remains CPU-bound?
8. How would you make sessions correct across multiple API workers?
9. How would you test a model adapter without downloading its model?
10. What bias/privacy risks remain even with careful UI wording?
11. How would you detect prompt injection in an uploaded resume?
12. What changes are required before production deployment?

## Strong answer sketches

**Recall@k vs grounding:** Recall@k asks whether labeled relevant chunks appear in the top k. Grounding
asks whether generated factual claims are supported by the supplied chunks. Good retrieval can coexist
with poor grounding, so both require separate labels and metrics.

**Multi-worker state:** Define a session repository protocol, store versioned JSON in PostgreSQL, update
with optimistic locking, store documents in encrypted object storage, use a shared vector index keyed by
tenant/session, and make answer submission idempotent. Never rely on a process-global dictionary.

**Prompt injection:** Treat documents as quoted data, delimit them, prohibit following their instructions,
restrict tool access, enforce schema/evidence allow-lists, scan suspicious patterns, and test adversarial
fixtures. Prompt instructions alone are not a security boundary.

**Evaluation:** Create labeled retrieval queries, consented audio with reference transcripts, answer
rubrics with two raters, repeated structured-output runs, and end-to-end latency traces. Report confidence
intervals and inter-rater agreement; never copy benchmark claims from a model card as system performance.

## Hands-on exercises

1. Trace one API answer from `routes.py` through `orchestrator.py` into a stored `InterviewTurn`.
2. Add a mock gateway that first returns an invalid score, then valid JSON; verify exactly one retry.
3. Add a retrieval case where lexical search fails on a paraphrase; compare semantic mode.
4. Replace the in-memory store with SQLite behind the same interface.
5. Add idempotency keys to answer submission and test duplicate requests.
6. Build a small labeled dataset and run an ablation: transcript only vs RAG vs RAG + cues.

## Resume bullet rule

Use the checked-in result: 100% Recall@3 on 20 synthetic retrieval cases, 0.925 MRR, 100% schema/citation
validity across six answer cases after bounded repair, and correct ordering of three strong/weak pairs.
Also state the 66.7% score-band result and 13.36/29.07-second median/p95 generation latency. Distinguish
implemented capability, measured synthetic performance, and work still required for production.
