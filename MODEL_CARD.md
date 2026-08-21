# SignalPrep model and system card

## Intended use

SignalPrep coaches a consenting user during mock interviews. It generates evidence-cited practice
questions, evaluates answer content, and may show explicitly uncertain communication cues. It is not
an employment-decision, ranking, surveillance, deception-detection, personality, disability, or
mental-health system.

## Default measured runtime

| Component | Default | Status |
|---|---|---|
| Structured generation | Ollama `qwen2.5:3b` | Real local inference; native JSON schema |
| Retrieval | `hybrid_ollama` | Dense + lexical fusion with explicit fallback |
| Embeddings | BGE base English GGUF, 109M | Local Ollama embeddings, 768 dimensions |
| Content scoring | Anchored LLM + observable calibration | Raw and calibrated scores retained |
| STT | faster-whisper `small` | Optional; unavailable unless installed |
| Speech cue classifier | none | Intentionally unconfigured |
| Text cue classifier | none | Intentionally unconfigured |

The checked-in synthetic v1 result measured 100% Recall@3, 0.925 MRR, 100% structured-output and
citation validity, 3/3 correct strong-over-weak score ordering, and 66.7% authored score-band
accuracy. Median/p95 generation latency was 13.36/29.07 seconds. These results describe one local
model/runtime and do not establish human agreement or production validity.

## Scoring boundary

The language model produces qualitative feedback and raw rubric scores. A transparent calibration
layer combines those scores with observable answer signals: word/sentence count, question overlap,
technical terms, validation language, decision/trade-off language, measurements, and outcomes. The
dashboard and artifacts retain raw model, observable, and calibrated scores. A larger human-rated
dataset is required to validate or learn these weights.

Communication cues never change content scores. Optional speech/text fusion remains a configurable
60/40 heuristic, not a learned or validated probability.

## Retrieval and injection boundary

Documents are untrusted. SignalPrep detects common instruction-override, prompt-exfiltration,
role-override, and tool-override patterns. Flagged chunks remain visible to audit evaluation but are
excluded from model context by default; flagged raw JD lines are quarantined before requirement
extraction. Citations must be from the supplied ID allowlist. Lexical support and semantic similarity
are reported separately; neither is presented as proof of entailment.

## Privacy and operational limitations

Sessions are in memory. Temporary audio uploads are deleted after the turn. Documents, transcripts,
keys, and raw audio are not deliberately logged. Production deployment still requires authentication,
tenant isolation, transport/storage encryption, malware scanning, durable deletion, quotas, audit
logging, monitoring, and independently reviewed evaluation gates.

Outputs can vary with model/runtime versions, microphone quality, accent, language, disability,
culture, transcript error, dataset bias, and domain shift. Model revisions and licenses must be
reviewed before redistribution or commercial use.
