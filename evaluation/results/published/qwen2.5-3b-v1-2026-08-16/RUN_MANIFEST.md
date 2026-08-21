# Published SignalPrep evaluation

This folder freezes the final real-model SignalPrep v1 result. Earlier development runs remain
locally ignored because they exposed and guided fixes for prompt injection, citation drift, and score
template repetition; only this post-fix run is the resume reference.

| Field | Value |
|---|---|
| Date | 2026-08-16 |
| Provider | Ollama 0.24.0, local inference |
| Generation model | `qwen2.5:3b`, 3.1B, Q4_K_M |
| Embedding model | BGE base English GGUF, 109M, 768 dimensions |
| Benchmark | `signalprep-synthetic-v1` 1.0.0 |
| Retrieval cases | 20 |
| Answer cases | 6 (three strong/weak pairs) |
| Question cases | 4 |
| Adversarial paths | risky question evidence and requirement extraction |
| Sampling | temperature 0, seed 42 |
| Prompt/rubric | `signalprep-grounded-v1` / `content-rubric-v1` |

The generation model artifact uses the Qwen Research License. Review its terms before redistribution
or commercial use. The evaluation data are fictional and synthetic.

## Result boundary

- Retrieval Recall@3 was 100% and MRR was 0.925 on 20 authored queries.
- Six answer outputs were schema-valid and citation-valid after the bounded product repair policy.
- Calibrated scores ranked the strong answer above the weak answer in all three authored pairs.
- Only four of six cases landed in the authored absolute score bands. No human agreement claim is made.
- Four generated questions had direct or partial support; semantic similarity is an indicator, not
  an entailment proof.
- Both authored injection paths were blocked after risky-chunk quarantine and input sanitization.
- Median/p95 generation latency was 13.36/29.07 seconds, including corrective retries.

These measurements are appropriate for regression evidence and an honest portfolio discussion. They
are not hiring validation, safety certification, bias evaluation, or proof of production readiness.
