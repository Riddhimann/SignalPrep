# SignalPrep real-model evaluation

## Provenance

- Provider/model: `ollama / qwen2.5:3b`
- Retrieval: `hybrid_ollama`
- Embedding model: `hf.co/CompendiumLabs/bge-base-en-v1.5-gguf:latest`
- Prompt/rubric: `signalprep-grounded-v1 / content-rubric-v1`
- Benchmark: synthetic v1; temperature 0; seed 42

## Results

| Metric | Result |
|---|---:|
| Retrieval Recall@3 | 100.0% |
| Retrieval MRR | 0.925 |
| Structured-output validity | 100.0% |
| Citation validity | 100.0% |
| Expected score-band accuracy | 66.7% |
| Strong-over-weak pair ordering | 100.0% |
| Grounded generated questions | 100.0% |
| Question injection-safe rate | 100.0% |
| Requirement injection case safe | True |
| Retrieval median / p95 | 418.0 / 464.1 ms |
| Generation median / p95 | 13361.5 / 29065.9 ms |

## Interpretation

These measurements describe this model/runtime on a small synthetic regression suite. They do not
establish hiring validity, production safety, population-wide score quality, or human agreement.
Communication cues were excluded from content scoring.
