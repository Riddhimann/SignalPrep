from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Sequence
from typing import Protocol

import httpx

from interview_coach.exceptions import ModelUnavailableError
from interview_coach.rag.safety import prompt_injection_flags
from interview_coach.schemas import DocumentChunk, RetrievedEvidence

TOKEN = re.compile(r"[a-zA-Z][a-zA-Z0-9+#.-]{1,}")


class Retriever(Protocol):
    backend_name: str

    def index(self, chunks: list[DocumentChunk]) -> None: ...

    def search(self, query: str, source: str | None, k: int) -> list[RetrievedEvidence]: ...


def _tokens(text: str) -> list[str]:
    return [token.lower() for token in TOKEN.findall(text)]


class LexicalVectorStore:
    """Transparent TF-IDF cosine fallback used by the offline demo."""

    backend_name = "lexical_demo"

    def __init__(self) -> None:
        self._chunks: list[DocumentChunk] = []
        self._vectors: list[dict[str, float]] = []

    def index(self, chunks: list[DocumentChunk]) -> None:
        self._chunks = list(chunks)
        document_frequency: Counter[str] = Counter()
        counts: list[Counter[str]] = []
        for chunk in chunks:
            count = Counter(_tokens(chunk.text))
            counts.append(count)
            document_frequency.update(count.keys())
        total = max(len(chunks), 1)
        self._vectors = []
        for count in counts:
            normed: dict[str, float] = {}
            for token, frequency in count.items():
                tf = 1 + math.log(frequency)
                idf = math.log((1 + total) / (1 + document_frequency[token])) + 1
                normed[token] = tf * idf
            norm = math.sqrt(sum(value * value for value in normed.values())) or 1
            self._vectors.append({key: value / norm for key, value in normed.items()})

    def search(self, query: str, source: str | None = None, k: int = 3) -> list[RetrievedEvidence]:
        query_counts = Counter(_tokens(query))
        query_norm = math.sqrt(sum(value * value for value in query_counts.values())) or 1
        scores: list[tuple[float, DocumentChunk]] = []
        for chunk, vector in zip(self._chunks, self._vectors, strict=True):
            if source and chunk.source != source:
                continue
            score = sum(
                (count / query_norm) * vector.get(token, 0) for token, count in query_counts.items()
            )
            if score > 0:
                scores.append((min(float(score), 1.0), chunk))
        scores.sort(key=lambda pair: pair[0], reverse=True)
        return [
            RetrievedEvidence(
                chunk_id=chunk.chunk_id,
                source=chunk.source,
                text=chunk.text,
                score=score,
                retrieval_method=self.backend_name,
                lexical_score=score,
                risk_flags=prompt_injection_flags(chunk.text),
            )
            for score, chunk in scores[:k]
        ]


class OllamaEmbeddingClient:
    def __init__(self, model_id: str, base_url: str, timeout_seconds: float) -> None:
        self.model_id = model_id
        self._url = base_url.rstrip("/") + "/api/embed"
        self._timeout = timeout_seconds

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        try:
            with httpx.Client(timeout=self._timeout) as client:
                response = client.post(
                    self._url,
                    json={
                        "model": self.model_id,
                        "input": list(texts),
                        "truncate": True,
                    },
                )
                response.raise_for_status()
                embeddings = response.json()["embeddings"]
            if not isinstance(embeddings, list) or len(embeddings) != len(texts):
                raise TypeError("embedding count did not match input count")
            return [[float(value) for value in vector] for vector in embeddings]
        except (httpx.HTTPError, KeyError, TypeError, ValueError) as exc:
            raise ModelUnavailableError(
                f"Ollama embedding model {self.model_id!r} is unavailable"
            ) from exc


def _normalized(vector: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


class HybridOllamaVectorStore:
    """Fuse local dense retrieval with transparent lexical matching."""

    backend_name = "hybrid_ollama"

    def __init__(
        self,
        model_id: str,
        base_url: str,
        timeout_seconds: float = 120,
        *,
        semantic_weight: float = 0.65,
    ) -> None:
        if semantic_weight < 0 or semantic_weight > 1:
            raise ValueError("semantic_weight must be between 0 and 1")
        self._semantic_weight = semantic_weight
        self._client = OllamaEmbeddingClient(model_id, base_url, timeout_seconds)
        self._lexical = LexicalVectorStore()
        self._chunks: list[DocumentChunk] = []
        self._embeddings: list[list[float]] = []

    def index(self, chunks: list[DocumentChunk]) -> None:
        self._chunks = list(chunks)
        self._lexical.index(chunks)
        self._embeddings = [
            _normalized(vector) for vector in self._client.embed([chunk.text for chunk in chunks])
        ]

    def search(self, query: str, source: str | None = None, k: int = 3) -> list[RetrievedEvidence]:
        if not self._chunks:
            return []
        query_vector = _normalized(self._client.embed([query])[0])
        lexical = {
            item.chunk_id: item.score
            for item in self._lexical.search(query, source=source, k=len(self._chunks))
        }
        ranked: list[tuple[float, float, float, DocumentChunk]] = []
        for chunk, vector in zip(self._chunks, self._embeddings, strict=True):
            if source and chunk.source != source:
                continue
            cosine = sum(left * right for left, right in zip(query_vector, vector, strict=True))
            semantic_score = max(0.0, min((cosine + 1.0) / 2.0, 1.0))
            lexical_score = lexical.get(chunk.chunk_id, 0.0)
            fused = (
                self._semantic_weight * semantic_score
                + (1.0 - self._semantic_weight) * lexical_score
            )
            ranked.append((fused, semantic_score, lexical_score, chunk))
        ranked.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
        return [
            RetrievedEvidence(
                chunk_id=chunk.chunk_id,
                source=chunk.source,
                text=chunk.text,
                score=fused,
                retrieval_method=self.backend_name,
                lexical_score=lexical_score,
                semantic_score=semantic_score,
                risk_flags=prompt_injection_flags(chunk.text),
            )
            for fused, semantic_score, lexical_score, chunk in ranked[:k]
        ]


class SemanticVectorStore:
    backend_name = "semantic"

    def __init__(self, model_id: str) -> None:
        try:
            import faiss  # type: ignore
            from sentence_transformers import SentenceTransformer  # type: ignore
        except ImportError as exc:
            raise ModelUnavailableError(
                "Semantic retrieval requires sentence-transformers and faiss-cpu"
            ) from exc
        self._faiss = faiss
        self._model = SentenceTransformer(model_id)
        self._chunks: list[DocumentChunk] = []
        self._index = None

    def index(self, chunks: list[DocumentChunk]) -> None:
        import numpy as np

        self._chunks = list(chunks)
        embeddings = self._model.encode(
            [chunk.text for chunk in chunks],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")
        self._index = self._faiss.IndexFlatIP(embeddings.shape[1])
        self._index.add(np.ascontiguousarray(embeddings))

    def search(self, query: str, source: str | None = None, k: int = 3) -> list[RetrievedEvidence]:
        import numpy as np

        if self._index is None:
            return []
        candidate_k = min(len(self._chunks), max(k * 4, k))
        vector = self._model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")
        scores, indices = self._index.search(np.ascontiguousarray(vector), candidate_k)
        results: list[RetrievedEvidence] = []
        for score, index in zip(scores[0], indices[0], strict=True):
            if index < 0:
                continue
            chunk = self._chunks[int(index)]
            if source and chunk.source != source:
                continue
            results.append(
                RetrievedEvidence(
                    chunk_id=chunk.chunk_id,
                    source=chunk.source,
                    text=chunk.text,
                    score=max(0.0, min((float(score) + 1) / 2, 1.0)),
                )
            )
            if len(results) == k:
                break
        return results


def create_vector_store(
    backend: str,
    embedding_model: str,
    *,
    ollama_url: str = "http://127.0.0.1:11434",
    timeout_seconds: float = 120,
) -> Retriever:
    if backend == "hybrid_ollama":
        return HybridOllamaVectorStore(
            embedding_model,
            ollama_url,
            timeout_seconds,
        )
    if backend == "semantic":
        return SemanticVectorStore(embedding_model)
    return LexicalVectorStore()
