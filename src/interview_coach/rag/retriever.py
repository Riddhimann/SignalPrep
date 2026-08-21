from __future__ import annotations

from interview_coach.exceptions import ModelUnavailableError
from interview_coach.rag.chunker import chunk_document
from interview_coach.rag.vector_store import Retriever, create_vector_store
from interview_coach.schemas import RetrievedEvidence
from interview_coach.storage.rag_source_store import RagSourceStore


class RagService:
    def __init__(
        self,
        backend: str,
        embedding_model: str,
        *,
        ollama_url: str = "http://127.0.0.1:11434",
        timeout_seconds: float = 120,
        source_store: RagSourceStore | None = None,
    ) -> None:
        self._backend = backend
        self._embedding_model = embedding_model
        self._ollama_url = ollama_url
        self._timeout_seconds = timeout_seconds
        self._source_store = source_store
        self._stores: dict[str, Retriever] = {}
        self._effective_modes: dict[str, str] = {}

    @property
    def mode(self) -> str:
        return self._backend

    def index_session(self, session_id: str, resume: str, job_description: str) -> int:
        count = self._build_index(session_id, resume, job_description)
        if self._source_store is not None:
            self._source_store.save(
                session_id,
                resume,
                job_description,
                self._effective_modes[session_id],
            )
        return count

    def _build_index(self, session_id: str, resume: str, job_description: str) -> int:
        chunks = chunk_document(resume, "resume") + chunk_document(
            job_description, "job_description"
        )
        store = create_vector_store(
            self._backend,
            self._embedding_model,
            ollama_url=self._ollama_url,
            timeout_seconds=self._timeout_seconds,
        )
        try:
            store.index(chunks)
        except ModelUnavailableError:
            if self._backend != "hybrid_ollama":
                raise
            store = create_vector_store("lexical_demo", self._embedding_model)
            store.index(chunks)
        self._stores[session_id] = store
        self._effective_modes[session_id] = store.backend_name
        return len(chunks)

    def _restore_index(self, session_id: str) -> bool:
        if session_id in self._stores:
            return True
        if self._source_store is None:
            return False
        state = self._source_store.load(session_id)
        if state is None:
            return False
        resume, job_description, _stored_mode = state
        self._build_index(session_id, resume, job_description)
        return True

    def effective_mode(self, session_id: str) -> str:
        self._restore_index(session_id)
        return self._effective_modes.get(session_id, self._backend)

    def degraded_message(self, session_id: str) -> str | None:
        if self._backend == "hybrid_ollama" and self.effective_mode(session_id) != self._backend:
            return "Hybrid embeddings were unavailable; retrieval fell back to lexical TF-IDF."
        return None

    def search(
        self,
        session_id: str,
        query: str,
        k_per_source: int = 3,
        include_flagged: bool = False,
    ) -> list[RetrievedEvidence]:
        if not self._restore_index(session_id):
            return []
        store = self._stores[session_id]
        candidate_k = k_per_source if include_flagged else max(k_per_source * 3, k_per_source)
        results = store.search(query, "job_description", candidate_k)
        results += store.search(query, "resume", candidate_k)
        if not include_flagged:
            results = [item for item in results if not item.risk_flags]
        seen: set[str] = set()
        deduplicated = [
            item for item in results if not (item.chunk_id in seen or seen.add(item.chunk_id))
        ]
        by_source: dict[str, int] = {}
        output: list[RetrievedEvidence] = []
        for item in deduplicated:
            count = by_source.get(item.source, 0)
            if count >= k_per_source:
                continue
            output.append(item)
            by_source[item.source] = count + 1
        return output

    def has_index(self, session_id: str) -> bool:
        return self._restore_index(session_id)
