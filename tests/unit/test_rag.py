from interview_coach.rag.chunker import chunk_document
from interview_coach.rag.parser import normalize_text
from interview_coach.rag.vector_store import LexicalVectorStore


def test_normalization_preserves_heading_breaks():
    assert normalize_text("SKILLS\r\n\r\n  Python   SQL ") == "SKILLS\n\nPython SQL"


def test_chunk_sources_never_mix():
    resume = chunk_document("Python API experience", "resume", target_words=3, overlap_words=1)
    jd = chunk_document(
        "Requires Python and SQL", "job_description", target_words=3, overlap_words=1
    )
    assert all(chunk.source == "resume" for chunk in resume)
    assert all(chunk.source == "job_description" for chunk in jd)


def test_lexical_retrieval_filters_source():
    chunks = chunk_document("Python FastAPI service", "resume") + chunk_document(
        "SQL machine learning role", "job_description"
    )
    store = LexicalVectorStore()
    store.index(chunks)
    results = store.search("Python experience", source="resume", k=3)
    assert results and all(item.source == "resume" for item in results)
