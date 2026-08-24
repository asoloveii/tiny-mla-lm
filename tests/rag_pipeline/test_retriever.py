from typing import List, Tuple

from langchain_core.cross_encoders import BaseCrossEncoder

from rag.retriever import get_retriever
from rag.vectorstore import build_vectorstore


def _store(tmp_path, sample_documents, fake_embeddings):
    return build_vectorstore(sample_documents, str(tmp_path / "store"), backend="faiss", embeddings=fake_embeddings)


def test_get_retriever_plain_mode_returns_topk(tmp_path, sample_documents, fake_embeddings):
    store = _store(tmp_path, sample_documents, fake_embeddings)
    retriever = get_retriever(store, k=2, use_reranker=False)

    results = retriever.invoke("Paris")

    assert len(results) == 2


class FakeCrossEncoder(BaseCrossEncoder):
    """Stand-in for HuggingFaceCrossEncoder — scores a pair high if the
    query text appears in the document, so CrossEncoderReranker's real
    sorting/truncation logic can be exercised without downloading an actual
    cross-encoder model. Must subclass BaseCrossEncoder: CrossEncoderReranker
    is a pydantic model whose `model` field is typed BaseCrossEncoder, so a
    plain duck-typed class fails pydantic's is_instance_of validation."""

    def __init__(self, model_name: str = None, **kwargs):
        super().__init__(**kwargs)

    def score(self, text_pairs: List[Tuple[str, str]]) -> List[float]:
        return [1.0 if q.lower() in d.lower() else 0.0 for q, d in text_pairs]


def test_get_retriever_reranker_mode_narrows_to_k(tmp_path, sample_documents, fake_embeddings, monkeypatch):
    store = _store(tmp_path, sample_documents, fake_embeddings)
    monkeypatch.setattr(
        "langchain_community.cross_encoders.HuggingFaceCrossEncoder", FakeCrossEncoder
    )

    retriever = get_retriever(store, k=1, use_reranker=True, rerank_fetch_k=3)
    results = retriever.invoke("Eiffel Tower Paris France")

    assert len(results) == 1
    assert "Paris" in results[0].page_content


def test_get_retriever_reranker_mode_falls_back_when_no_pair_matches(
    tmp_path, sample_documents, fake_embeddings, monkeypatch
):
    store = _store(tmp_path, sample_documents, fake_embeddings)
    monkeypatch.setattr(
        "langchain_community.cross_encoders.HuggingFaceCrossEncoder", FakeCrossEncoder
    )

    retriever = get_retriever(store, k=2, use_reranker=True, rerank_fetch_k=3)
    results = retriever.invoke("completely unrelated query about spacecraft")

    # no pair scores above 0, but the reranker should still return k docs
    # (just with tied/zero scores) rather than erroring or returning none
    assert len(results) == 2