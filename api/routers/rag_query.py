from functools import lru_cache

from fastapi import APIRouter, HTTPException

from api.schemas import AskRequest, AskResponse, SourceDocument
from rag.chain import build_chain
from rag.retriever import get_retriever
from rag.vectorstore import load_vectorstore

router = APIRouter()

VECTORSTORE_DIR = "./data/vectorstore"
VECTORSTORE_BACKEND = "faiss"


@lru_cache(maxsize=1)
def _get_vectorstore():
    return load_vectorstore(VECTORSTORE_DIR, backend=VECTORSTORE_BACKEND)


@lru_cache(maxsize=8)
def _get_retriever(k: int, use_reranker: bool):
    return get_retriever(_get_vectorstore(), k=k, use_reranker=use_reranker)


@router.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    try:
        retriever = _get_retriever(request.k, request.use_reranker)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=f"Vector store not available: {exc}") from exc

    chain = build_chain(retriever, max_tokens=request.max_tokens, temperature=request.temperature)
    
    try:
        result = chain.answer(request.question)
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"vLLM backend error: {exc}") from exc

    return AskResponse(
        answer=result.answer,
        sources=[
            SourceDocument(content=doc.page_content, source=doc.metadata.get("source", "unknown"))
            for doc in result.source_documents
        ],
    )
