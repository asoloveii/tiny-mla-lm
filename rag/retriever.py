from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore
from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker


DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def get_retriever(vectorstore: VectorStore,
                  k: int = 3,
                  use_reranker: bool = False,
                  rerank_fetch_k: int = 20,
                  reranker_model: str = DEFAULT_RERANKER_MODEL,) -> BaseRetriever:
    """
    Plain mode: vector similarity search returns the top k chunks directly.

    Reranked mode: vector search first casts a wider net (rerank_fetch_k,
    recall-oriented), then a cross-encoder rescores query+chunk pairs
    jointly and narrows it down to the best k (precision-oriented)
    """
    if not use_reranker:
        return vectorstore.as_retriever(search_kwargs={"k": k})

    from langchain_community.cross_encoders import HuggingFaceCrossEncoder

    base_retriever = vectorstore.as_retriever(search_kwargs={"k": rerank_fetch_k})
    cross_encoder = HuggingFaceCrossEncoder(model_name=reranker_model)
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=k)

    return ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever,
    )
