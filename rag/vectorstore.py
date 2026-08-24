import os
from typing import Iterable, Literal, Optional, Union

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

Backend = Literal["faiss", "chroma"]
VectorStore = Union[FAISS, Chroma]

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings(model_name: str = DEFAULT_EMBEDDING_MODEL) -> Embeddings:
    """Single place to construct the embedding model"""
    return HuggingFaceEmbeddings(model_name=model_name)


def build_vectorstore(documents: Iterable[Document],
                      persist_dir: str,
                      backend: Backend = "faiss",
                      embeddings: Optional[Embeddings] = None,) -> VectorStore:
    """Create a fresh vector store from documents and persist it to disk"""
    documents = list(documents)
    if not documents:
        raise ValueError("No documents provided to build_vectorstore()")

    embeddings = embeddings or get_embeddings()
    os.makedirs(persist_dir, exist_ok=True)

    if backend == "faiss":
        store = FAISS.from_documents(documents, embeddings)
        store.save_local(persist_dir)
        return store

    if backend == "chroma":
        return Chroma.from_documents(documents, embeddings, persist_directory=persist_dir)

    raise ValueError(f"Unknown backend: {backend!r} (expected 'faiss' or 'chroma')")


def load_vectorstore(persist_dir: str,
                    backend: Backend = "faiss",
                    embeddings: Optional[Embeddings] = None,) -> VectorStore:
    """Load a previously-persisted vector store for querying"""
    embeddings = embeddings or get_embeddings()

    if backend == "faiss":
        if not os.path.exists(os.path.join(persist_dir, "index.faiss")):
            raise FileNotFoundError(f"No FAISS index found at {persist_dir!r} - run ingest.py first")
        return FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)

    if backend == "chroma":
        if not os.path.isdir(persist_dir):
            raise FileNotFoundError(f"No Chroma collection found at {persist_dir!r} - run ingest.py first")
        return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

    raise ValueError(f"Unknown backend: {backend!r} (expected 'faiss' or 'chroma')")


def add_documents(
    store: VectorStore,
    documents: Iterable[Document],
    persist_dir: str,
    backend: Backend = "faiss",
) -> None:
    """Incrementally add documents to an already-loaded store"""
    documents = list(documents)
    if not documents:
        return
    store.add_documents(documents)
    if backend == "faiss":
        store.save_local(persist_dir)  
