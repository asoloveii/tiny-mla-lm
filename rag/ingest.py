import argparse
from typing import List

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .vectorstore import Backend, build_vectorstore, get_embeddings

LOADERS_BY_SUFFIX = {
    ".pdf": PyPDFLoader,
    ".txt": TextLoader,
    ".md": TextLoader,
}


def load_documents(source_dir: str) -> List[Document]:
    """Load every supported file found under source_dir"""
    docs: List[Document] = []
    for suffix, loader_cls in LOADERS_BY_SUFFIX.items():
        loader = DirectoryLoader(
            source_dir,
            glob=f"**/*{suffix}",
            loader_cls=loader_cls,
            show_progress=True,
        )
        docs.extend(loader.load())

    if not docs:
        raise ValueError(
            f"No documents found under {source_dir!r} "
            f"(supported extensions: {', '.join(LOADERS_BY_SUFFIX)})"
        )
    return docs


def chunk_documents(documents: List[Document],
                    chunk_size: int = 300,
                    chunk_overlap: int = 70,) -> List[Document]:
    """Split into overlapping chunks"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)


def ingest(source_dir: str,
           persist_dir: str,
           backend: Backend = "faiss",
           chunk_size: int = 1000,
           chunk_overlap: int = 150,) -> None:
    print(f"Loading documents from {source_dir} ...")
    raw_docs = load_documents(source_dir)
    print(f"Loaded {len(raw_docs)} documents")

    chunks = chunk_documents(raw_docs, chunk_size, chunk_overlap)
    print(f"Split into {len(chunks)} chunks")

    embeddings = get_embeddings()
    build_vectorstore(chunks, persist_dir, backend=backend, embeddings=embeddings)
    print(f"Wrote {backend} vector store to {persist_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="Directory of source documents")
    parser.add_argument("--persist-dir", required=True, help="Where to write the vector store")
    parser.add_argument("--backend", choices=["faiss", "chroma"], default="faiss")
    parser.add_argument("--chunk-size", type=int, default=300)
    parser.add_argument("--chunk-overlap", type=int, default=70)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    ingest(args.source, args.persist_dir, args.backend, args.chunk_size, args.chunk_overlap)
