import pytest

from rag.vectorstore import add_documents, build_vectorstore, load_vectorstore


def test_build_vectorstore_faiss_creates_persist_dir(tmp_path, sample_documents, fake_embeddings):
    persist_dir = tmp_path / "faiss_store"
    store = build_vectorstore(sample_documents, str(persist_dir), backend="faiss", embeddings=fake_embeddings)

    assert (persist_dir / "index.faiss").exists()
    assert store.index.ntotal == len(sample_documents)


def test_build_vectorstore_rejects_empty_documents(tmp_path, fake_embeddings):
    with pytest.raises(ValueError):
        build_vectorstore([], str(tmp_path / "empty"), backend="faiss", embeddings=fake_embeddings)


def test_build_vectorstore_unknown_backend_raises(tmp_path, sample_documents, fake_embeddings):
    with pytest.raises(ValueError):
        build_vectorstore(sample_documents, str(tmp_path), backend="not_a_backend", embeddings=fake_embeddings)


def test_load_vectorstore_faiss_roundtrip(tmp_path, sample_documents, fake_embeddings):
    persist_dir = tmp_path / "faiss_store"
    build_vectorstore(sample_documents, str(persist_dir), backend="faiss", embeddings=fake_embeddings)

    loaded = load_vectorstore(str(persist_dir), backend="faiss", embeddings=fake_embeddings)
    results = loaded.similarity_search("fox", k=1)

    assert len(results) == 1


def test_load_vectorstore_missing_faiss_index_raises(tmp_path, fake_embeddings):
    with pytest.raises(FileNotFoundError):
        load_vectorstore(str(tmp_path / "does_not_exist"), backend="faiss", embeddings=fake_embeddings)


def test_load_vectorstore_missing_chroma_dir_raises(tmp_path, fake_embeddings):
    with pytest.raises(FileNotFoundError):
        load_vectorstore(str(tmp_path / "does_not_exist"), backend="chroma", embeddings=fake_embeddings)


def test_add_documents_grows_faiss_store(tmp_path, sample_documents, fake_embeddings):
    persist_dir = tmp_path / "faiss_store"
    store = build_vectorstore(sample_documents[:2], str(persist_dir), backend="faiss", embeddings=fake_embeddings)

    add_documents(store, [sample_documents[2]], str(persist_dir), backend="faiss")

    reloaded = load_vectorstore(str(persist_dir), backend="faiss", embeddings=fake_embeddings)
    assert reloaded.index.ntotal == 3


def test_add_documents_noop_on_empty_list(tmp_path, sample_documents, fake_embeddings):
    persist_dir = tmp_path / "faiss_store"
    store = build_vectorstore(sample_documents, str(persist_dir), backend="faiss", embeddings=fake_embeddings)
    before = store.index.ntotal

    add_documents(store, [], str(persist_dir), backend="faiss")

    assert store.index.ntotal == before


def test_build_and_load_vectorstore_chroma_roundtrip(tmp_path, sample_documents, fake_embeddings):
    pytest.importorskip("chromadb")
    persist_dir = tmp_path / "chroma_store"
    build_vectorstore(sample_documents, str(persist_dir), backend="chroma", embeddings=fake_embeddings)

    loaded = load_vectorstore(str(persist_dir), backend="chroma", embeddings=fake_embeddings)
    results = loaded.similarity_search("Paris", k=1)

    assert len(results) == 1
