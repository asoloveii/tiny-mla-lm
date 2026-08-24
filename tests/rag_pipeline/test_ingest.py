import pytest
from langchain_core.documents import Document

from rag.ingest import chunk_documents, load_documents


def test_load_documents_reads_txt_and_md_recursively(docs_dir):
    docs = load_documents(str(docs_dir))
    sources = {d.metadata.get("source", "") for d in docs}

    assert len(docs) == 3
    assert any("a.txt" in s for s in sources)
    assert any("b.md" in s for s in sources)
    assert any("c.txt" in s for s in sources)


def test_load_documents_raises_on_empty_dir(tmp_path):
    with pytest.raises(ValueError):
        load_documents(str(tmp_path))


def test_load_documents_ignores_unsupported_extensions(tmp_path):
    (tmp_path / "notes.txt").write_text("this one counts")
    (tmp_path / "image.png").write_bytes(b"\x89PNG\r\n")  # unsupported extension

    docs = load_documents(str(tmp_path))

    assert len(docs) == 1
    assert docs[0].page_content == "this one counts"


def test_chunk_documents_splits_long_text():
    long_text = "sentence. " * 500  # long enough to require multiple chunks
    doc = Document(page_content=long_text, metadata={"source": "long.txt"})

    chunks = chunk_documents([doc], chunk_size=200, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(c.metadata["source"] == "long.txt" for c in chunks)


def test_chunk_documents_keeps_short_text_as_one_chunk():
    doc = Document(page_content="short text", metadata={"source": "short.txt"})
    chunks = chunk_documents([doc], chunk_size=1000, chunk_overlap=100)

    assert len(chunks) == 1
    assert chunks[0].page_content == "short text"


def test_chunk_documents_overlap_preserves_boundary_content():
    # two sentences long enough to be split into separate chunks; the
    # overlap should mean the boundary word isn't lost from both sides
    text = ("A" * 90) + ". " + ("B" * 90) + "."
    doc = Document(page_content=text, metadata={"source": "boundary.txt"})

    chunks = chunk_documents([doc], chunk_size=100, chunk_overlap=30)

    assert len(chunks) >= 2
    joined = " ".join(c.page_content for c in chunks)
    assert "A" * 90 in joined
    assert "B" * 90 in joined
