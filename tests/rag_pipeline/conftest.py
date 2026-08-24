import hashlib
import random
from typing import List

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings


class FakeEmbeddings(Embeddings):

    def __init__(self, dim: int = 16):
        self.dim = dim

    def _embed(self, text: str) -> List[float]:
        seed = int(hashlib.sha256(text.encode()).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)
        return [rng.uniform(-1, 1) for _ in range(self.dim)]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


@pytest.fixture
def fake_embeddings() -> FakeEmbeddings:
    return FakeEmbeddings()


@pytest.fixture
def sample_documents() -> List[Document]:
    return [
        Document(
            page_content="Cats are small domesticated carnivorous mammals.",
            metadata={"source": "cats.txt"},
        ),
        Document(
            page_content="Python is a popular programming language for AI.",
            metadata={"source": "python.txt"},
        ),
        Document(
            page_content="The Eiffel Tower is located in Paris, France.",
            metadata={"source": "paris.txt"},
        ),
    ]


@pytest.fixture
def docs_dir(tmp_path):
    """A small directory of real .txt/.md files for testing ingest.load_documents."""
    (tmp_path / "a.txt").write_text("The quick brown fox jumps over the lazy dog.")
    (tmp_path / "b.md").write_text("# Notes\n\nMarkdown files are loaded as plain text here.")
    sub = tmp_path / "subdir"
    sub.mkdir()
    (sub / "c.txt").write_text("Nested files should also be picked up by the recursive glob.")
    return tmp_path
