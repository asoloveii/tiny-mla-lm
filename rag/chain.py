from dataclasses import dataclass, field
from typing import List

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from api import client as api_client

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the question using only the "
    "information in the provided context. If the context doesn't contain "
    "the answer, say you don't know rather than guessing."
)

PROMPT_TEMPLATE = """{system_prompt}

Context:
{context}

Question: {question}

Answer:"""


def format_context(documents: List[Document]) -> str:
    """Number each chunk and tag its source"""
    blocks = []
    for i, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source", "unknown")
        blocks.append(f"[{i}] (source: {source})\n{doc.page_content}")
    return "\n\n".join(blocks)


def build_prompt(question: str, documents: List[Document]) -> str:
    return PROMPT_TEMPLATE.format(
        system_prompt=SYSTEM_PROMPT,
        context=format_context(documents),
        question=question,
    )


@dataclass
class RAGResult:
    answer: str
    source_documents: List[Document] = field(default_factory=list)


class RAGChain:
    def __init__(self, retriever: BaseRetriever, **generation_kwargs):
        self.retriever = retriever
        self.generation_kwargs = generation_kwargs  

    def answer(self, question: str) -> RAGResult:
        documents = self.retriever.invoke(question)
        prompt = build_prompt(question, documents)
        answer_text = api_client.generate(prompt, **self.generation_kwargs)
        return RAGResult(answer=answer_text, source_documents=documents)


def build_chain(retriever: BaseRetriever, **generation_kwargs) -> RAGChain:
    return RAGChain(retriever, **generation_kwargs)
