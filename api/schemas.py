from typing import List, Optional

from pydantic import BaseModel, Field


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    max_tokens: int = Field(512, gt=0, le=4096)
    temperature: float = Field(0.7, ge=0.0, le=2.0)
    stop: Optional[List[str]] = None


class GenerateResponse(BaseModel):
    text: str


class SourceDocument(BaseModel):
    content: str
    source: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    k: int = Field(3, gt=0, le=20)
    use_reranker: bool = False
    max_tokens: int = Field(512, gt=0, le=4096)
    temperature: float = Field(0.7, ge=0.0, le=2.0)


class AskResponse(BaseModel):
    answer: str
    sources: List[SourceDocument]
