from typing import Literal

from pydantic import BaseModel, Field

Mode = Literal["dense", "bm25", "hybrid", "hybrid_rerank"]


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    mode: Mode = "hybrid"
    top_k: int = Field(default=5, ge=1, le=20)


class Source(BaseModel):
    chunk_id: int
    source: str
    text: str
    score: float


class RetrieveResponse(BaseModel):
    sources: list[Source]
    latency_ms: dict[str, float]


class QueryResponse(RetrieveResponse):
    answer: str
