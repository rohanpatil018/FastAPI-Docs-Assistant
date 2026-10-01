import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.config import settings
from app.generate import Generator
from app.retrieve import Retriever
from app.schemas import QueryRequest, QueryResponse, RetrieveResponse

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load heavy models once at startup, not per request
    state["retriever"] = Retriever()
    state["generator"] = Generator() if settings.google_api_key else None
    yield
    state.clear()


app = FastAPI(title="RAG Service", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "chunks": len(state["retriever"].chunks),
        "llm_ready": state["generator"] is not None,
    }


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: QueryRequest):
    t0 = time.perf_counter()
    sources = state["retriever"].retrieve(req.question, req.mode, req.top_k)
    return {"sources": sources, "latency_ms": {"retrieval": _ms(t0)}}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if state["generator"] is None:
        raise HTTPException(503, "GOOGLE_API_KEY is not set")
    t0 = time.perf_counter()
    sources = state["retriever"].retrieve(req.question, req.mode, req.top_k)
    t1 = time.perf_counter()
    answer = state["generator"].answer(req.question, sources)
    t2 = time.perf_counter()
    return {
        "answer": answer,
        "sources": sources,
        "latency_ms": {
            "retrieval": round((t1 - t0) * 1000, 1),
            "generation": round((t2 - t1) * 1000, 1),
            "total": round((t2 - t0) * 1000, 1),
        },
    }


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)
