import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.generate import Generator
from app.retrieve import Retriever
from app.router_model import ROUTER_PATH, QueryRouter
from app.schemas import QueryRequest, QueryResponse, RetrieveResponse

state: dict = {}

CHITCHAT_REPLY = (
    "Hi! I answer questions about the FastAPI documentation. "
    "Ask me about routing, validation, dependencies, security or deployment."
)
OUT_OF_SCOPE_REPLY = (
    "That looks outside the FastAPI documentation, so I can't answer it reliably. "
    "Try a question about FastAPI itself, such as routing, validation, dependencies or deployment."
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load heavy models once at startup, not per request
    retriever = Retriever()
    state["retriever"] = retriever
    state["generator"] = Generator() if settings.google_api_key else None
    state["router"] = (
        QueryRouter(retriever.model) if os.path.exists(ROUTER_PATH) else None
    )
    yield
    state.clear()


app = FastAPI(title="RAG Service", lifespan=lifespan)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "chunks": len(state["retriever"].chunks),
        "llm_ready": state["generator"] is not None,
        "router_ready": state["router"] is not None,
    }


@app.post("/retrieve", response_model=RetrieveResponse)
def retrieve(req: QueryRequest):
    t0 = time.perf_counter()
    sources = state["retriever"].retrieve(req.question, req.mode, req.top_k)
    return {"sources": sources, "latency_ms": {"retrieval": _ms(t0)}}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    t0 = time.perf_counter()

    route, confidence = "retrieve", None
    if state["router"] is not None:
        route, confidence = state["router"].route(req.question)
    t_route = time.perf_counter()

    if route in ("chitchat", "out_of_scope"):
        return {
            "answer": CHITCHAT_REPLY if route == "chitchat" else OUT_OF_SCOPE_REPLY,
            "sources": [],
            "latency_ms": {"routing": round((t_route - t0) * 1000, 1)},
            "route": route,
            "route_confidence": confidence,
        }

    if state["generator"] is None:
        raise HTTPException(503, "GOOGLE_API_KEY is not set")
    sources = state["retriever"].retrieve(req.question, req.mode, req.top_k)
    t1 = time.perf_counter()
    answer = state["generator"].answer(req.question, sources)
    t2 = time.perf_counter()
    return {
        "answer": answer,
        "sources": sources,
        "latency_ms": {
            "routing": round((t_route - t0) * 1000, 1),
            "retrieval": round((t1 - t_route) * 1000, 1),
            "generation": round((t2 - t1) * 1000, 1),
            "total": round((t2 - t0) * 1000, 1),
        },
        "route": route,
        "route_confidence": confidence,
    }


def _ms(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000, 1)


# Serve the chat UI at / (registered last so the API routes keep priority)
app.mount(
    "/",
    StaticFiles(directory=Path(__file__).parent / "static", html=True),
    name="static",
)
