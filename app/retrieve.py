"""Dense, BM25, hybrid (reciprocal rank fusion) and reranked retrieval."""
import json
import re
from pathlib import Path

import numpy as np
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import QdrantClient
from rank_bm25 import BM25Okapi

from app.config import settings

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class Retriever:
    def __init__(self) -> None:
        self.model = TextEmbedding(settings.embed_model)
        self.reranker = TextCrossEncoder(settings.rerank_model)
        self.client = QdrantClient(path=settings.qdrant_path)
        self.chunks: list[dict] = json.loads(
            Path(settings.chunks_path).read_text(encoding="utf-8")
        )
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])

    def dense(self, query: str, k: int) -> list[dict]:
        vec = next(iter(self.model.embed([BGE_QUERY_PREFIX + query])))
        vec = vec / np.linalg.norm(vec)
        hits = self.client.query_points(
            settings.collection, query=vec.tolist(), limit=k
        ).points
        return [{**h.payload, "score": float(h.score)} for h in hits]

    def sparse(self, query: str, k: int) -> list[dict]:
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(scores)[::-1][:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top]

    def hybrid(self, query: str, k: int, rrf_k: int = 60) -> list[dict]:
        pool = max(k, settings.candidate_k)
        fused: dict[int, float] = {}
        by_id: dict[int, dict] = {}
        for results in (self.dense(query, pool), self.sparse(query, pool)):
            for rank, item in enumerate(results):
                cid = item["chunk_id"]
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (rrf_k + rank + 1)
                by_id[cid] = item
        ranked = sorted(fused, key=fused.get, reverse=True)[:k]
        return [{**by_id[cid], "score": fused[cid]} for cid in ranked]

    def rerank(self, query: str, candidates: list[dict], k: int) -> list[dict]:
        scores = list(self.reranker.rerank(query, [c["text"] for c in candidates]))
        order = np.argsort(scores)[::-1][:k]
        return [{**candidates[i], "score": float(scores[i])} for i in order]

    def retrieve(self, query: str, mode: str = "hybrid", k: int | None = None) -> list[dict]:
        k = k or settings.top_k
        if mode == "dense":
            return self.dense(query, k)
        if mode == "bm25":
            return self.sparse(query, k)
        if mode.startswith("hybrid_rerank"):
            # "hybrid_rerank@N" scores N candidates (used by the eval sweep);
            # plain "hybrid_rerank" uses settings.rerank_candidates.
            n = int(mode.split("@")[1]) if "@" in mode else settings.rerank_candidates
            return self.rerank(query, self.hybrid(query, n), k)
        return self.hybrid(query, k)