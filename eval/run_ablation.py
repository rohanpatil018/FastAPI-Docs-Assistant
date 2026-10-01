"""Measure retrieval quality for each mode against eval/testset.json.

Run with:  python -m eval.run_ablation
Stop the API server first (embedded Qdrant allows one process at a time).
"""
import json
import time
from pathlib import Path

from app.retrieve import Retriever

K = 10
MODES = ["dense", "bm25", "hybrid", "hybrid_rerank@10", "hybrid_rerank"]


def evaluate(retriever: Retriever, testset: list[dict], mode: str) -> dict:
    hit1 = hit5 = src_hit5 = 0
    rr = 0.0
    latencies = []
    for item in testset:
        t0 = time.perf_counter()
        results = retriever.retrieve(item["question"], mode, K)
        latencies.append((time.perf_counter() - t0) * 1000)
        ids = [r["chunk_id"] for r in results]
        if item["gold_chunk_id"] in ids:
            rank = ids.index(item["gold_chunk_id"]) + 1
            rr += 1.0 / rank
            hit1 += rank == 1
            hit5 += rank <= 5
        if item["gold_source"] in [r["source"] for r in results[:5]]:
            src_hit5 += 1
    n = len(testset)
    latencies.sort()
    return {
        "mode": mode,
        "hit@1": round(hit1 / n, 3),
        "hit@5": round(hit5 / n, 3),
        "mrr@10": round(rr / n, 3),
        "source_hit@5": round(src_hit5 / n, 3),
        "p50_ms": round(latencies[len(latencies) // 2], 1),
        "p95_ms": round(latencies[min(n - 1, int(0.95 * n))], 1),
    }


def main() -> None:
    testset = json.loads(Path("eval/testset.json").read_text(encoding="utf-8"))
    retriever = Retriever()
    retriever.retrieve("warm up", "hybrid", 1)  # keep model load out of the timings
    rows = [evaluate(retriever, testset, m) for m in MODES]
    Path("eval/results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    print(f"\nQuestions: {len(testset)}\n")
    print("| Mode | Hit@1 | Hit@5 | MRR@10 | Source Hit@5 | p50 ms | p95 ms |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(
            f"| {r['mode']} | {r['hit@1']} | {r['hit@5']} | {r['mrr@10']} | "
            f"{r['source_hit@5']} | {r['p50_ms']} | {r['p95_ms']} |"
        )


if __name__ == "__main__":
    main()
