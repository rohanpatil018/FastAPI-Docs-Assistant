"""Refresh the result tables in README.md from eval/results.json and eval/router_results.json.

Run with:  python -m eval.update_readme
Replaces the text between <!-- RESULTS:START/END --> and <!-- ROUTER:START/END --> markers.
"""
import json
import re
from pathlib import Path

LABELS = {
    "dense": "Dense (bge-small)",
    "bm25": "BM25",
    "hybrid": "Hybrid (RRF)",
    "hybrid_rerank": "Hybrid + rerank (serving default)",
}


def label(mode: str) -> str:
    if mode.startswith("hybrid_rerank@"):
        return f"Hybrid + rerank (top {mode.split('@')[1]})"
    return LABELS.get(mode, mode)


def retrieval_table(rows: list[dict]) -> str:
    lines = [
        "| Mode | Hit@1 | Hit@5 | MRR@10 | Source Hit@5 | p50 ms | p95 ms |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {label(r['mode'])} | {r['hit@1']} | {r['hit@5']} | {r['mrr@10']} | "
            f"{r['source_hit@5']} | {r['p50_ms']} | {r['p95_ms']} |"
        )
    return "\n".join(lines)


def router_table(res: dict) -> str:
    lines = ["| Class | Precision | Recall | F1 |", "|---|---|---|---|"]
    for name, m in res["per_class"].items():
        lines.append(f"| {name} | {m['precision']} | {m['recall']} | {m['f1-score']} |")
    lines.append("")
    lines.append(
        f"5-fold cross-validated accuracy: **{res['cv_accuracy']}** on {res['examples']} labelled examples. "
        f"Retrieval test-set questions routed to `retrieve`: **{res['testset_routed_to_retrieve']}**."
    )
    return "\n".join(lines)


def replace_block(text: str, name: str, body: str) -> tuple[str, bool]:
    pattern = re.compile(rf"(<!-- {name}:START -->\n).*?(\n<!-- {name}:END -->)", re.S)
    if not pattern.search(text):
        return text, False
    return pattern.sub(lambda m: m.group(1) + body + m.group(2), text), True


def main() -> None:
    readme = Path("README.md")
    text = readme.read_text(encoding="utf-8")

    results = Path("eval/results.json")
    if results.exists():
        text, ok = replace_block(text, "RESULTS", retrieval_table(json.loads(results.read_text(encoding="utf-8"))))
        print("Retrieval table updated." if ok else "RESULTS markers not found; skipped.")

    router = Path("eval/router_results.json")
    if router.exists():
        text, ok = replace_block(text, "ROUTER", router_table(json.loads(router.read_text(encoding="utf-8"))))
        print("Router table updated." if ok else "ROUTER markers not found; skipped.")

    readme.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()