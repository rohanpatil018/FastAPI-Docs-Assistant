"""Train and evaluate the query router (NumPy only).

Run with:  python -m eval.train_router
Reports 5-fold cross-validated metrics on eval/router_data.json, then fits on all data
and saves storage/router.npz. It also checks how many of the retrieval test-set
questions the router would let through (a false refusal there is a real failure).
"""
import json
from pathlib import Path

import numpy as np

from app.config import settings
from app.router_model import LABELS, ROUTER_PATH, QueryRouter, embed_texts, train_softmax


def stratified_folds(y: np.ndarray, k: int, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    fold = np.zeros(len(y), dtype=int)
    for c in np.unique(y):
        idx = rng.permutation(np.where(y == c)[0])
        fold[idx] = np.arange(len(idx)) % k
    return fold


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, dict, np.ndarray]:
    k = len(LABELS)
    cm = np.zeros((k, k), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    per_class = {}
    for i, name in enumerate(LABELS):
        tp = cm[i, i]
        prec = tp / max(cm[:, i].sum(), 1)
        rec = tp / max(cm[i].sum(), 1)
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[name] = {
            "precision": round(float(prec), 3),
            "recall": round(float(rec), 3),
            "f1-score": round(float(f1), 3),
        }
    return float(np.trace(cm) / cm.sum()), per_class, cm


def main(embedder=None) -> None:
    if embedder is None:
        from fastembed import TextEmbedding

        embedder = TextEmbedding(settings.embed_model)

    data = json.loads(Path("eval/router_data.json").read_text(encoding="utf-8"))
    X = embed_texts(embedder, [d["text"] for d in data])
    y = np.array([LABELS.index(d["label"]) for d in data])

    folds = stratified_folds(y, 5)
    pred = np.zeros_like(y)
    for f in range(5):
        tr, te = folds != f, folds == f
        W, b = train_softmax(X[tr], y[tr], len(LABELS))
        pred[te] = np.argmax(X[te] @ W + b, axis=1)

    acc, per_class, cm = metrics(y, pred)
    print(f"5-fold cross-validated accuracy: {acc:.3f} on {len(y)} examples\n")
    print(f"{'class':<14}{'precision':>10}{'recall':>9}{'f1':>8}")
    for name, m in per_class.items():
        print(f"{name:<14}{m['precision']:>10}{m['recall']:>9}{m['f1-score']:>8}")
    print("\nConfusion matrix (rows = true, cols = predicted):")
    print("labels:", LABELS)
    print(cm)

    W, b = train_softmax(X, y, len(LABELS))
    Path(ROUTER_PATH).parent.mkdir(parents=True, exist_ok=True)
    np.savez(ROUTER_PATH, W=W, b=b, classes=np.array(LABELS))

    router = QueryRouter(embedder)
    testset = json.loads(Path("eval/testset.json").read_text(encoding="utf-8"))
    passed = sum(router.route(t["question"])[0] == "retrieve" for t in testset)
    print(f"\nRetrieval test-set questions routed to 'retrieve': {passed}/{len(testset)}")

    Path("eval/router_results.json").write_text(
        json.dumps(
            {
                "examples": len(data),
                "cv_accuracy": round(acc, 3),
                "per_class": per_class,
                "confusion_matrix": {"labels": LABELS, "rows": cm.tolist()},
                "testset_routed_to_retrieve": f"{passed}/{len(testset)}",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {ROUTER_PATH} and eval/router_results.json")


if __name__ == "__main__":
    main()