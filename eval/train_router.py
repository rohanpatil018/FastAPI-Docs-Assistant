"""Train and evaluate the query router.

Run with:  python -m eval.train_router
Reports 5-fold cross-validated metrics on eval/router_data.json, then fits on all data
and saves storage/router.joblib. It also checks how many of the retrieval test-set
questions the router would let through (a false refusal there is a real failure).
"""
import json
from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from app.config import settings
from app.router_model import LABELS, ROUTER_PATH, QueryRouter, embed_texts


def main(embedder=None) -> None:
    if embedder is None:
        from fastembed import TextEmbedding

        embedder = TextEmbedding(settings.embed_model)

    data = json.loads(Path("eval/router_data.json").read_text(encoding="utf-8"))
    texts = [d["text"] for d in data]
    y = [d["label"] for d in data]
    X = embed_texts(embedder, texts)

    clf = LogisticRegression(C=10, max_iter=1000, class_weight="balanced")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pred = cross_val_predict(clf, X, y, cv=cv)

    report = classification_report(y, pred, labels=LABELS, output_dict=True, zero_division=0)
    print(classification_report(y, pred, labels=LABELS, zero_division=0))
    cm = confusion_matrix(y, pred, labels=LABELS)
    print("Confusion matrix (rows = true, cols = predicted):")
    print("labels:", LABELS)
    print(cm)

    clf.fit(X, y)
    Path(ROUTER_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, ROUTER_PATH)

    router = QueryRouter(embedder)
    testset = json.loads(Path("eval/testset.json").read_text(encoding="utf-8"))
    passed = sum(router.route(t["question"])[0] == "retrieve" for t in testset)
    print(f"\nRetrieval test-set questions routed to 'retrieve': {passed}/{len(testset)}")

    Path("eval/router_results.json").write_text(
        json.dumps(
            {
                "examples": len(data),
                "cv_accuracy": round(report["accuracy"], 3),
                "per_class": {k: {m: round(report[k][m], 3) for m in ("precision", "recall", "f1-score")} for k in LABELS},
                "confusion_matrix": {"labels": LABELS, "rows": cm.tolist()},
                "testset_routed_to_retrieve": f"{passed}/{len(testset)}",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Saved storage/router.joblib and eval/router_results.json")


if __name__ == "__main__":
    main()
