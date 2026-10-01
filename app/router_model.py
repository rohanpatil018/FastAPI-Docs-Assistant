"""Query router: classifies a question as retrieve / chitchat / out_of_scope.

Features are the same bge-small embeddings used for retrieval; the classifier is a
scikit-learn logistic regression trained by eval/train_router.py.
"""
import joblib
import numpy as np

ROUTER_PATH = "storage/router.joblib"
LABELS = ["retrieve", "chitchat", "out_of_scope"]
# If a non-retrieve prediction is less confident than this, retrieve anyway.
# Wrongly refusing a real question is worse than retrieving for a chat message.
THRESHOLD = 0.6


def embed_texts(embedder, texts: list[str]) -> np.ndarray:
    vecs = np.array(list(embedder.embed(texts)))
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


class QueryRouter:
    def __init__(self, embedder, path: str = ROUTER_PATH) -> None:
        self.embedder = embedder
        self.clf = joblib.load(path)

    def route(self, text: str) -> tuple[str, float]:
        proba = self.clf.predict_proba(embed_texts(self.embedder, [text]))[0]
        i = int(np.argmax(proba))
        label, conf = str(self.clf.classes_[i]), float(proba[i])
        if label != "retrieve" and conf < THRESHOLD:
            return "retrieve", conf
        return label, conf
