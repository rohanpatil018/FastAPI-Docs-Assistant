"""Query router: classifies a question as retrieve / chitchat / out_of_scope.

Features are the same bge-small embeddings used for retrieval; the classifier is a
softmax (multinomial logistic) regression implemented in NumPy and trained by
eval/train_router.py. Inference needs only NumPy, with no scikit-learn or SciPy.
"""
import numpy as np

ROUTER_PATH = "storage/router.npz"
LABELS = ["retrieve", "chitchat", "out_of_scope"]
# If a non-retrieve prediction is less confident than this, retrieve anyway.
# Wrongly refusing a real question is worse than retrieving for a chat message.
THRESHOLD = 0.6


def embed_texts(embedder, texts: list[str]) -> np.ndarray:
    vecs = np.array(list(embedder.embed(texts)))
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def train_softmax(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    l2: float = 1e-3,
    epochs: int = 600,
    lr: float = 0.05,
    balanced: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Full-batch Adam on class-weighted cross-entropy with L2 regularisation."""
    n, d = X.shape
    counts = np.bincount(y, minlength=n_classes).clip(min=1)
    class_w = n / (n_classes * counts) if balanced else np.ones(n_classes)
    sw = class_w[y]
    sw = sw / sw.sum()
    Y = np.eye(n_classes)[y]

    W, b = np.zeros((d, n_classes)), np.zeros(n_classes)
    mW, vW, mb, vb = np.zeros_like(W), np.zeros_like(W), np.zeros_like(b), np.zeros_like(b)
    b1, b2, eps = 0.9, 0.999, 1e-8
    for t in range(1, epochs + 1):
        G = (softmax(X @ W + b) - Y) * sw[:, None]
        gW, gb = X.T @ G + l2 * W, G.sum(axis=0)
        mW, mb = b1 * mW + (1 - b1) * gW, b1 * mb + (1 - b1) * gb
        vW, vb = b2 * vW + (1 - b2) * gW**2, b2 * vb + (1 - b2) * gb**2
        W -= lr * (mW / (1 - b1**t)) / (np.sqrt(vW / (1 - b2**t)) + eps)
        b -= lr * (mb / (1 - b1**t)) / (np.sqrt(vb / (1 - b2**t)) + eps)
    return W, b


class QueryRouter:
    def __init__(self, embedder, path: str = ROUTER_PATH) -> None:
        self.embedder = embedder
        data = np.load(path, allow_pickle=False)
        self.W, self.b = data["W"], data["b"]
        self.classes = [str(c) for c in data["classes"]]

    def route(self, text: str) -> tuple[str, float]:
        proba = softmax(embed_texts(self.embedder, [text]) @ self.W + self.b)[0]
        i = int(np.argmax(proba))
        label, conf = self.classes[i], float(proba[i])
        if label != "retrieve" and conf < THRESHOLD:
            return "retrieve", conf
        return label, conf