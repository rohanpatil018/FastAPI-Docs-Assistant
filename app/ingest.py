"""Build the index: load docs -> chunk -> embed -> Qdrant (+ chunks.json for BM25).

Run with:  python -m app.ingest
Stop the API server first: embedded Qdrant allows one process at a time.
"""
import json
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
import numpy as np
from fastembed import TextEmbedding
from data import load_env

from app.config import settings


def load_documents(docs_dir: str) -> list[dict]:
    docs = []
    for path in sorted(Path(docs_dir).rglob("*")):
        if path.suffix.lower() in {".md", ".txt"} and path.is_file():
            text = path.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                docs.append({"source": str(path.relative_to(docs_dir)), "text": text})
    return docs


def chunk_documents(docs: list[dict]) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = []
    for doc in docs:
        for piece in splitter.split_text(doc["text"]):
            chunks.append(
                {"chunk_id": len(chunks), "source": doc["source"], "text": piece}
            )
    return chunks


def build_index() -> None:
    docs = load_documents(settings.docs_dir)
    if not docs:
        raise SystemExit(f"No .md/.txt files found in {settings.docs_dir}")
    chunks = chunk_documents(docs)
    print(f"Loaded {len(docs)} documents -> {len(chunks)} chunks")

    model = TextEmbedding(settings.embed_model)
    vectors = np.array(list(model.embed([c["text"] for c in chunks], batch_size=64)))
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    Path(settings.chunks_path).parent.mkdir(parents=True, exist_ok=True)
    Path(settings.chunks_path).write_text(json.dumps(chunks), encoding="utf-8")

    client = QdrantClient(path=settings.qdrant_path)
    if client.collection_exists(settings.collection):
        client.delete_collection(settings.collection)
    client.create_collection(
        settings.collection,
        vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.COSINE),
    )
    points = [
        PointStruct(id=c["chunk_id"], vector=v.tolist(), payload=c)
        for c, v in zip(chunks, vectors)
    ]
    for i in range(0, len(points), 256):
        client.upsert(settings.collection, points=points[i : i + 256])
    client.close()
    print("Index built.")


if __name__ == "__main__":
    build_index()
