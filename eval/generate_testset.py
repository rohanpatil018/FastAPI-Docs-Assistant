"""Generate an eval set: one question per sampled chunk, with that chunk as the gold answer.

Run with:  python -m eval.generate_testset 40
Only reads storage/chunks.json, so the API server can stay running.
Resumes from eval/testset.json if it already exists.
Afterwards, open eval/testset.json and hand-edit ~15 questions.
"""
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import settings

N = int(sys.argv[1]) if len(sys.argv) > 1 else 40
OUT = Path("eval/testset.json")

PROMPT = (
    "Below is a passage from technical documentation. Write ONE question a developer "
    "might ask that this passage answers. The question must make sense on its own "
    "(never say 'the passage' or 'this document') and should not copy long phrases "
    "from the text. Return only the question.\n\nPassage:\n{text}"
)


def text_of(response) -> str:
    content = response.content
    if isinstance(content, list):
        content = "".join(
            p.get("text", "") if isinstance(p, dict) else str(p) for p in content
        )
    return content.strip()


def pick_chunks(chunks: list[dict], n: int) -> list[dict]:
    random.seed(7)
    pool = [c for c in chunks if len(c["text"]) >= 300 and "```" not in c["text"]]
    random.shuffle(pool)
    per_source: Counter = Counter()
    picked = []
    for c in pool:
        if per_source[c["source"]] >= 2:
            continue
        per_source[c["source"]] += 1
        picked.append(c)
        if len(picked) == n:
            break
    return picked


def main() -> None:
    if not settings.google_api_key:
        raise SystemExit("GOOGLE_API_KEY is not set in .env")

    chunks = json.loads(Path(settings.chunks_path).read_text(encoding="utf-8"))
    llm = ChatGoogleGenerativeAI(
        model=settings.gemini_model, google_api_key=settings.google_api_key
    )

    testset = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else []
    done = {t["gold_chunk_id"] for t in testset}
    print(f"Resuming with {len(testset)} questions already saved", flush=True)

    for chunk in pick_chunks(chunks, N):
        if chunk["chunk_id"] in done:
            continue

        question = None
        for attempt in range(5):
            try:
                question = text_of(llm.invoke(PROMPT.format(text=chunk["text"])))
                break
            except Exception as e:  # rate limits on the free tier are common
                print(f"  retry ({e.__class__.__name__})", flush=True)
                time.sleep(20 * (attempt + 1))
        if not question:
            continue

        testset.append(
            {
                "question": question,
                "gold_chunk_id": chunk["chunk_id"],
                "gold_source": chunk["source"],
            }
        )
        OUT.write_text(json.dumps(testset, indent=2), encoding="utf-8")
        print(f"{len(testset)}/{N}  {question}", flush=True)
        time.sleep(13)

    print(f"Saved {len(testset)} questions to {OUT}")


if __name__ == "__main__":
    main()