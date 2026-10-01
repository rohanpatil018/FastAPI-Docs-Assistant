# RAG service (work in progress)

Stage 1: dense / BM25 / hybrid retrieval + Gemini answers + FastAPI.

## Setup
```
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env    # add GOOGLE_API_KEY
```

## Corpus (FastAPI docs)
```
git clone --depth 1 --filter=blob:none --sparse https://github.com/fastapi/fastapi.git tmp_fastapi
cd tmp_fastapi && git sparse-checkout set docs/en/docs && cd ..
cp -r tmp_fastapi/docs/en/docs/* data/docs/      # Windows: xcopy /E
```

## Run
```
python -m app.ingest                  # build the index (stop the API first)
uvicorn app.main:app --reload         # then open http://localhost:8000/docs
```
