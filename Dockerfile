FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*

# Hugging Face Spaces runs containers as a non-root user with uid 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    FASTEMBED_CACHE_PATH=/home/user/app/.fastembed
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .

# Bake the corpus, models, vector index and router into the image at build time
RUN git clone --depth 1 --filter=blob:none --sparse https://github.com/fastapi/fastapi.git /tmp/fa \
    && cd /tmp/fa && git sparse-checkout set docs/en/docs \
    && mkdir -p /home/user/app/data/docs \
    && cp -r docs/en/docs/* /home/user/app/data/docs/ \
    && rm -rf /tmp/fa
RUN python -m app.ingest && python -m eval.train_router

EXPOSE 7860
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
