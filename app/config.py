from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    google_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"
    embed_model: str = "BAAI/bge-small-en-v1.5"

    docs_dir: str = "data/docs"
    qdrant_path: str = "storage/qdrant"
    chunks_path: str = "storage/chunks.json"
    collection: str = "docs"

    chunk_size: int = 800
    chunk_overlap: int = 100
    top_k: int = 5
    candidate_k: int = 20

    rerank_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"


settings = Settings()
