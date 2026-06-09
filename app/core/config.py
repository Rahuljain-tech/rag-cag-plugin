import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Ollama settings
    OLLAMA_HOST: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    OLLAMA_GEN_MODEL: str = os.getenv("OLLAMA_GEN_MODEL", "llama3")
    OLLAMA_EMBED_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
    
    # Cache settings
    CACHE_DB_PATH: str = os.getenv("CACHE_DB_PATH", "semantic_cache.db")
    CACHE_SIMILARITY_THRESHOLD: float = float(os.getenv("CACHE_SIMILARITY_THRESHOLD", "0.90"))

    # Metrics
    METRICS_DB_PATH: str = os.getenv("METRICS_DB_PATH", "metrics.db")
    METRICS_RECENT_LIMIT: int = int(os.getenv("METRICS_RECENT_LIMIT", "10"))

    # Ingest / chunking
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "800"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))
    INGEST_MAX_FILE_BYTES: int = int(os.getenv("INGEST_MAX_FILE_BYTES", "512000"))

    class Config:
        env_file = ".env"

settings = Settings()
