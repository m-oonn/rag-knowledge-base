import os
from pathlib import Path
from typing import Optional, Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ============================================================
    # App
    # ============================================================
    APP_NAME: str = "RAG-PDF-System"
    APP_ENV: str = "development"
    DEBUG: bool = True
    API_PREFIX: str = "/api/v1"

    # ============================================================
    # Paths
    # ============================================================
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    UPLOAD_DIR: Path = BASE_DIR / "data" / "raw"
    PROCESSED_DIR: Path = BASE_DIR / "data" / "processed"
    VECTOR_DIR: Path = BASE_DIR / "data" / "vectors"
    CHROMA_DB_DIR: Path = BASE_DIR / "data" / "chroma_db"

    # ============================================================
    # Security
    # ============================================================
    SECRET_KEY: str = "unsafe-secret-key"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440

    # ============================================================
    # Vector Store Mode
    #   "chroma"  → zero-dependency, local file-based
    #   "milvus"  → enterprise, requires Milvus + etcd + MinIO
    # ============================================================
    VECTOR_STORE: Literal["chroma", "milvus"] = "chroma"

    # ============================================================
    # LLM Provider
    #   "ollama"    → local free, requires Ollama running
    #   "deepseek"  → cheap cloud, requires DEEPSEEK_API_KEY
    #   "dashscope" → Alibaba Qwen, requires DASHSCOPE_API_KEY
    # ============================================================
    LLM_PROVIDER: Literal["ollama", "deepseek", "dashscope"] = "ollama"

    # ============================================================
    # Embedding Provider
    #   "sentence-transformers" → local free, all-MiniLM-L6-v2
    #   "dashscope"             → Alibaba text-embedding-v1
    # ============================================================
    EMBEDDING_PROVIDER: Literal["sentence-transformers", "dashscope"] = "sentence-transformers"

    # ============================================================
    # Rerank Provider
    #   "none"       → no reranking (fastest)
    #   "local"      → sentence-transformers CrossEncoder
    #   "dashscope"  → Alibaba gte-rerank
    # ============================================================
    RERANK_PROVIDER: Literal["none", "local", "dashscope"] = "local"

    # ============================================================
    # LLM Config
    # ============================================================
    DASHSCOPE_API_KEY: Optional[str] = None
    DEEPSEEK_API_KEY: Optional[str] = None
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com"

    # DashScope models
    LLM_MODEL: str = "qwen-max"            # used when provider=dashscope
    EMBEDDING_MODEL: str = "text-embedding-v1"
    EMBEDDING_BATCH_SIZE: int = 20
    EMBEDDING_MAX_BATCH_SIZE: int = 25

    # Ollama models
    OLLAMA_HOST: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "qwen2.5:7b"

    # DeepSeek models
    DEEPSEEK_MODEL: str = "deepseek-chat"

    # Sentence Transformers
    SENTENCE_TRANSFORMER_MODEL: str = "BAAI/bge-small-zh-v1.5"  # Chinese-optimized, 512-dim
    SENTENCE_TRANSFORMER_MODEL_EN: str = "all-MiniLM-L6-v2"       # English fallback, 384-dim
    RERANK_CROSS_ENCODER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    HF_ENDPOINT: str = "https://huggingface.co"

    # ============================================================
    # Chroma Config
    # ============================================================
    CHROMA_COLLECTION_NAME: str = "rag_documents"

    # ============================================================
    # Milvus Config (enterprise mode)
    # ============================================================
    MILVUS_HOST: str = "localhost"
    MILVUS_PORT: int = 19530
    MILVUS_COLLECTION_NAME: str = "rag_documents_v2"
    MILVUS_DIMENSION: int = 1536

    # ============================================================
    # Redis (optional, for session cache)
    # ============================================================
    REDIS_URL: str = "redis://localhost:6379/0"

    # ============================================================
    # RabbitMQ / Celery (optional, for async doc processing)
    # ============================================================
    RABBITMQ_HOST: str = "localhost"
    RABBITMQ_PORT: int = 5672
    RABBITMQ_USER: str = "guest"
    RABBITMQ_PASSWORD: str = "guest"
    ENABLE_CELERY: bool = False

    # ============================================================
    # MinIO (optional, for object storage)
    # ============================================================
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET_NAME: str = "rag-documents"
    MINIO_SECURE: bool = False
    ENABLE_MINIO: bool = False

    # ============================================================
    # SQL Database
    # ============================================================
    DATABASE_URL: str = "sqlite:///./rag_system.db"

    # ============================================================
    # RAG Config
    # ============================================================
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K: int = 20
    RERANK_TOP_N: int = 5

    # Rerank toggle
    ENABLE_RERANK: bool = True
    RERANK_MODEL: str = "gte-rerank"

    # Hybrid Search (BM25 + Vector + RRF)
    ENABLE_HYBRID_SEARCH: bool = True
    HYBRID_ALPHA: float = 0.5  # 0=all BM25, 1=all vector

    # Multi-hop
    ENABLE_MULTI_HOP: bool = False
    MAX_HOP: int = 3

    # Memory
    SHORT_TERM_MEMORY_TTL: int = 3600
    LONG_TERM_MEMORY_COLLECTION: str = "user_memory"
    MEMORY_HISTORY_LIMIT: int = 10

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()

os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
os.makedirs(settings.PROCESSED_DIR, exist_ok=True)
os.makedirs(settings.VECTOR_DIR, exist_ok=True)
