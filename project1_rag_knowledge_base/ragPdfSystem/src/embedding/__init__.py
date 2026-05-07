"""Embedding abstraction layer.

Supports two providers selected via settings.EMBEDDING_PROVIDER:
  - "sentence-transformers" → local, free, all-MiniLM-L6-v2 (default)
  - "dashscope"             → cloud, requires DASHSCOPE_API_KEY
"""

from src.embedding.base import BaseEmbedding
from src.settings import settings
from src.utils.logger import logger

_embedding_service = None


def get_embedding_service() -> BaseEmbedding:
    global _embedding_service
    if _embedding_service is not None:
        return _embedding_service

    provider = settings.EMBEDDING_PROVIDER

    if provider == "sentence-transformers":
        from src.embedding.sentence_embedding import SentenceTransformersEmbedding
        _embedding_service = SentenceTransformersEmbedding()
    elif provider == "dashscope":
        from src.embedding.dashscope_embedding import DashScopeEmbeddingService
        _embedding_service = DashScopeEmbeddingService()
    else:
        raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider}")

    return _embedding_service
