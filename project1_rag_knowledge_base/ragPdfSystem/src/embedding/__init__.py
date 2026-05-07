"""Embedding abstraction layer.

Supports two providers selected via settings.EMBEDDING_PROVIDER:
  - "sentence-transformers" → local, free, all-MiniLM-L6-v2 (default)
  - "dashscope"             → cloud, requires DASHSCOPE_API_KEY
"""

from src.embedding.base import BaseEmbedding
from src.settings import settings
from src.utils.logger import logger


def get_embedding_service() -> BaseEmbedding:
    provider = settings.EMBEDDING_PROVIDER

    if provider == "sentence-transformers":
        from src.embedding.sentence_embedding import SentenceTransformersEmbedding
        return SentenceTransformersEmbedding()

    if provider == "dashscope":
        from src.embedding.dashscope_embedding import DashScopeEmbeddingService
        return DashScopeEmbeddingService()

    raise ValueError(f"Unknown EMBEDDING_PROVIDER: {provider}")
