"""Local embedding using sentence-transformers (free, no API key needed).

Supports multiple models with auto-fallback:
  1. BAAI/bge-small-zh-v1.5  — Chinese-optimized, 512-dim (recommended)
  2. all-MiniLM-L6-v2         — English-optimized, 384-dim (fallback)

Interview talking point:
    "我用 BGE 模型做中文 Embedding，对比过 all-MiniLM，
     BGE 在中文语义相似度上提升明显，因为它的训练数据包含大量中文语料。"
"""

from typing import List

from src.embedding.base import BaseEmbedding
from src.settings import settings
from src.utils.logger import logger


class SentenceTransformersEmbedding(BaseEmbedding):
    def __init__(self):
        self.model = None
        self.model_name = None
        self._load()

    def _load(self):
        import os
        from sentence_transformers import SentenceTransformer

        if settings.HF_ENDPOINT:
            os.environ.setdefault("HF_ENDPOINT", settings.HF_ENDPOINT)

        # Try models in order: Chinese → English fallback
        candidates = [
            settings.SENTENCE_TRANSFORMER_MODEL,
            settings.SENTENCE_TRANSFORMER_MODEL_EN,
        ]

        for name in candidates:
            try:
                logger.info(f"Loading embedding model: {name}")
                self.model = SentenceTransformer(name)
                self.model_name = name
                logger.info(f"Embedding model loaded: {name}")
                return
            except Exception as e:
                logger.warning(f"Failed to load '{name}': {e}")

        raise RuntimeError("No embedding model could be loaded")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
