"""Local embedding using sentence-transformers (free, no API key needed)."""

from typing import List

from src.embedding.base import BaseEmbedding
from src.settings import settings
from src.utils.logger import logger


class SentenceTransformersEmbedding(BaseEmbedding):
    def __init__(self):
        self.model_name = settings.SENTENCE_TRANSFORMER_MODEL
        self.model = None
        self._load()

    def _load(self):
        try:
            import os
            from sentence_transformers import SentenceTransformer
            from src.settings import settings as s
            if s.HF_ENDPOINT:
                os.environ.setdefault("HF_ENDPOINT", s.HF_ENDPOINT)
            logger.info(f"Loading sentence-transformers: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            logger.info("sentence-transformers loaded")
        except Exception as e:
            logger.error(f"Failed to load sentence-transformers: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
