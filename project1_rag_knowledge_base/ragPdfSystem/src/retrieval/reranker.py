"""Reranker abstraction.

Supports three modes selected via settings.RERANK_PROVIDER:
  - "none"       → skip reranking (fastest)
  - "local"      → sentence-transformers CrossEncoder (free, local)
  - "dashscope"  → Alibaba gte-rerank (requires DASHSCOPE_API_KEY)
"""

from typing import List

from src.settings import settings
from src.utils.logger import logger
from src.models.vector import SearchResult


class BaseReranker:
    def rerank(self, query: str, documents: List[SearchResult]) -> List[SearchResult]:
        raise NotImplementedError


class NoReranker(BaseReranker):
    """Pass-through reranker that just truncates to top_n."""

    def rerank(self, query: str, documents: List[SearchResult]) -> List[SearchResult]:
        return documents[:settings.RERANK_TOP_N]


class LocalCrossEncoderReranker(BaseReranker):
    """Local reranking using sentence-transformers CrossEncoder.

    Uses HF_ENDPOINT from settings for China mirror compatibility.
    Falls back to NoReranker if model cannot be loaded.
    """

    _FALLBACK_MODELS = [
        "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "BAAI/bge-reranker-base",
        "BAAI/bge-reranker-v2-m3",
    ]

    def __init__(self):
        self.model = None
        self._load()

    def _load(self):
        import os
        from sentence_transformers import CrossEncoder

        # Set HF mirror for China users
        if settings.HF_ENDPOINT:
            os.environ.setdefault("HF_ENDPOINT", settings.HF_ENDPOINT)

        candidates = [settings.RERANK_CROSS_ENCODER_MODEL] + self._FALLBACK_MODELS
        for model_name in candidates:
            try:
                logger.info(f"Loading CrossEncoder: {model_name}")
                self.model = CrossEncoder(model_name)
                logger.info(f"CrossEncoder loaded: {model_name}")
                return
            except Exception as e:
                logger.warning(f"CrossEncoder '{model_name}' failed: {e}")

        logger.warning("All CrossEncoder models failed, reranking disabled")

    def rerank(self, query: str, documents: List[SearchResult]) -> List[SearchResult]:
        if not documents or not self.model:
            return documents[:settings.RERANK_TOP_N]

        pairs = [(query, doc.text) for doc in documents]
        scores = self.model.predict(pairs)

        for doc, score in zip(documents, scores):
            doc.score = round(float(score), 4)

        documents.sort(key=lambda d: d.score, reverse=True)
        return documents[:settings.RERANK_TOP_N]


class DashScopeReranker(BaseReranker):
    """Original DashScope reranker (kept intact)."""

    def __init__(self):
        self.api_key = settings.DASHSCOPE_API_KEY
        self.model = settings.RERANK_MODEL
        self.top_n = settings.RERANK_TOP_N

    def rerank(self, query: str, documents: List[SearchResult]) -> List[SearchResult]:
        if not documents:
            return []

        try:
            from dashscope import TextReRank
            from http import HTTPStatus

            doc_texts = [doc.text for doc in documents]
            resp = TextReRank.call(
                model=self.model,
                query=query,
                documents=doc_texts,
                top_n=self.top_n,
                api_key=self.api_key,
            )

            if resp.status_code == HTTPStatus.OK:
                reranked = []
                for item in resp.output.results:
                    idx = item.index
                    score = item.relevance_score
                    doc = documents[idx]
                    doc.score = score
                    reranked.append(doc)
                return reranked
            else:
                logger.error(f"DashScope Rerank error: {resp.code} - {resp.message}")
                return documents[:self.top_n]
        except Exception as e:
            logger.error(f"Rerank failed: {e}")
            return documents[:self.top_n]


def get_reranker() -> BaseReranker:
    if not settings.ENABLE_RERANK or settings.RERANK_PROVIDER == "none":
        return NoReranker()

    if settings.RERANK_PROVIDER == "local":
        return LocalCrossEncoderReranker()

    if settings.RERANK_PROVIDER == "dashscope":
        return DashScopeReranker()

    return NoReranker()
