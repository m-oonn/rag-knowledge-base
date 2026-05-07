"""Memory system with dual backend support.

Short-term memory:
  - Redis (enterprise) or in-memory dict (lightweight)
Long-term memory:
  - Milvus (enterprise) or Chroma (lightweight)
"""

import json
from typing import List, Dict, Any, Optional

from src.settings import settings
from src.utils.logger import logger


class MemorySystem:
    def __init__(self):
        self.history_limit = settings.MEMORY_HISTORY_LIMIT
        self.ttl = settings.SHORT_TERM_MEMORY_TTL

        # Short-term: try Redis, fallback to in-memory dict
        self._redis = None
        try:
            import redis
            self._redis = redis.from_url(settings.REDIS_URL)
            self._redis.ping()
            logger.info("Memory: using Redis for short-term storage")
        except Exception:
            logger.info("Memory: Redis unavailable, using in-memory dict")
            self._local_store: Dict[str, list] = {}

        # Long-term: use the shared vector store factory
        try:
            from src.database.vector_db import get_vector_store
            self._vector_store = get_vector_store()
        except Exception:
            self._vector_store = None

    # ── Short-term memory ────────────────────────────────────

    def get_short_term_memory(self, session_id: str, limit: int = 10) -> List[Dict[str, str]]:
        if limit <= 0:
            return []

        if self._redis:
            try:
                key = f"session:{session_id}:history"
                data = self._redis.lrange(key, 0, limit - 1)
                messages = [json.loads(d) for d in data]
                messages.reverse()
                return messages
            except Exception as e:
                logger.error(f"Redis get memory error: {e}")

        # Local fallback
        messages = self._local_store.get(session_id, [])
        return list(reversed(messages[-limit:]))

    def add_short_term_memory(self, session_id: str, role: str, content: str):
        message = {"role": role, "content": content}

        if self._redis:
            try:
                key = f"session:{session_id}:history"
                self._redis.lpush(key, json.dumps(message))
                self._redis.ltrim(key, 0, self.history_limit - 1)
                self._redis.expire(key, self.ttl)
                return
            except Exception:
                pass

        # Local fallback
        if session_id not in self._local_store:
            self._local_store[session_id] = []
        self._local_store[session_id].append(message)
        # Trim
        if len(self._local_store[session_id]) > self.history_limit:
            self._local_store[session_id] = self._local_store[session_id][-self.history_limit:]

    def clear_short_term_memory(self, session_id: str):
        if self._redis:
            try:
                self._redis.delete(f"session:{session_id}:history")
                return
            except Exception:
                pass
        self._local_store.pop(session_id, None)

    # ── Long-term memory ─────────────────────────────────────

    def retrieve_long_term_memory(self, query_vector: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        if self._vector_store:
            try:
                results = self._vector_store.search(query_vector, top_k=top_k)
                return [{"id": r.id, "text": r.text, "score": r.score} for r in results]
            except Exception as e:
                logger.error(f"Long-term memory retrieval error: {e}")
        return []
