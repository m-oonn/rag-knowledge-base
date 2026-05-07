"""Vector database abstraction layer.

Supports two backends selected via settings.VECTOR_STORE:
  - "chroma"  → embedded, zero-dependency (default)
  - "milvus"  → enterprise, requires Milvus + etcd + MinIO

Usage:
    from src.database.vector_db import get_vector_store
    store = get_vector_store()
    store.insert(records)
    results = store.search(query_vector, top_k=5)
"""

from typing import List, Optional, Protocol, runtime_checkable

from src.settings import settings
from src.utils.logger import logger
from src.models.vector import VectorRecord, SearchResult


# ================================================================
# Interface
# ================================================================

@runtime_checkable
class VectorStore(Protocol):
    """Protocol that every vector store backend must satisfy."""

    def insert(self, records: List[VectorRecord]) -> None: ...
    def search(
        self, vector: List[float], top_k: int = 10, expr: Optional[str] = None
    ) -> List[SearchResult]: ...


# ================================================================
# Chroma backend
# ================================================================

class ChromaStore:
    """ChromaDB vector store (embedded, no external service needed)."""

    def __init__(self):
        import chromadb
        self.collection_name = settings.CHROMA_COLLECTION_NAME
        self._path = str(settings.CHROMA_DB_DIR)
        self.client = chromadb.PersistentClient(path=self._path)
        self.collection = self._get_or_create_collection()
        logger.info(f"Chroma store ready at {self._path}")

    def _get_or_create_collection(self):
        # Try to use sentence-transformers embedding function if available
        ef = None
        if settings.EMBEDDING_PROVIDER == "sentence-transformers":
            try:
                from chromadb.utils.embedding_functions import (
                    SentenceTransformerEmbeddingFunction,
                )
                ef = SentenceTransformerEmbeddingFunction(
                    model_name=settings.SENTENCE_TRANSFORMER_MODEL
                )
            except Exception:
                pass

        if ef:
            return self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=ef,
                metadata={"hnsw:space": "cosine"},
            )
        return self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def insert(self, records: List[VectorRecord]) -> None:
        if not records:
            return
        ids = [r.id for r in records]
        documents = [r.metadata.get("text", "") for r in records]
        metadatas = [{**r.metadata, "kb_id": r.metadata.get("kb_id", 0)} for r in records]
        embeddings = [r.values for r in records]

        # Batch insert
        batch_size = 40
        for i in range(0, len(ids), batch_size):
            end = i + batch_size
            self.collection.add(
                ids=ids[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
                embeddings=embeddings[i:end],
            )
        logger.info(f"Chroma: inserted {len(records)} records")

    def search(
        self, vector: List[float], top_k: int = 10, expr: Optional[str] = None
    ) -> List[SearchResult]:
        where = None
        if expr:
            # Simple kb_id filter conversion: "kb_id in [1,2]" or "kb_id == 1"
            where = _parse_expr_to_chroma_where(expr)

        n = max(1, min(top_k, max(self.collection.count(), 1)))
        results = self.collection.query(
            query_embeddings=[vector],
            n_results=n,
            include=["documents", "metadatas", "distances"],
            where=where,
        )

        search_results: List[SearchResult] = []
        if results and results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                doc_id = results["ids"][0][i]
                text = results["documents"][0][i] if results["documents"] else ""
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0
                similarity = max(0.0, 1.0 - distance)

                search_results.append(SearchResult(
                    id=doc_id,
                    score=round(similarity, 4),
                    text=text,
                    metadata=meta,
                ))
        return search_results


def _parse_expr_to_chroma_where(expr: str) -> Optional[dict]:
    """Convert simple Milvus-style expr to Chroma where filter.

    Supports: "kb_id == 1" and "kb_id in [1,2]"
    """
    import re

    # "kb_id == 1"
    eq_match = re.match(r"kb_id\s*==\s*(\d+)", expr)
    if eq_match:
        return {"kb_id": int(eq_match.group(1))}

    # "kb_id in [1, 2, 3]"
    in_match = re.match(r"kb_id\s+in\s+\[([^\]]+)\]", expr)
    if in_match:
        ids = [int(x.strip()) for x in in_match.group(1).split(",")]
        return {"kb_id": {"$in": ids}}

    return None


# ================================================================
# Milvus backend (original, kept intact)
# ================================================================

class MilvusStore:
    """Original Milvus client (enterprise mode)."""

    def __init__(self):
        self.host = settings.MILVUS_HOST
        self.port = settings.MILVUS_PORT
        self.collection_name = settings.MILVUS_COLLECTION_NAME
        self.dim = settings.MILVUS_DIMENSION
        self.collection = None
        self.has_kb_id = False
        self._connect()
        self._init_collection()

    def _connect(self):
        from pymilvus import connections
        import time as _time
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if connections.has_connection("default"):
                    connections.disconnect("default")
                connections.connect(
                    "default", host=self.host, port=self.port, timeout=10
                )
                logger.info(f"Connected to Milvus at {self.host}:{self.port}")
                return
            except Exception as e:
                logger.warning(f"Milvus connect attempt {attempt+1}/{max_retries}: {e}")
                if attempt < max_retries - 1:
                    _time.sleep(5)
        raise RuntimeError("Could not connect to Milvus")

    def _init_collection(self):
        from pymilvus import (
            Collection, CollectionSchema, DataType, FieldSchema, utility,
        )
        if not utility.has_collection(self.collection_name):
            fields = [
                FieldSchema(name="id", dtype=DataType.VARCHAR, max_length=512, is_primary=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="metadata", dtype=DataType.JSON),
                FieldSchema(name="kb_id", dtype=DataType.INT64),
            ]
            schema = CollectionSchema(fields, "RAG Document Collection")
            self.collection = Collection(self.collection_name, schema)
            self.collection.create_index(
                field_name="embedding",
                index_params={"metric_type": "L2", "index_type": "IVF_FLAT", "params": {"nlist": 1024}},
            )
            self.collection.create_index(field_name="kb_id", index_name="kb_id_index")
        else:
            self.collection = Collection(self.collection_name)
        self.has_kb_id = any(f.name == "kb_id" for f in self.collection.schema.fields)
        self.collection.load()

    def insert(self, records: List[VectorRecord]) -> None:
        if not records:
            return
        ids = [r.id for r in records]
        embeddings = [r.values for r in records]
        texts = [r.metadata.get("text", "") for r in records]
        metadatas = [r.metadata for r in records]
        data = [ids, embeddings, texts, metadatas]
        if self.has_kb_id:
            kb_ids = [int(r.metadata.get("kb_id", 0)) for r in records]
            data.append(kb_ids)
        self.collection.insert(data)
        logger.info(f"Milvus: inserted {len(records)} records")

    def search(
        self, vector: List[float], top_k: int = 10, expr: Optional[str] = None
    ) -> List[SearchResult]:
        from pymilvus import Collection
        search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
        output_fields = ["text", "metadata"]
        if self.has_kb_id:
            output_fields.append("kb_id")
        if not self.has_kb_id and expr and "kb_id" in expr:
            expr = None

        results = self.collection.search(
            data=[vector],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=output_fields,
        )
        search_results: List[SearchResult] = []
        for hits in results:
            for hit in hits:
                search_results.append(SearchResult(
                    id=hit.id,
                    score=hit.score,
                    text=hit.entity.get("text"),
                    metadata=hit.entity.get("metadata"),
                ))
        return search_results


# ================================================================
# Factory
# ================================================================

def get_vector_store() -> VectorStore:
    if settings.VECTOR_STORE == "chroma":
        return ChromaStore()
    elif settings.VECTOR_STORE == "milvus":
        return MilvusStore()
    raise ValueError(f"Unknown VECTOR_STORE: {settings.VECTOR_STORE}")
