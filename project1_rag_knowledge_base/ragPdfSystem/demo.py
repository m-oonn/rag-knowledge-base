#!/usr/bin/env python3
"""One-click RAG demo — no server needed. Shows the full pipeline end-to-end.

Run: python demo.py
"""

import os
import sys
import time
import hashlib
from pathlib import Path
from io import BytesIO

project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))


def banner(text: str):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")


def step(text: str):
    print(f"\n  [{text}]")


def ok(text: str):
    print(f"    ✅ {text}")


# ================================================================
# Step 1: Setup
# ================================================================
banner("Step 1: Initialize Components")

from src.settings import settings
step("Configuration")
ok(f"LLM:       {settings.LLM_PROVIDER}")
ok(f"Vector DB: {settings.VECTOR_STORE}")
ok(f"Embedding: {settings.EMBEDDING_PROVIDER}")
ok(f"Rerank:    {settings.RERANK_PROVIDER}")
ok(f"Hybrid:    {settings.ENABLE_HYBRID_SEARCH}")

step("Loading embedding model...")
from src.embedding import get_embedding_service
embed = get_embedding_service()
ok("Embedding model loaded")

step("Initializing ChromaDB...")
from src.database.vector_db import get_vector_store
store = get_vector_store()
ok("ChromaDB ready")

step("Loading LLM client...")
from src.llm.llm_client import LLMClient
llm = LLMClient()
ok(f"LLM ready: {settings.LLM_PROVIDER}")

# ================================================================
# Step 2: Create sample documents
# ================================================================
banner("Step 2: Create Sample Documents")

sample_docs = {
    "python_intro.md": """# Python Programming Language

Python is a high-level, interpreted programming language created by Guido van Rossum.
First released in 1991, Python emphasizes code readability with its notable use of
significant whitespace.

Python is dynamically typed and garbage-collected. It supports multiple programming
paradigms including procedural, object-oriented, and functional programming.

Python 3.12 introduced new features including better error messages, f-string improvements,
and a new type parameter syntax for generics.
""",
    "machine_learning.md": """# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems to learn
and improve from experience without being explicitly programmed.

Key types of machine learning:
- Supervised learning: Training on labeled data
- Unsupervised learning: Finding patterns in unlabeled data
- Reinforcement learning: Learning through trial and error

Popular frameworks include PyTorch, TensorFlow, and scikit-learn.
Deep learning uses neural networks with many layers for complex pattern recognition.
""",
    "fastapi_guide.md": """# FastAPI Framework

FastAPI is a modern, fast (high-performance) web framework for building APIs with Python.
It is based on standard Python type hints.

Key features:
- Automatic API documentation (Swagger UI and ReDoc)
- Data validation via Pydantic
- Async support with asyncio
- Dependency injection system
- Built-in OAuth2 and JWT authentication

FastAPI is one of the fastest Python frameworks, on par with Node.js and Go.
It's used by companies like Netflix, Uber, and Microsoft.
""",
}

docs_dir = settings.UPLOAD_DIR
docs_dir.mkdir(parents=True, exist_ok=True)

from src.processors.text_chunker import TextChunker
from src.models.document import Document, DocumentMetadata
from src.models.vector import VectorRecord
from src.database.models import KnowledgeDocument
import uuid

chunker = TextChunker(chunk_size=settings.CHUNK_SIZE, chunk_overlap=settings.CHUNK_OVERLAP)
total_chunks = 0

for filename, content in sample_docs.items():
    # Save file
    filepath = docs_dir / filename
    filepath.write_text(content, encoding="utf-8")
    ok(f"Created: {filename} ({len(content)} chars)")

    # Parse → Chunk → Embed → Store
    doc = Document(
        filename=filename,
        content=content,
        metadata=DocumentMetadata(source=filename),
    )
    chunks = chunker.split_document(doc)
    texts = [c.text for c in chunks]
    embeddings = embed.embed_documents(texts)

    records = []
    for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
        vec_id = str(uuid.uuid4())
        records.append(VectorRecord(
            id=vec_id,
            values=emb,
            metadata={
                "text": chunk.text,
                "source": filename,
                "chunk_index": i,
                "kb_id": 1,
            },
        ))

    store.insert(records)
    total_chunks += len(chunks)
    ok(f"  → {len(chunks)} chunks indexed")

ok(f"Total: {len(sample_docs)} documents → {total_chunks} chunks")

# ================================================================
# Step 3: Test queries
# ================================================================
banner("Step 3: RAG Queries")

test_queries = [
    "Who created Python and when?",
    "What is machine learning?",
    "What are FastAPI's key features?",
    "What version of Python introduced f-string improvements?",
]

# Initialize reranker
from src.retrieval.reranker import get_reranker
reranker = get_reranker()

for query in test_queries:
    step(f"Query: {query}")

    # 1. Embed query
    query_vec = embed.embed_query(query)

    # 2. Vector search
    t0 = time.time()
    results = store.search(query_vec, top_k=5)
    search_time = time.time() - t0

    # 3. Rerank
    if results and settings.ENABLE_RERANK:
        results = reranker.rerank(query, results)

    # 4. Build prompt and generate
    if results:
        context = "\n\n".join(f"[{i+1}] {r.text[:300]}" for i, r in enumerate(results))
        prompt = f"""基于以下上下文回答问题。如无相关信息请说明。

上下文：
{context}

问题：{query}

回答："""
        t1 = time.time()
        answer = llm.generate_custom_response(prompt)
        gen_time = time.time() - t1

        print(f"    📎 Retrieved {len(results)} docs in {search_time:.2f}s")
        for r in results[:3]:
            print(f"       [{r.score:.4f}] {r.text[:80]}...")
        print(f"    🤖 Answer ({gen_time:.2f}s): {answer[:200]}")
    else:
        print("    ❌ No results found")

# ================================================================
# Step 4: Summary
# ================================================================
banner("Demo Complete")

print(f"""
  Pipeline tested:
    1. Document creation → {len(sample_docs)} files
    2. Text chunking → {total_chunks} chunks
    3. Embedding → {settings.EMBEDDING_PROVIDER}
    4. Vector storage → {settings.VECTOR_STORE} (chroma)
    5. Retrieval → Top-5 search
    6. Reranking → {settings.RERANK_PROVIDER}
    7. LLM generation → {settings.LLM_PROVIDER}

  Start the API server:  uvicon src.main:app --reload
  API docs:              http://localhost:8000/docs
  Streamlit UI:          streamlit run streamlit_app.py
""")
