"""
Day 12 Demo：文本 Embedding
运行方式：python day12_1_embedding.py
前置条件：pip install sentence-transformers chromadb numpy

学习目标：
1. 用 sentence-transformers 生成文本向量
2. 计算文本间的语义相似度
3. 将 Embedding 集成到 Chroma
"""

import numpy as np
import time

# ============================================================
# Part 1：加载 Embedding 模型
# ============================================================

print("=" * 55)
print("Part 1: Load Embedding Model")
print("=" * 55)

try:
    from sentence_transformers import SentenceTransformer
    print("\n  Loading model: all-MiniLM-L6-v2 ...")
    start = time.time()
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"  [OK] Model loaded in {time.time()-start:.1f}s")
    USE_ST = True
except ImportError:
    print("\n  [WARN] sentence-transformers not installed")
    print("  Run: pip install sentence-transformers")
    print("  Using TF-IDF fallback for demo...")
    USE_ST = False

# ============================================================
# Part 2：生成向量
# ============================================================

print("\n" + "=" * 55)
print("Part 2: Generate Embeddings")
print("=" * 55)

texts = [
    "Python是一种高级编程语言",
    "Java也是一种编程语言",
    "今天天气真好",
    "FastAPI是Python的Web框架",
    "明天会下雨吗",
]

if USE_ST:
    vectors = embed_model.encode(texts)
    print(f"\n  Texts: {len(texts)}")
    print(f"  Vector shape: {vectors.shape}")
    print(f"  Vector dim: {vectors.shape[1]}")
    print(f"  First vector (first 10 values): {vectors[0][:10].round(4)}")
else:
    from sklearn.feature_extraction.text import TfidfVectorizer
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(texts).toarray()
    print(f"\n  [Fallback] Using TF-IDF, dim={vectors.shape[1]}")

# ============================================================
# Part 3：语义相似度矩阵
# ============================================================

print("\n" + "=" * 55)
print("Part 3: Similarity Matrix")
print("=" * 55)

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

print(f"\n  {'':20s}", end="")
for i in range(len(texts)):
    print(f"  T{i+1}", end="")
print()

for i in range(len(texts)):
    label = texts[i][:18] + ".." if len(texts[i]) > 18 else texts[i]
    print(f"  {label:20s}", end="")
    for j in range(len(texts)):
        sim = cosine_sim(vectors[i], vectors[j])
        print(f" {sim:.2f}", end="")
    print()

print("\n  Observations:")
print("  - T1(Python) and T2(Java) are similar (both programming)")
print("  - T1(Python) and T3(weather) are very different")
print("  - T3(weather) and T5(rain) are similar (both weather)")

# ============================================================
# Part 4：语义搜索
# ============================================================

print("\n" + "=" * 55)
print("Part 4: Semantic Search")
print("=" * 55)

queries = ["编程语言", "天气预报", "Web开发"]

for q in queries:
    if USE_ST:
        q_vec = embed_model.encode(q)
    else:
        q_vec = vectorizer.transform([q]).toarray()[0]

    scores = [cosine_sim(q_vec, v) for v in vectors]
    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)

    print(f"\n  Query: '{q}'")
    for idx, score in ranked[:3]:
        print(f"    [{score:.3f}] {texts[idx]}")

# ============================================================
# Part 5：批量 Embedding 性能
# ============================================================

print("\n" + "=" * 55)
print("Part 5: Batch Performance")
print("=" * 55)

if USE_ST:
    batch_texts = [f"这是第{i}个测试文本，用于测试批量Embedding性能。" for i in range(100)]
    start = time.time()
    batch_vectors = embed_model.encode(batch_texts, batch_size=32, show_progress_bar=False)
    elapsed = time.time() - start
    print(f"\n  Embedded {len(batch_texts)} texts in {elapsed:.2f}s")
    print(f"  Speed: {len(batch_texts)/elapsed:.0f} texts/sec")
    print(f"  Output shape: {batch_vectors.shape}")
else:
    print("\n  [SKIP] Need sentence-transformers for benchmark")

# ============================================================
# Part 6：集成 Chroma
# ============================================================

print("\n" + "=" * 55)
print("Part 6: Integrate with Chroma")
print("=" * 55)

import chromadb

if USE_ST:
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
    ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
else:
    ef = None  # Chroma 会用默认的 Embedding

client = chromadb.Client()
collection = client.create_collection("embedding_demo", embedding_function=ef)

# 添加文档（Chroma 自动调用 Embedding 模型）
knowledge = [
    "Python的装饰器用@语法糖实现，本质是高阶函数",
    "FastAPI通过Depends实现依赖注入，管理数据库连接等资源",
    "asyncio.gather可以并发执行多个协程，提高IO密集型任务的效率",
    "Pydantic的BaseModel用于数据验证，支持类型注解和自定义验证规则",
    "RAG系统将检索到的文档作为上下文，让LLM基于真实数据回答问题",
    "向量数据库使用余弦相似度或欧几里得距离来衡量文本的语义相关性",
]

collection.add(
    ids=[f"k{i}" for i in range(len(knowledge))],
    documents=knowledge,
    metadatas=[{"topic": "python"} for _ in knowledge],
)

# 查询
test_queries = ["怎么做数据验证？", "什么是RAG？", "异步编程怎么用？"]
for q in test_queries:
    results = collection.query(query_texts=[q], n_results=2)
    print(f"\n  Q: {q}")
    for doc, dist in zip(results["documents"][0], results["distances"][0]):
        print(f"    [{1-dist:.3f}] {doc[:60]}...")

print("\n" + "=" * 55)
print("Day 12 Summary")
print("=" * 55)
print("""
  1. sentence-transformers: local, free embedding model
  2. model.encode(texts) -> numpy array of vectors
  3. Cosine similarity measures semantic closeness
  4. Batch encoding is much faster than one-by-one
  5. Integrate with Chroma via SentenceTransformerEmbeddingFunction
  6. Tomorrow: LangChain RAG pipeline
""")
