"""
Day 10 Demo 1：RAG 核心概念实战
运行方式：python day10_1_rag_concepts.py

前置条件：
  pip install numpy scikit-learn  (sklearn 用于计算余弦相似度)

学习目标：
1. 亲手体验 Embedding（文本→向量）的过程
2. 理解余弦相似度怎么计算
3. 用最原始的方式实现一个 mini 检索系统
4. 理解 RAG 的完整流程（不依赖任何框架）
"""

import numpy as np

# ============================================================
# 第一部分：理解向量和相似度
# ============================================================

print("=" * 55)
print("Part 1: Vectors and Similarity")
print("=" * 55)

# 假设我们有一个极简的 Embedding 模型，只用 3 个维度表示文本：
# 维度0 = 编程相关程度
# 维度1 = AI相关程度
# 维度2 = 日常生活程度

# 手工构造几个"假"向量来理解原理
texts_and_vectors = {
    "Python编程入门":   np.array([0.9, 0.3, 0.1]),
    "FastAPI开发指南":  np.array([0.85, 0.2, 0.05]),
    "深度学习原理":     np.array([0.5, 0.95, 0.05]),
    "RAG技术详解":      np.array([0.6, 0.9, 0.1]),
    "今天吃什么":       np.array([0.05, 0.02, 0.98]),
    "天气预报":         np.array([0.02, 0.01, 0.95]),
}

print("\n  Texts and their vectors (3D simplified):\n")
for text, vec in texts_and_vectors.items():
    print(f"    {text:15s} -> {vec}")


# 余弦相似度计算
def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    余弦相似度 = A·B / (|A| × |B|)

    这就是向量数据库内部做的核心计算。
    值域：-1 到 1，越接近 1 越相似。
    """
    dot_product = np.dot(a, b)           # 点积
    norm_a = np.linalg.norm(a)           # A的模长
    norm_b = np.linalg.norm(b)           # B的模长
    return dot_product / (norm_a * norm_b)


# 计算查询和所有文档的相似度
query_text = "怎么学Python？"
query_vec = np.array([0.88, 0.25, 0.15])  # 假设查询的向量

print(f"\n  Query: '{query_text}' -> {query_vec}")
print(f"\n  Similarity scores:\n")

scores = []
for text, vec in texts_and_vectors.items():
    sim = cosine_similarity(query_vec, vec)
    scores.append((text, sim))
    bar = "#" * int(sim * 30)
    print(f"    {sim:.4f} {bar:30s} {text}")

# 排序找 Top-3
scores.sort(key=lambda x: x[1], reverse=True)
print(f"\n  Top-3 most relevant:")
for i, (text, sim) in enumerate(scores[:3], 1):
    print(f"    {i}. {text} (score: {sim:.4f})")

print()
print("  --> This is what vector DB does: find most similar documents!")
print()


# ============================================================
# 第二部分：文本分割（Chunking）演示
# ============================================================

print("=" * 55)
print("Part 2: Text Chunking")
print("=" * 55)

sample_doc = """# FastAPI 介绍

FastAPI 是一个现代、快速的 Web 框架，用于构建 Python API。它基于标准的 Python 类型提示，使用 Pydantic 进行数据验证。

## 主要特性

FastAPI 支持异步编程（async/await），性能媲美 NodeJS 和 Go 框架。它自动生成交互式 API 文档（Swagger UI），大大减少了文档维护的工作量。

## 安装方法

使用 pip 安装 FastAPI 和 uvicorn：pip install fastapi uvicorn。创建一个 main.py 文件，定义路由和处理函数。

## 依赖注入

FastAPI 的依赖注入系统通过 Depends 实现。你可以定义依赖函数来管理数据库连接、用户认证等通用逻辑，然后在路由中声明使用。这避免了重复代码，提高了可维护性。

## 数据验证

Pydantic 模型是 FastAPI 的核心。定义一个继承 BaseModel 的类，FastAPI 就会自动验证请求数据的类型和格式。验证失败时自动返回 422 错误和详细的错误信息。"""


def chunk_by_fixed_size(text: str, chunk_size: int = 150, overlap: int = 30) -> list[str]:
    """
    固定大小分割。

    chunk_size: 每段的最大字符数
    overlap: 相邻段重叠的字符数（防止切断上下文）
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - overlap  # 下一段从重叠位置开始
    return chunks


def chunk_by_paragraph(text: str) -> list[str]:
    """
    按段落分割（以空行分隔）。
    保持段落完整性，不会切断句子。
    """
    paragraphs = text.split("\n\n")
    return [p.strip() for p in paragraphs if p.strip()]


def chunk_by_heading(text: str) -> list[str]:
    """
    按标题分割（Markdown ## 标题）。
    每个章节作为一个 chunk，语义最完整。
    """
    chunks = []
    current = ""
    for line in text.split("\n"):
        if line.startswith("## ") and current.strip():
            chunks.append(current.strip())
            current = line + "\n"
        else:
            current += line + "\n"
    if current.strip():
        chunks.append(current.strip())
    return chunks


# 对比三种分割方式
print(f"\n  Original document: {len(sample_doc)} chars")

print(f"\n  --- Method 1: Fixed size (150 chars, 30 overlap) ---")
chunks1 = chunk_by_fixed_size(sample_doc, 150, 30)
for i, c in enumerate(chunks1):
    print(f"    Chunk {i+1} ({len(c)} chars): {c[:60]}...")

print(f"\n  --- Method 2: By paragraph ---")
chunks2 = chunk_by_paragraph(sample_doc)
for i, c in enumerate(chunks2):
    print(f"    Chunk {i+1} ({len(c)} chars): {c[:60]}...")

print(f"\n  --- Method 3: By heading (recommended for Markdown) ---")
chunks3 = chunk_by_heading(sample_doc)
for i, c in enumerate(chunks3):
    print(f"    Chunk {i+1} ({len(c)} chars): {c[:60]}...")

print()
print("  --> Fixed size: simple but may break sentences")
print("  --> By paragraph: preserves structure")
print("  --> By heading: best for Markdown (used in LangChain)")
print()


# ============================================================
# 第三部分：手写 Mini RAG（不用任何框架）
# ============================================================

print("=" * 55)
print("Part 3: Mini RAG from Scratch")
print("=" * 55)

# 这里用一个超简单的方法模拟 Embedding：TF-IDF
# 真实项目用 sentence-transformers 或 OpenAI Embedding API
# 但原理是一样的：文本 → 向量 → 相似度检索

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine


class MiniRAG:
    """
    最简 RAG 实现 —— 不依赖 LangChain、Chroma 等任何框架。

    目的：让你彻底理解 RAG 每一步在做什么。

    流程：
    1. 文档 → 分割成 chunks
    2. chunks → TF-IDF 向量化（模拟 Embedding）
    3. 用户问题 → 向量化 → 找最相似的 chunks
    4. 相似 chunks + 问题 → 组装 Prompt
    """

    def __init__(self):
        self.chunks: list[str] = []
        self.vectorizer = TfidfVectorizer()
        self.vectors = None

    def add_document(self, text: str):
        """添加文档：分割 + 向量化"""
        # 按段落分割
        new_chunks = [p.strip() for p in text.split("\n\n") if p.strip() and len(p.strip()) > 20]
        self.chunks.extend(new_chunks)

        # 重新向量化所有 chunks
        self.vectors = self.vectorizer.fit_transform(self.chunks)

        print(f"  Added document: {len(new_chunks)} chunks, total {len(self.chunks)} chunks")

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """检索最相关的 chunks"""
        # 把查询也向量化
        query_vec = self.vectorizer.transform([query])

        # 计算和所有 chunks 的相似度
        similarities = sklearn_cosine(query_vec, self.vectors).flatten()

        # 取 Top-K
        top_indices = similarities.argsort()[::-1][:top_k]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # 过滤掉完全不相关的
                results.append({
                    "content": self.chunks[idx],
                    "score": float(similarities[idx]),
                })
        return results

    def build_prompt(self, query: str, docs: list[dict]) -> str:
        """组装 RAG Prompt"""
        context = "\n\n".join([
            f"[Document {i+1}, relevance: {d['score']:.2f}]\n{d['content']}"
            for i, d in enumerate(docs)
        ])

        return f"""Based on the following documents, answer the question.
If the documents don't contain relevant info, say "Not found".
Answer in Chinese.

---Documents---
{context}
---End---

Question: {query}
Answer:"""


# 使用 Mini RAG
print()
rag = MiniRAG()

# 添加文档
rag.add_document(sample_doc)

# 添加更多文档
extra_doc = """# Python 异步编程

async/await 是 Python 的异步编程语法。异步函数用 async def 定义，在等待 IO 操作时用 await 让出控制权。

asyncio.gather 可以同时执行多个异步任务。asyncio.create_task 创建后台任务。

异步编程适合 IO 密集型场景：网络请求、数据库查询、文件读写。不适合 CPU 密集型计算。

FastAPI 原生支持异步，用 async def 定义路由函数可以获得最佳性能。"""

rag.add_document(extra_doc)

# 测试检索
test_queries = [
    "FastAPI怎么做数据验证？",
    "什么是异步编程？",
    "怎么安装FastAPI？",
    "如何做机器学习？",  # 文档里没有的内容
]

print()
for query in test_queries:
    print(f"  Q: {query}")
    results = rag.search(query, top_k=2)

    if results:
        for i, r in enumerate(results):
            print(f"     [{i+1}] score={r['score']:.3f}: {r['content'][:80]}...")
    else:
        print(f"     No relevant documents found")

    # 组装 Prompt（打印出来看看）
    if query == test_queries[0]:
        prompt = rag.build_prompt(query, results)
        print(f"\n  === Generated RAG Prompt (for first query) ===")
        print(f"  {prompt[:300]}...")
        print(f"  === End Prompt ===")

    print()


# ============================================================
# 第四部分：可视化向量空间（如果有 matplotlib）
# ============================================================

print("=" * 55)
print("Part 4: Vector Space Visualization")
print("=" * 55)

try:
    import matplotlib
    matplotlib.use("Agg")  # 不弹窗，直接保存
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    # 用 PCA 把高维 TF-IDF 向量降到 2D
    if rag.vectors is not None and rag.vectors.shape[0] >= 3:
        pca = PCA(n_components=2)
        coords_2d = pca.fit_transform(rag.vectors.toarray())

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))

        for i, (x, y) in enumerate(coords_2d):
            label = rag.chunks[i][:20] + "..."
            ax.scatter(x, y, s=100, zorder=5)
            ax.annotate(label, (x, y), fontsize=8, ha="center", va="bottom")

        ax.set_title("Document Chunks in Vector Space (PCA 2D)")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        ax.grid(True, alpha=0.3)

        output_path = "rag_vector_space.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=100)
        plt.close()
        print(f"\n  Chart saved to: {output_path}")
        print("  --> Similar documents are close together in the vector space!")
    else:
        print("\n  Not enough data for visualization")

except ImportError:
    print("\n  [SKIP] matplotlib not installed, skipping visualization")
    print("  Run: pip install matplotlib")

print()


# ============================================================
# 总结
# ============================================================

print("=" * 55)
print("Day 10 Summary")
print("=" * 55)
print("""
What you learned today:

1. RAG = Retrieval + Augmented Generation
   - Offline: doc -> split -> embed -> store in vector DB
   - Online:  query -> embed -> search -> prompt -> LLM answer

2. Embedding = text -> vector (numbers)
   - Similar texts have similar vectors

3. Cosine Similarity = measure how close two vectors are
   - 1.0 = identical, 0.0 = unrelated

4. Chunking = split documents into smaller pieces
   - Fixed size / by paragraph / by heading
   - chunk_size=500-1000, overlap=50-200

5. Mini RAG from scratch (no frameworks!)
   - TF-IDF as simple embedding
   - Cosine similarity for retrieval
   - Prompt assembly with retrieved docs

Tomorrow (Day 11): Chroma vector database
  - Replace our MiniRAG with a real vector DB
  - Persistent storage, proper CRUD operations
""")
