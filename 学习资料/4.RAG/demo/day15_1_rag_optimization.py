"""
Day 15 Demo 1: RAG 优化实战
运行方式: python day15_1_rag_optimization.py

前置条件:
  pip install numpy scikit-learn
  (全部用标准库 + numpy/sklearn，不需要额外安装)

学习目标:
1. 手写 BM25 算法，理解关键词检索原理
2. 对比纯向量检索 vs 纯 BM25 vs 混合检索的效果
3. 实现 RRF（倒数排名融合）合并多路检索结果
4. 模拟重排序（Reranking）过程
5. 体验查询改写对检索结果的改善
6. 用测试集评估检索准确率（Recall@K, Precision@K, MRR）
"""

import math
import re
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# 准备知识库文档（模拟 RAG 的文档库）
# ============================================================

# 模拟一个小型知识库，每个文档有 id、内容和标题
DOCUMENTS = [
    {
        "id": "doc_01",
        "title": "FastAPI 安装指南",
        "content": "FastAPI 的安装非常简单。使用 pip install fastapi uvicorn 即可完成安装。"
                   "uvicorn 是 ASGI 服务器，用于运行 FastAPI 应用。"
                   "建议在虚拟环境中安装，避免依赖冲突。"
    },
    {
        "id": "doc_02",
        "title": "FastAPI 路由定义",
        "content": "FastAPI 使用装饰器定义路由。@app.get 处理 GET 请求，"
                   "@app.post 处理 POST 请求。路径参数用花括号定义，"
                   "查询参数通过函数参数自动识别。支持异步路由函数 async def。"
    },
    {
        "id": "doc_03",
        "title": "Pydantic 数据验证",
        "content": "Pydantic 是 FastAPI 的数据验证核心。定义继承 BaseModel 的类，"
                   "FastAPI 自动验证请求体的数据类型和格式。"
                   "验证失败返回 422 Unprocessable Entity 错误和详细信息。"
    },
    {
        "id": "doc_04",
        "title": "Chroma 向量数据库",
        "content": "Chroma 是嵌入式向量数据库，Python 原生支持。"
                   "使用 PersistentClient 可以持久化存储数据到磁盘。"
                   "persist_directory 参数指定存储路径。"
                   "支持 add、query、update、delete 四种基本操作。"
    },
    {
        "id": "doc_05",
        "title": "RAG 检索增强生成",
        "content": "RAG 全称 Retrieval-Augmented Generation，检索增强生成。"
                   "核心思路是先从知识库检索相关文档，再把文档和问题一起交给 LLM 生成回答。"
                   "相比纯 LLM，RAG 能减少幻觉并提供最新信息。"
    },
    {
        "id": "doc_06",
        "title": "文本分割策略",
        "content": "RecursiveCharacterTextSplitter 是 LangChain 推荐的分割器。"
                   "它按多级分隔符递归分割：先按段落，再按句子，最后按字符。"
                   "chunk_size 通常设 500-1000，chunk_overlap 设 50-200。"
    },
    {
        "id": "doc_07",
        "title": "Embedding 向量化",
        "content": "Embedding 把文本转成固定维度的向量。语义相近的文本，向量距离也近。"
                   "常用模型：sentence-transformers（本地免费）、OpenAI text-embedding-ada-002。"
                   "中文推荐 BAAI/bge-base-zh-v1.5 模型。"
    },
    {
        "id": "doc_08",
        "title": "Python 异步编程",
        "content": "Python 的 async/await 是异步编程的核心语法。"
                   "asyncio 模块提供事件循环和异步原语。"
                   "异步适合 IO 密集型场景：网络请求、数据库查询。"
                   "FastAPI 原生支持异步路由。"
    },
    {
        "id": "doc_09",
        "title": "LangChain 框架概览",
        "content": "LangChain 是构建 LLM 应用的框架。核心组件包括："
                   "Chain（链式调用）、Agent（自主决策）、Memory（对话记忆）。"
                   "它提供了统一的接口对接各种 LLM 和向量数据库。"
    },
    {
        "id": "doc_10",
        "title": "API 错误处理",
        "content": "FastAPI 中常见的 HTTP 错误码：400 Bad Request 请求格式错误，"
                   "401 Unauthorized 未认证，403 Forbidden 无权限，"
                   "404 Not Found 资源不存在，422 Unprocessable Entity 数据验证失败，"
                   "500 Internal Server Error 服务器内部错误。"
    },
]

print("=" * 60)
print("RAG 优化实战 - 知识库已加载")
print("=" * 60)
print(f"\n  知识库文档数: {len(DOCUMENTS)}")
for doc in DOCUMENTS:
    print(f"    [{doc['id']}] {doc['title']}")


# ============================================================
# === Part 1 === 手写 BM25 检索
# ============================================================

print("\n" + "=" * 60)
print("Part 1: 手写 BM25 检索")
print("=" * 60)


def tokenize_chinese(text: str) -> list:
    """
    简单的中文分词（按字符+英文单词分割）。
    实际项目中应使用 jieba 等专业分词库。
    这里为了零依赖，用正则简单处理。
    """
    # 提取中文字符（每个字作为一个token）和英文单词
    tokens = re.findall(r"[\u4e00-\u9fff]|[a-zA-Z0-9_]+", text.lower())
    return tokens


class SimpleBM25:
    """
    手写 BM25 实现。
    不依赖任何外部库，帮助理解 BM25 算法的核心原理。

    BM25 公式:
      score(q, d) = SUM[ IDF(qi) * TF(qi, d) ]

      IDF(qi) = log((N - n(qi) + 0.5) / (n(qi) + 0.5) + 1)
      TF(qi, d) = (f(qi,d) * (k1+1)) / (f(qi,d) + k1*(1-b+b*|d|/avgdl))
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75):
        """
        参数:
            k1: 控制词频饱和度，通常 1.2-2.0
            b:  控制文档长度归一化，0=不考虑长度, 1=完全归一化
        """
        self.k1 = k1
        self.b = b
        self.doc_count = 0       # 文档总数 N
        self.avg_doc_len = 0     # 平均文档长度 avgdl
        self.doc_lengths = []    # 每篇文档的长度
        self.doc_freqs = {}      # 每个词出现在多少篇文档中 n(qi)
        self.term_freqs = []     # 每篇文档中每个词的频率 f(qi, d)
        self.doc_ids = []        # 文档 ID 列表

    def fit(self, documents: list):
        """
        建立索引。类似于向量检索中的"存入向量库"。
        """
        self.doc_count = len(documents)
        self.doc_ids = [doc["id"] for doc in documents]

        total_len = 0
        for doc in documents:
            # 分词
            tokens = tokenize_chinese(doc["content"])
            self.doc_lengths.append(len(tokens))
            total_len += len(tokens)

            # 统计词频
            tf = Counter(tokens)
            self.term_freqs.append(tf)

            # 统计文档频率（每个词出现在多少文档中）
            for token in set(tokens):
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

        self.avg_doc_len = total_len / self.doc_count if self.doc_count > 0 else 0
        print(f"  BM25 索引构建完成: {self.doc_count} 篇文档, "
              f"词汇量 {len(self.doc_freqs)}, 平均长度 {self.avg_doc_len:.1f} tokens")

    def _idf(self, term: str) -> float:
        """计算逆文档频率 IDF"""
        n = self.doc_freqs.get(term, 0)  # 包含该词的文档数
        # 加 1 防止 log(0)，这是 BM25 的标准变体
        return math.log((self.doc_count - n + 0.5) / (n + 0.5) + 1)

    def _tf_score(self, term: str, doc_idx: int) -> float:
        """计算调整后的词频分数 TF"""
        f = self.term_freqs[doc_idx].get(term, 0)  # 原始词频
        doc_len = self.doc_lengths[doc_idx]
        # BM25 的 TF 公式：词频有饱和效应，不是线性增长
        numerator = f * (self.k1 + 1)
        denominator = f + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
        return numerator / denominator if denominator > 0 else 0

    def search(self, query: str, top_k: int = 5) -> list:
        """
        检索：计算查询和每篇文档的 BM25 分数，返回 Top-K。
        """
        query_tokens = tokenize_chinese(query)
        scores = []

        for doc_idx in range(self.doc_count):
            score = 0.0
            for token in query_tokens:
                idf = self._idf(token)
                tf = self._tf_score(token, doc_idx)
                score += idf * tf
            scores.append((doc_idx, score))

        # 按分数降序排序
        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for doc_idx, score in scores[:top_k]:
            results.append({
                "doc_id": self.doc_ids[doc_idx],
                "score": score,
                "rank": len(results) + 1,
            })
        return results


# 构建 BM25 索引
print()
bm25 = SimpleBM25(k1=1.5, b=0.75)
bm25.fit(DOCUMENTS)

# 测试 BM25 检索
test_query = "FastAPI 怎么安装"
print(f"\n  查询: '{test_query}'")
bm25_results = bm25.search(test_query, top_k=5)
print(f"  BM25 检索结果:")
for r in bm25_results:
    doc = next(d for d in DOCUMENTS if d["id"] == r["doc_id"])
    print(f"    #{r['rank']} [{r['doc_id']}] score={r['score']:.4f}  {doc['title']}")


# ============================================================
# === Part 2 === 向量检索 vs BM25 vs 混合检索对比
# ============================================================

print("\n" + "=" * 60)
print("Part 2: 三种检索方式对比")
print("=" * 60)

# --- 2.1 构建向量检索 ---
print("\n--- 2.1 构建向量检索索引 (TF-IDF 模拟 Embedding) ---")

# 用 TF-IDF 向量模拟真实的 Embedding 向量
# 实际项目中这里用 sentence-transformers 生成真正的语义向量
vectorizer = TfidfVectorizer()
doc_texts = [doc["content"] for doc in DOCUMENTS]
doc_vectors = vectorizer.fit_transform(doc_texts)
print(f"  向量维度: {doc_vectors.shape[1]}")


def vector_search(query: str, top_k: int = 5) -> list:
    """向量检索（TF-IDF 模拟）"""
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, doc_vectors).flatten()
    top_indices = similarities.argsort()[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_indices, 1):
        results.append({
            "doc_id": DOCUMENTS[idx]["id"],
            "score": float(similarities[idx]),
            "rank": rank,
        })
    return results


# --- 2.2 三种方式对比 ---
print("\n--- 2.2 对比实验 ---")

test_queries = [
    "FastAPI 怎么安装",
    "persist_directory 参数",     # 精确关键词
    "怎么让AI回答更准确",         # 语义化表达
    "422 错误怎么解决",
]

for query in test_queries:
    print(f"\n  查询: '{query}'")

    # 向量检索
    vec_results = vector_search(query, top_k=3)
    # BM25 检索
    bm25_results = bm25.search(query, top_k=3)

    print(f"  {'向量检索 Top-3':30s} | {'BM25 检索 Top-3':30s}")
    print(f"  {'-'*30} | {'-'*30}")
    for i in range(3):
        # 向量结果
        if i < len(vec_results):
            vr = vec_results[i]
            v_doc = next(d for d in DOCUMENTS if d["id"] == vr["doc_id"])
            v_str = f"{vr['score']:.3f} {v_doc['title'][:15]}"
        else:
            v_str = "-"

        # BM25 结果
        if i < len(bm25_results):
            br = bm25_results[i]
            b_doc = next(d for d in DOCUMENTS if d["id"] == br["doc_id"])
            b_str = f"{br['score']:.3f} {b_doc['title'][:15]}"
        else:
            b_str = "-"

        print(f"  {i+1}. {v_str:28s} | {i+1}. {b_str:28s}")

print("""
  观察:
  - "persist_directory 参数" → BM25 能精确匹配关键词，向量检索可能漏掉
  - "怎么让AI回答更准确" → 向量检索理解语义，BM25 靠关键词可能匹配不上
  - 两种方式互补，所以混合检索效果更好!
""")


# ============================================================
# === Part 3 === RRF 倒数排名融合
# ============================================================

print("=" * 60)
print("Part 3: RRF 倒数排名融合（合并两路检索结果）")
print("=" * 60)


def reciprocal_rank_fusion(result_lists: list, k: int = 60) -> list:
    """
    倒数排名融合（Reciprocal Rank Fusion）。

    将多个检索器的结果融合为一个排名。
    核心公式: RRF_score(d) = SUM[ 1 / (k + rank_i(d)) ]

    参数:
        result_lists: 多个检索结果列表，每个列表的元素必须有 doc_id 和 rank
        k: 常数，通常取 60，防止排名第1的权重过大

    返回:
        按 RRF 分数排序的融合结果
    """
    rrf_scores = {}  # doc_id -> RRF 分数

    for results in result_lists:
        for item in results:
            doc_id = item["doc_id"]
            rank = item["rank"]
            # 累加每个检索器对该文档的 RRF 贡献
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + 1.0 / (k + rank)

    # 按 RRF 分数降序排序
    sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

    return [
        {"doc_id": doc_id, "rrf_score": score, "rank": rank + 1}
        for rank, (doc_id, score) in enumerate(sorted_results)
    ]


# 对每个测试查询演示 RRF 融合
print()
query = "persist_directory 参数怎么配置"
print(f"  查询: '{query}'\n")

vec_results = vector_search(query, top_k=5)
bm25_results = bm25.search(query, top_k=5)

print(f"  向量检索排名:")
for r in vec_results[:5]:
    doc = next(d for d in DOCUMENTS if d["id"] == r["doc_id"])
    print(f"    #{r['rank']} {doc['title']} (score={r['score']:.4f})")

print(f"\n  BM25 检索排名:")
for r in bm25_results[:5]:
    doc = next(d for d in DOCUMENTS if d["id"] == r["doc_id"])
    print(f"    #{r['rank']} {doc['title']} (score={r['score']:.4f})")

# RRF 融合
rrf_results = reciprocal_rank_fusion([vec_results, bm25_results], k=60)

print(f"\n  RRF 融合排名:")
for r in rrf_results[:5]:
    doc = next(d for d in DOCUMENTS if d["id"] == r["doc_id"])
    # 找到该文档在两个检索器中的排名
    v_rank = next((x["rank"] for x in vec_results if x["doc_id"] == r["doc_id"]), "-")
    b_rank = next((x["rank"] for x in bm25_results if x["doc_id"] == r["doc_id"]), "-")
    print(f"    #{r['rank']} {doc['title']} "
          f"(rrf={r['rrf_score']:.6f}, vec_rank={v_rank}, bm25_rank={b_rank})")

print("""
  RRF 的优势:
  - 不需要归一化不同检索器的分数（分数量纲不同）
  - 两个检索器都认为好的文档，排名自然靠前
  - k=60 是经验值，让排名之间的差距更平滑
""")


# ============================================================
# === Part 4 === 模拟重排序（Reranking）
# ============================================================

print("=" * 60)
print("Part 4: 模拟重排序（Reranking）")
print("=" * 60)

print("""
  重排序的思路:
  1. 粗排: 快速从大量文档中召回 Top-20（用向量检索或混合检索）
  2. 精排: 对 Top-20 用更精确的模型重新打分（Cross-encoder）

  这里我们用一个简单的评分函数模拟 Cross-encoder:
  - 关键词完全匹配加分
  - 标题包含查询词加分
  - 内容覆盖度加分
""")


def simulate_reranker(query: str, candidates: list) -> list:
    """
    模拟 Cross-encoder 重排序。

    真实项目中这里用 CrossEncoder 模型（如 bge-reranker-base）。
    这里用规则模拟，帮助理解重排序做了什么。
    """
    query_tokens = set(tokenize_chinese(query))
    scored = []

    for candidate in candidates:
        doc = next(d for d in DOCUMENTS if d["id"] == candidate["doc_id"])
        content_tokens = set(tokenize_chinese(doc["content"]))
        title_tokens = set(tokenize_chinese(doc["title"]))

        # 评分规则（模拟 Cross-encoder 的细粒度理解）
        score = 0.0

        # 1. 查询词在内容中的覆盖率
        overlap = query_tokens & content_tokens
        coverage = len(overlap) / len(query_tokens) if query_tokens else 0
        score += coverage * 0.4

        # 2. 查询词在标题中的命中
        title_overlap = query_tokens & title_tokens
        title_score = len(title_overlap) / len(query_tokens) if query_tokens else 0
        score += title_score * 0.3

        # 3. 内容长度适中加分（太短信息不够，太长噪声多）
        content_len = len(doc["content"])
        if 80 <= content_len <= 300:
            score += 0.2
        elif content_len > 300:
            score += 0.1

        # 4. 精确短语匹配加分（模拟 Cross-encoder 对短语的理解）
        for token in query_tokens:
            if len(token) > 2 and token in doc["content"].lower():
                score += 0.1

        scored.append({
            "doc_id": candidate["doc_id"],
            "original_rank": candidate["rank"],
            "rerank_score": round(score, 4),
        })

    # 按重排序分数排序
    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    for i, item in enumerate(scored):
        item["new_rank"] = i + 1

    return scored


# 演示重排序
query = "FastAPI 422 数据验证错误"
print(f"\n  查询: '{query}'")

# 粗排（混合检索）
vec_r = vector_search(query, top_k=5)
bm25_r = bm25.search(query, top_k=5)
rrf_r = reciprocal_rank_fusion([vec_r, bm25_r], k=60)

print(f"\n  粗排结果（RRF 混合检索 Top-5）:")
for r in rrf_r[:5]:
    doc = next(d for d in DOCUMENTS if d["id"] == r["doc_id"])
    print(f"    #{r['rank']} {doc['title']}")

# 精排（重排序）
reranked = simulate_reranker(query, rrf_r[:5])
print(f"\n  精排结果（重排序后）:")
for r in reranked:
    doc = next(d for d in DOCUMENTS if d["id"] == r["doc_id"])
    change = r["original_rank"] - r["new_rank"]
    arrow = "^" * change if change > 0 else "v" * (-change) if change < 0 else "="
    print(f"    #{r['new_rank']} {doc['title']} "
          f"(score={r['rerank_score']:.4f}, was #{r['original_rank']} {arrow})")

print("""
  观察:
  - 重排序可能改变文档的顺序
  - 和查询更精确匹配的文档排名会上升
  - 在真实项目中，Cross-encoder 的排序变化会更显著
""")


# ============================================================
# === Part 5 === 查询改写（Query Rewriting）
# ============================================================

print("=" * 60)
print("Part 5: 查询改写")
print("=" * 60)

print("""
  查询改写的目标: 把用户的口语化/模糊查询转化为更适合检索的查询

  在真实项目中，查询改写通常由 LLM 完成。
  这里我们用规则模拟，展示改写前后的检索效果差异。
""")

# 简单的改写规则（实际项目用 LLM）
REWRITE_RULES = {
    # 缩写 -> 全称
    "api": "API 接口",
    "db": "数据库 database",
    "llm": "大语言模型 LLM",
    "rag": "RAG 检索增强生成",
    # 口语 -> 正式
    "咋": "怎么",
    "啥": "什么",
    "搞": "实现",
    "弄": "配置",
}

# 同义词扩展
SYNONYMS = {
    "安装": "安装 install 部署 setup",
    "错误": "错误 error 报错 异常",
    "配置": "配置 设置 参数 config",
    "数据库": "数据库 database 存储 storage",
}


def rewrite_query(query: str) -> str:
    """
    查询改写: 缩写扩展 + 口语转正式 + 同义词扩展。

    在真实项目中，这个函数会调用 LLM:
      prompt = "请将用户问题改写为适合检索的查询: {query}"
    """
    rewritten = query.lower()

    # 1. 缩写和口语替换
    for old, new in REWRITE_RULES.items():
        rewritten = rewritten.replace(old, new)

    # 2. 同义词扩展（把同义词追加到查询后面）
    expanded_terms = []
    for term, synonyms in SYNONYMS.items():
        if term in rewritten:
            expanded_terms.append(synonyms)

    if expanded_terms:
        rewritten = rewritten + " " + " ".join(expanded_terms)

    return rewritten


# 演示查询改写效果
rewrite_examples = [
    "fastapi 咋装",
    "rag 是啥",
    "db 配置",
    "422 报错",
]

print()
for original in rewrite_examples:
    rewritten = rewrite_query(original)
    print(f"  原始查询: '{original}'")
    print(f"  改写后:   '{rewritten}'")

    # 对比改写前后的检索结果
    results_before = vector_search(original, top_k=3)
    results_after = vector_search(rewritten, top_k=3)

    before_top = next(d for d in DOCUMENTS if d["id"] == results_before[0]["doc_id"])
    after_top = next(d for d in DOCUMENTS if d["id"] == results_after[0]["doc_id"])

    print(f"  改写前 Top-1: {before_top['title']} (score={results_before[0]['score']:.4f})")
    print(f"  改写后 Top-1: {after_top['title']} (score={results_after[0]['score']:.4f})")
    score_change = results_after[0]["score"] - results_before[0]["score"]
    if score_change > 0.01:
        print(f"  效果: [OK] 分数提升 +{score_change:.4f}")
    elif score_change < -0.01:
        print(f"  效果: 分数下降 {score_change:.4f}")
    else:
        print(f"  效果: 分数基本不变")
    print()


# ============================================================
# === Part 6 === 检索评估：Recall@K, Precision@K, MRR
# ============================================================

print("=" * 60)
print("Part 6: 检索效果评估")
print("=" * 60)

print("""
  评估方法: 准备一组测试问题，每个问题标注"正确答案在哪些文档中"，
  然后对比检索结果和标注答案，计算量化指标。
""")

# 测试集（手动标注：每个问题的相关文档 ID）
TEST_SET = [
    {
        "query": "怎么安装FastAPI",
        "relevant_docs": ["doc_01"],  # 正确答案在 doc_01
    },
    {
        "query": "Pydantic 数据验证",
        "relevant_docs": ["doc_03"],
    },
    {
        "query": "Chroma persist_directory 持久化",
        "relevant_docs": ["doc_04"],
    },
    {
        "query": "RAG 是什么 检索增强",
        "relevant_docs": ["doc_05"],
    },
    {
        "query": "chunk_size 文本分割",
        "relevant_docs": ["doc_06"],
    },
    {
        "query": "Embedding 向量化模型",
        "relevant_docs": ["doc_07"],
    },
    {
        "query": "async await 异步",
        "relevant_docs": ["doc_08"],
    },
    {
        "query": "422 错误处理",
        "relevant_docs": ["doc_03", "doc_10"],  # 两篇都相关
    },
]


def evaluate_retrieval(search_fn, test_set: list, top_k: int = 5) -> dict:
    """
    评估检索效果。

    指标:
      Recall@K: 检索到的相关文档数 / 所有相关文档数
      Precision@K: 检索到的相关文档数 / K
      MRR: 1 / 第一个相关文档的排名（Mean Reciprocal Rank）

    参数:
        search_fn: 检索函数，接受 (query, top_k) 返回结果列表
        test_set: 测试集
        top_k: 检索返回的文档数
    """
    total_recall = 0
    total_precision = 0
    total_rr = 0  # Reciprocal Rank 累加
    query_count = len(test_set)

    details = []
    for test in test_set:
        query = test["query"]
        relevant = set(test["relevant_docs"])

        # 执行检索
        results = search_fn(query, top_k)
        retrieved_ids = [r["doc_id"] for r in results]

        # 计算命中
        hits = relevant & set(retrieved_ids)

        # Recall@K
        recall = len(hits) / len(relevant) if relevant else 0
        total_recall += recall

        # Precision@K
        precision = len(hits) / top_k
        total_precision += precision

        # MRR: 第一个相关文档的排名
        rr = 0
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in relevant:
                rr = 1.0 / (i + 1)
                break
        total_rr += rr

        details.append({
            "query": query,
            "recall": recall,
            "precision": precision,
            "rr": rr,
            "hits": len(hits),
            "total_relevant": len(relevant),
        })

    metrics = {
        "recall@k": total_recall / query_count,
        "precision@k": total_precision / query_count,
        "mrr": total_rr / query_count,
        "details": details,
    }
    return metrics


# --- 评估三种检索方式 ---

# 1. 纯向量检索
print("\n--- 评估: 纯向量检索 ---")
vec_metrics = evaluate_retrieval(vector_search, TEST_SET, top_k=3)

# 2. 纯 BM25
print("--- 评估: 纯 BM25 ---")
bm25_metrics = evaluate_retrieval(bm25.search, TEST_SET, top_k=3)


# 3. 混合检索（RRF 融合）
def hybrid_search(query: str, top_k: int = 5) -> list:
    """混合检索: 向量 + BM25 + RRF 融合"""
    vec_r = vector_search(query, top_k=top_k)
    bm25_r = bm25.search(query, top_k=top_k)
    rrf_r = reciprocal_rank_fusion([vec_r, bm25_r], k=60)
    return rrf_r[:top_k]


print("--- 评估: 混合检索 (向量 + BM25 + RRF) ---")
hybrid_metrics = evaluate_retrieval(hybrid_search, TEST_SET, top_k=3)

# 汇总对比
print(f"""
  ╔══════════════════╦══════════╦══════════════╦════════╗
  ║ 检索方式         ║ Recall@3 ║ Precision@3  ║  MRR   ║
  ╠══════════════════╬══════════╬══════════════╬════════╣
  ║ 纯向量检索       ║  {vec_metrics['recall@k']:.4f}  ║    {vec_metrics['precision@k']:.4f}    ║ {vec_metrics['mrr']:.4f} ║
  ║ 纯 BM25          ║  {bm25_metrics['recall@k']:.4f}  ║    {bm25_metrics['precision@k']:.4f}    ║ {bm25_metrics['mrr']:.4f} ║
  ║ 混合检索(RRF)    ║  {hybrid_metrics['recall@k']:.4f}  ║    {hybrid_metrics['precision@k']:.4f}    ║ {hybrid_metrics['mrr']:.4f} ║
  ╚══════════════════╩══════════╩══════════════╩════════╝

  指标解读:
  - Recall@3:    在 Top-3 结果中找到了多少比例的相关文档（越高越好）
  - Precision@3: Top-3 结果中有多少比例是相关的（越高越好）
  - MRR:         第一个正确结果平均排在第几位（越高=越靠前）
""")

# 逐条详情
print("  逐条详情（混合检索）:")
for d in hybrid_metrics["details"]:
    status = "[OK]" if d["recall"] == 1.0 else "[MISS]"
    print(f"    {status} '{d['query'][:20]:20s}' "
          f"recall={d['recall']:.2f} hits={d['hits']}/{d['total_relevant']}")


# ============================================================
# 总结
# ============================================================

print("\n" + "=" * 60)
print("Day 15 Summary")
print("=" * 60)
print(f"""
  今天学到了:

  1. BM25 关键词检索
     - 基于 TF(词频) * IDF(逆文档频率) 的经典算法
     - 擅长精确关键词匹配，和向量检索互补

  2. 混合检索 + RRF 融合
     - 向量检索 + BM25 两路召回
     - RRF 公式: score = SUM[1/(60+rank)]
     - 不需要归一化分数，简单有效

  3. 重排序 (Reranking)
     - 粗排(Bi-encoder) -> 精排(Cross-encoder)
     - 两阶段架构: 先快后精

  4. 查询改写
     - 缩写扩展、口语转正式、同义词扩展
     - 实际项目中用 LLM 做改写效果更好

  5. 检索评估
     - Recall@K: 找全了多少
     - Precision@K: 找对了多少
     - MRR: 第一个正确结果排多前

  明天: Day 16 - 把所有技术整合成完整的 Mini RAG 系统!
""")
