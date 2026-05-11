# Day 3：检索链路 — 混合检索 + 重排序

> 纯手机端阅读，不需对照任何文件。这是面试最核心的一天，每个技术点都可能被追问。

---

## 一、先搞清楚问题：纯向量检索有什么短板

### 1.1 一个具体例子

知识库里有两句话：

```
A: "Python 3.12 新增了 f-string 内嵌表达式功能和更友好的错误提示。"
B: "Python 2.7 的 f-string 支持有限，建议升级到 3.6 以上版本。"
```

用户问："Python 3.12 的 f-string 有什么改进？"

**纯向量检索：** 把问题向量化，找相近的句子。因为 A 和 B 在语义上非常像（都在讲 f-string、Python 版本），向量检索可能把两者都排进 Top-5，甚至 B 排得比 A 更高——但这显然不对。用户关心的是 3.12，不是 2.7。

**问题根因：** 向量检索看的是"整体语义"，不是"精确关键词"。它知道两句话都关于 Python 和 f-string，但它不知道"3.12"才是关键区别。

### 1.2 向量 vs 关键词 — 各自强项

| | 向量检索 | BM25 关键词检索 |
|------|---------|--------------|
| 擅长 | 语义相近的词（"汽车"↔"轿车"） | 精确匹配的字符串（"3.12"） |
| 不擅长 | 专有名词、编号、版本号 | 同义词、改写、模糊查询 |
| 例子 | 搜"好吃的"→ 找到"美食"、"佳肴" | 搜"3.12"→ 找到所有含"3.12"的文档 |
| 例子盲区 | 搜"Python3.12"→ 可能返回 Python2.7 | 搜"汽车"→ 找不到"轿车" |

**结论：两条管道各有所长，各有所短。把它们的结果融合起来，取长补短。**

---

## 二、BM25 原理（不需要背公式，理解直觉）

### 2.1 BM25 在算什么

BM25 做三件事：

**① IDF：这个词有多稀有？**

"Python"这个词几乎每篇文档都有 → IDF 低 → 权重低（因为它区分不了文档）。

"Guido"这个词只在少数文档出现 → IDF 高 → 权重高（因为它能精确缩小范围）。

**② TF：这个词在这篇文档里出现了几次？**

但不是简单计数——出现 1 次和出现 2 次差别很大，但出现 10 次和出现 11 次差别就很小了。BM25 用 S 形曲线"压制"高频词的贡献。

**③ 文档长度归一化：越长不代表越相关**

一篇 10000 字的文档提到 3 次"Python"，和一篇 100 字的文档提到 3 次"Python"——后者显然更相关。BM25 用平均文档长度做归一化。

### 2.2 BM25 公式（看一遍就行，不用背）

```
score = IDF × (TF × (k1+1)) / (TF + k1×(1-b+b×|doc|/avg_len))

IDF: 这个词的区分度（越稀有越高）
TF:  词频（出现次数）
k1:  词频饱和度参数（默认 1.5）
b:   文档长度影响系数（默认 0.75）
```

---

## 三、RRF 融合 — 怎么把两份排名合成一份

### 3.1 最简单的想法（但不好）

向量检索排名第 1 的给 1 分，第 2 的给 0.9 分……BM25 排名第 1 的也给 1 分……然后把两个分数加起来排序。

**问题：** 两个检索器的打分机制完全不同。向量给 0.85 分和 BM25 给 0.85 分代表的含义完全不同。直接加权重尺度不对等。

### 3.2 RRF 的做法

RRF 不关心绝对分数，它只关心排名——你在第几。

```
RRF_score(文档) = Σ 1/(k + rank_i)
```

其中 k 是平滑常数，通常取 60。

**一个例子：**

文章 X 在向量检索中排第 1 名，在 BM25 中排第 3 名。

```
RRF_score(X) = 1/(60+1) + 1/(60+3) = 1/61 + 1/63 = 0.0164 + 0.0159 = 0.0323
```

文章 Y 在向量检索中排第 2 名，在 BM25 中排第 1 名。

```
RRF_score(Y) = 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.0161 + 0.0164 = 0.0325
```

Y 的总分略高于 X → Y 排在 X 前面。

### 3.3 k=60 的作用

```
k=0  时：第1名得分1.0，第2名0.5，第3名0.33，第10名0.1
       排名影响巨大，第1名和第2名差距是0.5

k=60 时：第1名得分0.0164，第2名0.0161，第3名0.0159
         第1名和第10名(0.0143)差距只有0.0021
         排名影响被"平滑"了，不会因为一个检索器把某文档排了第1
         另一个排了第50就完全被淹没
```

k 越大越"民主"——排名差异被压缩。60 是经验值。

**面试一句话：** "RRF 用排名代替绝对分数做融合，消除了不同检索器打分量纲不一致的问题。"

---

## 四、两阶段检索：粗排 + 精排

### 4.1 为什么不直接用 CrossEncoder 搜全部

CrossEncoder 把问题和文档拼在一起算分，精度很高。但代价是：每对 (query, doc) 都要过一次完整的 Transformer 模型。

1000 个文档 × 每个 10ms = 10 秒。RAG 问答总共要求在 2 秒内出答案，这不可接受。

**方案：先快（粗排）再准（精排）。**

### 4.2 Bi-encoder（粗排）

```
问题 "Python是谁"  → 独自编码 → [0.11, -0.02, ...]
文档 "Python由Guido" → 独自编码 → [0.12, -0.03, ...]
                             ↓
                    两个向量算距离 → 近似分数

速度：可以预计算所有文档的向量，查询时只算一个问题向量，毫秒级
精度：中等。问题和文档独立编码，缺少交叉信息
```

### 4.3 Cross-encoder（精排）

```
输入: "Python是谁 [SEP] Python由Guido于1991年创建"
  → Transformer 完整处理 → 一个精确的相似度分数

速度：每对 (query, doc) 都要重新算，秒级
精度：很高。问题和文档一起过模型，能捕捉细微的语义关系
```

### 4.4 两阶段的数字

```
1000 个文档
  → Bi-encoder 快速扫描 (0.01s)
  → 粗排 Top-20
  → Cross-encoder 对 20 个精读 (0.5s)
  → 精排 Top-5
  → 总耗时 0.51s ← 可行

如果全用 Cross-encoder：
1000 个文档 × 10ms/个 = 10s ← 不可接受
```

---

## 五、五份源码完整内嵌

### 5.1 BaseRetriever（抽象基类）

```python
from abc import ABC, abstractmethod
from typing import List
from src.models.vector import SearchResult

class BaseRetriever(ABC):
    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> List[SearchResult]:
        """所有检索器必须实现这个方法。输入问题，输出排序好的文档列表。"""
        pass
```

### 5.2 VectorRetriever（纯向量检索）

```python
from typing import List, Optional
from src.retrieval.base_retriever import BaseRetriever
from src.database.vector_db import get_vector_store
from src.embedding import get_embedding_service
from src.models.vector import SearchResult

class VectorRetriever(BaseRetriever):
    def __init__(self):
        self.vector_store = get_vector_store()        # ChromaStore 单例
        self.embedding_service = get_embedding_service()  # Embedding 单例

    def retrieve(self, query: str, top_k: int = 10,
                 kb_id: Optional[int] = None,
                 kb_ids: Optional[List[int]] = None) -> List[SearchResult]:
        # ① 问题 → 向量
        query_vector = self.embedding_service.embed_query(query)
        # "Python 谁发明的" → [0.11, -0.02, 0.43, ...] (512 个浮点数)

        if not query_vector:
            return []

        # ② 构建过滤条件（按知识库筛选）
        expr = None
        if kb_ids:
            expr = f"kb_id in {kb_ids}"    # 多知识库："kb_id in [1, 2, 3]"
        elif kb_id is not None:
            expr = f"kb_id == {kb_id}"      # 单知识库："kb_id == 1"

        # ③ 向量检索
        results = self.vector_store.search(query_vector, top_k=top_k, expr=expr)
        # ChromaDB 返回距离最近的 top_k 个文档

        return results
```

### 5.3 HybridRetriever（混合检索 — 核心中的核心）

```python
"""混合检索：BM25 关键词 + 向量语义 + RRF 融合。"""

import math
from collections import defaultdict
from typing import List, Optional

from src.retrieval.vector_retriever import VectorRetriever
from src.models.vector import SearchResult
from src.settings import settings
from src.utils.logger import logger

# ═══════════════════════════════════════
# BM25 关键词检索器
# ═══════════════════════════════════════

class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1          # 词频饱和度（默认1.5）
        self.b = b            # 文档长度归一化（默认0.75）
        self.documents = []   # 所有文档原文
        self.doc_len = []     # 每篇文档长度
        self.avgdl = 0        # 平均文档长度
        self.idf = {}         # 每个词的 IDF
        self._term_doc_freqs = []  # 每篇文档中每个词的频率

    def index_documents(self, documents: List[str], metadatas: List[dict]):
        """构建 BM25 索引。"""
        self.documents = documents
        self.doc_len = [len(d) for d in documents]
        self.avgdl = sum(self.doc_len) / max(len(self.doc_len), 1)

        total_docs = len(documents)
        self._term_doc_freqs = []

        for doc_text in documents:
            terms = self._tokenize(doc_text)      # 分词
            term_freqs = defaultdict(int)
            seen = set()
            for t in terms:
                term_freqs[t] += 1                # 本篇文章中这个词出现了几次
                if t not in seen:
                    self.df[t] = self.df.get(t, 0) + 1
                    seen.add(t)
            self._term_doc_freqs.append(term_freqs)

        # 计算 IDF：每出现一篇文档就积累一次，最后统一算
        for term, freq in self.df.items():
            self.idf[term] = math.log(
                1 + (total_docs - freq + 0.5) / (freq + 0.5)
            )

    def _tokenize(self, text: str) -> List[str]:
        """中英混合分词。英文按空格拆，中文按字拆。"""
        import re
        tokens = []
        for chunk in re.split(r'[^a-zA-Z0-9一-鿿]+', text.lower()):
            if not chunk:
                continue
            if re.match(r'^[一-鿿]+$', chunk):
                tokens.extend(list(chunk))          # 中文：逐字
            else:
                tokens.append(chunk)                 # 英文：单词
        return tokens

    def search(self, query: str, top_k: int = 20) -> List[SearchResult]:
        """BM25 关键词搜索。"""
        if not self.documents:
            return []

        query_terms = self._tokenize(query)
        scores = []

        for i, doc_text in enumerate(self.documents):
            score = 0.0
            for term in query_terms:
                if term not in self.idf:
                    continue
                tf = self._term_doc_freqs[i].get(term, 0)
                if tf == 0:
                    continue
                # BM25 公式
                num = tf * (self.k1 + 1)
                den = tf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                score += self.idf[term] * num / den
            if score > 0:
                scores.append((i, score))

        scores.sort(key=lambda x: x[1], reverse=True)   # 分数高的排前面
        results = []
        for idx, score in scores[:top_k]:
            results.append(SearchResult(
                id=f"bm25_{idx}",
                score=round(score, 4),
                text=self.documents[idx],
                metadata={}
            ))
        return results

# ═══════════════════════════════════════
# 混合检索器（BM25 + 向量 + RRF）
# ═══════════════════════════════════════

class HybridRetriever:
    RRF_K = 60   # RRF 平滑常数

    def __init__(self):
        self.vector_retriever = VectorRetriever()   # 向量检索
        self.bm25_retriever = BM25Retriever()       # 关键词检索

    def index_chunks(self, documents: List[str], metadatas: List[dict]):
        """给 BM25 喂文档数据。向量检索用 Chroma，不需要额外的索引步骤。"""
        self.bm25_retriever.index_documents(documents, metadatas)

    def retrieve(self, query: str, top_k: int = 10,
                 kb_ids: Optional[List[int]] = None,
                 alpha: float = 0.5) -> List[SearchResult]:
        """
        混合检索主方法。

        alpha=0.5：向量和 BM25 各占一半权重
        alpha=0：纯 BM25（只看关键词）
        alpha=1：纯向量（只看语义）
        """
        fetch_k = top_k * 3  # 多取一些给 RRF 融合选

        # ── 管道 1：向量检索 ──
        vector_results = self.vector_retriever.retrieve(
            query, top_k=fetch_k, kb_ids=kb_ids
        )

        # ── 管道 2：BM25 检索 ──
        bm25_results = self.bm25_retriever.search(query, top_k=fetch_k)

        # 其中一个为空就直接返回另一个
        if not bm25_results:
            return vector_results[:top_k]
        if not vector_results:
            return bm25_results[:top_k]

        # ── RRF 融合 ──
        rrf_scores = {}

        # 向量结果：按排名贡献分数
        for rank, result in enumerate(vector_results):
            key = result.text[:200]   # 取前200字符做去重 key
            rrf_scores[key] = {
                "score": 1.0 / (self.RRF_K + rank + 1) * alpha,
                "result": result,
            }

        # BM25 结果：同样按排名贡献分数
        for rank, result in enumerate(bm25_results):
            key = result.text[:200]
            bm25_rrf = 1.0 / (self.RRF_K + rank + 1) * (1 - alpha)
            if key in rrf_scores:
                rrf_scores[key]["score"] += bm25_rrf  # 同时出现→加分
            else:
                rrf_scores[key] = {
                    "score": bm25_rrf,
                    "result": result,
                }

        # 按 RRF 总分降序排列
        fused = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)

        results = []
        for item in fused[:top_k]:
            item["result"].score = round(item["score"], 4)
            results.append(item["result"])

        return results
```

**每一个关键决策：**

1. **为什么要 `fetch_k = top_k * 3`？** 两个检索器各取了 15-30 个结果，RRF 融合时才有足够的候选去重排。只取 5 个的话没法挑。
2. **为什么用 `text[:200]` 做 key？** 两个管道返回的文档可能内容相同但 metadata 不同。用文本片段做去重 key。
3. **alpha 默认 0.5 什么意思？** 向量和 BM25 各一半权重。如果用户问的是精确数字（"3.12 更新了什么"），可以提高 BM25 权重。

### 5.4 Reranker（重排序 — 精排）

```python
"""重排序器：三种模式，策略模式实现。"""

from typing import List
from src.settings import settings
from src.utils.logger import logger
from src.models.vector import SearchResult

class BaseReranker:
    def rerank(self, query: str, documents: List[SearchResult]) -> List[SearchResult]:
        raise NotImplementedError

# ── 模式 1：不重排，直接截断（最快）──
class NoReranker(BaseReranker):
    def rerank(self, query: str, docs: List[SearchResult]) -> List[SearchResult]:
        return docs[:settings.RERANK_TOP_N]

# ── 模式 2：本地 CrossEncoder（免费，推荐）──
class LocalCrossEncoderReranker(BaseReranker):
    _FALLBACK_MODELS = [
        "cross-encoder/ms-marco-MiniLM-L-6-v2",    # 最常用的轻量模型
        "BAAI/bge-reranker-base",                   # BGE 系列中文重排序
        "BAAI/bge-reranker-v2-m3",                  # 更新版
    ]

    def __init__(self):
        self.model = None
        self._load()

    def _load(self):
        import os
        from sentence_transformers import CrossEncoder

        # 优先用 HF 国内镜像
        if settings.HF_ENDPOINT:
            os.environ.setdefault("HF_ENDPOINT", settings.HF_ENDPOINT)

        # 按候选列表依次尝试，全部失败则 reranker 降级为空操作
        candidates = [settings.RERANK_CROSS_ENCODER_MODEL] + self._FALLBACK_MODELS
        for name in candidates:
            try:
                self.model = CrossEncoder(name)
                logger.info(f"CrossEncoder loaded: {name}")
                return
            except Exception as e:
                logger.warning(f"CrossEncoder '{name}' failed: {e}")

    def rerank(self, query: str, docs: List[SearchResult]) -> List[SearchResult]:
        if not docs or not self.model:
            return docs[:settings.RERANK_TOP_N]

        # CrossEncoder：把 问题+文档 拼在一起打分
        pairs = [(query, doc.text) for doc in docs]
        scores = self.model.predict(pairs)
        # 输入: [("Python是谁", "Python由Guido创建"), ("Python是谁", "Java是...")]
        # 输出: [9.5, 2.1]

        for doc, score in zip(docs, scores):
            doc.score = round(float(score), 4)

        docs.sort(key=lambda d: d.score, reverse=True)   # 高分排前
        return docs[:settings.RERANK_TOP_N]              # 取前 N

# ── 模式 3：DashScope 云端重排序（企业级）──
class DashScopeReranker(BaseReranker):
    def rerank(self, query: str, docs: List[SearchResult]) -> List[SearchResult]:
        from dashscope import TextReRank
        from http import HTTPStatus

        doc_texts = [doc.text for doc in docs]
        resp = TextReRank.call(
            model=settings.RERANK_MODEL,     # gte-rerank
            query=query,
            documents=doc_texts,
            top_n=settings.RERANK_TOP_N,
            api_key=settings.DASHSCOPE_API_KEY,
        )
        if resp.status_code == HTTPStatus.OK:
            reranked = []
            for item in resp.output.results:
                doc = docs[item.index]
                doc.score = item.relevance_score
                reranked.append(doc)
            return reranked
        return docs[:settings.RERANK_TOP_N]

# ── 工厂函数 ──
def get_reranker() -> BaseReranker:
    if not settings.ENABLE_RERANK or settings.RERANK_PROVIDER == "none":
        return NoReranker()
    if settings.RERANK_PROVIDER == "local":
        return LocalCrossEncoderReranker()
    if settings.RERANK_PROVIDER == "dashscope":
        return DashScopeReranker()
    return NoReranker()
```

**设计亮点：**

- **策略模式** — `NoReranker`、`LocalCrossEncoderReranker`、`DashScopeReranker` 都有相同的 `rerank()` 接口，切换不需要改调用方代码
- **自动降级** — 三个模型依次尝试，全部失败也不崩溃，用空操作代替
- **Fallback 模型列表** — 即使 CrossEncoder 没缓存，也会尝试 BGE 系列做备选

---

## 六、一条完整的检索链路（从问题到 Top-5）

```
用户问 "Python是谁发明的？"
  │
  ├─→ VectorRetriever.retrieve(query, top_k=15)
  │     ├─ embed_query("Python是谁发明的") → [0.11, -0.02, ...]
  │     └─ ChromaStore.search(vector, top_k=15, kb_ids=[1])
  │         返回：[A(0.88), B(0.85), C(0.80), D(0.75), E(0.70), ...]
  │
  ├─→ BM25Retriever.search(query, top_k=15)
  │     ├─ 分词：["python", "是", "谁", "发", "明", "的"]
  │     ├─ 对每篇文档算 BM25 分数（IDF × TF × 长度归一化）
  │     返回：[C(23.5), A(21.2), F(18.7), G(15.3), E(14.1), ...]
  │
  ├─→ RRF 融合
  │     A: 1/61 + 1/62 = 0.0164+0.0161 = 0.0325
  │     C: 1/63 + 1/61 = 0.0159+0.0164 = 0.0323
  │     ...
  │     融合排名：[A, C, B, E, F, D, G, ...]
  │
  ├─→ CrossEncoderReranker.rerank()
  │     对前 20 个逐个精排打分：
  │     (Python是谁, A) → 9.5   (Python是谁, C) → 8.2
  │     (Python是谁, B) → 7.1   (Python是谁, E) → 6.8
  │     ...
  │     精排后 Top-5：[A, C, B, E, F]
  │
  └─→ 返回 Top-5 SearchResult → 送入 LLM 生成回答
```

---

## 七、动手练习

### 练习 1：对比三种检索结果（20 分钟）

写一个脚本，建一份包含专有名词（产品型号、版本号）的文档，分别用纯向量、纯 BM25、混合检索搜同一个问题。对比三份结果。

```python
# 提纲（在项目根目录下运行）
import sys; sys.path.insert(0, '.')
from src.retrieval.vector_retriever import VectorRetriever
from src.retrieval.hybrid_retriever import BM25Retriever, HybridRetriever

# 准备数据
docs = [
    "Samsung S24 Ultra 配备了骁龙8Gen3处理器和200MP摄像头。",
    "Samsung S23 Ultra 配备骁龙8Gen2，拍照效果优秀。",
    "iPhone 15 Pro Max 使用A17 Pro芯片，钛金属框架。",
    "华为 P70 Pro 支持卫星通信，搭载麒麟9010。",
]

# 索引
bm25 = BM25Retriever()
bm25.index_documents(docs, [{}]*len(docs))

vec_retriever = VectorRetriever()
hybrid = HybridRetriever()
hybrid.index_chunks(docs, [{}]*len(docs))

# 查询
query = "S24 Ultra 的处理器是什么？"

print("=== 纯 BM25 ===")
for r in bm25.search(query, top_k=3):
    print(f"  [{r.score:.4f}] {r.text[:80]}")

print("=== 纯向量 ===")
for r in vec_retriever.retrieve(query, top_k=3):
    print(f"  [{r.score:.4f}] {r.text[:80]}")

print("=== 混合检索 ===")
for r in hybrid.retrieve(query, top_k=3):
    print(f"  [{r.score:.4f}] {r.text[:80]}")
```

**预期结果：** BM25 能精确定位"S24"，向量可能把 S23 也排进来（语义相近），混合后综合排名最优。

### 练习 2：开关重排序看差异（10 分钟）

运行 `python demo.py`，然后改 `.env` 把 `ENABLE_RERANK=false`，再跑一次。对比同一个问题的检索结果排名是否变化。

```env
# 开重排序（更准但更慢）
ENABLE_RERANK=true
RERANK_PROVIDER=local

# 关重排序（更快但粗排直接截断）
ENABLE_RERANK=false
```

### 练习 3：调整 alpha 权重（10 分钟）

在 `HybridRetriever.retrieve()` 里 alpha 控制 BM25 和向量的权重：

```python
# alpha=0.7 → 向量权重 70%，BM25 30%（偏语义）
results = hybrid.retrieve(query, top_k=5, alpha=0.7)

# alpha=0.3 → BM25 权重 70%，向量 30%（偏关键词）
results = hybrid.retrieve(query, top_k=5, alpha=0.3)
```

写一个小脚本，对同一个查询用 alpha=0、0.5、1 分别检索，对比三份结果有什么不同。

---

## 八、面试速记

### Q1：混合检索是什么？为什么比纯向量好？

**一句话：** BM25 做关键词精确匹配，向量做语义模糊匹配，RRF 融合两份排名。搜"Python 3.12"时 BM25 保证找到含"3.12"的文档，向量补上语义相关的上下文。

### Q2：RRF 为什么用 1/(k+rank) 而不是简单加权？

**一句话：** 两个检索器的绝对分数不在同一尺度（向量给 0.85，BM25 给 23.5），直接相加没有意义。RRF 只关心排名，消除了量纲差异。k=60 平滑第 1 名和第 10 名的差距。

### Q3：Bi-encoder vs Cross-encoder 的区别？为什么分两步？

**一句话：** Bi-encoder 分开编码（快，可以从几千选 20），Cross-encoder 联合编码（准，从 20 选 5）。全用 Cross-encoder 太慢（1000 文档 × 10ms = 10s），两阶段兼顾速度与精度。

### Q4：杂交检索中 alpha 参数的作用？

**一句话：** 控制向量和 BM25 的权重分配。alpha=0.5 等权，alpha=1 纯向量，alpha=0 纯 BM25。精确关键词查询可以降 alpha 给 BM25 更多权重。

### Q5：Reranker 的三个模式怎么选？

**一句话：** none=最快（直接截断），local=免费且准（CrossEncoder），dashscope=企业级最准。开发和面试用 local 就行。

---

## 九、验收清单

- [ ] 能举例说明纯向量检索的短板（具体例子）
- [ ] 能用自己的话解释 BM25 在算什么（不需要背公式）
- [ ] 能解释 RRF 为什么用排名而不是分数
- [ ] 能讲清楚 Bi-encoder（粗排）vs Cross-encoder（精排）的区别
- [ ] 能说出为什么分两步（速度 vs 精度权衡）
- [ ] 能画出混合检索 → RRF → CrossEncoder 的完整流程图
- [ ] 练习 1 跑过并对比了三份检索结果
- [ ] 练习 2 开关重排序后看到结果变化
- [ ] 5 道面试速记题全部能用自己的话讲 1 分钟
- [ ] 能一句话回答"你的检索方案是什么"
