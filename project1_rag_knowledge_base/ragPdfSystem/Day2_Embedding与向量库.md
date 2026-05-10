# Day 2：Embedding 与向量数据库

> 纯手机端阅读，不需对照任何文件。读完能理解 Embedding 原理、写出 Chroma 操作代码、说出模型选型理由。

---

## 一、Embedding 是什么

### 1.1 从"查字典"讲起

你有一句话："今天天气真不错，适合出去跑步"。

计算机不认识"意思"，它只认识数字。Embedding 就是翻译官——把人类的话翻译成计算机能运算的数字。

```
"今天天气不错" → [0.12, -0.03, 0.45, 0.08, -0.21, ...]  ← 512 个浮点数
"今天天气很差" → [0.10, -0.05, 0.42, 0.09, -0.18, ...]  ← 和上面很接近
"Python 是编程语言" → [-0.35, 0.22, -0.08, 0.41, 0.15, ...]  ← 和上面差很远
```

两句话意思越近，它们对应的向量在空间中距离越近；意思越远，距离越远。

### 1.2 为什么不能直接用文字搜索

你搜"汽车"，用 Ctrl+F 去文档里找——能找到所有包含"汽车"两个字的句子。

但你搜"汽车"，文档里写的是"轿车"、"跑车"、"SUV"——Ctrl+F 一个都找不到。

Embedding 解决了这个问题：**"汽车"和"轿车"在向量空间中挨得很近。** 你搜"汽车"，系统自动把"轿车"相关的也找出来。

这就是"语义搜索"——搜的是意思，不是字符串。

### 1.3 训练原理（一句话版）

Embedding 模型通过对比学习训练：给模型一对句子，告诉它这两个意思一样还是不一样。一样就拉近它们向量，不一样就推远。训练几百万次后，模型学会了"什么样的句子意思相近"。

---

## 二、项目中用了哪些 Embedding 模型

### 2.1 模型对比

| 模型 | 维度 | 语言 | 大小 | 适用 |
|------|------|------|------|------|
| BAAI/bge-small-zh-v1.5 | 512 | 中英 | ~100MB | **中文文档首选** |
| all-MiniLM-L6-v2 | 384 | 英语 | ~80MB | 英文文档、速度快 |
| text-embedding-v1 (DashScope) | 1536 | 中英 | 云端 | 企业级，需付费 |

**维度越高 = 表达能力越强，但计算越慢。** 512 是中英双语场景的甜点。

### 2.2 为什么中文文档用 BGE 而不是 all-MiniLM

`all-MiniLM-L6-v2` 是用英英文语料训练的。它理解"Python is great"没问题，但面对"这是一段关于机器学习的中文介绍"时，它的向量质量会明显下降——就像让一个只学过英语的人去翻译中文。

`BAAI/bge-small-zh-v1.5` 是中英双语训练的，中文语义理解好得多。

**面试这么说：**"我对比过 BGE 和 all-MiniLM，BGE 在中文语义相似度任务上效果更好，因为它的训练数据包含大量中文语料。"

---

## 三、Embedding 源码完整内嵌

### 3.1 抽象基类 base.py

```python
from abc import ABC, abstractmethod
from typing import List

class BaseEmbedding(ABC):
    """所有 Embedding 实现的抽象基类。定义了必须实现的两个方法。"""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量向量化：多段文本 → 多个向量。用于离线入库。"""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """单条向量化：一个问题 → 一个向量。用于在线查询。"""
        pass
```

**为什么分成两个方法？**

`embed_documents` 一次处理多条文本，批量计算更快（GPU 并行）。
`embed_query` 只处理一条，返回单个向量。

底层调的是同一个模型，只是接口不同。

### 3.2 sentence_embedding.py（本地模型完整版）

```python
"""本地 Embedding：免费，不需要 API Key。

模型加载顺序：先尝试 BGE 中文模型，失败则回退到 all-MiniLM 英文模型。
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

        # 优先用国内 HF 镜像（settings.HF_ENDPOINT 从 .env 读取）
        if settings.HF_ENDPOINT:
            os.environ.setdefault("HF_ENDPOINT", settings.HF_ENDPOINT)

        # 候选模型列表：中文首选 → 英文备选
        candidates = [
            settings.SENTENCE_TRANSFORMER_MODEL,      # 默认 BAAI/bge-small-zh-v1.5
            settings.SENTENCE_TRANSFORMER_MODEL_EN,   # 备选 all-MiniLM-L6-v2
        ]

        for name in candidates:
            try:
                logger.info(f"Loading embedding model: {name}")
                self.model = SentenceTransformer(name)
                self.model_name = name
                logger.info(f"Embedding loaded: {name}")
                return
            except Exception as e:
                logger.warning(f"Failed to load '{name}': {e}")

        # 两个都失败则抛异常
        raise RuntimeError("No embedding model could be loaded")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量向量化，用于离线入库。"""
        if not texts:
            return []
        embeddings = self.model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()
        # normalize_embeddings=True：将所有向量归一化到单位长度。
        # 归一化后，两个向量的内积 = 余弦相似度，计算更快。

    def embed_query(self, text: str) -> List[float]:
        """单条向量化，用于在线查询。"""
        embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
```

**关键细节：**

- `normalize_embeddings=True` 把向量长度归一化为 1。此时两点内积 = 余弦相似度。
- `candidates` 列表实现自动降级：BGE 下载失败就切 all-MiniLM，不会崩溃。

### 3.3 工厂函数 + 单例 `__init__.py`

```python
"""Embedding 层入口。根据 settings.EMBEDDING_PROVIDER 返回对应实现。"""

_embedding_service = None   # 全局单例

def get_embedding_service() -> BaseEmbedding:
    global _embedding_service
    if _embedding_service is not None:      # 已创建过，直接返回
        return _embedding_service

    provider = settings.EMBEDDING_PROVIDER

    if provider == "sentence-transformers":
        from src.embedding.sentence_embedding import SentenceTransformersEmbedding
        _embedding_service = SentenceTransformersEmbedding()
    elif provider == "dashscope":
        from src.embedding.dashscope_embedding import DashScopeEmbeddingService
        _embedding_service = DashScopeEmbeddingService()
    else:
        raise ValueError(f"Unknown provider: {provider}")

    return _embedding_service
```

**为什么单例？** 加载一次 Embedding 模型需要 100MB 内存和 10-30 秒下载。如果每次请求都 new 一个，服务会在 10 秒内 OOM。

**切换模型只需改 .env：**
```env
EMBEDDING_PROVIDER=sentence-transformers    # 本地免费
EMBEDDING_PROVIDER=dashscope                # 云端付费
```

---

## 四、向量数据库 ChromaDB

### 4.1 传统数据库 vs 向量数据库

```
传统数据库（MySQL/SQLite）:
  查询：WHERE name = 'Python'         → 精确匹配，是或否
  返回：完全匹配的行

向量数据库（ChromaDB/Milvus）:
  查询：这个 512 维向量附近有什么？    → 近似匹配，多或少
  返回：距离最近的 K 个结果，按相似度排序
```

**本质区别：** 传统数据库做"等于判断"，向量数据库做"距离计算"。

### 4.2 为什么选 ChromaDB

| ChromaDB | Milvus | FAISS |
|----------|--------|-------|
| 嵌入式，import 即可 | 需要独立部署 etcd+MinIO | 纯算法库，无持久化 |
| 自动持久化到磁盘 | 分布式，支持数十亿向量 | 需自己管理存储 |
| Python 原生 API | 需配环境变量 | C++ 底层，Python 封装 |
| **适合小团队开发** | 适合生产大规模 | 适合研究对比 |

**面试这么说：**"开发时用 Chroma 因为零部署，生产环境可以切 Milvus 支持更大规模。"

### 4.3 核心概念

```
ChromaDB
  └── Client（客户端）
        └── Collection（集合，类似 SQL 的表）
              ├── id（唯一标识）
              ├── embedding（向量）
              ├── document（原文，可选）
              └── metadata（元数据，可选）
```

一个项目可以建多个 Collection。本项目只用了一个：`rag_documents`。

### 4.4 两个核心操作

**insert：** 把向量 + 文档 + 元数据一起存进去。

```python
collection.add(
    ids=["doc1_0", "doc1_1"],                     # 唯一 ID
    documents=["Python是编程语言", "FastAPI是框架"], # 原文（检索时返回）
    embeddings=[[0.12, -0.03, ...], [0.08, 0.11, ...]],  # 512 维向量
    metadatas=[                                     # 附带信息（过滤用）
        {"source": "python.md", "page": 1},
        {"source": "fastapi.md", "page": 2}
    ]
)
```

**query：** 给一个查询向量，找最近的 K 个。

```python
results = collection.query(
    query_embeddings=[[0.11, -0.02, ...]],  # 查询向量
    n_results=5,                              # 返回 Top-5
    include=["documents", "metadatas", "distances"]  # 要返回的字段
)

# 返回结构：
# results["ids"][0]        → ["doc1_0", "doc1_1", ...]  ← 5 个 ID
# results["documents"][0]  → ["Python是...", "FastAPI...", ...]  ← 5 段原文
# results["distances"][0]  → [0.12, 0.35, ...]  ← 5 个距离（越小越相似）
```

**distances 越小 = 越相似。** 余弦空间下，0 是完全相同，1 是完全不相关。

---

## 五、ChromaStore 源码完整内嵌

```python
"""src/database/vector_db.py 的 Chroma 部分"""

from typing import List, Optional, Protocol, runtime_checkable
from src.settings import settings
from src.utils.logger import logger
from src.models.vector import VectorRecord, SearchResult

# ── 接口协议：所有向量库后端必须实现这两个方法 ──
@runtime_checkable
class VectorStore(Protocol):
    def insert(self, records: List[VectorRecord]) -> None: ...
    def search(self, vector: List[float], top_k: int = 10,
               expr: Optional[str] = None) -> List[SearchResult]: ...

# ── ChromaDB 实现 ──
class ChromaStore:
    def __init__(self):
        import chromadb
        self.collection_name = settings.CHROMA_COLLECTION_NAME
        self._path = str(settings.CHROMA_DB_DIR)               # data/chroma_db/
        self.client = chromadb.PersistentClient(path=self._path)
        # PersistentClient：数据存磁盘，重启不丢
        # 对比：chromadb.Client() 是内存模式，重启就没了
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        """获取已有集合，不存在则创建。"""
        # 1. 先尝试获取已有的
        try:
            col = self.client.get_collection(
                name=self.collection_name,
                embedding_function=None,   # ← 关键：不用 Chroma 自带的 Embedding
            )
            return col
        except Exception:
            pass

        # 2. 不存在则创建新的
        return self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},  # 用余弦距离
        )
        # 为什么 embedding_function=None？
        # 我们自己向量化后传入，不用 Chroma 自动向量化。
        # 好处：切模型时不需要重建集合。如果用 Chroma 自带的，
        # 它会锁定维度（比如 384），换 BGE (512维) 就会报错。

    def insert(self, records: List[VectorRecord]) -> None:
        """批量插入。"""
        if not records:
            return

        ids = [r.id for r in records]
        documents = [r.metadata.get("text", "") for r in records]
        metadatas = [{**r.metadata, "kb_id": r.metadata.get("kb_id", 0)}
                     for r in records]
        embeddings = [r.values for r in records]

        # 分批插入，每批最多 40 个（Chroma 的限制）
        batch_size = 40
        for i in range(0, len(ids), batch_size):
            end = i + batch_size
            self.collection.add(
                ids=ids[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
                embeddings=embeddings[i:end],
            )

    def search(self, vector: List[float], top_k=10,
               expr: Optional[str] = None) -> List[SearchResult]:
        """向量检索。expr 可选过滤条件，如 'kb_id == 1'。"""
        # 处理过滤条件
        where = None
        if expr:
            where = _parse_expr_to_chroma_where(expr)  # "kb_id == 1" → {"kb_id": 1}

        # 确保 n_results ≥ 1，避免空集合时 Chroma 报错
        n = max(1, min(top_k, max(self.collection.count(), 1)))

        results = self.collection.query(
            query_embeddings=[vector],
            n_results=n,
            include=["documents", "metadatas", "distances"],
            where=where,
        )

        # 整理返回结果
        search_results = []
        if results and results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                distance = results["distances"][0][i]  # 余弦距离
                similarity = max(0.0, 1.0 - distance)   # 转为相似度
                search_results.append(SearchResult(
                    id=results["ids"][0][i],
                    score=round(similarity, 4),
                    text=results["documents"][0][i],
                    metadata=results["metadatas"][0][i],
                ))
        return search_results

# ── 单例工厂函数 ──
_vector_store = None

def get_vector_store() -> VectorStore:
    global _vector_store
    if _vector_store is not None:
        return _vector_store
    if settings.VECTOR_STORE == "chroma":
        _vector_store = ChromaStore()
    elif settings.VECTOR_STORE == "milvus":
        _vector_store = MilvusStore()
    return _vector_store
```

**三个关键设计决策：**

1. **`embedding_function=None`** — 自己传 embedding，不绑定特定模型维度，切模型不需要重建
2. **`distance → similarity`** — Chroma 返回的是余弦距离（越小越像），我们转成相似度（越大越像）
3. **`max(1, ...)`** — 防止空集合时 `n_results=0` 导致报错

### 5.1 VectorRecord 和 SearchResult 数据结构

```python
# 入库时用 VectorRecord
VectorRecord(
    id="abc123",                     # 唯一 ID
    values=[0.12, -0.03, ...],       # 512 维向量
    metadata={                        # 附带信息
        "text": "Python是...",       # 原文
        "source": "python.md",       # 来源文件
        "chunk_index": 0,            # 第几个 chunk
        "kb_id": 1,                  # 属于哪个知识库
    }
)

# 检索返回 SearchResult
SearchResult(
    id="abc123",
    score=0.8765,                    # 相似度（越大越好）
    text="Python是编程语言...",
    metadata={"source": "python.md", "page": 3}
)
```

---

## 六、动手练习

### 练习 1：写一个 Embedding 实验脚本（15 分钟）

在终端里打开 Python，逐行输入：

```python
from sentence_transformers import SentenceTransformer
import numpy as np

# 加载模型（用已缓存的小模型）
model = SentenceTransformer('all-MiniLM-L6-v2')

# 向量化三段文本
texts = [
    "今天天气很好，适合出去玩",
    "今天天气不错，可以出门逛逛",
    "Python是一种高级编程语言"
]
embeddings = model.encode(texts, normalize_embeddings=True)

# 计算两两之间的余弦相似度
def cos_sim(a, b):
    return np.dot(a, b)  # 归一化后直接点积就是余弦相似度

print(f"句1 vs 句2: {cos_sim(embeddings[0], embeddings[1]):.4f}")  # 应该 > 0.8
print(f"句1 vs 句3: {cos_sim(embeddings[0], embeddings[2]):.4f}")  # 应该 < 0.5
```

**预期结果：** 句1 和句2 相似度很高（都关于天气），句1 和句3 很低（完全不同话题）。

### 练习 2：切换 Embedding 模型对比（10 分钟）

```python
# 用项目配置
import sys; sys.path.insert(0, '.')
from src.embedding import get_embedding_service

# 默认是 BGE（512 维）
emb = get_embedding_service()
vec = emb.embed_query("测试")
print(f"维度: {len(vec)}")   # 如果是 BGE → 512，如果是 all-MiniLM → 384
```

改 `.env` 里的 `SENTENCE_TRANSFORMER_MODEL=all-MiniLM-L6-v2`，重启 Python 再跑一次，观察维度变化。

### 练习 3：操作 ChromaDB 增删查（15 分钟）

```python
import chromadb

# 创建客户端（内存模式，不写磁盘）
client = chromadb.Client()

# 创建集合
collection = client.create_collection("test_collection")

# 插入 3 条数据
collection.add(
    ids=["1", "2", "3"],
    documents=["Python编程语言", "Java编程语言", "今天天气很好"],
    # 不传 embeddings 的话 Chroma 会用内置模型自动向量化
)

# 查询
results = collection.query(
    query_texts=["编程语言有哪些"],  # 自动向量化查询
    n_results=2
)

print("查询结果:")
for i, doc in enumerate(results["documents"][0]):
    dist = results["distances"][0][i]
    print(f"  [{1-dist:.4f}] {doc}")   # 1-distance = 相似度
```

**观察：** "Python编程语言"和"Java编程语言"应该排在前面，"今天天气很好"在最后或不出现在 Top-2。

---

## 七、面试速记

### Q：Embedding 是什么？

**一句话：** 把文本映射到高维向量空间，语义相近的文本向量距离近。

**类比：** 地图上的坐标。意思相近的句子（"汽车"和"轿车"）坐标靠近，意思无关的（"汽车"和"蛋糕"）坐标远离。

### Q：为什么 BGE 比 all-MiniLM 好？

**一句话：** BGE 是中英双语训练的，all-MiniLM 只用英文训练。中文场景 BGE 的语义理解明显更好。

### Q：Chroma 和 Milvus 怎么选？

**一句话：** Chroma 嵌入式零部署，适合开发和小项目；Milvus 分布式高性能，适合生产大规模。我做了双模式，改配置即可切换。

### Q：为什么用单例模式管理 Embedding？

**一句话：** Embedding 模型加载需要 100MB+ 内存，单例保证全局只加载一次，避免 OOM。

### Q：为什么 embedding_function=None？

**一句话：** 我们自己传向量，不绑定特定维度。换模型时不需要重建 Chroma 集合。

---

## 八、验收清单

- [ ] 能用自己的话解释 Embedding 原理（用"地图坐标"类比）
- [ ] 能说出 BGE (512维) 和 all-MiniLM (384维) 的区别
- [ ] 能说出 embedding_documents 和 embed_query 分别什么时候用
- [ ] 能写出 Chroma 的 add 和 query 代码
- [ ] 能解释 distances 越小 = 越相似，以及为什么转成 similarity
- [ ] 能解释为什么单例模式
- [ ] 能解释 embedding_function=None 的原因
- [ ] 练习 1、2、3 都亲自跑过
- [ ] 能回答上面 5 道面试速记题
