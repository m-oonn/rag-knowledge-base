# Day 1：架构与核心链路

> 纯手机端阅读，不需对照任何文件。读完能画出全链路图、讲清楚代码每一行。

---

## 一、从零理解 RAG

你开了一家图书馆，读者问"Python 是谁发明的？"

**纯 LLM：** 你凭记忆回答，记错了就胡说。

**RAG：** 先翻到 Python 那本书的相关段落，对着内容回答。

RAG 一句话：**让 LLM 带着参考资料回答，而不是瞎编。**

它同时解决三个问题：

- **幻觉** — LLM 编造不存在的 API、人名、数据
- **知识过时** — LLM 训练数据截止 2023 年，不知道之后的事
- **不可溯源** — 回答完不知道从哪看来的，没法验证真假

---

## 二、两条管线

把图书馆数字化，分成两件事：

**离线（准备知识库）：** 书 → 拆成卡片 → 编号 → 按编号存柜子

**在线（回答提问）：** 有人问 → 从柜子找相关卡片 → 拼一起给 LLM

---

### 2.1 离线管线：文档 → 向量库（4 步）

```
PDF 文件
  ↓ ① 解析  提取纯文本（去掉排版、表格、图片）
  ↓ ② 分块  切成 1000 字小段，相邻段重叠 200 字
  ↓ ③ 向量化 每段文字变成 512 个数字的向量
  ↓ ④ 存储  向量 + 原文 + 来源 存入 ChromaDB
```

**每一步为什么不能跳过：**

| 跳过哪步 | 会怎样 |
|---------|-------|
| 不解析 | LLM 读不懂 PDF 二进制数据，只能读纯文本 |
| 不分块 | 一篇 5000 字的文章塞进 Embedding，语义被稀释；检索时定位不到具体段落 |
| 不向量化 | 文字本身不能做数学运算。搜"汽车"永远找不到"轿车" |
| 不存向量库 | 每次查询都要重新向量化所有文档，几万条数据根本跑不动 |

---

### 2.2 在线管线：问题 → 答案（5 步）

```
用户问 "Python 是谁创造的？"
  ↓ ① 向量化    把问题也变成 512 维向量（和文档用同一个模型）
  ↓ ② 粗排检索  ChromaDB 找向量最相近的 20 个 chunk
  ↓ ③ 精排重排  BM25 关键词补位 + RRF 融合 + CrossEncoder 精选 Top-5
  ↓ ④ 拼接 Prompt  5段文档 + 问题 + 约束规则 拼成一段文本
  ↓ ⑤ LLM 生成  发给 DeepSeek → 返回带引用的回答
```

**每一步为什么不能跳过：**

| 跳过哪步 | 会怎样 |
|---------|-------|
| 不向量化问题 | 没法做语义检索，只能关键词匹配 |
| 不粗排 | 不知道找哪些文档，无头苍蝇 |
| 不混合检索 | 精确关键词（如"Python 3.12"）搜不到，纯向量只懂语义不懂字符串 |
| 不重排序 | 粗排不够准，第 5 名可能比第 1 名更适合回答 |
| 不拼 Prompt | LLM 不知道你有哪些资料，也不知道该遵守什么规则 |
| 不调 LLM | 就没有回答 |

---

### 2.3 完整架构图（一张图看懂）

```
┌───────── 离线（准备知识库）─────────┐
│                                      │
│  PDF/DOCX → 解析 → 分块 → 向量化     │
│                            ↓         │
│                        ChromaDB      │
│                                      │
└──────────────────────────────────────┘
              ↓ 知识库就绪
┌───────── 在线（回答提问）─────────┐
│                                      │
│  用户问题 → 向量化 → ChromaDB 检索   │
│                    ↘                 │
│                     BM25 关键词检索  │
│                        ↓             │
│                    RRF 融合排名      │
│                        ↓             │
│                  CrossEncoder 精排   │
│                        ↓             │
│                  Prompt 拼接         │
│                        ↓             │
│                  LLM 生成回答        │
│                                      │
└──────────────────────────────────────┘
```

---

## 三、两个核心技术（面试必问）

### 3.1 混合检索 — 解决"找不找得到"

**纯向量的盲区：** 精确关键词。搜"Python 3.12 更新日志"，向量可能返回"Python 2.7 更新日志"，因为它们语义非常相近。

**BM25 的强项：** 精确匹配。"3.12"这个字符串在哪些文档里出现？直接匹配出来。

```
向量检索：找语义像的 → 各种"更新日志"都返回，不管版本号
BM25 检索：找词对上的 → "3.12"这个精确字符串出现在哪
RRF 融合：两份结果综合排名 → 既语义相关，又精确匹配
```

RRF（Reciprocal Rank Fusion）公式：`score = 1/(k+rank)`，k 通常取 60。一个文档在两个检索器中排名都靠前，融合分数就高。

**面试金句：**"纯向量检索对精确关键词匹配有短板，我用 BM25 + RRF 混合检索解决了。"

### 3.2 两阶段检索 — 解决"排得对不对"

```
粗排（Bi-encoder）→ 问题+文档各自单独计算 → 快但不准 → 从几千取 20 个
精排（Cross-encoder）→ 问题+文档拼一起算分 → 慢但精准 → 从 20 取 5 个
```

为什么分两步？CrossEncoder 对每对文档都要重新计算，对全部 1000 个文档做一次太慢。先快筛再精挑。

**面试金句：**"我用两阶段检索，Bi-encoder 粗排保证覆盖，Cross-encoder 精排保证精度。"

---

## 四、Demo 源码逐行讲解

以下是 `demo.py` 的完整源码。你不用打开任何文件，读完这段就能理解每一步。

### Step 1：初始化组件（第 32-58 行）

```python
# 从 .env 文件读取配置
from src.settings import settings
# 打印当前配置：LLM=deepseek, Vector=chroma, Embedding=sentence-transformers...

# 加载 Embedding 模型（首次从 HuggingFace 下载 ~100MB）
from src.embedding import get_embedding_service
embed = get_embedding_service()
# get_embedding_service() 是单例：全局只创建一次，重复调用返回同一个对象
# 内部优先尝试 BAAI/bge-small-zh-v1.5（中文），失败则用 all-MiniLM-L6-v2（英文）

# 初始化 ChromaDB——连接本地文件 chroma_db/ 文件夹
from src.database.vector_db import get_vector_store
store = get_vector_store()
# 同样是单例。ChromaStore 内部创建 Chroma 客户端，获取或创建集合

# 初始化 LLM 客户端——自动检测使用哪个模型
from src.llm.llm_client import LLMClient
llm = LLMClient()
# 根据 settings.LLM_PROVIDER 决定用 DeepSeek/Ollama/Qwen
```

### Step 2：创建文档 + 入库（第 60-154 行）

```python
# 三篇样例文档，直接写在代码里（不打文件）
sample_docs = {
    "python_intro.md": """# Python...
Python 是 Guido van Rossum 于 1991 年创建的高级编程语言...""",
    "machine_learning.md": """# 机器学习...
机器学习使系统能从经验中学习...""",
    "fastapi_guide.md": """# FastAPI...
FastAPI 是高性能 Python Web 框架...""",
}

# 逐篇处理
for filename, content in sample_docs.items():
    # ① 保存文件到磁盘
    filepath.write_text(content)

    # ② 创建 Document 对象（Pydantic 模型）
    doc = Document(filename=filename, content=content,
                   metadata=DocumentMetadata(source=filename))

    # ③ 分割文本：用 RecursiveCharacterTextSplitter
    chunks = chunker.split_document(doc)
    # 输入：一篇 500 字的 Document
    # 输出：1-2 个 Chunk（每段 ≤ 1000 字，间距 200 字重叠）

    # ④ 向量化：全部分 Chunk 的文本 → 512 维向量
    texts = [c.text for c in chunks]
    embeddings = embed.embed_documents(texts)
    # 输入：["Python 是...", "机器学习是..."]
    # 输出：[[0.12, -0.03, ...], [0.08, 0.11, ...]]

    # ⑤ 构建 VectorRecord 列表
    for chunk, emb in zip(chunks, embeddings):
        records.append(VectorRecord(
            id=str(uuid.uuid4()),     # 唯一 ID
            values=emb,               # 512 个浮点数的向量
            metadata={                # 附带信息
                "text": chunk.text,   # 原文（检索时返回的）
                "source": filename,   # 来源文件
                "chunk_index": i,     # 第几个 chunk
                "kb_id": 1,           # 属于哪个知识库
            }
        ))

    # ⑥ 批量存入 ChromaDB
    store.insert(records)
```

### Step 3：RAG 问答（第 156-205 行）

```python
test_queries = [
    "Who created Python and when?",
    "What is machine learning?",
    "What are FastAPI's key features?",
]

for query in test_queries:
    # ① 问题向量化——和文档用同一个 Embedding 模型
    query_vec = embed.embed_query(query)
    # "Who created Python" → [0.11, -0.02, 0.43, ...] (512个浮点数)

    # ② 向量检索——在 ChromaDB 中找最相近的 5 个
    results = store.search(query_vec, top_k=5)
    # 返回 List[SearchResult]，每个有：
    #   id: 向量 ID
    #   score: 相似度（越大越相关）
    #   text: 文档原文
    #   metadata: 来源文件名、页码等

    # ③ 重排序——CrossEncoder 对 5 个结果重新打分排序
    if results and settings.ENABLE_RERANK:
        results = reranker.rerank(query, results)
    # CrossEncoder 把问题+文档拼在一起精读一遍，给更准的分数

    # ④ 拼接 Prompt
    context = "\n\n".join(
        f"[{i+1}] {r.text[:300]}" for i, r in enumerate(results)
    )
    prompt = f"""基于以下上下文回答问题。如无相关信息请说明。

上下文：
{context}

问题：{query}

回答："""

    # ⑤ 调 LLM 生成回答
    answer = llm.generate_custom_response(prompt)
    # 内部根据 .env 的 LLM_PROVIDER 决定调谁：
    #   ollama    → POST localhost:11434/api/generate
    #   deepseek  → POST api.deepseek.com/chat/completions
    #   dashscope → LangChain ChatTongyi
```

### 运行输出示例

```
============================================================
  Step 1: Initialize Components
============================================================
    ✅ LLM:       deepseek
    ✅ Vector DB: chroma
    ✅ Embedding: sentence-transformers
    ✅ Rerank:    local
    ✅ Hybrid:    True

============================================================
  Step 2: Create Sample Documents
============================================================
    ✅ Created: python_intro.md (525 chars)
    ✅   → 1 chunks indexed
    ✅ Created: machine_learning.md (538 chars)
    ✅   → 1 chunks indexed
    ✅ Created: fastapi_guide.md (489 chars)
    ✅   → 1 chunks indexed
    ✅ Total: 3 documents → 3 chunks

============================================================
  Step 3: RAG Queries
============================================================

  [Query: Who created Python and when?]
    📎 Retrieved 5 docs in 0.00s
       [8.6854] Python was created by Guido van Rossum...
       [7.3350] Python is a high-level, interpreted...
    🤖 Answer (0.77s): Python由Guido van Rossum于1991年创建。
```

解读输出：
- `[8.6854]` — CrossEncoder 打分，分数越高越相关
- `Retrieved ... in 0.00s` — 向量检索几乎不耗时
- `Answer (0.77s)` — LLM 生成花了 0.77 秒

---

## 五、三大核心源码完整内嵌

以下是你项目最核心的三段代码，直接读，不要跳。

### 5.1 项目入口 main.py（完整版）

```python
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.settings import settings
from src.database.sql_session import engine, Base
from src.utils.logger import logger

def create_app() -> FastAPI:
    """创建 FastAPI 应用，注册所有路由。"""
    app = FastAPI(
        title=settings.APP_NAME,
        version="1.0.0",
        description="RAG Knowledge Base Q&A System",
    )

    # CORS 中间件：允许前端跨域访问
    app.add_middleware(CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # 导入所有数据库模型 → 自动建 13 张表
    from src.database.models import (
        User, KnowledgeBase, KnowledgeDocument, DocumentChunk,
        GeneratedQAPair, ChatSession, ChatInteraction,
        EvaluationTask, EvaluationResult, EvaluationDatasetItem,
        Assistant, Agent,
    )
    Base.metadata.create_all(bind=engine)

    # ── 注册 5 个核心路由 ──
    from src.api.routers import health, auth, knowledge_base, chat, evaluation

    app.include_router(health.router,
        prefix=settings.API_PREFIX + "/health", tags=["Health"])
    app.include_router(auth.router,
        prefix=settings.API_PREFIX + "/auth", tags=["Auth"])
    app.include_router(knowledge_base.router,
        prefix=settings.API_PREFIX + "/knowledge-bases", tags=["Knowledge Base"])
    app.include_router(chat.router,
        prefix=settings.API_PREFIX + "/chat", tags=["RAG Chat"])
    app.include_router(evaluation.router,
        prefix=settings.API_PREFIX + "/evaluations", tags=["Evaluation"])

    # ── 注册 5 个扩展路由（try/except 包裹，依赖缺失不崩溃）──
    for router_name, module_path, prefix_suffix, tag in [
        ("assistant", "src.api.routers.assistant", "/assistants", "Assistant"),
        ("agent",     "src.api.routers.agent",      "/agents",    "Agent"),
        ("monitor",   "src.api.routers.monitor",    "/monitor",   "Monitor"),
        ("query",     "src.api.routers.query",      "/query",     "Query"),
        ("loadfile",  "src.api.routers.loadfile",   "/upload",    "File Management"),
    ]:
        try:
            mod = __import__(module_path, fromlist=["router"])
            app.include_router(mod.router,
                prefix=settings.API_PREFIX + prefix_suffix, tags=[tag])
        except Exception as e:
            logger.warning(f"Router '{router_name}' skipped: {e}")

    # ── MinIO 存储：只在开启时加载 ──
    if settings.ENABLE_MINIO:
        try:
            from src.api.routers import storage
            app.include_router(storage.router,
                prefix=settings.API_PREFIX + "/storage", tags=["MinIO Storage"])
        except Exception as e:
            logger.warning(f"MinIO storage router skipped: {e}")
    else:
        logger.info("MinIO disabled (ENABLE_MINIO=False)")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Vector store: {settings.VECTOR_STORE}")
    logger.info(f"LLM provider: {settings.LLM_PROVIDER}")
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
```

**三个关键设计：**

- **try/except 包裹扩展路由** — 缺 MinIO/Celery 时服务器不崩溃，只跳过那个路由
- **if ENABLE_MINIO** — 不需要的服务完全不加载，轻量模式下没有任何额外依赖
- **每个 Router 独立文件** — auth 挂了不影响 chat，低耦合

### 5.2 RAG 问答核心 rag_service.py（精简关键部分）

```python
# ── 单例模式：全局只创建一个实例 ──
_rag_service_instance = None

def get_rag_service():
    global _rag_service_instance
    if _rag_service_instance is None:       # 第一次调用才创建
        _rag_service_instance = RAGService()
    return _rag_service_instance            # 之后直接返回同一个

# ── 初始化：装配所有组件 ──
class RAGService:
    def __init__(self):
        self.retriever = VectorRetriever()  # 向量检索器
        self.llm_client = LLMClient()       # LLM 客户端
        self.reranker = get_reranker()      # CrossEncoder 重排序
        self.analyzer = QuestionAnalyzer()  # 多跳问题分析
        self.memory = MemorySystem()        # 对话记忆
        self._hybrid = None                 # 混合检索器（懒加载）
        self._bm25_dirty = True             # BM25 需同步标记

# ── 核心方法：query() ──
    def query(self, query_text, top_k=5, session_id="default",
              kb_ids=None, assistant_config=None):

        # ① 获取这个会话的历史对话
        history = self.memory.get_short_term_memory(session_id)
        # 为什么需要？用户问"那 Java 呢"，需要上文知道他在问什么

        # ② 没选知识库 → 普通聊天模式，不走 RAG
        if not kb_ids:
            answer = self.llm_client.generate_general_response(query_text, "")
            self.memory.add_short_term_memory(session_id, "user", query_text)
            self.memory.add_short_term_memory(session_id, "assistant", answer)
            return {"query": query_text, "answer": answer, "source_documents": []}

        # ③ 问题分析：是不是复杂问题需要多跳？
        analysis = self.analyzer.analyze(query_text)
        # "Python 谁发明的？" → is_multi_hop: False
        # "对比 Python 和 Java 性能" → is_multi_hop: True

        # ④ 根据分析结果选检索策略
        if settings.ENABLE_MULTI_HOP and analysis.get("is_multi_hop"):
            result = self._multi_hop_query(...)    # 多跳：拆子问题逐步检索
        else:
            result = self._single_hop_query(query_text, top_k,
                                             history_str, kb_ids, system_prompt)

        # ⑤ 更新对话记忆
        self.memory.add_short_term_memory(session_id, "user", query_text)
        self.memory.add_short_term_memory(session_id, "assistant", result["answer"])
        return result

# ── 干活的方法：_single_hop_query() ──
    def _single_hop_query(self, query_text, top_k, history_str,
                          kb_ids, system_prompt):

        # ① 检索（粗排）
        initial_k = top_k * 2  # 为什么 ×2？给重排序多备一些候选
        retriever = self._get_retriever()
        # _get_retriever() 返回：
        #   ENABLE_HYBRID_SEARCH=true  → HybridRetriever (BM25+向量+RRF)
        #   false                      → VectorRetriever (纯向量)

        search_results = retriever.retrieve(
            query_text, top_k=initial_k, kb_ids=kb_ids
        )
        # 返回 List[SearchResult]，每个有 score、text、metadata

        # ② 重排序（精排）
        if settings.ENABLE_RERANK and search_results:
            search_results = self.reranker.rerank(query_text, search_results)
            # CrossEncoder 把问题+每个文档拼一起精读一遍
            # 20 个中取 5 个最相关的
        else:
            search_results = search_results[:top_k]  # 直接截前 5

        # ③ 拼接上下文
        context = self._format_context(search_results)
        # 输出："[1] Python 由 Guido 创建...\n[2] Guido 是 Python 之父..."

        # ④ 调用 LLM 生成回答
        answer = self.llm_client.generate_response(query_text, context)
        # 内部拼接完整 Prompt → 发给 DeepSeek/Ollama → 返回回答

        # ⑤ 返回结果（带来源）
        return self._format_response(query_text, answer, search_results)
```

**9 个步骤总结：**

```
❶ 获取历史 → ❷ 判断有无KB → ❸ 分析问题 → ❹ 选检索器
→ ❺ 检索粗排 → ❻ 重排序 → ❼ 拼上下文 → ❽ LLM 生成 → ❾ 返回
```

### 5.3 文档处理链路 knowledge_base.py（简化版）

```python
def _process_document_async(doc_id: int):
    """处理一个文档：解析 → 分块 → 向量化 → 存储。"""
    db = SessionLocal()

    # ── ① 查文档记录 + 标"处理中" ──
    doc = db.query(KnowledgeDocument).filter(
        KnowledgeDocument.id == doc_id
    ).first()
    doc.status = 1   # 1 = Processing（处理中）
    db.commit()

    # ── ② 解析文档：自动识别 PDF/DOCX/MD/TXT ──
    parser = MultiDocParser()
    parsed_doc = parser.parse(doc.file_path)
    content = parsed_doc.content           # 纯文本内容
    if not content:
        doc.status = 3                     # 3 = Failed（失败）
        doc.error_msg = "文件为空"
        db.commit()
        return

    # ── ③ 文本分割 ──
    chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
    # 内部用 RecursiveCharacterTextSplitter
    # 分隔符优先级：段落 → 换行 → 中文句号 → 英文句号 → 空格 → 硬切
    doc_model = Document(filename=doc.filename, content=content, ...)
    chunks = chunker.split_document(doc_model)
    # 返回 List[Chunk]：每个 Chunk 有 text、page、chunk_index

    # ── ④ 向量化 ──
    emb_svc = get_embedding_service()      # 单例 Embedding 服务
    texts = [c.text for c in chunks]       # 提取所有 chunk 文本
    embeddings = emb_svc.embed_documents(texts)
    # 输入：["Python 是一种...", "它由 Guido..."]
    # 输出：[[0.12, -0.03, ...], [0.08, 0.11, ...]]

    # ── ⑤ 存入 ChromaDB ──
    store = get_vector_store()             # 单例 ChromaStore
    records = []
    for chunk, emb in zip(chunks, embeddings):
        records.append(VectorRecord(
            id=str(uuid.uuid4()),
            values=emb,                    # 512 维向量
            metadata={
                "text": chunk.text,        # 原文
                "source": doc.filename,    # 来源文件
                "chunk_index": i,          # 第几个 chunk
                "kb_id": doc.kb_id,        # 属于哪个知识库
            }
        ))
    store.insert(records)                  # 批量插入

    # ── ⑥ 标记完成 ──
    doc.status = 2        # 2 = Completed（完成）
    doc.chunk_count = len(chunks)
    db.commit()
    db.close()
```

**文档状态码：** 0=上传中 → 1=处理中 → 2=完成 → 3=失败

---

## 六、URL 与代码对应表

当你访问 `http://localhost:8000/api/v1/chat/` 时：

```
POST /api/v1/auth/register
  → src/api/routers/auth.py → 在 users 表创建用户

POST /api/v1/auth/login/access-token
  → src/api/routers/auth.py → 验密码 → 返回 JWT

POST /api/v1/knowledge-bases/
  → src/api/routers/knowledge_base.py → 创建知识库

POST /api/v1/knowledge-bases/{id}/upload
  → src/api/routers/knowledge_base.py → 上传文件 → 调 _process_document_async

POST /api/v1/chat/
  → src/api/routers/chat.py → 调 RAGService.query() → 返回回答

POST /api/v1/chat/stream
  → src/api/routers/chat.py → SSE 流式输出（逐 token 推送）

GET /api/v1/health/
  → src/api/routers/health.py → {"status": "ok"}
```

所有后续请求都要在 Header 里带：`Authorization: Bearer <token>`

---

## 七、动手练习

### 练习 1：画全链路图

不看文档，白纸画两张图：

**图 1：离线管线**
```
PDF → [  ] → [  ] → [  ] → [  ] → 知识库就绪
```
每个空填：步骤名 + 用哪个类

**图 2：在线管线**
```
用户问题 → [  ] → [  ] → [  ] → [  ] → [  ] → 回答
```
每个空填：步骤名 + 用哪个类

画完对照第二部分的图。

### 练习 2：用自己的话讲每段代码

指着上面"五大核心源码"中的每段代码，出声读并用你自己的话解释。重点：

- `_single_hop_query` 里为什么 `top_k * 2`？
- `_get_retriever()` 可能返回哪两种类型？
- `reranker.rerank()` 做了什么？
- BM25 和向量检索各有什么优缺点？

### 练习 3：口头问答

依次回答，每个用一句话：

1. RAG 解决 LLM 的哪三个问题？
2. Embedding 是什么？用你理解的类比讲出来。
3. 混合检索比纯向量好在哪？举一个具体例子。
4. Bi-encoder 和 Cross-encoder 的根本区别是什么？为什么分两步？
5. `doc.status=0,1,2,3` 分别代表什么？
6. main.py 的 try/except（扩展路由部分）是干什么的？

---

## 八、验收清单

- [ ] 不看文档能画出离线 4 步 + 在线 5 步
- [ ] 能解释 RAG 解决的三个问题
- [ ] 能用"地图坐标"类比解释 Embedding
- [ ] 能举例说明混合检索比纯向量好的场景
- [ ] 能说清 Bi-encoder vs Cross-encoder 区别
- [ ] 能独立列出 `_single_hop_query` 的步骤
- [ ] 能说出 `_process_document_async` 的 6 步
- [ ] 知道 7 个 API 端点对应哪个文件
- [ ] 对着镜子讲一遍完整流程（3 分钟）
- [ ] 练习题的 6 个口头问答全部答对
