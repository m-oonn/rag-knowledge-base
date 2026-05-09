# Day 1：架构与核心链路 — 完整学习文档

> 4 小时。读完这个你就能画出完整架构图，讲清楚一个请求怎么走。

---

## 第一部分：从零理解 RAG（30 分钟）

### 1.1 用"图书馆"理解问题

你开了一家图书馆。有个读者问："Python 是谁发明的？"

**做法 A（纯 LLM）：** 你凭记忆回答。万一记错了就胡说八道。

**做法 B（RAG）：** 先翻书找到 Python 相关段落，对着抄下来的内容回答。

RAG 解决一句话：**让 LLM 带着参考资料回答，而不是凭记忆瞎编。**

它同时解决三个问题：

| 问题 | 举例 |
|------|------|
| 幻觉 | LLM 编造不存在的 API、人名、数据 |
| 知识过时 | LLM 训练数据截止 2023 年，不知道 2024 年的事 |
| 不可溯源 | LLM 回答完你不知道它从哪看来的 |

---

### 1.2 把"翻书回答"拆成两步

**离线（建索引）：** 把书拆成卡片 → 每张卡片编号 → 按编号存到柜子里。
**在线（查问答）：** 有人提问 → 从柜子找最相关的几张卡片 → 拼在一起给 LLM。

这两步对应项目中的两条独立管线。

---

### 1.3 离线管线：文档 → 向量库（4 步）

```
PDF 文件
  ↓ ① 解析（MultiDocParser）：提取纯文本
  ↓ ② 分块（TextChunker）：切成 1000 字小段，重叠 200 字
  ↓ ③ 向量化（Embedding）：每段文字 → 512 维数字向量
  ↓ ④ 存储（ChromaStore）：向量 + 原文 + 元数据 存入向量库
```

每一步为什么不能跳过：

| 步骤 | 跳过会怎样 |
|------|-----------|
| 解析 | LLM 读不懂 PDF 二进制数据 |
| 分块 | 5000 字一次性 Embedding 语义被稀释，检索定位不到具体段落 |
| 向量化 | 文字本身没法做数学运算，搜"汽车"找不到"轿车" |
| 存向量库 | 每次查询都重新向量化所有文档，慢到不可用 |

---

### 1.4 在线管线：问题 → 答案（5 步）

```
用户问题 "Python 是谁创造的？"
  ↓ ① 向量化：问题 → 512 维向量（用同一个 Embedding 模型）
  ↓ ② 粗排检索：ChromaDB 找最相近的 20 个 chunk
  ↓ ③ 混合检索 + 精排：BM25 关键词补位 → RRF 融合 → CrossEncoder 精选 Top-5
  ↓ ④ 拼接 Prompt：Top-5 段落 + 用户问题 + 约束规则 拼成一段文本
  ↓ ⑤ LLM 生成：发给 DeepSeek/Ollama → 返回带引用的回答
```

每一步为什么不能跳过：

| 步骤 | 跳过会怎样 |
|------|-----------|
| 向量化 | 无法做语义检索 |
| 粗排 | 不知道查什么文档 |
| 混合检索 | 精确关键词（如"Python 3.12"）可能找不到 |
| 重排序 | 粗排精度不够，第 5 名可能比第 1 名好 |
| Prompt 拼接 | LLM 不知道你的知识库内容和约束条件 |
| LLM 生成 | 没有回答 |

---

## 第二部分：混合检索和重排序详解（20 分钟）

### 2.1 混合检索解决"找不找得到"

**纯向量的盲区：** 精确关键词。你搜"Python 3.12 更新日志"，向量可能返回"Python 2.7 更新日志"——语义上都是"Python 更新日志"，但关键词不对。

**BM25 的强项：** 精确匹配"3.12"这个字符串出现在哪些文档里。

```
向量检索：找语义像的 → "更新日志"相关都返回，不管版本号
BM25 检索：找词对上的 → "3.12"这个词出现在哪里
RRF 融合：两份结果综合排名 → 既有语义相关性又有精确匹配
```

**面试金句：** "纯向量检索对精确关键词匹配有短板，我用 BM25 + RRF 混合检索解决了这个问题。"

### 2.2 重排序解决"排得对不对"

粗排是批量处理的，精度有限。CrossEncoder 做更仔细的检查：把问题和每个候选文档**拼在一起**打分。

```
粗排：Bi-encoder → 问题+文档各自独立计算 → 快但粗糙 → 取 20 个
精排：Cross-encoder → 问题+文档拼接后联合计算 → 慢但精准 → 取 5 个
```

**为什么分两步：** CrossEncoder 很贵，不可能对全部 1000 个文档都做一次。所以先快筛（向量粗排 20 个），再精挑（CrossEncoder 精选 5 个）。

**面试金句：** "我用两阶段检索，Bi-encoder 粗排保证覆盖，Cross-encoder 精排保证精度。"

---

## 第三部分：从 Demo 看懂全链路（30 分钟）

运行 `python demo.py`，对照下面的注释理解每一步：

```python
# ================================================================
# Step 1: Initialize Components
# ================================================================
# 输出：
#   ✅ LLM:       deepseek       ← 用的是 DeepSeek API（.env 里配的）
#   ✅ Vector DB: chroma         ← 向量库是 Chroma（嵌入式，零部署）
#   ✅ Embedding: sentence-transformers  ← 本地 BGE/all-MiniLM 模型
#   ✅ Rerank:    local          ← CrossEncoder 本地重排序
#   ✅ Hybrid:    True           ← 混合检索已开启

# 这一步干了什么：
# ① 从 .env 读取所有配置 → settings.py
# ② 加载 Embedding 模型（如果是首次，从 HuggingFace 下载 ~100MB）
# ③ 初始化 ChromaDB 客户端 → 连接或创建 chroma_db/ 文件夹
# ④ 初始化 LLM 客户端 → 检查 DeepSeek API Key 是否已配置
```

```python
# ================================================================
# Step 2: Create Sample Documents
# ================================================================
# 输出：
#   ✅ Created: python_intro.md (525 chars)
#   ✅   → 1 chunks indexed
#   ✅ Created: machine_learning.md (538 chars)
#   ✅   → 1 chunks indexed
#   ✅ Created: fastapi_guide.md (489 chars)
#   ✅   → 1 chunks indexed
#   ✅ Total: 3 documents → 3 chunks

# 这一步干了什么（对应离线管线 4 步）：
# ① MultiDocParser 解析文件 → 提取纯文本
# ② TextChunker 分割 → 3 篇文章因为都短于 1000 字，所以各成 1 个 chunk
# ③ Embedding 向量化 → 每个 chunk → 384/512 维向量
# ④ ChromaStore.insert() → 存入 ChromaDB
```

```python
# ================================================================
# Step 3: RAG Queries
# ================================================================
# 输出示例：
#  [Query: Who created Python and when?]
#     📎 Retrieved 5 docs in 0.00s
#        [8.6854] Python was created by Guido van Rossum...
#        [7.3350] Python is a high-level, interpreted...
#     🤖 Answer (0.77s): Python由Guido van Rossum于1991年创建。

# 每一步对应代码中的位置：
# "Retrieved 5 docs"     → rag_service.py 第 159 行：retriever.retrieve()
# "[8.6854]" 分数         → reranker.py 的 CrossEncoder 打分（分数越高越相关）
# "Answer (0.77s)"       → llm_client.py 调用 DeepSeek，耗时 0.77 秒
```

---

## 第四部分：核心源码逐行阅读（60 分钟）

### 4.1 项目入口：`src/main.py`（74 行）

**你打开这个文件，对照下面阅读。**

```
第 14 行：def create_app() → 创建 FastAPI 应用
第 22-27 行：CORS 中间件 → 允许前端跨域访问
第 30-36 行：导入所有数据库模型 → 自动创建 13 张表
第 39 行：导入 5 个核心路由
第 41-45 行：注册 5 个核心路由（health, auth, kb, chat, evaluation）
第 48-59 行：注册 5 个扩展路由（assistant, agent, monitor, query, loadfile）
        每个用 try/except 包裹 → 依赖缺失时优雅跳过，不崩溃
第 62-69 行：MinIO 存储 → 只在 ENABLE_MINIO=true 时加载
第 74 行：app = create_app() → 模块加载时立即创建
第 76-82 行：uvicorn.run() → 启动服务器（仅直接运行 main.py 时触发）
```

**关键设计思想：**

| 设计 | 体现在哪 | 为什么这样做 |
|------|---------|------------|
| 优雅降级 | 第 55-59 行 try/except | 缺 MinIO/Celery 不崩溃 |
| 条件加载 | 第 62 行 if ENABLE_MINIO | 不需要的服务不加载 |
| 模块化 | 每个 Router 独立文件 | 一个路由挂了不影响其他 |

---

### 4.2 RAG 核心：`src/services/rag_service.py`（~200 行）

**单例模式（第 12-19 行）：**

```python
_rag_service_instance = None   # 全局变量，初始为空

def get_rag_service():
    global _rag_service_instance
    if _rag_service_instance is None:    # 第一次调用才创建
        _rag_service_instance = RAGService()
    return _rag_service_instance         # 之后直接返回同一个
```

为什么单例？RAGService 创建时会加载 Embedding 模型（~100MB）、LLM 客户端、CrossEncoder（~500MB）。创建一次就够了，每次请求 new 一个会 OOM。

**__init__ 初始化（第 23-30 行）：**

```python
self.retriever = VectorRetriever()   # 向量检索器（内含 ChromaStore + Embedding）
self.llm_client = LLMClient()        # LLM 客户端（自动检测 DeepSeek/Ollama）
self.reranker = get_reranker()       # CrossEncoder 重排序器
self.analyzer = QuestionAnalyzer()   # 多跳问题分析器
self.memory = MemorySystem()         # 对话记忆（Redis 或内存）
self._hybrid = None                  # 混合检索器（懒加载）
self._bm25_dirty = True              # BM25 索引标记（需要同步时设为 True）
```

**核心方法 `query()`（第 62-143 行）— 最重要的方法：**

```
query(query_text, top_k, session_id, kb_ids, assistant_config)

第 72 行：提取 system_prompt（如果有）
第 77 行：提取 memory_config（对话记忆设置）
第 83 行：get_short_term_memory(session_id) → 获取这个会话的历史对话
          为什么需要历史？用户问"那Java呢"，需要上文才知道他问什么
第 101 行：if not kb_ids → 没选知识库就当普通聊天，不走 RAG
第 131 行：analyzer.analyze(query) → 判断是不是复杂问题（需不需要多跳）
第 134 行：ENABLE_MULTI_HOP？→ 多跳 or 单跳
第 140-141 行：更新对话记忆 → 把这次问答存起来
```

**`_single_hop_query()`（第 145-170 行）— 真正干活的方法：**

```
第 155 行：initial_k = top_k * 2  ← 为什么*2？
           因为重排序会砍掉一半。给 CrossEncoder 20 个才够它挑出 5 个最好的。
           
第 156 行：retriever = self._get_retriever()
           ENABLE_HYBRID_SEARCH=true → HybridRetriever (BM25+向量+RRF)
           false → VectorRetriever (纯向量)

第 159 行：retriever.retrieve(query_text, top_k=20, kb_ids=[...])
           返回 List[SearchResult]：每个有 score, text, metadata

第 162-165 行：reranker.rerank() 或直接截断
           ENABLE_RERANK=true → CrossEncoder 对 20 个精排取 5 个
           false → 直接取前 5 个

第 168 行：self._format_context(search_results)
           把 5 个 SearchResult 拼成：
           "[1] Python由Guido创建...\n[2] Guido是Python之父..."

然后调用 llm_client.generate_response(query, context) → 返回 answer + sources
```

---

### 4.3 文档处理：`src/api/routers/knowledge_base.py` — `_process_document_async()`

**第 33-34 行：模式选择**

```python
if _has_celery_task:
    return _celery_process_task.delay(doc_id)  # 企业模式：异步 Celery
else:
    # 轻量模式：同步处理（你的当前模式）
```

**第 46-52 行：标记"处理中"**

```python
db = SessionLocal()                              # 打开数据库连接
doc = db.query(KnowledgeDocument).filter(...)     # 查到这个文档记录
doc.status = 1  # Processing                     # 状态改为 1（处理中）
db.commit()                                       # 保存
```

**第 54-62 行：解析文档**

```python
parser = MultiDocParser()                        # 创建解析器
parsed_doc = parser.parse(filepath)              # 自动识别 PDF/DOCX/MD/TXT
content = parsed_doc.content                     # 拿到纯文本
if not content:                                   # 空文件？
    doc.status = 3; doc.error_msg = "Empty..."   # 标记失败
```

**第 64-74 行：分割**

```python
chunker = TextChunker(chunk_size=1000, chunk_overlap=200)
doc_model = DocModel(...)                        # 包装成 Document 对象
chunks = chunker.split_document(doc_model)       # 分割 → List[Chunk]
```

**第 76-90 行：向量化 + 存储**

```python
emb_svc = get_embedding_service()                # 单例 Embedding
store = get_vector_store()                       # 单例 ChromaStore
texts = [c.text for c in chunks]                 # 提取所有 chunk 文本
embeddings = emb_svc.embed_documents(texts)     # 批量向量化

# 构建 VectorRecord 列表
for c, emb in zip(chunks, embeddings):
    records.append(VectorRecord(
        id=str(uuid.uuid4()),
        values=emb,                               # 512 维向量
        metadata={                                # 原文、来源、页码、KB ID
            "text": c.text, "source": doc.filename,
            "chunk_index": i, "kb_id": doc.kb_id,
        }
    ))
store.insert(records)                            # 批量插入 ChromaDB
```

---

## 第五部分：路由注册 — URL 怎么对应代码（10 分钟）

当你访问 `http://localhost:8000/api/v1/chat/` 时，请求怎么找到处理函数？

```
                              main.py（总控）
                                  │
        ┌─────────────┬───────────┼───────────┬──────────────┐
        ▼             ▼           ▼           ▼              ▼
    /health       /auth      /knowledge    /chat       /evaluation
 health.py      auth.py     -bases       chat.py    evaluation.py
                             kb.py

    /health/     /auth/     /knowledge    /chat/     /evaluations/
                 register   -bases/       (RAG问答)   (RAGAS评测)
                 login      创建/删除/     stream
                           上传/文档      (SSE流式)
                           管理
```

**注册代码（main.py 第 41-45 行）：**

```python
app.include_router(health.router,       prefix="/api/v1/health",         ...)
app.include_router(auth.router,         prefix="/api/v1/auth",           ...)
app.include_router(knowledge_base.router, prefix="/api/v1/knowledge-bases", ...)
app.include_router(chat.router,         prefix="/api/v1/chat",           ...)
app.include_router(evaluation.router,   prefix="/api/v1/evaluations",   ...)
```

**完整的 API 表：**

| URL | 文件 | 功能 |
|-----|------|------|
| `GET /api/v1/health/` | health.py | 健康检查 |
| `POST /api/v1/auth/register` | auth.py | 注册 |
| `POST /api/v1/auth/login/access-token` | auth.py | 登录 |
| `POST /api/v1/knowledge-bases/` | knowledge_base.py | 创建知识库 |
| `POST /api/v1/knowledge-bases/{id}/upload` | knowledge_base.py | 上传文档 |
| `GET /api/v1/knowledge-bases/{id}/documents` | knowledge_base.py | 文档列表 |
| `POST /api/v1/chat/` | chat.py | RAG 问答 |
| `POST /api/v1/chat/stream` | chat.py | SSE 流式问答 |
| `GET /api/v1/chat/sessions` | chat.py | 对话历史 |
| `POST /api/v1/evaluations/` | evaluation.py | 创建评测 |
| `GET /api/v1/evaluations/{id}/report` | evaluation.py | 评测报告 |

---

## 第六部分：动手练习（30 分钟）

### 练习 1：在白纸上画全链路（15 分钟）

不要看文档，画两张图：

**图 1：离线管线**
```
PDF → [  ] → [  ] → [  ] → [  ] → 知识库就绪
```
填空：每一步叫什么？用哪个类/文件？

**图 2：在线管线**
```
用户问题 → [  ] → [  ] → [  ] → [  ] → [  ] → 回答
```
填空：每一步叫什么？用哪个类/文件？

画完对照这个文档的第一部分，看漏了什么。

---

### 练习 2：对照源码讲一遍（10 分钟）

打开 `src/services/rag_service.py`，找到 `_single_hop_query` 方法（第 145 行）。

用手指着代码，一行一行读出声，用你自己的话解释每行在干什么。

重点解释：
- 第 155 行为什么 `top_k * 2`？
- 第 156 行返回的 retriever 可能是什么类型？
- 第 162 行 reranker.rerank() 做了什么？

---

### 练习 3：验证理解（5 分钟）

回答下面三个问题。每个问题用一句话：

1. RAG 为什么需要 Embedding？（提示：文字本身没法...）
2. 混合检索比纯向量好在哪？（提示：精确关键词...）
3. 为什么需要重排序？（提示：粗排精度不够...）
4. main.py 里的 try/except（第 55-59 行）是干什么的？
5. _process_document_async 函数里 doc.status=1,2,3 分别代表什么？

---

## 第七部分：验收清单

完成以下全部才算 Day 1 通过：

- [ ] 能不看文档画出离线 4 步 + 在线 5 步的完整流程图
- [ ] 能解释 RAG 同时解决 LLM 的哪三个问题
- [ ] 能解释 Embedding 的原理（用"地图坐标"的类比）
- [ ] 能解释混合检索为什么比纯向量好（举一个具体例子）
- [ ] 能解释 Bi-encoder vs Cross-encoder 的区别和为什么分两步
- [ ] 能独立说出 `_single_hop_query` 的 9 个步骤
- [ ] 能说出 `_process_document_async` 的 6 个步骤
- [ ] 知道 11 个 API 端点对应的 URL 和文件
- [ ] 对着镜子讲一遍 RAG 全流程（3 分钟）

---

## 遇到问题怎么办

| 问题 | 排查步骤 |
|------|---------|
| Demo 卡在"Loading embedding model" | HF 下载超时。设 `$env:HF_ENDPOINT="https://hf-mirror.com"` 或切 `all-MiniLM-L6-v2` |
| 服务器起不来 | 检查端口 8000 是否被占用：`netstat -ano | findstr 8000`，关掉占用的进程 |
| 文档上传后 status=3 | 查 SQLite `SELECT error_msg FROM knowledge_documents WHERE id=N` |
| 检索返回空 | 检查 Chroma 集合是否有数据：`python -c "from src.database.vector_db import get_vector_store; print(get_vector_store().collection.count())"` |
