"""
Day 13 Demo：LangChain RAG 流水线
运行方式：python day13_1_langchain_rag.py
前置条件：pip install langchain langchain-community chromadb sentence-transformers

学习目标：
1. 用 LangChain 组件搭建完整 RAG 流水线
2. 理解 Document Loader → Splitter → Embedding → VectorStore → Retriever
3. 构建 RetrievalQA 并测试
"""

import os
import tempfile

# ============================================================
# Part 1：准备示例文档
# ============================================================

print("=" * 55)
print("Part 1: Create Sample Documents")
print("=" * 55)

# 创建临时目录存放示例文档
doc_dir = tempfile.mkdtemp(prefix="rag_docs_")

sample_docs = {
    "fastapi_guide.md": """# FastAPI 开发指南

## 什么是 FastAPI
FastAPI 是一个现代、快速的 Python Web 框架，用于构建 API。它基于 Python 类型提示，使用 Pydantic 进行数据验证，自动生成 Swagger 文档。

## 路由定义
使用装饰器定义路由：@app.get("/path") 定义 GET 请求，@app.post("/path") 定义 POST 请求。路径参数用花括号：/users/{user_id}。

## 依赖注入
FastAPI 的 Depends 机制实现依赖注入。常用于数据库连接管理：定义 get_db 函数用 yield 返回连接，在路由中通过 db=Depends(get_db) 获取。

## 数据验证
Pydantic BaseModel 定义数据模型。FastAPI 自动验证请求体的 JSON 数据，类型不匹配返回 422 错误。Field() 可以添加额外验证规则。

## 异步支持
FastAPI 原生支持 async/await。IO 密集型路由用 async def，CPU 密集型用普通 def。FastAPI 会自动处理线程池。
""",
    "rag_intro.md": """# RAG 技术介绍

## RAG 原理
RAG（Retrieval-Augmented Generation）检索增强生成，通过从知识库中检索相关文档来增强大模型的回答质量。解决了大模型知识过时和幻觉问题。

## 核心流程
1. 文档预处理：将 PDF/Word/Markdown 文档解析为纯文本
2. 文本分割：将长文本切分成 500-1000 字符的小段（chunk）
3. 向量化：用 Embedding 模型将每个 chunk 转成向量
4. 存储：将向量存入 Chroma 等向量数据库
5. 检索：用户提问时，将问题也向量化，找最相似的 chunk
6. 生成：将检索到的 chunk 作为上下文，和问题一起发给 LLM

## 优化方向
混合检索：结合 BM25 关键词检索和向量语义检索。重排序（Reranker）：用交叉编码器对检索结果重新排序。查询改写：优化用户问题以提高检索效果。
""",
    "python_async.md": """# Python 异步编程

## 为什么需要异步
Web 应用大量时间在等待：等 API 响应、等数据库查询、等文件读写。异步让程序在等待时去处理其他请求。

## 核心语法
async def 定义协程函数，await 在等待时让出控制权。asyncio.run() 启动事件循环。asyncio.gather() 并发执行多个协程。

## 在 FastAPI 中使用
FastAPI 路由支持 async def。调用异步数据库、异步 HTTP 客户端（httpx）时用 async def。纯 CPU 计算用普通 def，FastAPI 自动放到线程池。

## 常见错误
不要在 async def 中用 time.sleep()（会阻塞事件循环），要用 asyncio.sleep()。不要忘记 await，否则得到的是协程对象而不是结果。
""",
}

for filename, content in sample_docs.items():
    filepath = os.path.join(doc_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  Created: {filename} ({len(content)} chars)")

print(f"  Doc directory: {doc_dir}")
print()

# ============================================================
# Part 2：加载文档
# ============================================================

print("=" * 55)
print("Part 2: Load Documents")
print("=" * 55)

from langchain_community.document_loaders import TextLoader, DirectoryLoader

# 方法1：单个文件
loader = TextLoader(os.path.join(doc_dir, "fastapi_guide.md"), encoding="utf-8")
single_doc = loader.load()
print(f"\n  Single file: {len(single_doc)} document(s)")
print(f"  Content preview: {single_doc[0].page_content[:80]}...")
print(f"  Metadata: {single_doc[0].metadata}")

# 方法2：整个目录
dir_loader = DirectoryLoader(doc_dir, glob="**/*.md", loader_cls=TextLoader, loader_kwargs={"encoding": "utf-8"})
all_docs = dir_loader.load()
print(f"\n  Directory load: {len(all_docs)} document(s)")
for doc in all_docs:
    src = os.path.basename(doc.metadata.get("source", "unknown"))
    print(f"    {src}: {len(doc.page_content)} chars")

print()

# ============================================================
# Part 3：文本分割
# ============================================================

print("=" * 55)
print("Part 3: Text Splitting")
print("=" * 55)

from langchain_text_splitters import RecursiveCharacterTextSplitter

# RecursiveCharacterTextSplitter 的分割优先级：
# 先尝试按 \n\n 分 → 不行按 \n 分 → 不行按空格分 → 最后按字符分
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,       # 每个 chunk 最大 300 字符
    chunk_overlap=50,     # 相邻 chunk 重叠 50 字符（防止切断上下文）
    length_function=len,
    separators=["\n\n", "\n", "。", " ", ""],  # 中文优化：加上句号作为分隔符
)

chunks = splitter.split_documents(all_docs)
print(f"\n  Original documents: {len(all_docs)}")
print(f"  After splitting: {len(chunks)} chunks")
print(f"  Chunk size range: {min(len(c.page_content) for c in chunks)}-{max(len(c.page_content) for c in chunks)} chars")

print(f"\n  First 5 chunks:")
for i, chunk in enumerate(chunks[:5]):
    src = os.path.basename(chunk.metadata.get("source", ""))
    print(f"    [{i}] ({len(chunk.page_content)} chars) [{src}] {chunk.page_content[:60]}...")

print()

# ============================================================
# Part 4：创建向量库
# ============================================================

print("=" * 55)
print("Part 4: Create Vector Store")
print("=" * 55)

import chromadb

# 尝试加载 sentence-transformers Embedding
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("\n  [OK] Using sentence-transformers embedding")
except Exception:
    # 如果装不了，用 Chroma 默认的
    embeddings = None
    print("\n  [WARN] Using Chroma default embedding")

from langchain_community.vectorstores import Chroma

# 从文档创建向量库（自动 Embedding + 存储）
if embeddings:
    vectorstore = Chroma.from_documents(chunks, embeddings)
else:
    vectorstore = Chroma.from_documents(chunks, collection_name="rag_demo")

print(f"  [OK] Vector store created with {len(chunks)} chunks")

# 创建检索器
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print(f"  [OK] Retriever ready (top-3)")
print()

# ============================================================
# Part 5：检索测试
# ============================================================

print("=" * 55)
print("Part 5: Retrieval Test")
print("=" * 55)

test_queries = [
    "FastAPI怎么做数据验证？",
    "RAG的核心流程是什么？",
    "异步编程有哪些常见错误？",
    "怎么做依赖注入？",
]

for query in test_queries:
    docs = retriever.invoke(query)
    print(f"\n  Q: {query}")
    for i, doc in enumerate(docs):
        src = os.path.basename(doc.metadata.get("source", ""))
        print(f"    [{i+1}] [{src}] {doc.page_content[:80]}...")

print()

# ============================================================
# Part 6：完整 RAG Chain
# ============================================================

print("=" * 55)
print("Part 6: Complete RAG Chain")
print("=" * 55)

# 尝试连接 LLM
llm_available = False
try:
    from langchain_community.chat_models import ChatOpenAI
    import requests
    # 检查 Ollama
    requests.get("http://localhost:11434/api/tags", timeout=2)
    llm = ChatOpenAI(
        base_url="http://localhost:11434/v1",
        api_key="ollama",
        model="qwen2.5:7b",
        temperature=0.3,
    )
    llm_available = True
    print("\n  [OK] LLM: Ollama")
except Exception:
    try:
        from dotenv import load_dotenv
        load_dotenv()
        ds_key = os.getenv("DEEPSEEK_API_KEY")
        if ds_key:
            llm = ChatOpenAI(
                base_url="https://api.deepseek.com",
                api_key=ds_key,
                model="deepseek-chat",
                temperature=0.3,
            )
            llm_available = True
            print("\n  [OK] LLM: DeepSeek")
    except Exception:
        pass

if llm_available:
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough

    # RAG Prompt 模板
    rag_prompt = ChatPromptTemplate.from_template("""你是一个知识库问答助手。根据以下参考文档回答用户问题。

参考文档：
{context}

规则：
1. 只根据参考文档回答，不要用你自己的知识
2. 如果文档中没有相关信息，说"知识库中未找到相关信息"
3. 用中文回答，简洁清晰

用户问题：{question}""")

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    # 构建 RAG Chain（LCEL 语法）
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    # 测试完整 RAG
    rag_questions = [
        "FastAPI的依赖注入是怎么实现的？",
        "RAG系统有哪些优化方向？",
        "为什么不能在异步函数里用time.sleep？",
    ]

    for q in rag_questions:
        print(f"\n  Q: {q}")
        answer = rag_chain.invoke(q)
        print(f"  A: {answer[:200]}")

else:
    print("\n  [SKIP] No LLM available (Ollama or DeepSeek)")
    print("  The retrieval part above still works!")
    print("  To test full RAG: start Ollama or set DEEPSEEK_API_KEY")

# 清理临时文件
import shutil
shutil.rmtree(doc_dir)

print("\n" + "=" * 55)
print("Day 13 Summary")
print("=" * 55)
print("""
  LangChain RAG Pipeline:
  1. DirectoryLoader -> load all .md files
  2. RecursiveCharacterTextSplitter -> split into chunks
  3. Chroma.from_documents -> embed + store
  4. vectorstore.as_retriever -> create retriever
  5. RAG Chain: retriever | prompt | llm | parser

  This is the EXACT pipeline you'll build in Project 1!
  Next: Day 14 (Document Parsing) + Day 15 (RAG Optimization)
""")
