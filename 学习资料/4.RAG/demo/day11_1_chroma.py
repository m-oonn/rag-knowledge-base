"""
Day 11 Demo：Chroma 向量数据库
运行方式：python day11_1_chroma.py
前置条件：pip install chromadb

学习目标：
1. 掌握 Chroma 的增删改查
2. 掌握元数据过滤
3. 掌握持久化存储
4. 构建一个实用的文档检索系统
"""

import chromadb
import os
import shutil

# ============================================================
# Part 1：基础操作 - 增删改查
# ============================================================

print("=" * 55)
print("Part 1: Chroma CRUD")
print("=" * 55)

# 内存模式（数据不持久化，程序退出就没了）
client = chromadb.Client()

# 创建集合（类似数据库的"表"）
collection = client.create_collection(
    name="python_docs",
    metadata={"description": "Python学习文档"},
)

# 批量添加文档
# Chroma 会自动用内置的 Embedding 模型将文档转成向量
docs = [
    "Python是一种解释型、面向对象的高级编程语言，由Guido van Rossum于1991年创造。",
    "FastAPI是一个现代、快速的Web框架，基于Python类型提示，自动生成API文档。",
    "装饰器是Python的一种设计模式，它允许在不修改原函数代码的情况下扩展函数的功能。",
    "async/await是Python的异步编程语法，适合IO密集型场景如网络请求和数据库查询。",
    "Pydantic是一个数据验证库，FastAPI用它来自动验证请求参数的类型和格式。",
    "LangChain是一个用于构建大语言模型应用的框架，支持RAG、Agent等模式。",
    "Chroma是一个轻量级向量数据库，支持相似度检索和元数据过滤。",
    "RAG（检索增强生成）通过检索相关文档来增强大模型的回答质量。",
    "Docker是一个容器化平台，可以将应用及其依赖打包成容器，实现一键部署。",
    "Git是一个分布式版本控制系统，用于跟踪代码变更和团队协作。",
]

metadatas = [
    {"source": "python_basics", "category": "language", "difficulty": "beginner"},
    {"source": "fastapi_docs", "category": "framework", "difficulty": "intermediate"},
    {"source": "python_basics", "category": "language", "difficulty": "intermediate"},
    {"source": "python_basics", "category": "language", "difficulty": "intermediate"},
    {"source": "fastapi_docs", "category": "framework", "difficulty": "beginner"},
    {"source": "ai_docs", "category": "ai", "difficulty": "intermediate"},
    {"source": "ai_docs", "category": "ai", "difficulty": "beginner"},
    {"source": "ai_docs", "category": "ai", "difficulty": "intermediate"},
    {"source": "devops_docs", "category": "devops", "difficulty": "beginner"},
    {"source": "devops_docs", "category": "devops", "difficulty": "beginner"},
]

ids = [f"doc_{i}" for i in range(len(docs))]

collection.add(documents=docs, metadatas=metadatas, ids=ids)
print(f"\n  [OK] Added {collection.count()} documents")

# ============================================================
# Part 2：相似度查询
# ============================================================

print("\n" + "=" * 55)
print("Part 2: Similarity Search")
print("=" * 55)

queries = [
    "怎么学Python编程？",
    "FastAPI怎么做数据验证？",
    "什么是向量数据库？",
    "怎么部署应用？",
]

for q in queries:
    results = collection.query(query_texts=[q], n_results=3)
    print(f"\n  Q: {q}")
    for i, (doc, dist) in enumerate(zip(results["documents"][0], results["distances"][0])):
        print(f"    [{i+1}] (dist={dist:.4f}) {doc[:60]}...")

# ============================================================
# Part 3：元数据过滤
# ============================================================

print("\n" + "=" * 55)
print("Part 3: Metadata Filtering")
print("=" * 55)

# 只搜索 AI 相关文档
print("\n  --- Filter: category=ai ---")
results = collection.query(
    query_texts=["检索增强"],
    n_results=3,
    where={"category": "ai"},
)
for doc in results["documents"][0]:
    print(f"    {doc[:70]}...")

# 只搜索入门难度
print("\n  --- Filter: difficulty=beginner ---")
results = collection.query(
    query_texts=["Python"],
    n_results=3,
    where={"difficulty": "beginner"},
)
for doc in results["documents"][0]:
    print(f"    {doc[:70]}...")

# 组合过滤
print("\n  --- Filter: category=ai AND difficulty=intermediate ---")
results = collection.query(
    query_texts=["AI开发"],
    n_results=3,
    where={"$and": [{"category": "ai"}, {"difficulty": "intermediate"}]},
)
for doc in results["documents"][0]:
    print(f"    {doc[:70]}...")

# 文档内容过滤
print("\n  --- Filter: document contains 'FastAPI' ---")
results = collection.query(
    query_texts=["Web开发"],
    n_results=3,
    where_document={"$contains": "FastAPI"},
)
for doc in results["documents"][0]:
    print(f"    {doc[:70]}...")

# ============================================================
# Part 4：更新和删除
# ============================================================

print("\n" + "=" * 55)
print("Part 4: Update & Delete")
print("=" * 55)

# 更新文档
collection.update(
    ids=["doc_0"],
    documents=["Python是全球最流行的编程语言之一，广泛用于AI、Web开发、数据分析等领域。"],
    metadatas=[{"source": "python_basics", "category": "language", "difficulty": "beginner", "updated": "true"}],
)
print(f"  [OK] Updated doc_0")

# 查看更新后的结果
result = collection.get(ids=["doc_0"])
print(f"  Updated content: {result['documents'][0][:60]}...")

# 删除文档
collection.delete(ids=["doc_9"])
print(f"  [OK] Deleted doc_9, remaining: {collection.count()} docs")

# ============================================================
# Part 5：持久化存储
# ============================================================

print("\n" + "=" * 55)
print("Part 5: Persistent Storage")
print("=" * 55)

persist_dir = "./chroma_test_data"

# 清理旧数据
if os.path.exists(persist_dir):
    shutil.rmtree(persist_dir)

# 创建持久化客户端
persistent_client = chromadb.PersistentClient(path=persist_dir)
persistent_col = persistent_client.create_collection("persistent_docs")

# 添加数据
persistent_col.add(
    ids=["p1", "p2", "p3"],
    documents=[
        "RAG系统需要向量数据库来存储文档的Embedding向量",
        "Chroma支持持久化存储，数据保存在磁盘上",
        "PersistentClient会自动将数据写入指定目录",
    ],
)
print(f"  [OK] Saved {persistent_col.count()} docs to {persist_dir}")

# 模拟重启：重新加载
del persistent_client, persistent_col

reloaded_client = chromadb.PersistentClient(path=persist_dir)
reloaded_col = reloaded_client.get_collection("persistent_docs")
print(f"  [OK] Reloaded: {reloaded_col.count()} docs (data persisted!)")

# 验证数据还在
results = reloaded_col.query(query_texts=["向量数据库"], n_results=2)
for doc in results["documents"][0]:
    print(f"    Found: {doc[:60]}...")

# 清理测试数据
shutil.rmtree(persist_dir)
print(f"  [OK] Cleaned up test data")

# ============================================================
# Part 6：实际场景 - 文档检索系统
# ============================================================

print("\n" + "=" * 55)
print("Part 6: Practical Document Retrieval")
print("=" * 55)


def build_knowledge_base(collection_obj):
    """构建一个模拟的知识库"""
    knowledge = {
        "FastAPI路由": "FastAPI使用装饰器定义路由，支持GET/POST/PUT/DELETE等HTTP方法。@app.get('/users/{user_id}')定义一个GET路由，路径参数user_id会自动转换类型。",
        "FastAPI依赖注入": "FastAPI通过Depends实现依赖注入。定义一个函数返回数据库连接，然后在路由参数中声明db=Depends(get_db)，FastAPI会自动调用并传入。",
        "RAG流程": "RAG的完整流程：1.文档分割成chunks 2.每个chunk用Embedding模型转成向量 3.存入向量数据库 4.用户提问时，将问题也转成向量 5.在向量库中找最相似的chunks 6.将chunks和问题一起发给LLM生成回答。",
        "Prompt工程": "好的Prompt应该包含：角色定义、任务说明、规则约束、输出格式。使用Few-Shot给出示例可以显著提高输出稳定性。temperature参数控制创造性，代码类任务建议设为0-0.3。",
        "异步编程": "Python的async/await让程序在等待IO时去处理其他任务。asyncio.gather()可以并发执行多个协程。FastAPI原生支持异步，用async def定义路由可获得最佳性能。",
    }

    collection_obj.add(
        ids=[f"kb_{i}" for i in range(len(knowledge))],
        documents=list(knowledge.values()),
        metadatas=[{"title": k, "source": "knowledge_base"} for k in knowledge.keys()],
    )
    return len(knowledge)


# 构建知识库
kb_collection = client.create_collection("knowledge_base")
count = build_knowledge_base(kb_collection)
print(f"\n  [OK] Knowledge base built: {count} entries")

# 模拟 RAG 检索
user_questions = [
    "FastAPI怎么定义路由？",
    "RAG系统是怎么工作的？",
    "怎么写好的Prompt？",
]

for question in user_questions:
    results = kb_collection.query(query_texts=[question], n_results=2)
    print(f"\n  User: {question}")
    print(f"  Retrieved:")
    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    )):
        print(f"    [{i+1}] [{meta['title']}] (score={1-dist:.3f})")
        print(f"        {doc[:80]}...")


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 55)
print("Day 11 Summary")
print("=" * 55)
print("""
  1. chromadb.Client() = in-memory (dev/testing)
  2. chromadb.PersistentClient(path=) = on-disk (production)
  3. collection.add() / .query() / .update() / .delete()
  4. query: n_results=K, where={metadata filter}
  5. Chroma auto-embeds documents (default model)
  6. Tomorrow: use sentence-transformers for better embeddings
""")
