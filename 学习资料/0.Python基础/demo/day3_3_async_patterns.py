"""
Day 3 Demo 3：异步编程模式 + FastAPI 预览
运行方式：python day3_3_async_patterns.py

学习目标：
1. 掌握异步迭代器（async for）
2. 掌握异步上下文管理器（async with）
3. 理解异步生成器（流式输出的基础！）
4. 预览 FastAPI 中的异步写法
"""

import asyncio
import time


# ============================================================
# 第一部分：异步上下文管理器（async with）
# ============================================================

print("=" * 50)
print("第一部分：async with —— 异步资源管理")
print("=" * 50)


class AsyncDBConnection:
    """
    模拟异步数据库连接。

    和昨天学的上下文管理器一样，只是方法变成了异步的：
    - __enter__  → __aenter__   (加了 a = async)
    - __exit__   → __aexit__
    - with       → async with

    用途：
    - 异步数据库连接（FastAPI 中的 SQLAlchemy async session）
    - 异步 HTTP 客户端（httpx.AsyncClient）
    - 异步文件操作（aiofiles）
    """

    def __init__(self, db_name):
        self.db_name = db_name

    async def __aenter__(self):
        """异步进入：建立连接"""
        print(f"  📂 正在连接数据库 {self.db_name}...")
        await asyncio.sleep(0.3)  # 模拟连接耗时
        print(f"  ✅ 数据库连接成功")
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """异步退出：关闭连接"""
        print(f"  📁 正在关闭数据库连接...")
        await asyncio.sleep(0.1)  # 模拟关闭耗时
        print(f"  ✅ 数据库连接已关闭")
        return False

    async def query(self, sql):
        """模拟异步查询"""
        await asyncio.sleep(0.2)
        return [{"id": 1, "name": "张三"}, {"id": 2, "name": "李四"}]


async def main_async_with():
    # 用 async with 自动管理数据库连接生命周期
    async with AsyncDBConnection("ai_app.db") as db:
        results = await db.query("SELECT * FROM users")
        print(f"  查询结果: {results}")
    # 退出 async with 后，连接自动关闭


asyncio.run(main_async_with())
print()


# ============================================================
# 第二部分：异步生成器（流式输出的基础！）
# ============================================================

print("=" * 50)
print("第二部分：异步生成器 —— 流式输出基础")
print("=" * 50)


async def stream_llm_response(prompt: str):
    """
    模拟大模型的流式输出。

    这是你后面做项目必须掌握的核心概念！

    大模型不是一次性返回所有文字，而是一个字一个字（token）地生成。
    用异步生成器可以实现"边生成边返回"：
    - 用户看到回答是逐渐出现的（像 ChatGPT 那样）
    - 不需要等全部生成完才显示
    """
    response = "Python是一种解释型、面向对象的高级编程语言，广泛用于AI开发。"

    # yield 逐字返回（模拟 token 流式输出）
    for char in response:
        await asyncio.sleep(0.05)  # 模拟每个 token 的生成时间
        yield char  # async def + yield = 异步生成器


async def main_stream():
    print("\n  模拟大模型流式输出:")
    print("  AI: ", end="", flush=True)

    # async for 遍历异步生成器
    full_response = ""
    async for token in stream_llm_response("什么是Python？"):
        print(token, end="", flush=True)  # 逐字打印
        full_response += token

    print()  # 换行
    print(f"\n  完整回答（{len(full_response)}字）: {full_response}")


asyncio.run(main_stream())
print()


# ============================================================
# 第三部分：异步迭代器（async for）
# ============================================================

print("=" * 50)
print("第三部分：async for —— 异步迭代处理")
print("=" * 50)


async def fetch_documents_batch(batch_size=3):
    """
    模拟分批获取文档。

    场景：RAG 系统中有大量文档需要处理，
    不能一次性全部加载到内存，需要分批异步获取。
    """
    all_docs = [
        "Python基础教程.pdf",
        "FastAPI文档.md",
        "LangChain指南.pdf",
        "RAG原理.md",
        "向量数据库介绍.pdf",
        "Agent开发.md",
        "Prompt工程.pdf",
        "Docker入门.md",
    ]

    for i in range(0, len(all_docs), batch_size):
        batch = all_docs[i:i + batch_size]
        await asyncio.sleep(0.5)  # 模拟获取一批文档的耗时
        yield batch  # 每次返回一批


async def main_async_for():
    print("\n  分批处理文档:")
    batch_num = 0

    async for doc_batch in fetch_documents_batch(3):
        batch_num += 1
        print(f"  第{batch_num}批: {doc_batch}")
        # 处理这一批文档...
        for doc in doc_batch:
            # 模拟处理每个文档
            pass
        print(f"  第{batch_num}批处理完成")

    print(f"\n  共处理 {batch_num} 批文档")


asyncio.run(main_async_for())
print()


# ============================================================
# 第四部分：异步队列（生产者-消费者模式）
# ============================================================

print("=" * 50)
print("第四部分：异步队列 —— 生产者消费者模式")
print("=" * 50)


async def document_producer(queue: asyncio.Queue):
    """
    生产者：模拟用户不断上传文档。

    场景：RAG 系统中，用户上传文档，系统在后台异步处理。
    """
    documents = ["报告1.pdf", "手册2.docx", "笔记3.md", "论文4.pdf", "教程5.md"]

    for doc in documents:
        await asyncio.sleep(random.uniform(0.2, 0.5))
        await queue.put(doc)
        print(f"  [上传] {doc} 已加入队列（队列大小: {queue.qsize()}）")

    # 放入结束信号
    await queue.put(None)


async def document_consumer(queue: asyncio.Queue, worker_id: int):
    """
    消费者：从队列中取出文档并处理。
    """
    while True:
        doc = await queue.get()

        if doc is None:
            # 收到结束信号，把它放回去让其他消费者也能看到
            await queue.put(None)
            break

        print(f"  [Worker-{worker_id}] 处理: {doc}")
        await asyncio.sleep(random.uniform(0.3, 0.8))  # 模拟处理耗时
        print(f"  [Worker-{worker_id}] 完成: {doc}")
        queue.task_done()


import random


async def main_queue():
    """
    2 个 worker 同时处理用户上传的文档。
    比单 worker 快，但不会无限创建连接。
    """
    queue = asyncio.Queue(maxsize=5)  # 最多缓存 5 个

    print("\n--- 2 个 Worker 并行处理文档队列 ---\n")
    start = time.time()

    # 同时运行生产者和两个消费者
    await asyncio.gather(
        document_producer(queue),
        document_consumer(queue, 1),
        document_consumer(queue, 2),
    )

    elapsed = time.time() - start
    print(f"\n  全部处理完成！耗时: {elapsed:.1f}秒")


asyncio.run(main_queue())
print()


# ============================================================
# 第五部分：FastAPI 异步写法预览
# ============================================================

print("=" * 50)
print("第五部分：FastAPI 异步写法预览（明天要用！）")
print("=" * 50)

print("""
以下是 FastAPI 中你会写的典型异步代码，看看就好，明天详细学：

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import httpx

app = FastAPI()

# ==========================================
# 1. 异步路由 —— 最基本的用法
# ==========================================
@app.get("/users/{user_id}")
async def get_user(user_id: int):
    # 异步查询数据库
    user = await db.fetch_one("SELECT * FROM users WHERE id = :id", {"id": user_id})
    return user

# ==========================================
# 2. 异步依赖注入 —— 数据库连接管理
# ==========================================
async def get_db():
    db = AsyncSession()
    try:
        yield db          # async with 的生成器版本
    finally:
        await db.close()  # 异步关闭

@app.get("/items")
async def get_items(db = Depends(get_db)):
    return await db.execute("SELECT * FROM items")

# ==========================================
# 3. 流式输出 —— 大模型回答逐字返回
# ==========================================
async def generate_stream(question: str):
    async for token in llm.stream(question):  # 异步生成器！
        yield f"data: {token}\\n\\n"          # SSE 格式

@app.get("/chat/stream")
async def chat_stream(question: str):
    return StreamingResponse(
        generate_stream(question),
        media_type="text/event-stream"
    )

# ==========================================
# 4. 并发调用多个服务
# ==========================================
@app.get("/dashboard")
async def dashboard():
    # 同时获取多个数据源
    user, orders, stats = await asyncio.gather(
        get_user_info(),
        get_recent_orders(),
        get_statistics(),
    )
    return {"user": user, "orders": orders, "stats": stats}
```

看到了吗？今天学的所有东西都会在 FastAPI 中用到：
- async def → 异步路由
- await → 异步数据库查询、API调用
- async for → 流式输出
- async with / yield → 资源管理
- asyncio.gather → 并发查询
""")


# ============================================================
# 总结
# ============================================================

print("=" * 50)
print("📝 Day 3 完整总结")
print("=" * 50)
print("""
今天学了异步编程的三大块：

【基础篇】
  - async def 定义协程，await 让出控制权
  - asyncio.run() 运行，asyncio.gather() 并发
  - asyncio.create_task() 后台任务

【实战篇】
  - wait_for() 超时控制
  - Semaphore 并发限制
  - run_in_executor 同步→异步

【高级篇】
  - async with 异步资源管理
  - async for + 异步生成器 = 流式输出
  - 异步队列 = 生产者消费者模式

明天学 FastAPI 时，所有这些都会真正用起来！
""")
