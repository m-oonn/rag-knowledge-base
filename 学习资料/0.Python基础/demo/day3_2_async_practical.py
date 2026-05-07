"""
Day 3 Demo 2：异步编程实战
运行方式：python day3_2_async_practical.py

学习目标：
1. 掌握异步超时控制（wait_for）
2. 掌握并发限制（Semaphore）
3. 理解 run_in_executor（同步代码异步化）
4. 模拟真实的 AI 应用异步场景
"""

import asyncio
import time
import random


# ============================================================
# 第一部分：异步超时控制
# ============================================================

print("=" * 50)
print("第一部分：异步超时控制（wait_for）")
print("=" * 50)


async def slow_api_call():
    """模拟一个可能很慢的 API 调用"""
    delay = random.uniform(1, 5)  # 随机 1-5 秒
    print(f"  API 调用中... (将耗时 {delay:.1f}秒)")
    await asyncio.sleep(delay)
    return "API 返回数据"


async def main_timeout():
    """
    场景：调用大模型 API 时设置超时。
    如果 3 秒内没有响应，就放弃。

    在 AI 项目中非常重要：
    - 大模型 API 有时候会很慢
    - 用户不能无限等待
    - 需要设置合理的超时时间
    """
    print("\n--- 尝试调用 API（超时 3 秒）---")

    try:
        # wait_for: 最多等 3 秒
        result = await asyncio.wait_for(slow_api_call(), timeout=3.0)
        print(f"  ✅ 成功: {result}")
    except asyncio.TimeoutError:
        print("  ❌ 超时！API 响应太慢，已放弃")
        print("  → 实际项目中可以：返回缓存结果 / 切换到更快的模型 / 提示用户重试")


# 运行几次看效果（有时成功有时超时）
asyncio.run(main_timeout())
print()


# ============================================================
# 第二部分：并发限制（Semaphore）
# ============================================================

print("=" * 50)
print("第二部分：Semaphore —— 限制并发数量")
print("=" * 50)


async def call_api_with_limit(semaphore, task_id):
    """
    用信号量限制同时进行的 API 调用数量。

    为什么要限制？
    - API 有速率限制（rate limit），比如每秒最多 5 次
    - 同时发起太多请求会被 API 封禁
    - 本地 Ollama 同时处理太多请求会很慢
    """
    async with semaphore:  # 获取信号量（如果已满就等待）
        print(f"  任务{task_id}: 开始调用 API")
        await asyncio.sleep(random.uniform(0.5, 1.5))  # 模拟 API 耗时
        print(f"  任务{task_id}: 完成")
        return f"任务{task_id}的结果"


async def main_semaphore():
    """
    场景：有 10 个文档需要调用 API 处理，但 API 限制同时最多 3 个请求。
    用 Semaphore 控制并发数。
    """
    # 最多同时 3 个并发
    semaphore = asyncio.Semaphore(3)

    print(f"\n--- 10 个任务，最多 3 个同时执行 ---")
    start = time.time()

    tasks = [call_api_with_limit(semaphore, i) for i in range(10)]
    results = await asyncio.gather(*tasks)

    elapsed = time.time() - start
    print(f"\n  全部完成！耗时: {elapsed:.1f}秒")
    print(f"  （如果不限制并发，约1秒；限制3并发，约3-4秒；串行需要10秒）")


asyncio.run(main_semaphore())
print()


# ============================================================
# 第三部分：run_in_executor（同步代码异步化）
# ============================================================

print("=" * 50)
print("第三部分：run_in_executor —— 同步变异步")
print("=" * 50)


def heavy_computation(n):
    """
    模拟 CPU 密集型操作（同步的，没有异步版本）。

    在 AI 项目中的例子：
    - 计算文本的 Embedding 向量
    - 解析 PDF 文件
    - 图片预处理
    这些操作通常没有异步版本，需要用 run_in_executor 包装。
    """
    print(f"  [线程] 开始计算 n={n}...")
    total = sum(i * i for i in range(n))
    time.sleep(1)  # 模拟额外耗时
    print(f"  [线程] 计算完成 n={n}")
    return total


async def main_executor():
    """
    run_in_executor 的作用：
    - 把同步的阻塞函数放到线程池中执行
    - 主事件循环不会被阻塞
    - 其他异步任务可以继续运行
    """
    loop = asyncio.get_event_loop()

    print("\n--- 用 run_in_executor 并行执行同步代码 ---")
    start = time.time()

    # 把同步函数放到线程池中并行执行
    # None 表示使用默认线程池
    result1, result2, result3 = await asyncio.gather(
        loop.run_in_executor(None, heavy_computation, 100000),
        loop.run_in_executor(None, heavy_computation, 200000),
        loop.run_in_executor(None, heavy_computation, 300000),
    )

    elapsed = time.time() - start
    print(f"\n  结果: {result1}, {result2}, {result3}")
    print(f"  耗时: {elapsed:.1f}秒（串行需要 ~3秒，并行更快）")


asyncio.run(main_executor())
print()


# ============================================================
# 第四部分：模拟真实 AI 应用场景
# ============================================================

print("=" * 50)
print("第四部分：模拟 RAG 问答系统的异步流程")
print("=" * 50)


# --- 模拟各个组件 ---

async def embed_query(query: str) -> list:
    """模拟：将用户问题转换为向量"""
    print(f"  [Embedding] 正在向量化: '{query}'")
    await asyncio.sleep(0.3)  # 模拟 Embedding API 调用
    vector = [0.1, 0.2, 0.3]  # 假的向量
    print(f"  [Embedding] 完成")
    return vector


async def search_vector_db(vector: list, top_k: int = 3) -> list:
    """模拟：在向量数据库中检索相关文档"""
    print(f"  [检索] 在向量库中搜索 top_{top_k}...")
    await asyncio.sleep(0.5)  # 模拟数据库查询
    docs = [
        {"content": "Python是一种解释型语言...", "score": 0.95},
        {"content": "Python支持多种编程范式...", "score": 0.87},
        {"content": "Python由Guido创造...", "score": 0.82},
    ]
    print(f"  [检索] 找到 {len(docs)} 个相关文档")
    return docs


async def call_llm(prompt: str) -> str:
    """模拟：调用大语言模型生成回答"""
    print(f"  [LLM] 正在生成回答...")
    await asyncio.sleep(2)  # 模拟大模型响应（通常最慢）
    answer = "Python是一种广泛使用的高级编程语言，由Guido van Rossum于1991年创造。"
    print(f"  [LLM] 回答生成完成")
    return answer


async def save_history(question: str, answer: str):
    """模拟：异步保存对话记录"""
    print(f"  [保存] 正在保存对话记录...")
    await asyncio.sleep(0.2)
    print(f"  [保存] 对话记录已保存")


async def check_cache(query: str) -> str | None:
    """模拟：检查缓存中是否有相同问题的答案"""
    print(f"  [缓存] 检查缓存...")
    await asyncio.sleep(0.1)
    print(f"  [缓存] 未命中")
    return None


# --- 完整的异步问答流程 ---

async def rag_answer(question: str) -> dict:
    """
    完整的 RAG 问答流程（异步版本）。

    这就是你后面项目一要实现的核心逻辑！
    异步的好处：
    1. 检查缓存和 Embedding 可以同时进行
    2. 保存记录放后台，不耽误返回结果给用户
    3. 多个用户同时提问时，不会互相阻塞
    """
    print(f"\n{'─' * 40}")
    print(f"用户提问: {question}")
    print(f"{'─' * 40}")
    start = time.time()

    # 步骤1: 同时检查缓存 + 向量化问题
    cache_result, query_vector = await asyncio.gather(
        check_cache(question),
        embed_query(question),
    )

    # 如果缓存命中直接返回
    if cache_result:
        return {"answer": cache_result, "source": "cache"}

    # 步骤2: 检索相关文档
    docs = await search_vector_db(query_vector)

    # 步骤3: 构建 prompt 并调用 LLM
    context = "\n".join([d["content"] for d in docs])
    prompt = f"根据以下资料回答问题：\n{context}\n\n问题：{question}"
    answer = await call_llm(prompt)

    # 步骤4: 后台保存记录（不等待）
    asyncio.create_task(save_history(question, answer))

    elapsed = time.time() - start
    print(f"\n  📊 总耗时: {elapsed:.2f}秒")

    return {
        "answer": answer,
        "sources": [d["content"][:20] + "..." for d in docs],
        "time": f"{elapsed:.2f}s"
    }


async def main_rag():
    result = await rag_answer("什么是Python？")
    print(f"\n  最终结果:")
    print(f"    回答: {result['answer']}")
    print(f"    来源: {result['sources']}")
    print(f"    耗时: {result['time']}")

    # 等一下让后台的 save_history 完成
    await asyncio.sleep(0.5)


asyncio.run(main_rag())
print()


# ============================================================
# 第五部分：模拟多用户并发
# ============================================================

print("=" * 50)
print("第五部分：模拟多用户同时提问")
print("=" * 50)


async def simple_qa(user, question, delay):
    """简化版问答，模拟不同用户的请求"""
    print(f"  [{user}] 提问: {question}")
    await asyncio.sleep(delay)  # 模拟处理时间
    print(f"  [{user}] 收到回答")
    return f"{user}的回答"


async def main_concurrent_users():
    """
    场景：3个用户同时发送请求。
    异步服务器可以同时处理所有请求。
    """
    start = time.time()

    # 3个用户同时请求
    results = await asyncio.gather(
        simple_qa("用户A", "什么是RAG？", 2),
        simple_qa("用户B", "Python怎么学？", 1.5),
        simple_qa("用户C", "什么是Agent？", 1),
    )

    elapsed = time.time() - start
    print(f"\n  3个用户同时处理完成！")
    print(f"  总耗时: {elapsed:.1f}秒（如果排队需要 4.5秒）")


asyncio.run(main_concurrent_users())
print()


# ============================================================
# 总结
# ============================================================

print("=" * 50)
print("📝 异步编程实战总结")
print("=" * 50)
print("""
1. asyncio.wait_for(coro, timeout) → API 调用超时控制
2. asyncio.Semaphore(n) → 限制并发数量（防止 API 限流）
3. loop.run_in_executor() → 同步代码异步化（PDF解析等）
4. asyncio.create_task() → 后台任务（保存日志等不需要等结果的操作）

实际 AI 项目中的异步模式：
- 缓存检查 + Embedding 同时进行
- 多个 API 并发调用
- 保存记录放后台不阻塞响应
- 超时控制防止无限等待
- 限流控制防止 API 封禁

明天学 FastAPI 时，这些模式会直接用到！
""")
