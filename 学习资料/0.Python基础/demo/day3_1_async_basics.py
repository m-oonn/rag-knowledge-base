"""
Day 3 Demo 1：async/await 基础
运行方式：python day3_1_async_basics.py

学习目标：
1. 理解 async def 和普通 def 的区别
2. 理解 await 的作用
3. 学会用 asyncio.run() 运行协程
"""

import asyncio
import time


# ============================================================
# 第一部分：普通函数 vs 异步函数
# ============================================================

print("=" * 50)
print("第一部分：普通函数 vs 异步函数")
print("=" * 50)


# 普通函数：直接调用就能执行
def normal_greet(name):
    return f"你好，{name}！"


# 异步函数：加了 async 关键字
async def async_greet(name):
    return f"你好，{name}！"


# 普通函数直接调用
result1 = normal_greet("张三")
print(f"普通函数结果: {result1}")

# 异步函数直接调用 → 得到的是协程对象，不是结果！
result2 = async_greet("张三")
print(f"异步函数直接调用: {result2}")
print(f"类型: {type(result2)}")
# ↑ 输出类似: <coroutine object async_greet at 0x...>
# 说明：异步函数需要通过 asyncio.run() 或 await 来执行

# 正确运行异步函数
result3 = asyncio.run(async_greet("张三"))
print(f"通过 asyncio.run() 运行: {result3}")
print()


# ============================================================
# 第二部分：await 的作用
# ============================================================

print("=" * 50)
print("第二部分：await —— 等待时让出控制权")
print("=" * 50)


async def make_coffee():
    """
    模拟做咖啡的过程。

    await asyncio.sleep(n) 模拟一个耗时的 IO 操作（如网络请求）。
    在 sleep 期间，CPU 可以去处理其他协程。
    """
    print("  1. 磨咖啡豆...")
    await asyncio.sleep(1)  # 模拟耗时操作，让出控制权 1 秒
    print("  2. 冲泡咖啡...")
    await asyncio.sleep(2)  # 再等 2 秒
    print("  3. 咖啡完成！")
    return "一杯热美式"


async def main_coffee():
    print("\n开始做咖啡:")
    start = time.time()
    coffee = await make_coffee()
    elapsed = time.time() - start
    print(f"  得到: {coffee}")
    print(f"  总耗时: {elapsed:.1f}秒")


asyncio.run(main_coffee())
print()


# ============================================================
# 第三部分：同步 vs 异步 —— 性能对比
# ============================================================

print("=" * 50)
print("第三部分：同步 vs 异步的性能差距")
print("=" * 50)


# --- 同步版本：一个一个排队 ---
def sync_call_api(name, seconds):
    """同步版本的 API 调用模拟"""
    print(f"  [同步] {name}: 开始请求...")
    time.sleep(seconds)  # 同步等待，会阻塞！
    print(f"  [同步] {name}: 完成！")
    return f"{name}的结果"


print("\n--- 同步方式（一个一个来）---")
start = time.time()

# 同步：必须一个一个来
r1 = sync_call_api("查数据库", 1)
r2 = sync_call_api("调大模型", 2)
r3 = sync_call_api("存记录", 0.5)

elapsed = time.time() - start
print(f"  同步总耗时: {elapsed:.1f}秒（1+2+0.5=3.5秒）")


# --- 异步版本：同时执行 ---
async def async_call_api(name, seconds):
    """异步版本的 API 调用模拟"""
    print(f"  [异步] {name}: 开始请求...")
    await asyncio.sleep(seconds)  # 异步等待，不阻塞！
    print(f"  [异步] {name}: 完成！")
    return f"{name}的结果"


async def main_async_demo():
    print("\n--- 异步方式（同时执行）---")
    start = time.time()

    # 异步：用 gather 同时发起所有请求
    results = await asyncio.gather(
        async_call_api("查数据库", 1),
        async_call_api("调大模型", 2),
        async_call_api("存记录", 0.5),
    )

    elapsed = time.time() - start
    print(f"  异步总耗时: {elapsed:.1f}秒（取最慢的=2秒）")
    print(f"  结果: {results}")


asyncio.run(main_async_demo())
print()


# ============================================================
# 第四部分：asyncio.gather 详解
# ============================================================

print("=" * 50)
print("第四部分：asyncio.gather —— 并发利器")
print("=" * 50)


async def fetch_user(user_id):
    """模拟从数据库获取用户"""
    await asyncio.sleep(0.5)
    users = {1: "张三", 2: "李四", 3: "王五"}
    return users.get(user_id, "未知用户")


async def fetch_orders(user_id):
    """模拟获取用户订单"""
    await asyncio.sleep(0.8)
    return [f"订单{user_id}01", f"订单{user_id}02"]


async def fetch_balance(user_id):
    """模拟查询余额"""
    await asyncio.sleep(0.3)
    return 1000 + user_id * 100


async def main_gather():
    """
    场景：用户打开个人中心页面，需要同时获取：
    - 用户信息
    - 订单列表
    - 账户余额

    同步需要: 0.5 + 0.8 + 0.3 = 1.6秒
    异步只需: max(0.5, 0.8, 0.3) = 0.8秒
    """
    user_id = 1
    start = time.time()

    # gather 同时执行三个查询
    user, orders, balance = await asyncio.gather(
        fetch_user(user_id),
        fetch_orders(user_id),
        fetch_balance(user_id),
    )

    elapsed = time.time() - start
    print(f"\n  用户: {user}")
    print(f"  订单: {orders}")
    print(f"  余额: {balance}")
    print(f"  耗时: {elapsed:.2f}秒（三个请求同时进行）")


asyncio.run(main_gather())
print()


# ============================================================
# 第五部分：asyncio.create_task —— 后台任务
# ============================================================

print("=" * 50)
print("第五部分：create_task —— 后台任务")
print("=" * 50)


async def background_save(data):
    """模拟后台保存数据（不需要等结果）"""
    await asyncio.sleep(1)
    print(f"  [后台] 数据已保存: {data}")


async def main_task():
    """
    场景：用户发送消息后，需要保存记录，但不需要等保存完成再回复用户。
    用 create_task 把保存操作放到后台。
    """
    print("\n  用户发来消息...")

    # 创建后台任务（立即开始执行，但不等待结果）
    task = asyncio.create_task(background_save("用户的消息"))

    # 立即处理其他事情（生成回复）
    print("  立即生成回复（不等待保存完成）")
    await asyncio.sleep(0.1)  # 模拟生成回复
    print("  回复已发送给用户！")

    # 确保后台任务完成（程序退出前要等）
    await task
    print("  所有任务完成")


asyncio.run(main_task())
print()


# ============================================================
# 第六部分：常见错误
# ============================================================

print("=" * 50)
print("第六部分：常见错误与注意事项")
print("=" * 50)

# --- 错误1：在异步函数里用 time.sleep ---
print("\n--- 错误示范：time.sleep vs asyncio.sleep ---")


async def bad_sleep():
    """❌ 错误：time.sleep 会阻塞整个事件循环"""
    start = time.time()

    async def task(name, seconds):
        print(f"  {name} 开始")
        time.sleep(seconds)  # ❌ 阻塞！其他协程无法运行
        print(f"  {name} 结束")

    # 即使用 gather，time.sleep 也会让它们变成串行
    await asyncio.gather(task("A", 1), task("B", 1))
    print(f"  耗时: {time.time() - start:.1f}秒（本应1秒，实际2秒）")


async def good_sleep():
    """✅ 正确：asyncio.sleep 不阻塞"""
    start = time.time()

    async def task(name, seconds):
        print(f"  {name} 开始")
        await asyncio.sleep(seconds)  # ✅ 不阻塞，让出控制权
        print(f"  {name} 结束")

    await asyncio.gather(task("A", 1), task("B", 1))
    print(f"  耗时: {time.time() - start:.1f}秒（真正并发，只要1秒）")


print("\n用 time.sleep（错误）:")
asyncio.run(bad_sleep())

print("\n用 asyncio.sleep（正确）:")
asyncio.run(good_sleep())


# --- 错误2：忘记 await ---
print("\n\n--- 错误示范：忘记 await ---")


async def get_data():
    await asyncio.sleep(0.1)
    return "重要数据"


async def demo_forget_await():
    # ❌ 忘了 await
    result_bad = get_data()
    print(f"  忘了 await: {result_bad}")
    print(f"  类型: {type(result_bad)}")
    # 关闭这个没被 await 的协程，避免警告
    result_bad.close()

    # ✅ 正确写法
    result_good = await get_data()
    print(f"  正确 await: {result_good}")
    print(f"  类型: {type(result_good)}")


asyncio.run(demo_forget_await())
print()


# ============================================================
# 总结
# ============================================================

print("=" * 50)
print("📝 异步编程基础总结")
print("=" * 50)
print("""
1. async def 定义异步函数（协程）
2. await 在等待时让出控制权（只能在 async def 内用）
3. asyncio.run() 是运行协程的入口
4. asyncio.gather() 并发执行多个协程
5. asyncio.create_task() 创建后台任务
6. 用 asyncio.sleep() 而不是 time.sleep()
7. 别忘了 await！

明天学 FastAPI 时，这些都会用到：
  async def endpoint() + await db.query() + await llm.call()
""")
