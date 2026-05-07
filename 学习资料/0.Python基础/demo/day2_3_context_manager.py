"""
Day 2 Demo 3：上下文管理器（Context Manager）
运行方式：python day2_3_context_manager.py

学习目标：
1. 理解 with 语句的执行流程
2. 会用类方式和生成器方式自定义上下文管理器
3. 理解为什么需要上下文管理器（资源安全管理）
4. 了解在 FastAPI/AI 项目中的实际应用
"""

import time
import os
import tempfile
from contextlib import contextmanager


# ============================================================
# 第一部分：with 语句基础
# ============================================================

print("=" * 50)
print("第一部分：with 语句基础")
print("=" * 50)

# 创建一个临时文件用于演示
test_file = "test_context.txt"

# --- 不用 with（不安全的写法）---
print("\n--- 不用 with（不安全）---")
f = open(test_file, "w")
f.write("Hello, World!\n")
f.close()  # 必须手动关闭，如果上面出错了，这行不会执行
print("手动关闭文件: 成功")

# --- 用 with（安全的写法）---
print("\n--- 用 with（安全）---")
with open(test_file, "w") as f:
    f.write("Hello, Context Manager!\n")
    # 退出 with 块时，文件自动关闭
    print(f"文件是否关闭: {f.closed}")  # False，还在 with 块内

print(f"退出 with 后，文件是否关闭: {f.closed}")  # True，自动关闭了

# 读取验证
with open(test_file, "r") as f:
    content = f.read()
    print(f"文件内容: {content.strip()}")

# 清理测试文件
os.remove(test_file)
print()


# ============================================================
# 第二部分：自定义上下文管理器（类方式）
# ============================================================

print("=" * 50)
print("第二部分：类方式上下文管理器")
print("=" * 50)


class Timer:
    """
    计时器上下文管理器。

    原理：
    - __enter__: 进入 with 块时调用，返回值赋给 as 后面的变量
    - __exit__: 退出 with 块时调用（无论是否出错都会调用）

    用途：
    - 测量代码执行时间
    - AI 项目中测量 API 调用耗时
    """

    def __init__(self, label="代码"):
        self.label = label
        self.elapsed = 0

    def __enter__(self):
        """
        进入 with 块时执行。
        返回 self，这样 with Timer() as t 中的 t 就是 Timer 实例。
        """
        self.start = time.time()
        print(f"[Timer] ⏱ 开始计时: {self.label}")
        return self  # 返回值会赋给 as 后面的变量

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        退出 with 块时执行（无论是否出错）。

        参数说明：
        - exc_type: 异常类型（没有异常时为 None）
        - exc_val: 异常值
        - exc_tb: 异常的 traceback

        返回值：
        - True: 吞掉异常，不再抛出
        - False: 让异常继续传播（通常用 False）
        """
        self.elapsed = time.time() - self.start
        print(f"[Timer] ⏱ {self.label} 耗时: {self.elapsed:.4f}秒")

        if exc_type is not None:
            print(f"[Timer] ⚠ 发生了异常: {exc_val}")

        return False  # 不吞掉异常


# 正常使用
print()
with Timer("列表求和") as t:
    total = sum(range(1_000_000))

print(f"结果: {total}, 耗时: {t.elapsed:.4f}秒")

# 嵌套使用
print()
with Timer("外层操作"):
    with Timer("内层操作1"):
        _ = [i ** 2 for i in range(100_000)]
    with Timer("内层操作2"):
        _ = sorted(range(100_000), reverse=True)

print()


# ============================================================
# 第三部分：自定义上下文管理器（生成器方式，更简洁）
# ============================================================

print("=" * 50)
print("第三部分：生成器方式上下文管理器")
print("=" * 50)


@contextmanager
def timer(label="代码"):
    """
    用 @contextmanager 装饰器创建上下文管理器。

    比类方式简洁得多！
    - yield 之前的代码 = __enter__（进入时执行）
    - yield 的值 = as 后面的变量
    - yield 之后的代码 = __exit__（退出时执行）

    记忆：yield 把函数劈成两半 —— 上半进入，下半退出。
    """
    start = time.time()
    print(f"[timer] ⏱ 开始: {label}")
    try:
        yield  # 把控制权交给 with 块内的代码
    finally:
        # finally 确保即使出错也会执行（和 __exit__ 一样）
        elapsed = time.time() - start
        print(f"[timer] ⏱ {label} 耗时: {elapsed:.4f}秒")


print()
with timer("字典推导"):
    data = {str(i): i ** 2 for i in range(100_000)}

print()


# ============================================================
# 第四部分：实用上下文管理器示例
# ============================================================

print("=" * 50)
print("第四部分：实用示例")
print("=" * 50)


# --- 示例1：临时工作目录 ---
@contextmanager
def working_directory(path):
    """
    临时切换工作目录。

    用途：需要在某个目录下执行操作，完成后自动回到原目录。
    无论中间是否出错，都能保证回到原目录。
    """
    original = os.getcwd()
    try:
        os.chdir(path)
        print(f"  切换到: {os.getcwd()}")
        yield
    finally:
        os.chdir(original)
        print(f"  回到: {os.getcwd()}")


print("\n--- 临时切换目录 ---")
print(f"当前目录: {os.getcwd()}")
with working_directory(tempfile.gettempdir()):
    print(f"  临时目录中的文件数: {len(os.listdir('.'))}")
print(f"恢复目录: {os.getcwd()}")


# --- 示例2：数据库连接模拟 ---
@contextmanager
def database_connection(db_name):
    """
    模拟数据库连接管理。

    在 FastAPI 项目中，数据库连接管理是上下文管理器最重要的用途：
    - 请求开始时打开连接
    - 请求结束时关闭连接
    - 出错时回滚事务

    FastAPI 的 Depends + yield 就是利用了这个机制。
    """
    print(f"  📂 打开数据库连接: {db_name}")
    connection = {"db": db_name, "connected": True}  # 模拟连接对象
    try:
        yield connection
    except Exception as e:
        print(f"  ↩️ 回滚事务: {e}")
    finally:
        connection["connected"] = False
        print(f"  📁 关闭数据库连接: {db_name}")


print("\n--- 数据库连接（正常）---")
with database_connection("ai_app.db") as db:
    print(f"  查询数据... (连接状态: {db['connected']})")
    result = "查询到 10 条记录"
    print(f"  {result}")

print(f"  连接状态: {db['connected']}")  # False，已关闭


print("\n--- 数据库连接（出错）---")
try:
    with database_connection("ai_app.db") as db:
        print(f"  插入数据...")
        raise ValueError("数据格式错误！")  # 模拟错误
except ValueError:
    print("  外部捕获了异常")

print()


# --- 示例3：API 调用追踪器（AI 项目常用）---
@contextmanager
def api_tracker(api_name):
    """
    API 调用追踪器。

    在 AI 项目中跟踪每次大模型 API 的调用：
    - 记录调用开始时间
    - 记录 token 使用量
    - 记录调用结果
    """
    stats = {
        "api": api_name,
        "start_time": time.time(),
        "tokens_used": 0,
        "success": False
    }
    print(f"\n  📡 调用 {api_name}...")
    try:
        yield stats  # 把 stats 字典交给 with 块，让它填充数据
        stats["success"] = True
    except Exception as e:
        stats["error"] = str(e)
        raise
    finally:
        stats["duration"] = time.time() - stats["start_time"]
        status = "✅" if stats["success"] else "❌"
        print(f"  {status} {api_name}: {stats['duration']:.3f}秒, "
              f"tokens={stats['tokens_used']}")


print("--- API 调用追踪 ---")
with api_tracker("Claude API") as stats:
    # 模拟 API 调用
    time.sleep(0.1)
    stats["tokens_used"] = 150  # 记录 token 使用量

print()


# ============================================================
# 第五部分：with 的高级用法
# ============================================================

print("=" * 50)
print("第五部分：高级用法")
print("=" * 50)

# --- 同时管理多个资源 ---
print("\n--- 同时打开多个文件 ---")

# 创建两个测试文件
with open("input.txt", "w") as f:
    f.write("Hello from input!")

# 同时打开多个文件（Python 3.10+ 写法）
with open("input.txt", "r") as fin, open("output.txt", "w") as fout:
    content = fin.read()
    fout.write(f"处理后: {content}")
    print(f"  读取: {content}")
    print(f"  写入: 处理后: {content}")

# 清理
os.remove("input.txt")
os.remove("output.txt")
print()


# ============================================================
# 第六部分：FastAPI 中的上下文管理器（预览）
# ============================================================

print("=" * 50)
print("第六部分：FastAPI 中的用法（预览）")
print("=" * 50)

print("""
在 FastAPI 中，上下文管理器通过 Depends + yield 实现：

```python
from fastapi import FastAPI, Depends

app = FastAPI()

# 用 yield 的函数就是上下文管理器
def get_db():
    db = SessionLocal()       # 打开连接（__enter__）
    try:
        yield db              # 把连接交给路由函数使用
    finally:
        db.close()            # 关闭连接（__exit__）

@app.get("/users")
def get_users(db = Depends(get_db)):
    # db 由上下文管理器自动管理
    # 路由执行完毕后，db.close() 自动调用
    return db.query(User).all()
```

核心思想：
- 每个请求自动获得一个数据库连接
- 请求结束后自动关闭连接
- 出错时也能保证连接关闭
- 不需要手动管理连接生命周期
""")


# ============================================================
# 总结
# ============================================================

print("=" * 50)
print("📝 上下文管理器总结")
print("=" * 50)
print("""
1. with 语句 = 自动调用 __enter__ 和 __exit__
2. 确保资源"用完必定清理"，即使出错也不会泄漏
3. 两种自定义方式：
   - 类方式：定义 __enter__ 和 __exit__ 方法
   - 生成器方式：@contextmanager + yield（更简洁）
4. yield 把函数劈成两半：上半进入，下半退出
5. FastAPI 的 Depends + yield 就是上下文管理器的应用
6. 常见场景：文件操作、数据库连接、API 会话、计时器
""")
