"""
Day 2 Demo 1：装饰器（Decorator）
运行方式：python day2_1_decorator.py

学习目标：
1. 理解装饰器的本质（函数接收函数，返回函数）
2. 会写基础装饰器和带参数的装饰器
3. 知道 @wraps 的作用
4. 了解实际应用场景
"""

import time
from functools import wraps


# ============================================================
# 第一部分：最简单的装饰器
# ============================================================

def simple_decorator(func):
    """
    最简单的装饰器：在函数执行前后打印日志。

    原理：
    - 接收一个函数 func 作为参数
    - 定义一个新函数 wrapper，在里面调用原函数
    - 返回这个新函数
    """
    def wrapper(*args, **kwargs):
        # *args, **kwargs 让装饰器能处理任意参数的函数
        print(f"[LOG] 准备调用函数: {func.__name__}")
        result = func(*args, **kwargs)  # 调用原始函数
        print(f"[LOG] 函数 {func.__name__} 执行完毕")
        return result
    return wrapper


# @ 语法糖：等价于 greet = simple_decorator(greet)
@simple_decorator
def greet(name):
    """打招呼函数"""
    print(f"你好，{name}！")
    return f"Hello, {name}"


print("=" * 50)
print("第一部分：最简单的装饰器")
print("=" * 50)

# 调用 greet 时，实际执行的是 wrapper
result = greet("张三")
print(f"返回值: {result}")
print()


# ============================================================
# 第二部分：实用装饰器 —— 计时器
# ============================================================

def timer(func):
    """
    计时装饰器：测量函数执行时间。

    实际用途：
    - 性能分析，找出慢函数
    - AI 项目中经常用来测量 API 调用耗时
    """
    @wraps(func)  # 保留原函数的名字和文档字符串
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        print(f"[TIMER] {func.__name__} 耗时: {elapsed:.4f}秒")
        return result
    return wrapper


@timer
def slow_function():
    """模拟一个耗时操作"""
    total = sum(range(1_000_000))  # 计算 0 到 99万 的和
    return total


print("=" * 50)
print("第二部分：计时装饰器")
print("=" * 50)

result = slow_function()
print(f"计算结果: {result}")
print()


# ============================================================
# 第三部分：@wraps 的重要性
# ============================================================

print("=" * 50)
print("第三部分：@wraps 的作用")
print("=" * 50)

# slow_function 用了 @wraps，所以保留了原函数信息
print(f"slow_function 的名字: {slow_function.__name__}")
print(f"slow_function 的文档: {slow_function.__doc__}")

# greet 没用 @wraps，函数信息丢失了
print(f"greet 的名字: {greet.__name__}")  # 会显示 "wrapper" 而不是 "greet"
print()


# ============================================================
# 第四部分：带参数的装饰器
# ============================================================

def retry(max_attempts=3, delay=1):
    """
    重试装饰器：函数执行失败时自动重试。

    三层嵌套结构：
    - 最外层 retry(max_attempts, delay): 接收装饰器自身的参数
    - 中间层 decorator(func): 接收被装饰的函数
    - 最内层 wrapper(*args, **kwargs): 实际执行逻辑

    实际用途：
    - API 调用失败重试（网络不稳定时）
    - 数据库连接重试
    - AI 项目中调用大模型 API 经常需要重试
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    print(f"[RETRY] {func.__name__} 第{attempt}次失败: {e}")
                    if attempt < max_attempts:
                        print(f"[RETRY] {delay}秒后重试...")
                        time.sleep(delay)
            # 所有重试都失败了
            raise last_exception
        return wrapper
    return decorator


# 模拟一个不稳定的函数（前两次失败，第三次成功）
call_count = 0

@retry(max_attempts=3, delay=0.5)
def unstable_api_call():
    """模拟不稳定的 API 调用"""
    global call_count
    call_count += 1
    if call_count < 3:
        raise ConnectionError("网络连接失败")
    return "API 调用成功！"


print("=" * 50)
print("第四部分：带参数的装饰器（重试）")
print("=" * 50)

result = unstable_api_call()
print(f"最终结果: {result}")
print()


# ============================================================
# 第五部分：多个装饰器叠加
# ============================================================

def bold(func):
    """给返回值加粗（模拟 HTML）"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        return f"<b>{func(*args, **kwargs)}</b>"
    return wrapper


def italic(func):
    """给返回值加斜体"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        return f"<i>{func(*args, **kwargs)}</i>"
    return wrapper


# 多个装饰器从下往上执行：先 italic，再 bold
# 等价于: formatted_text = bold(italic(formatted_text))
@bold
@italic
def formatted_text(text):
    return text


print("=" * 50)
print("第五部分：多个装饰器叠加")
print("=" * 50)

result = formatted_text("Hello World")
print(f"结果: {result}")
# 输出: <b><i>Hello World</i></b>
# 执行顺序：先 italic 包了一层 <i>，再 bold 包了一层 <b>
print()


# ============================================================
# 第六部分：实战 —— API 调用日志装饰器
# ============================================================

def log_api_call(func):
    """
    API 调用日志装饰器。

    这个装饰器在你后面做 AI 项目时非常有用：
    - 记录每次 API 调用的参数
    - 记录调用耗时
    - 记录是否成功
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # 记录调用开始
        print(f"\n{'─' * 40}")
        print(f"📡 调用 API: {func.__name__}")
        print(f"   参数: args={args}, kwargs={kwargs}")

        start = time.time()
        try:
            result = func(*args, **kwargs)
            elapsed = time.time() - start
            print(f"   ✅ 成功 | 耗时: {elapsed:.3f}秒")
            print(f"   返回: {result[:50]}..." if len(str(result)) > 50 else f"   返回: {result}")
            return result
        except Exception as e:
            elapsed = time.time() - start
            print(f"   ❌ 失败 | 耗时: {elapsed:.3f}秒 | 错误: {e}")
            raise
    return wrapper


@log_api_call
def call_llm(prompt, model="qwen2.5:7b"):
    """模拟调用大语言模型 API"""
    time.sleep(0.1)  # 模拟网络延迟
    return f"AI 回复: 你问的是「{prompt}」，这是一个很好的问题..."


print("=" * 50)
print("第六部分：API 调用日志装饰器（实战）")
print("=" * 50)

response = call_llm("什么是装饰器？", model="claude-3")
print(f"\n收到回复: {response}")


# ============================================================
# 总结
# ============================================================

print("\n" + "=" * 50)
print("📝 装饰器总结")
print("=" * 50)
print("""
1. 装饰器 = 接收函数，返回新函数
2. @decorator 是语法糖，等价于 func = decorator(func)
3. 用 *args, **kwargs 让装饰器适配任意参数
4. 用 @wraps 保留原函数信息
5. 带参数的装饰器需要三层嵌套
6. 多个装饰器从下往上执行
7. FastAPI 的 @app.get() 就是装饰器
""")
