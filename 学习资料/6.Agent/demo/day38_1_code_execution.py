"""
Day 38 Demo：代码执行工具
运行方式：python day38_1_code_execution.py

学习目标：
1. 实现安全的代码执行沙箱
2. 捕获 stdout、返回值、错误
3. 执行 pandas/matplotlib 代码
4. 实现错误自修复循环
"""

import io
import contextlib
import traceback
import time

# ============================================================
# Part 1：基础代码执行
# ============================================================

print("=" * 55)
print("Part 1: Basic Code Execution")
print("=" * 55)


def execute_code(code: str, allowed_modules=None) -> dict:
    """
    安全执行 Python 代码。

    返回 dict:
      - output: stdout 输出
      - result: 最后一个表达式的值（如果有）
      - error: 错误信息（如果有）
      - success: 是否成功
    """
    # 构建安全的全局变量
    safe_globals = {
        "__builtins__": {
            "print": print, "len": len, "range": range, "enumerate": enumerate,
            "zip": zip, "map": map, "filter": filter, "sorted": sorted,
            "int": int, "float": float, "str": str, "bool": bool,
            "list": list, "dict": dict, "tuple": tuple, "set": set,
            "min": min, "max": max, "sum": sum, "abs": abs, "round": round,
            "isinstance": isinstance, "type": type, "True": True, "False": False, "None": None,
        }
    }

    # 允许的模块
    if allowed_modules is None:
        allowed_modules = ["pandas", "numpy", "math", "json", "datetime"]

    for mod_name in allowed_modules:
        try:
            safe_globals[mod_name.split(".")[-1]] = __import__(mod_name)
        except ImportError:
            pass

    # 别名
    if "pandas" in str(safe_globals):
        import pandas
        safe_globals["pd"] = pandas
    if "numpy" in str(safe_globals):
        import numpy
        safe_globals["np"] = numpy

    # 捕获 stdout
    stdout_capture = io.StringIO()
    result = {"output": "", "result": None, "error": None, "success": False}

    try:
        with contextlib.redirect_stdout(stdout_capture):
            exec(code, safe_globals)
        result["output"] = stdout_capture.getvalue()
        result["success"] = True
    except Exception as e:
        result["output"] = stdout_capture.getvalue()
        result["error"] = f"{type(e).__name__}: {e}"
        result["success"] = False

    return result


# 测试基础执行
test_codes = [
    # 正常代码
    'print("Hello from sandbox!")\nprint(2 + 3)',
    # 使用允许的模块
    'import math\nprint(f"Pi = {math.pi:.4f}")',
    # 列表操作
    'data = [3, 1, 4, 1, 5, 9]\nprint(f"Sorted: {sorted(data)}")\nprint(f"Sum: {sum(data)}")',
    # 错误代码
    'x = 1 / 0',
    # 尝试危险操作（被阻止）
    'import os\nos.system("echo hacked!")',
]

for i, code in enumerate(test_codes):
    result = execute_code(code)
    status = "[OK]  " if result["success"] else "[FAIL]"
    print(f"\n  {status} Test {i+1}:")
    print(f"    Code: {code[:60]}...")
    if result["output"]:
        print(f"    Output: {result['output'].strip()[:80]}")
    if result["error"]:
        print(f"    Error: {result['error'][:80]}")

# ============================================================
# Part 2：执行 pandas 数据分析
# ============================================================

print("\n" + "=" * 55)
print("Part 2: Execute pandas Analysis")
print("=" * 55)

pandas_code = '''
import pandas as pd
import numpy as np

# 创建示例销售数据
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=12, freq='ME')
data = {
    'date': dates,
    'revenue': np.random.randint(40000, 70000, 12),
    'orders': np.random.randint(100, 300, 12),
}
df = pd.DataFrame(data)

print("=== Sales Data ===")
print(df.to_string(index=False))
print()
print(f"Total revenue: {df['revenue'].sum():,}")
print(f"Average monthly revenue: {df['revenue'].mean():,.0f}")
print(f"Best month: {df.loc[df['revenue'].idxmax(), 'date'].strftime('%Y-%m')}")
print(f"Revenue trend: {'increasing' if df['revenue'].iloc[-1] > df['revenue'].iloc[0] else 'decreasing'}")
'''

result = execute_code(pandas_code)
print(f"\n  pandas code execution:")
print(f"  Success: {result['success']}")
if result["output"]:
    for line in result["output"].strip().split("\n"):
        print(f"    {line}")

# ============================================================
# Part 3：执行 matplotlib 图表生成
# ============================================================

print("\n" + "=" * 55)
print("Part 3: Execute Chart Generation")
print("=" * 55)

chart_code = '''
import pandas as pd
import numpy as np

# 创建数据
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun']
revenue = [45000, 52000, 48000, 61000, 58000, 65000]

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(months, revenue, color='steelblue')
    ax.set_title('Monthly Revenue 2024')
    ax.set_ylabel('Revenue')
    plt.tight_layout()
    plt.savefig('agent_chart.png', dpi=100)
    plt.close()
    print("Chart saved to: agent_chart.png")
except ImportError:
    print("matplotlib not available, skipping chart")

# 文字分析结果（即使没有 matplotlib 也能输出）
print(f"Revenue summary:")
print(f"  Min: {min(revenue):,} ({months[revenue.index(min(revenue))]})")
print(f"  Max: {max(revenue):,} ({months[revenue.index(max(revenue))]})")
print(f"  Trend: {'Upward' if revenue[-1] > revenue[0] else 'Downward'}")
'''

result = execute_code(chart_code, allowed_modules=["pandas", "numpy", "matplotlib", "matplotlib.pyplot"])
print(f"\n  Chart code execution: {'[OK]' if result['success'] else '[FAIL]'}")
if result["output"]:
    for line in result["output"].strip().split("\n"):
        print(f"    {line}")

# 清理生成的图片
import os
if os.path.exists("agent_chart.png"):
    os.remove("agent_chart.png")
    print("    (chart file cleaned up)")

# ============================================================
# Part 4：错误自修复循环
# ============================================================

print("\n" + "=" * 55)
print("Part 4: Auto-Fix Loop (Error Self-Repair)")
print("=" * 55)


def execute_with_retry(code: str, max_retries: int = 3) -> dict:
    """
    执行代码，失败时自动修复重试。

    真实项目中：
    1. 执行代码
    2. 如果失败，把错误信息发给 LLM
    3. LLM 修改代码
    4. 重新执行
    5. 最多重试 max_retries 次

    这里用规则模拟 LLM 的修复过程。
    """
    for attempt in range(max_retries + 1):
        result = execute_code(code)

        if result["success"]:
            print(f"    Attempt {attempt + 1}: [OK]")
            return result

        print(f"    Attempt {attempt + 1}: [FAIL] {result['error']}")

        if attempt < max_retries:
            # 模拟 LLM 修复（真实项目：把错误发给 LLM 让它改）
            code = auto_fix(code, result["error"])
            print(f"    Auto-fixing...")

    return result


def auto_fix(code: str, error: str) -> str:
    """模拟 LLM 修复代码（规则版）"""
    if "ZeroDivisionError" in error:
        return code.replace("/ 0", "/ 1  # fixed: avoid division by zero")
    if "NameError" in error and "pd" in error:
        return "import pandas as pd\n" + code
    if "KeyError" in error:
        return code.replace("['nonexistent']", "['revenue']  # fixed: correct column name")
    return code  # 无法修复


# 测试自修复
print("\n  --- Test 1: Division by zero ---")
execute_with_retry("result = 100 / 0\nprint(f'Result: {result}')")

print("\n  --- Test 2: Missing import ---")
execute_with_retry("df = pd.DataFrame({'a': [1,2,3]})\nprint(df)")

print()


# ============================================================
# Part 5：完整的 Agent 代码执行流程
# ============================================================

print("=" * 55)
print("Part 5: Full Agent Code Execution Flow")
print("=" * 55)

def agent_code_flow(user_request: str):
    """
    完整流程：用户需求 → 生成代码 → 执行 → 返回结果。

    这就是你项目二"数据分析Agent"的核心模式！
    """
    print(f"\n  User: {user_request}")

    # Step 1: LLM 理解需求并生成代码（这里模拟）
    print("  [Agent] Understanding request...")
    if "average" in user_request.lower() or "平均" in user_request:
        generated_code = '''
import pandas as pd
import numpy as np
np.random.seed(42)
df = pd.DataFrame({
    'month': range(1, 13),
    'revenue': np.random.randint(40000, 70000, 12)
})
avg = df['revenue'].mean()
print(f"Average monthly revenue: {avg:,.0f}")
print(f"Months above average: {(df['revenue'] > avg).sum()}")
'''
    else:
        generated_code = '''
print("Analysis: The data shows a positive trend.")
print("Key finding: Revenue increased by 15% over the period.")
'''

    print(f"  [Agent] Generated code ({len(generated_code.strip().splitlines())} lines)")

    # Step 2: 执行代码
    print("  [Agent] Executing code...")
    result = execute_with_retry(generated_code)

    # Step 3: 解读结果
    if result["success"]:
        print(f"  [Agent] Results:")
        for line in result["output"].strip().split("\n"):
            print(f"    {line}")
        print(f"  [Agent] Analysis complete!")
    else:
        print(f"  [Agent] Execution failed after retries: {result['error']}")


agent_code_flow("Calculate the average monthly revenue")
agent_code_flow("Show me the data trend summary")


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 55)
print("Day 38 Summary")
print("=" * 55)
print("""
  Code Execution Tool:
  1. exec() in sandbox with restricted globals
  2. Capture stdout + handle errors
  3. Allow only safe modules (pandas, numpy, math)
  4. Auto-fix loop: error -> LLM fixes code -> retry
  5. Full flow: user request -> generate code -> execute -> results

  This is the CORE of your Data Analysis Agent (Project 2)!
  Tomorrow: Day 39 - pandas data analysis tools
""")
