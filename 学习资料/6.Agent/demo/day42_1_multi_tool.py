"""
Day 42 Demo：多工具协调
运行方式：python day42_1_multi_tool.py

学习目标：
1. 多工具注册和管理
2. Agent 自动选择工具
3. 工具链编排（顺序执行）
4. 错误恢复和重试
"""

import json
import time
import pandas as pd
import numpy as np
import os

# ============================================================
# Part 1：工具注册中心
# ============================================================

print("=" * 55)
print("Part 1: Tool Registry")
print("=" * 55)


class ToolRegistry:
    """
    工具注册中心：管理所有可用工具。
    Agent 通过这个类发现和调用工具。
    """

    def __init__(self):
        self.tools = {}

    def register(self, name: str, description: str, func, parameters: dict = None):
        self.tools[name] = {
            "name": name,
            "description": description,
            "func": func,
            "parameters": parameters or {},
        }

    def list_tools(self) -> str:
        """返回工具列表描述（给 LLM 看的）"""
        lines = []
        for name, tool in self.tools.items():
            lines.append(f"  - {name}: {tool['description']}")
        return "\n".join(lines)

    def call(self, name: str, **kwargs) -> str:
        """调用指定工具"""
        if name not in self.tools:
            return f"[FAIL] Tool '{name}' not found. Available: {list(self.tools.keys())}"
        try:
            result = self.tools[name]["func"](**kwargs)
            return str(result)
        except Exception as e:
            return f"[FAIL] {type(e).__name__}: {e}"


# 创建工具注册中心
registry = ToolRegistry()

# 注册工具
registry.register("read_data", "Read CSV file and return overview",
    lambda file_path: f"Loaded {file_path}: {pd.read_csv(file_path).shape}" if os.path.exists(file_path) else "File not found")

registry.register("analyze", "Analyze data: calculate statistics",
    lambda data_desc: f"Statistics: mean=5200, median=4800, std=1500, trend=upward")

registry.register("create_chart", "Create a chart from data",
    lambda chart_type, title: f"Chart created: {title} ({chart_type}), saved to chart.png")

registry.register("write_report", "Write analysis report",
    lambda findings: f"Report written: {len(findings)} chars of findings documented")

registry.register("calculate", "Calculate math expression",
    lambda expression: str(eval(expression, {"__builtins__": {}}, {"abs": abs, "round": round})))

print(f"\n  Registered tools:\n{registry.list_tools()}")


# ============================================================
# Part 2：工具链编排
# ============================================================

print("\n" + "=" * 55)
print("Part 2: Tool Chain Orchestration")
print("=" * 55)


def run_tool_chain(registry: ToolRegistry, chain: list[dict]) -> list[dict]:
    """
    按顺序执行一系列工具调用。

    chain 格式: [{"tool": "name", "args": {...}}, ...]
    返回每步的结果。
    """
    results = []
    for i, step in enumerate(chain):
        tool_name = step["tool"]
        args = step.get("args", {})
        print(f"\n  Step {i+1}: {tool_name}({args})")

        result = registry.call(tool_name, **args)
        results.append({"step": i+1, "tool": tool_name, "result": result})

        if "[FAIL]" in result:
            print(f"    [FAIL] {result}")
            print(f"    Chain stopped at step {i+1}")
            break
        else:
            print(f"    [OK] {result[:80]}")

    return results


# 模拟数据分析工具链
analysis_chain = [
    {"tool": "analyze", "args": {"data_desc": "sales data"}},
    {"tool": "create_chart", "args": {"chart_type": "line", "title": "Monthly Revenue Trend"}},
    {"tool": "create_chart", "args": {"chart_type": "bar", "title": "Product Comparison"}},
    {"tool": "write_report", "args": {"findings": "Revenue shows upward trend. Product A leads with 35% market share."}},
]

print("\n  Running analysis chain:")
results = run_tool_chain(registry, analysis_chain)
print(f"\n  Chain completed: {len(results)} steps executed")


# ============================================================
# Part 3：智能工具选择（模拟 LLM 决策）
# ============================================================

print("\n" + "=" * 55)
print("Part 3: Smart Tool Selection")
print("=" * 55)


def select_tools(user_request: str) -> list[dict]:
    """
    根据用户需求自动选择工具链。
    真实项目中这个决策由 LLM 做。
    """
    request_lower = user_request.lower()

    chain = []

    # 判断是否需要读数据
    if any(kw in request_lower for kw in ["data", "csv", "file", "数据", "文件"]):
        chain.append({"tool": "read_data", "args": {"file_path": "sales.csv"}})

    # 判断是否需要分析
    if any(kw in request_lower for kw in ["analyze", "statistics", "trend", "分析", "统计", "趋势"]):
        chain.append({"tool": "analyze", "args": {"data_desc": "loaded data"}})

    # 判断是否需要图表
    if any(kw in request_lower for kw in ["chart", "plot", "graph", "图", "图表", "可视化"]):
        chart_type = "line" if "trend" in request_lower or "趋势" in request_lower else "bar"
        chain.append({"tool": "create_chart", "args": {"chart_type": chart_type, "title": "Analysis Chart"}})

    # 判断是否需要报告
    if any(kw in request_lower for kw in ["report", "summary", "报告", "总结"]):
        chain.append({"tool": "write_report", "args": {"findings": "Analysis results compiled."}})

    # 判断是否是计算
    if any(kw in request_lower for kw in ["calculate", "计算", "多少"]):
        chain.append({"tool": "calculate", "args": {"expression": "100 * 1.15"}})

    if not chain:
        chain.append({"tool": "analyze", "args": {"data_desc": "general query"}})

    return chain


# 测试智能选择
test_requests = [
    "Analyze the sales data and show me a trend chart",
    "Give me a summary report of the statistics",
    "Calculate 15% growth on base 100",
    "Load the CSV file and create a bar chart",
]

for req in test_requests:
    chain = select_tools(req)
    tools_used = [s["tool"] for s in chain]
    print(f"\n  Request: {req}")
    print(f"  Selected tools: {' -> '.join(tools_used)}")


# ============================================================
# Part 4：错误恢复
# ============================================================

print("\n" + "=" * 55)
print("Part 4: Error Recovery")
print("=" * 55)


def run_with_recovery(registry: ToolRegistry, chain: list[dict], max_retries: int = 2) -> list:
    """带错误恢复的工具链执行"""
    results = []
    for i, step in enumerate(chain):
        tool_name = step["tool"]
        args = step.get("args", {})
        success = False

        for attempt in range(max_retries + 1):
            result = registry.call(tool_name, **args)

            if "[FAIL]" not in result:
                print(f"  Step {i+1}: {tool_name} -> [OK]")
                results.append(result)
                success = True
                break
            else:
                print(f"  Step {i+1}: {tool_name} -> [FAIL] (attempt {attempt+1})")
                if attempt < max_retries:
                    # 模拟修复策略
                    if "not found" in result.lower():
                        args["file_path"] = "backup_" + args.get("file_path", "data.csv")
                        print(f"           Trying alternative: {args}")

        if not success:
            print(f"  Step {i+1}: Giving up after {max_retries + 1} attempts")
            results.append(f"FAILED: {result}")

    return results


# 测试错误恢复
error_chain = [
    {"tool": "read_data", "args": {"file_path": "nonexistent.csv"}},  # 会失败
    {"tool": "analyze", "args": {"data_desc": "data"}},
]

print("\n  Running chain with recovery:")
run_with_recovery(registry, error_chain)


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 55)
print("Day 42 Summary")
print("=" * 55)
print("""
  Multi-tool coordination:
  1. ToolRegistry: register, discover, and call tools
  2. Tool chains: sequential execution of tool steps
  3. Smart selection: map user intent to tool chain
  4. Error recovery: retry with modified args

  Tomorrow: Day 43 - State management (checkpoints)
""")
