"""
Day 43 Demo：状态管理
运行方式：python day43_1_state_management.py

学习目标：
1. 理解 Agent 状态的概念
2. 实现状态在工具链中的传递
3. 实现简单的 Checkpoint（中断恢复）
4. 模拟多用户状态隔离
"""

import json
import time
import copy
from datetime import datetime

# ============================================================
# Part 1：Agent 状态定义
# ============================================================

print("=" * 55)
print("Part 1: Agent State Definition")
print("=" * 55)


class AgentState:
    """
    Agent 状态：跟踪执行过程中的所有信息。

    在 LangGraph 中这用 TypedDict 定义，这里用类方便演示。
    """

    def __init__(self, thread_id: str = "default"):
        self.thread_id = thread_id
        self.messages = []          # 对话历史
        self.current_step = "init"  # 当前步骤
        self.data = {}              # 中间数据（如加载的 DataFrame 信息）
        self.results = []           # 累计的分析结果
        self.tool_calls = []        # 工具调用历史
        self.created_at = datetime.now()

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content, "time": str(datetime.now())})

    def add_result(self, step: str, result: str):
        self.results.append({"step": step, "result": result, "time": str(datetime.now())})

    def add_tool_call(self, tool: str, args: dict, result: str):
        self.tool_calls.append({"tool": tool, "args": args, "result": result})

    def summary(self) -> str:
        return (f"Thread: {self.thread_id}\n"
                f"Step: {self.current_step}\n"
                f"Messages: {len(self.messages)}\n"
                f"Results: {len(self.results)}\n"
                f"Tool calls: {len(self.tool_calls)}")

    def to_dict(self) -> dict:
        """序列化为字典（用于 Checkpoint 保存）"""
        return {
            "thread_id": self.thread_id,
            "messages": self.messages,
            "current_step": self.current_step,
            "data": self.data,
            "results": self.results,
            "tool_calls": self.tool_calls,
            "created_at": str(self.created_at),
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentState":
        """从字典恢复状态"""
        state = cls(thread_id=data["thread_id"])
        state.messages = data["messages"]
        state.current_step = data["current_step"]
        state.data = data["data"]
        state.results = data["results"]
        state.tool_calls = data["tool_calls"]
        return state


# 演示状态传递
state = AgentState(thread_id="user_001")
state.add_message("user", "Analyze my sales data")
state.current_step = "loading_data"
state.data = {"file": "sales.csv", "rows": 200, "columns": 5}
state.add_tool_call("read_csv", {"path": "sales.csv"}, "200 rows loaded")
state.current_step = "analyzing"
state.add_result("data_loaded", "200 rows, 5 columns")

print(f"\n  State after loading data:")
print(f"  {state.summary()}")


# ============================================================
# Part 2：状态在工具链中传递
# ============================================================

print("\n" + "=" * 55)
print("Part 2: State Passing Through Tool Chain")
print("=" * 55)


def step_load_data(state: AgentState) -> AgentState:
    """步骤1：加载数据"""
    state.current_step = "load_data"
    state.data["loaded"] = True
    state.data["shape"] = "200 rows x 5 cols"
    state.add_tool_call("read_csv", {"path": "sales.csv"}, "OK")
    state.add_result("load", "Data loaded: 200 rows")
    print(f"    [Step 1] Load data -> OK")
    return state


def step_analyze(state: AgentState) -> AgentState:
    """步骤2：分析数据"""
    state.current_step = "analyze"
    state.data["stats"] = {"mean": 5200, "trend": "upward"}
    state.add_tool_call("analyze", {"type": "describe"}, "mean=5200")
    state.add_result("analyze", "Mean revenue: 5200, trend: upward")
    print(f"    [Step 2] Analyze -> mean=5200, trend=upward")
    return state


def step_chart(state: AgentState) -> AgentState:
    """步骤3：生成图表"""
    state.current_step = "chart"
    state.data["chart_path"] = "trend.png"
    state.add_tool_call("create_chart", {"type": "line"}, "trend.png")
    state.add_result("chart", "Chart saved: trend.png")
    print(f"    [Step 3] Create chart -> trend.png")
    return state


def step_report(state: AgentState) -> AgentState:
    """步骤4：生成报告"""
    state.current_step = "report"
    # 汇总之前所有结果生成报告
    all_findings = "; ".join([r["result"] for r in state.results])
    state.add_result("report", f"Report: {all_findings}")
    print(f"    [Step 4] Generate report -> compiled {len(state.results)} findings")
    state.current_step = "complete"
    return state


# 执行完整工具链
state = AgentState(thread_id="analysis_001")
state.add_message("user", "Analyze sales and create a report with charts")

print(f"\n  Running analysis pipeline:")
pipeline = [step_load_data, step_analyze, step_chart, step_report]

for step_func in pipeline:
    state = step_func(state)

print(f"\n  Final state:")
print(f"  {state.summary()}")
print(f"  Results:")
for r in state.results:
    print(f"    - [{r['step']}] {r['result'][:60]}")


# ============================================================
# Part 3：Checkpoint（中断恢复）
# ============================================================

print("\n" + "=" * 55)
print("Part 3: Checkpoint (Save & Resume)")
print("=" * 55)


class SimpleCheckpointer:
    """
    简单的 Checkpoint 系统：保存和恢复 Agent 状态。

    真实项目中用 LangGraph 的 MemorySaver 或 SqliteSaver。
    这里用内存字典模拟原理。
    """

    def __init__(self):
        self.checkpoints = {}  # thread_id -> state_dict

    def save(self, state: AgentState):
        """保存当前状态"""
        self.checkpoints[state.thread_id] = state.to_dict()
        print(f"    [Checkpoint] Saved state for {state.thread_id} at step '{state.current_step}'")

    def load(self, thread_id: str) -> AgentState | None:
        """恢复状态"""
        if thread_id in self.checkpoints:
            state = AgentState.from_dict(self.checkpoints[thread_id])
            print(f"    [Checkpoint] Restored state for {thread_id} at step '{state.current_step}'")
            return state
        return None

    def list_threads(self) -> list:
        return list(self.checkpoints.keys())


# 模拟中断恢复场景
ckpt = SimpleCheckpointer()

# 开始执行
print("\n  --- First run (will interrupt at step 2) ---")
state = AgentState(thread_id="task_42")
state = step_load_data(state)
ckpt.save(state)  # 保存检查点
state = step_analyze(state)
ckpt.save(state)  # 保存检查点

print("    [!] Simulating interruption (e.g., server restart)...")
del state  # 状态丢失

# 恢复执行
print("\n  --- Resuming from checkpoint ---")
state = ckpt.load("task_42")
print(f"    Resuming from step: {state.current_step}")
print(f"    Previous results: {len(state.results)}")

# 从中断处继续
state = step_chart(state)
state = step_report(state)
print(f"\n    [OK] Task completed after resume!")
print(f"    Total results: {len(state.results)}")


# ============================================================
# Part 4：多用户状态隔离
# ============================================================

print("\n" + "=" * 55)
print("Part 4: Multi-user State Isolation")
print("=" * 55)

ckpt = SimpleCheckpointer()

# 用户 A 的状态
state_a = AgentState(thread_id="user_alice")
state_a.add_message("user", "Analyze product sales")
state_a.data["query"] = "product analysis"
state_a = step_load_data(state_a)
ckpt.save(state_a)

# 用户 B 的状态（完全独立）
state_b = AgentState(thread_id="user_bob")
state_b.add_message("user", "Show regional revenue")
state_b.data["query"] = "regional analysis"
state_b = step_load_data(state_b)
state_b = step_analyze(state_b)
ckpt.save(state_b)

print(f"\n  Active threads: {ckpt.list_threads()}")
print(f"  Alice: step={ckpt.load('user_alice').current_step}, results={len(ckpt.load('user_alice').results)}")
print(f"  Bob:   step={ckpt.load('user_bob').current_step}, results={len(ckpt.load('user_bob').results)}")
print(f"\n  --> Each user has independent state!")


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 55)
print("Day 43 Summary")
print("=" * 55)
print("""
  State Management:
  1. AgentState: track messages, data, results, tool calls
  2. State passes through each step of the pipeline
  3. Checkpoint: save/restore state for interruption recovery
  4. Thread isolation: each user/task has independent state

  In LangGraph:
  - State = TypedDict flowing through graph nodes
  - Checkpoint = MemorySaver or SqliteSaver
  - thread_id in config for multi-user isolation

  Tomorrow: Day 44 - Comprehensive data analysis Agent demo!
""")
