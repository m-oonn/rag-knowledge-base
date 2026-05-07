"""
Day 36 Demo 1: LangGraph 进阶
运行方式: python day36_1_langgraph_advanced.py

前置条件:
  pip install langgraph langchain-core langchain-openai
  可选（LLM 提供者，任选一个）:
    - Ollama 运行中（免费，推荐调试）
    - 或 .env 中配置 DEEPSEEK_API_KEY

学习目标:
1. Agent 循环：LLM -> should_call_tool? -> tool -> LLM（循环）
2. 工具调用 Agent：用 LangGraph 实现 ReAct 模式
3. Human-in-the-loop：图暂停等待用户确认
4. 错误处理：节点失败时路由到 error_handler
5. 用多个查询测试 Agent（有些需要工具，有些不需要）
"""

import os
import sys
import json
import math
from typing import TypedDict, Annotated
from datetime import datetime

# === 检查核心依赖 ===
try:
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages
except ImportError:
    print("[FAIL] langgraph 未安装")
    print("  请运行: pip install langgraph langchain-core")
    sys.exit(1)

try:
    from langchain_core.messages import (
        HumanMessage, AIMessage, ToolMessage, SystemMessage
    )
    from langchain_core.tools import tool
except ImportError:
    print("[FAIL] langchain-core 未安装")
    print("  请运行: pip install langchain-core")
    sys.exit(1)


print("=" * 60)
print("  Day 36: LangGraph 进阶")
print("=" * 60)
print()


# === 自动检测 LLM（Ollama > DeepSeek > 模拟） ===
def get_llm():
    """
    按优先级尝试：Ollama -> DeepSeek -> 模拟模式
    返回: (llm_object_or_None, mode_string)
    """
    # 1. 尝试 Ollama
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=2)
        models = [m["name"] for m in resp.json().get("models", [])]
        if models:
            from langchain_openai import ChatOpenAI
            model_name = models[0]
            llm = ChatOpenAI(
                model=model_name,
                base_url="http://localhost:11434/v1",
                api_key="ollama",
                temperature=0,
            )
            print(f"  [OK] LLM: Ollama ({model_name})")
            return llm, "ollama"
    except Exception:
        pass

    # 2. 尝试 DeepSeek
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    ds_key = os.getenv("DEEPSEEK_API_KEY")
    if ds_key:
        try:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model="deepseek-chat",
                base_url="https://api.deepseek.com",
                api_key=ds_key,
                temperature=0,
            )
            print("  [OK] LLM: DeepSeek")
            return llm, "deepseek"
        except Exception:
            pass

    # 3. 模拟模式
    print("  [WARN] 没有可用的 LLM，使用模拟模式")
    print("         (部分功能用预设回复演示)")
    return None, "simulated"


llm, llm_mode = get_llm()
print()


# ============================================================
# Part 1: Agent 循环 —— 工具调用 Agent（核心）
# ============================================================
print("=" * 60)
print("  Part 1: Tool-Calling Agent (ReAct Loop)")
print("=" * 60)

# --- 定义工具 ---

@tool
def calculator(expression: str) -> str:
    """计算数学表达式。支持加减乘除、幂运算、开方等。
    Args:
        expression: 数学表达式，如 '2+3*4' 或 'sqrt(16)'
    """
    try:
        # 安全的数学计算（只允许数学函数）
        allowed = {
            "sqrt": math.sqrt, "abs": abs, "pow": pow,
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "log": math.log, "pi": math.pi, "e": math.e,
        }
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


@tool
def get_current_time() -> str:
    """获取当前日期和时间。不需要任何参数。"""
    now = datetime.now()
    return f"当前时间: {now.strftime('%Y年%m月%d日 %H:%M:%S')} (星期{['一','二','三','四','五','六','日'][now.weekday()]})"


@tool
def search_knowledge(query: str) -> str:
    """在知识库中搜索信息。可以查询 Python、AI、LangGraph 相关知识。
    Args:
        query: 搜索关键词
    """
    # 模拟知识库
    knowledge_base = {
        "python": "Python 是一种高级编程语言，以简洁的语法和丰富的库著称。广泛用于 Web 开发、数据分析、AI 等领域。",
        "langgraph": "LangGraph 是基于图结构的 AI 工作流框架，支持有状态、可循环的 Agent 构建。核心概念包括 StateGraph、Node、Edge。",
        "rag": "RAG（检索增强生成）是一种先检索知识库文档再生成回答的技术，解决了 LLM 知识过时和幻觉问题。",
        "agent": "AI Agent 是能自主使用工具完成任务的智能体。ReAct 模式是最常用的 Agent 架构：思考 -> 行动 -> 观察 -> 循环。",
        "fastapi": "FastAPI 是现代 Python Web 框架，支持异步、自动文档生成、类型验证。基于 Starlette 和 Pydantic。",
    }

    query_lower = query.lower()
    results = []
    for key, value in knowledge_base.items():
        if key in query_lower or query_lower in key:
            results.append(value)

    if results:
        return "知识库查询结果:\n" + "\n".join(results)
    else:
        return f"知识库中未找到与 '{query}' 相关的信息。"


# 所有工具列表
all_tools = [calculator, get_current_time, search_knowledge]
tool_map = {t.name: t for t in all_tools}


# --- 定义 Agent State ---
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# --- 定义节点 ---

def llm_node(state: AgentState) -> dict:
    """
    LLM 节点：调用 LLM，让它决定是否调工具。
    如果使用模拟模式，则根据消息内容返回预设回复。
    """
    messages = state["messages"]

    if llm is not None:
        # 真实 LLM 调用：绑定工具后调用
        llm_with_tools = llm.bind_tools(all_tools)
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}
    else:
        # 模拟模式：根据最后一条消息内容决定是否"调用工具"
        last_msg = messages[-1]
        content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

        # 检查是否有来自工具的返回（说明已经调用过工具了）
        if isinstance(last_msg, ToolMessage):
            # 工具已返回结果，直接总结
            tool_result = last_msg.content
            response = AIMessage(content=f"根据工具返回的信息：{tool_result}")
            return {"messages": [response]}

        # 判断是否需要调工具
        if any(w in content for w in ["计算", "算", "+", "-", "*", "/", "加", "减", "乘", "除"]):
            # 模拟 tool_call：调 calculator
            expr = "2+3*4"  # 模拟解析出的表达式
            for word in ["计算", "算一下", "算"]:
                if word in content:
                    # 尝试提取表达式
                    idx = content.find(word) + len(word)
                    expr = content[idx:].strip().strip("。？?！!") or "2+3*4"
                    break
            response = AIMessage(
                content="",
                tool_calls=[{
                    "id": "sim_call_1",
                    "name": "calculator",
                    "args": {"expression": expr},
                }]
            )
            return {"messages": [response]}

        elif any(w in content for w in ["时间", "几点", "日期", "今天"]):
            response = AIMessage(
                content="",
                tool_calls=[{
                    "id": "sim_call_2",
                    "name": "get_current_time",
                    "args": {},
                }]
            )
            return {"messages": [response]}

        elif any(w in content for w in ["什么是", "查一下", "搜索", "知识"]):
            query = content.replace("什么是", "").replace("查一下", "").replace("？", "").strip()
            response = AIMessage(
                content="",
                tool_calls=[{
                    "id": "sim_call_3",
                    "name": "search_knowledge",
                    "args": {"query": query or "python"},
                }]
            )
            return {"messages": [response]}

        else:
            # 不需要工具，直接回复
            response = AIMessage(content=f"你好！你说的是：'{content}'。有什么我可以帮助你的吗？")
            return {"messages": [response]}


def tool_node(state: AgentState) -> dict:
    """
    工具执行节点：解析 AI 消息中的 tool_calls，执行对应工具。
    相当于 langgraph.prebuilt.ToolNode 的手写版本，方便理解原理。
    """
    messages = state["messages"]
    last_msg = messages[-1]

    # AI 消息中可能包含多个 tool_call
    tool_calls = getattr(last_msg, "tool_calls", [])
    results = []

    for tc in tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]
        tool_id = tc.get("id", "unknown")

        print(f"    [Tool] Calling: {tool_name}({tool_args})")

        # 查找并执行工具
        if tool_name in tool_map:
            try:
                result = tool_map[tool_name].invoke(tool_args)
            except Exception as e:
                result = f"工具执行失败: {e}"
        else:
            result = f"未知工具: {tool_name}"

        print(f"    [Tool] Result:  {result[:80]}...")

        # 返回 ToolMessage（包含 tool_call_id，这样 LLM 知道这是哪个调用的结果）
        results.append(ToolMessage(content=str(result), tool_call_id=tool_id))

    return {"messages": results}


def should_continue(state: AgentState) -> str:
    """
    条件边路由函数：
    检查最后一条 AI 消息是否包含 tool_calls
    - 有 tool_calls --> 去 tool_node（继续循环）
    - 无 tool_calls --> 去 END（结束）
    """
    messages = state["messages"]
    last_msg = messages[-1]

    # 检查是否有 tool_calls
    tool_calls = getattr(last_msg, "tool_calls", [])
    if tool_calls:
        return "call_tools"
    else:
        return "end"


# --- 构建 Agent 图 ---
agent_graph = StateGraph(AgentState)

# 添加节点
agent_graph.add_node("llm", llm_node)
agent_graph.add_node("tools", tool_node)

# 添加边
agent_graph.add_edge(START, "llm")

# 条件边：LLM 输出后判断是否需要调工具
agent_graph.add_conditional_edges(
    "llm",
    should_continue,
    {
        "call_tools": "tools",  # 有 tool_calls --> 去执行工具
        "end": END,             # 没有 tool_calls --> 结束
    }
)

# 工具执行完回到 LLM（形成循环）
agent_graph.add_edge("tools", "llm")

# 编译
agent_app = agent_graph.compile()

# --- 测试 Agent ---
print("\n  Testing Agent with multiple queries:\n")

test_queries = [
    "你好！",                         # 不需要工具
    "帮我算一下 2+3*4",              # 需要 calculator
    "现在几点了？",                   # 需要 get_current_time
    "什么是 RAG？",                   # 需要 search_knowledge
    "今天天气真好",                   # 不需要工具
]

for i, query in enumerate(test_queries, 1):
    print(f"  --- Query {i}: '{query}' ---")
    result = agent_app.invoke({
        "messages": [HumanMessage(content=query)]
    })

    # 打印最终 AI 回复（最后一条非 ToolMessage 的消息）
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            print(f"  Agent: {msg.content[:100]}")
            break
    print()

print("  --> Agent 循环：LLM 决定调工具 -> 执行工具 -> 结果返回 LLM -> 再次判断")
print("  --> 不需要工具时直接回答，需要工具时先调工具再总结")
print()


# ============================================================
# Part 2: 手写 ToolNode vs langgraph.prebuilt.ToolNode
# ============================================================
print("=" * 60)
print("  Part 2: ToolNode 原理")
print("=" * 60)

print("""
  Part 1 中我们手写了 tool_node 函数，它的逻辑是：
    1. 从 AI 消息中读取 tool_calls
    2. 根据 tool_name 找到对应的工具函数
    3. 用 tool_args 调用工具
    4. 把结果封装成 ToolMessage 返回

  LangGraph 内置了 ToolNode，功能一样但更健壮：

    from langgraph.prebuilt import ToolNode
    tool_node = ToolNode(tools=[calculator, get_current_time, search_knowledge])

  两者等价，但 ToolNode 处理了更多边界情况（并发调用、错误格式等）。
  学习时手写一遍理解原理，生产代码用 ToolNode。
""")


# ============================================================
# Part 3: Human-in-the-loop（人工确认）
# ============================================================
print("=" * 60)
print("  Part 3: Human-in-the-loop (simulated)")
print("=" * 60)

# 注意：真正的 human-in-the-loop 需要 checkpointer + interrupt
# 这里用一个简化版来演示概念：
# confirmation_node 检查 state 中的 approved 字段


class HumanLoopState(TypedDict):
    task: str               # 要执行的任务描述
    plan: str               # Agent 拟定的计划
    approved: bool          # 是否被人工批准
    result: str             # 执行结果


def plan_node(state: HumanLoopState) -> dict:
    """规划节点：Agent 拟定执行计划"""
    task = state["task"]
    plan = f"针对任务 '{task}'，计划如下：\n  1. 分析需求\n  2. 执行操作\n  3. 返回结果"
    print(f"    [Plan] {plan}")
    return {"plan": plan}


def confirmation_node(state: HumanLoopState) -> dict:
    """
    人工确认节点（简化版）。
    真实项目中这里会用 interrupt() 暂停图，等待用户输入。
    这里模拟为：自动批准（演示流程）。
    """
    plan = state["plan"]
    # 在真实场景中：
    #   user_input = interrupt({"plan": plan, "question": "是否批准？(yes/no)"})
    #   approved = user_input == "yes"

    # 模拟用户批准
    approved = True
    status = "approved" if approved else "rejected"
    print(f"    [Human] Plan {status} (simulated)")
    return {"approved": approved}


def execute_node(state: HumanLoopState) -> dict:
    """执行节点：只在批准后执行"""
    return {"result": f"[OK] 任务 '{state['task']}' 执行成功！"}


def reject_node(state: HumanLoopState) -> dict:
    """拒绝节点：人工拒绝后的处理"""
    return {"result": f"[CANCELLED] 任务 '{state['task']}' 被人工取消。"}


def route_approval(state: HumanLoopState) -> str:
    """根据 approved 字段路由"""
    if state.get("approved"):
        return "go_execute"
    return "go_reject"


# 构建 Human-in-the-loop 图
hil_graph = StateGraph(HumanLoopState)
hil_graph.add_node("plan", plan_node)
hil_graph.add_node("confirm", confirmation_node)
hil_graph.add_node("execute", execute_node)
hil_graph.add_node("reject", reject_node)

hil_graph.add_edge(START, "plan")
hil_graph.add_edge("plan", "confirm")
hil_graph.add_conditional_edges("confirm", route_approval, {
    "go_execute": "execute",
    "go_reject": "reject",
})
hil_graph.add_edge("execute", END)
hil_graph.add_edge("reject", END)

hil_app = hil_graph.compile()

# 测试
print("\n  Test: Human-in-the-loop\n")
result = hil_app.invoke({
    "task": "删除所有临时文件",
    "plan": "",
    "approved": False,
    "result": "",
})
print(f"    [Result] {result['result']}")

print("""
  --> 真实场景中 confirmation_node 会暂停图（interrupt），
      等待用户在 Web UI 或命令行确认后继续。
      需要 checkpointer（如 MemorySaver）来持久化暂停状态。
""")


# ============================================================
# Part 4: 错误处理
# ============================================================
print("=" * 60)
print("  Part 4: 错误处理 (Error Handling)")
print("=" * 60)


class ErrorState(TypedDict):
    input_data: str
    result: str
    error: str
    retry_count: int


def risky_node(state: ErrorState) -> dict:
    """
    有风险的节点：可能失败。
    模拟场景：解析 JSON，如果格式不对就报错。
    """
    data = state["input_data"]
    retry = state.get("retry_count", 0)
    try:
        parsed = json.loads(data)
        result = f"[OK] JSON 解析成功: keys={list(parsed.keys())}"
        print(f"    [Risky] {result}")
        return {"result": result, "error": ""}
    except json.JSONDecodeError as e:
        error_msg = f"JSON 解析失败: {e}"
        print(f"    [Risky] [FAIL] {error_msg} (retry #{retry})")
        return {"error": error_msg, "retry_count": retry + 1}


def error_handler_node(state: ErrorState) -> dict:
    """错误处理节点：记录错误，尝试修复或给出提示"""
    error = state["error"]
    retry = state.get("retry_count", 0)

    if retry <= 2:
        # 尝试修复：用默认 JSON 替代
        print(f"    [ErrorHandler] Attempting fix (retry #{retry})...")
        return {
            "input_data": '{"fixed": true, "note": "auto-fixed by error handler"}',
            "error": "",
        }
    else:
        # 超过重试次数，返回最终错误
        result = f"[FAIL] 处理失败（重试 {retry} 次后放弃）: {error}"
        print(f"    [ErrorHandler] {result}")
        return {"result": result, "error": ""}


def check_error(state: ErrorState) -> str:
    """条件边：检查是否有错误"""
    if state.get("error") and state.get("retry_count", 0) <= 3:
        return "has_error"
    return "no_error"


# 构建错误处理图
error_graph = StateGraph(ErrorState)
error_graph.add_node("risky", risky_node)
error_graph.add_node("error_handler", error_handler_node)

error_graph.add_edge(START, "risky")
error_graph.add_conditional_edges("risky", check_error, {
    "has_error": "error_handler",
    "no_error": END,
})
# 错误处理后回到 risky 重试（形成循环）
error_graph.add_edge("error_handler", "risky")

error_app = error_graph.compile()

# 测试 1：正常输入
print("\n  Test 1: Valid JSON input\n")
result = error_app.invoke({
    "input_data": '{"name": "LangGraph", "version": "0.2"}',
    "result": "",
    "error": "",
    "retry_count": 0,
})
print(f"    Final: {result['result']}")

# 测试 2：错误输入（会触发错误处理 + 重试）
print("\n  Test 2: Invalid JSON input (triggers error handler)\n")
result = error_app.invoke({
    "input_data": "this is not json!",
    "result": "",
    "error": "",
    "retry_count": 0,
})
print(f"    Final: {result['result']}")

print("""
  --> risky_node 失败时，条件边路由到 error_handler
  --> error_handler 尝试修复数据，然后回到 risky_node 重试
  --> 如果修复成功，risky_node 正常完成 -> END
  --> 如果超过重试次数，error_handler 返回最终错误 -> risky 直接通过 -> END
""")


# ============================================================
# Part 5: 综合测试 —— Agent 处理多种查询
# ============================================================
print("=" * 60)
print("  Part 5: Agent 综合测试")
print("=" * 60)

comprehensive_queries = [
    "帮我算一下 sqrt(144) + 10",     # 工具：calculator
    "现在是什么时间？",                # 工具：get_current_time
    "什么是 LangGraph？",              # 工具：search_knowledge
    "你好，介绍一下你自己",            # 不需要工具
    "查一下 FastAPI",                  # 工具：search_knowledge
]

print("\n  Running comprehensive test...\n")

for i, query in enumerate(comprehensive_queries, 1):
    print(f"  {'='*50}")
    print(f"  Query {i}: {query}")
    print(f"  {'-'*50}")

    result = agent_app.invoke({
        "messages": [HumanMessage(content=query)]
    })

    # 打印对话轨迹
    msg_count = len(result["messages"])
    for j, msg in enumerate(result["messages"]):
        if isinstance(msg, HumanMessage):
            print(f"    [{j+1}/{msg_count}] User:  {msg.content[:80]}")
        elif isinstance(msg, AIMessage):
            if msg.content:
                print(f"    [{j+1}/{msg_count}] AI:    {msg.content[:80]}")
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    print(f"    [{j+1}/{msg_count}] AI->Tool: {tc['name']}({tc['args']})")
        elif isinstance(msg, ToolMessage):
            print(f"    [{j+1}/{msg_count}] Tool:  {msg.content[:80]}")

    # 需要调工具的标记
    needed_tool = any(
        isinstance(m, ToolMessage) for m in result["messages"]
    )
    print(f"    Used tools: {'Yes' if needed_tool else 'No'}")
    print()


# === 总结 ===
print("=" * 60)
print("  Day 36 Summary")
print("=" * 60)
print(f"""
  What you learned today:

  1. Agent Loop (ReAct pattern):
     - LLM -> should_continue? -> tools -> LLM (cycle)
     - Conditional edge checks for tool_calls
     - tools -> llm edge creates the loop

  2. Tool-Calling Agent:
     - Define tools with @tool decorator
     - LLM node: llm.bind_tools(tools).invoke(messages)
     - Tool node: parse tool_calls, execute, return ToolMessage
     - Condition: has tool_calls? -> tools : END

  3. Human-in-the-loop:
     - Plan -> Confirm (interrupt) -> Execute or Reject
     - Needs checkpointer (MemorySaver) for real interrupt

  4. Error Handling:
     - try/except in nodes, error info in state
     - Conditional edge routes to error_handler
     - Error handler -> retry loop

  5. LLM mode used: {llm_mode}

  Tomorrow (Day 37): 综合练习 - 完整 Agent
    - 把今天学的全部整合成一个可交互的 Agent
    - 3 个工具 + 对话记忆 + ReAct 循环 + verbose 模式
""")
