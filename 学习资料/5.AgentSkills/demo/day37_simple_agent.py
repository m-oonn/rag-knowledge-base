"""
Day 37 Demo: 综合练习 - 完整 Agent
运行方式: python day37_simple_agent.py

前置条件:
  pip install langgraph langchain-core langchain-openai python-dotenv

  LLM 提供者（任选一个，自动检测）:
    - Ollama 运行中（推荐，免费）: ollama serve
    - 或 .env 中配置 DEEPSEEK_API_KEY
    - 都没有也能运行（模拟模式，用预设回复演示流程）

功能:
  - 3 个工具：calculator, get_current_time, search_knowledge
  - 对话记忆：messages 列表保存完整对话历史
  - ReAct 循环：LLM 自主决定调工具还是直接回答
  - Verbose 模式：显示 Thought / Action / Observation 轨迹
  - 优雅降级：Ollama > DeepSeek > 模拟模式
  - 这是项目二（数据分析 Agent）的原型
"""

import os
import sys
import math
import json
from typing import TypedDict, Annotated
from datetime import datetime

# ============================================================
# 依赖检查
# ============================================================

try:
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages
    from langchain_core.messages import (
        HumanMessage, AIMessage, ToolMessage, SystemMessage
    )
    from langchain_core.tools import tool
except ImportError:
    print("[FAIL] 缺少依赖，请运行:")
    print("  pip install langgraph langchain-core langchain-openai")
    sys.exit(1)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv 不是必须的


# ============================================================
# 配置
# ============================================================

# Verbose 模式：显示 Thought/Action/Observation 轨迹
VERBOSE = True

# Agent 系统提示词
SYSTEM_PROMPT = """你是一个有用的 AI 助手，可以使用工具来帮助用户。

## 可用工具
- calculator: 计算数学表达式（加减乘除、开方、三角函数等）
- get_current_time: 获取当前日期和时间
- search_knowledge: 在知识库中搜索 Python、AI、LangGraph 等技术知识

## 规则
1. 如果用户的问题需要计算，使用 calculator 工具
2. 如果用户询问时间或日期，使用 get_current_time 工具
3. 如果用户询问技术知识，使用 search_knowledge 工具
4. 如果不需要工具，直接回答
5. 用中文回答
6. 回答简洁清晰"""


# ============================================================
# LLM 初始化：自动检测 Ollama > DeepSeek > 模拟模式
# ============================================================

def init_llm():
    """
    按优先级检测可用的 LLM：
    1. Ollama（本地免费）
    2. DeepSeek（API Key）
    3. 模拟模式（预设回复，用于演示流程）

    返回: (llm_object 或 None, mode_string)
    """
    # --- 尝试 Ollama ---
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
            return llm, f"Ollama ({model_name})"
    except Exception:
        pass

    # --- 尝试 DeepSeek ---
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
            return llm, "DeepSeek"
        except Exception:
            pass

    # --- 模拟模式 ---
    return None, "Simulated (no LLM available)"


# ============================================================
# 工具定义
# ============================================================

@tool
def calculator(expression: str) -> str:
    """计算数学表达式。支持加减乘除、幂运算、开方、三角函数等。
    Args:
        expression: 数学表达式字符串，如 '2+3*4', 'sqrt(16)', 'pow(2,10)'
    """
    try:
        # 白名单：只允许安全的数学函数
        safe_env = {
            "__builtins__": {},
            "sqrt": math.sqrt, "abs": abs, "pow": pow, "round": round,
            "sin": math.sin, "cos": math.cos, "tan": math.tan,
            "log": math.log, "log2": math.log2, "log10": math.log10,
            "pi": math.pi, "e": math.e,
            "ceil": math.ceil, "floor": math.floor,
        }
        result = eval(expression, safe_env)
        return f"计算结果: {expression} = {result}"
    except ZeroDivisionError:
        return f"计算错误: 除以零 ({expression})"
    except Exception as e:
        return f"计算错误: {e} (表达式: {expression})"


@tool
def get_current_time() -> str:
    """获取当前日期和时间。不需要任何参数。"""
    now = datetime.now()
    weekdays = ["一", "二", "三", "四", "五", "六", "日"]
    return (
        f"当前时间: {now.strftime('%Y年%m月%d日 %H:%M:%S')} "
        f"(星期{weekdays[now.weekday()]})"
    )


@tool
def search_knowledge(query: str) -> str:
    """在知识库中搜索技术信息。可查询 Python、AI、LangGraph、RAG 等相关知识。
    Args:
        query: 搜索关键词，如 'python', 'langgraph', 'rag'
    """
    # 模拟知识库（实际项目中替换为 RAG 检索）
    knowledge_base = {
        "python": (
            "Python 是一种高级编程语言，以简洁优雅的语法著称。"
            "广泛应用于 Web 开发（Django/FastAPI）、数据分析（Pandas/NumPy）、"
            "机器学习（PyTorch/TensorFlow）、自动化脚本等领域。"
            "Python 3.12+ 引入了更好的类型提示和性能优化。"
        ),
        "langgraph": (
            "LangGraph 是基于图结构的 AI 工作流编排框架，由 LangChain 团队开发。"
            "核心概念：StateGraph（图容器）、Node（处理节点）、Edge（连接边）。"
            "支持条件路由、循环（Agent 模式）、Human-in-the-loop。"
            "是构建复杂 AI Agent 的推荐框架。"
        ),
        "rag": (
            "RAG（Retrieval-Augmented Generation）= 检索增强生成。"
            "流程：用户问题 -> Embedding -> 向量检索 -> 相关文档 -> Prompt 拼接 -> LLM 生成回答。"
            "解决了 LLM 知识过时和幻觉（编造内容）问题。"
            "常用技术栈：LangChain + ChromaDB + Sentence-Transformers。"
        ),
        "agent": (
            "AI Agent = 能自主使用工具完成任务的智能体。"
            "ReAct 模式：Thought(思考) -> Action(调用工具) -> Observation(观察结果) -> 循环。"
            "LangGraph 中用条件边 + 循环实现 Agent 模式。"
            "工具类型：计算器、搜索引擎、代码执行器、API 调用等。"
        ),
        "fastapi": (
            "FastAPI 是现代 Python Web 框架，基于 Starlette（异步）和 Pydantic（数据验证）。"
            "特点：自动生成 API 文档（Swagger UI）、原生异步支持、类型安全。"
            "性能可媲美 Node.js 和 Go 框架。"
            "适合构建 AI 应用的后端 API 服务。"
        ),
        "prompt": (
            "Prompt Engineering = 通过设计输入文本控制 LLM 输出质量。"
            "核心技巧：System Prompt（角色定义）、Few-Shot（示例学习）、"
            "Chain of Thought（思维链推理）、结构化输出（JSON 模板）。"
            "RAG 和 Agent 的效果很大程度取决于 Prompt 设计。"
        ),
        "embedding": (
            "Embedding = 将文本转化为固定维度的向量（数字数组）。"
            "语义相近的文本，向量距离也近（余弦相似度接近 1）。"
            "常用模型：OpenAI text-embedding-ada-002、sentence-transformers。"
            "是 RAG 检索的基础技术。"
        ),
    }

    query_lower = query.lower().strip()
    results = []

    # 精确匹配
    for key, value in knowledge_base.items():
        if key in query_lower or query_lower in key:
            results.append(f"[{key}] {value}")

    # 模糊匹配：检查知识库内容是否包含查询词
    if not results:
        for key, value in knowledge_base.items():
            if query_lower in value.lower():
                results.append(f"[{key}] {value}")

    if results:
        return "知识库查询结果:\n" + "\n\n".join(results)
    else:
        return f"知识库中未找到与 '{query}' 相关的信息。可查询的主题包括：{', '.join(knowledge_base.keys())}"


# 所有工具
ALL_TOOLS = [calculator, get_current_time, search_knowledge]
TOOL_MAP = {t.name: t for t in ALL_TOOLS}


# ============================================================
# Agent State 定义
# ============================================================

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # 对话历史（自动追加）


# ============================================================
# Agent 节点定义
# ============================================================

# 全局变量，在 main() 中初始化
_llm = None
_llm_mode = ""


def llm_node(state: AgentState) -> dict:
    """
    LLM 节点：
    1. 将对话历史（包括系统提示）发送给 LLM
    2. LLM 返回文本回复或 tool_calls
    3. 如果是模拟模式，根据规则生成预设回复
    """
    messages = state["messages"]

    # --- 真实 LLM 调用 ---
    if _llm is not None:
        llm_with_tools = _llm.bind_tools(ALL_TOOLS)
        response = llm_with_tools.invoke(messages)

        # Verbose: 显示 LLM 的思考
        if VERBOSE:
            if response.content:
                _print_trace("Thought", response.content[:120])
            if hasattr(response, "tool_calls") and response.tool_calls:
                for tc in response.tool_calls:
                    args_str = ", ".join(f'{k}="{v}"' for k, v in tc["args"].items())
                    _print_trace("Action", f"{tc['name']}({args_str})")

        return {"messages": [response]}

    # --- 模拟模式 ---
    return _simulated_llm(messages)


def _simulated_llm(messages: list) -> dict:
    """
    模拟 LLM：根据消息内容用规则判断是否需要调用工具。
    当没有可用 LLM 时使用，保证 demo 流程能走通。
    """
    last_msg = messages[-1]
    content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)

    # 如果上一条是 ToolMessage，说明工具已执行，直接总结
    if isinstance(last_msg, ToolMessage):
        tool_result = last_msg.content
        summary = f"根据查询结果：{tool_result[:200]}"
        if VERBOSE:
            _print_trace("Thought", "工具已返回结果，我来总结回复用户")
        return {"messages": [AIMessage(content=summary)]}

    # 判断是否需要调工具
    content_lower = content.lower()

    # 规则 1：数学计算
    math_triggers = ["计算", "算一下", "算", "等于多少", "+", "-", "*", "/", "加", "减", "乘", "除", "开方", "sqrt"]
    if any(w in content_lower for w in math_triggers):
        # 尝试从用户消息中提取表达式
        expression = _extract_expression(content)
        if VERBOSE:
            _print_trace("Thought", f"用户需要计算，提取表达式: {expression}")
            _print_trace("Action", f'calculator(expression="{expression}")')
        return {"messages": [AIMessage(
            content="",
            tool_calls=[{"id": "sim_calc", "name": "calculator", "args": {"expression": expression}}]
        )]}

    # 规则 2：时间查询
    time_triggers = ["时间", "几点", "日期", "今天", "现在", "星期几"]
    if any(w in content_lower for w in time_triggers):
        if VERBOSE:
            _print_trace("Thought", "用户在询问时间")
            _print_trace("Action", "get_current_time()")
        return {"messages": [AIMessage(
            content="",
            tool_calls=[{"id": "sim_time", "name": "get_current_time", "args": {}}]
        )]}

    # 规则 3：知识搜索
    knowledge_triggers = ["什么是", "查一下", "搜索", "知识", "介绍", "解释", "了解"]
    if any(w in content_lower for w in knowledge_triggers):
        # 提取搜索关键词
        query = content
        for trigger in knowledge_triggers:
            query = query.replace(trigger, "")
        query = query.strip().strip("？?。！!")
        if not query:
            query = "python"
        if VERBOSE:
            _print_trace("Thought", f"用户想查询知识，搜索关键词: {query}")
            _print_trace("Action", f'search_knowledge(query="{query}")')
        return {"messages": [AIMessage(
            content="",
            tool_calls=[{"id": "sim_search", "name": "search_knowledge", "args": {"query": query}}]
        )]}

    # 规则 4：不需要工具，直接回复
    if any(w in content_lower for w in ["你好", "hi", "hello", "嗨"]):
        reply = "你好！我是 AI 助手。我可以帮你计算数学问题、查询时间、搜索技术知识。请问有什么可以帮你的？"
    elif any(w in content_lower for w in ["谢谢", "感谢", "thanks"]):
        reply = "不客气！如果还有其他问题，随时问我。"
    elif any(w in content_lower for w in ["再见", "bye", "拜拜"]):
        reply = "再见！祝你学习顺利！"
    else:
        reply = f"收到你的消息：'{content}'。我可以帮你计算数学问题、查询时间、搜索技术知识。试试问我一个具体问题？"

    if VERBOSE:
        _print_trace("Thought", "不需要工具，直接回复")
    return {"messages": [AIMessage(content=reply)]}


def _extract_expression(text: str) -> str:
    """从用户消息中提取数学表达式（简单规则）"""
    # 移除常见的非数学部分
    for prefix in ["帮我计算", "帮我算一下", "帮我算", "计算一下", "计算", "算一下", "算"]:
        if prefix in text:
            text = text[text.index(prefix) + len(prefix):]
            break

    # 清理
    expr = text.strip().strip("。？?！!，,")

    # 中文运算符替换
    replacements = {"加": "+", "减": "-", "乘": "*", "乘以": "*", "除以": "/", "除": "/", "的": "**", "次方": ""}
    for cn, en in replacements.items():
        expr = expr.replace(cn, en)

    return expr.strip() if expr.strip() else "1+1"


def tool_node(state: AgentState) -> dict:
    """
    工具执行节点：
    1. 从最后一条 AI 消息中提取 tool_calls
    2. 根据 tool_name 查找并执行对应工具
    3. 将结果封装为 ToolMessage 返回
    """
    messages = state["messages"]
    last_msg = messages[-1]
    tool_calls = getattr(last_msg, "tool_calls", [])

    results = []
    for tc in tool_calls:
        tool_name = tc["name"]
        tool_args = tc["args"]
        tool_id = tc.get("id", f"call_{tool_name}")

        # 执行工具
        if tool_name in TOOL_MAP:
            try:
                result = TOOL_MAP[tool_name].invoke(tool_args)
            except Exception as e:
                result = f"[FAIL] 工具执行失败: {e}"
        else:
            result = f"[FAIL] 未知工具: {tool_name}"

        # Verbose: 显示工具执行结果
        if VERBOSE:
            _print_trace("Observation", str(result)[:150])

        results.append(ToolMessage(content=str(result), tool_call_id=tool_id))

    return {"messages": results}


def should_continue(state: AgentState) -> str:
    """
    条件边路由函数：
    - 最后一条消息有 tool_calls --> "call_tools" （继续循环）
    - 否则 --> "end" （结束）
    """
    messages = state["messages"]
    if not messages:
        return "end"

    last_msg = messages[-1]
    tool_calls = getattr(last_msg, "tool_calls", [])

    if tool_calls:
        return "call_tools"
    return "end"


# ============================================================
# 辅助函数
# ============================================================

def _print_trace(label: str, content: str):
    """打印 Verbose 轨迹"""
    prefix_map = {
        "Thought":     "  [Thought]     ",
        "Action":      "  [Action]      ",
        "Observation": "  [Observation] ",
    }
    prefix = prefix_map.get(label, f"  [{label}] ")
    # 内容可能多行，缩进对齐
    lines = content.split("\n")
    print(f"{prefix}{lines[0]}")
    indent = " " * len(prefix)
    for line in lines[1:]:
        print(f"{indent}{line}")


def print_banner():
    """打印启动 banner"""
    print()
    print("=" * 60)
    print("  AI Agent - Day 37 综合练习")
    print("  (项目二原型)")
    print("=" * 60)
    print(f"  LLM:   {_llm_mode}")
    print(f"  Tools: calculator, get_current_time, search_knowledge")
    print(f"  Mode:  {'Verbose (show trace)' if VERBOSE else 'Normal'}")
    print("=" * 60)
    print()
    print("  Commands:")
    print("    quit / exit  - 退出")
    print("    verbose      - 切换 Verbose 模式")
    print("    clear        - 清除对话历史")
    print("    tools        - 显示可用工具")
    print()


# ============================================================
# 构建 Agent 图
# ============================================================

def build_agent_graph() -> object:
    """
    构建 ReAct Agent 图：
      START -> llm_node -> should_continue? -> tool_node -> llm_node (循环)
                                            -> END
    """
    graph = StateGraph(AgentState)

    # 添加节点
    graph.add_node("llm", llm_node)
    graph.add_node("tools", tool_node)

    # 添加边
    graph.add_edge(START, "llm")
    graph.add_conditional_edges("llm", should_continue, {
        "call_tools": "tools",
        "end": END,
    })
    graph.add_edge("tools", "llm")  # 工具结果回到 LLM，形成循环

    # 编译
    app = graph.compile()
    return app


# ============================================================
# 交互循环
# ============================================================

def run_interactive(agent_app):
    """
    交互式对话循环：
    - 维护 messages 列表作为对话记忆
    - 每次用户输入后调用 agent_app.invoke()
    - 累积 messages 实现多轮对话
    """
    global VERBOSE

    # 初始对话历史（包含系统提示）
    conversation_messages = [SystemMessage(content=SYSTEM_PROMPT)]

    print("Agent> 你好！我是 AI 助手。输入问题开始对话，输入 quit 退出。")
    print()

    while True:
        try:
            user_input = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break

        if not user_input:
            continue

        # --- 特殊命令 ---
        if user_input.lower() in ("quit", "exit", "q"):
            print("再见！祝你学习顺利！")
            break

        if user_input.lower() == "verbose":
            VERBOSE = not VERBOSE
            print(f"  Verbose mode: {'ON' if VERBOSE else 'OFF'}")
            continue

        if user_input.lower() == "clear":
            conversation_messages = [SystemMessage(content=SYSTEM_PROMPT)]
            print("  [OK] 对话历史已清除")
            continue

        if user_input.lower() == "tools":
            print("  Available tools:")
            for t in ALL_TOOLS:
                print(f"    - {t.name}: {t.description[:60]}")
            continue

        # --- 添加用户消息到历史 ---
        conversation_messages.append(HumanMessage(content=user_input))

        # --- 调用 Agent ---
        try:
            result = agent_app.invoke({"messages": conversation_messages})

            # 更新对话历史（用 Agent 返回的完整 messages）
            conversation_messages = result["messages"]

            # 提取最终 AI 回复（最后一条有 content 的 AIMessage）
            final_reply = None
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage) and msg.content:
                    final_reply = msg.content
                    break

            if final_reply:
                print(f"\nAgent> {final_reply}")
            else:
                print("\nAgent> (no response generated)")

        except Exception as e:
            print(f"\n  [FAIL] Agent error: {e}")
            # 发生错误时移除最后一条用户消息，避免污染历史
            if conversation_messages and isinstance(conversation_messages[-1], HumanMessage):
                conversation_messages.pop()

        print()


# ============================================================
# 非交互模式：自动演示
# ============================================================

def run_demo(agent_app):
    """
    自动演示模式：用预设问题测试 Agent。
    适合快速验证 Agent 是否正常工作。
    """
    demo_queries = [
        "你好！",
        "帮我算一下 123 * 456",
        "现在是什么时间？",
        "什么是 LangGraph？",
        "什么是 RAG？",
        "帮我算一下 sqrt(144) + pow(2, 10)",
        "谢谢你的帮助！",
    ]

    print("  Running demo with preset queries...\n")

    # 保持对话历史以测试多轮记忆
    conversation_messages = [SystemMessage(content=SYSTEM_PROMPT)]

    for i, query in enumerate(demo_queries, 1):
        print(f"  {'='*55}")
        print(f"  [{i}/{len(demo_queries)}] You> {query}")
        print(f"  {'-'*55}")

        conversation_messages.append(HumanMessage(content=query))

        try:
            result = agent_app.invoke({"messages": conversation_messages})
            conversation_messages = result["messages"]

            # 提取最终回复
            for msg in reversed(result["messages"]):
                if isinstance(msg, AIMessage) and msg.content:
                    print(f"  Agent> {msg.content[:200]}")
                    break
        except Exception as e:
            print(f"  [FAIL] {e}")
            if conversation_messages and isinstance(conversation_messages[-1], HumanMessage):
                conversation_messages.pop()

        print()

    # 统计
    total_msgs = len(conversation_messages)
    tool_msgs = sum(1 for m in conversation_messages if isinstance(m, ToolMessage))
    ai_msgs = sum(1 for m in conversation_messages if isinstance(m, AIMessage))
    print(f"  {'='*55}")
    print(f"  Demo complete!")
    print(f"  Total messages: {total_msgs}")
    print(f"  AI messages: {ai_msgs}, Tool calls: {tool_msgs}")
    print(f"  {'='*55}")
    print()


# ============================================================
# 主入口
# ============================================================

def main():
    global _llm, _llm_mode

    # 1. 初始化 LLM
    _llm, _llm_mode = init_llm()

    # 2. 打印 banner
    print_banner()

    # 3. 构建 Agent 图
    agent_app = build_agent_graph()

    # 4. 打印图结构
    print("  Graph structure:")
    try:
        graph_obj = agent_app.get_graph()
        for edge in graph_obj.edges:
            if hasattr(edge, "source") and hasattr(edge, "target"):
                print(f"    {edge.source} --> {edge.target}")
            elif isinstance(edge, (tuple, list)) and len(edge) >= 2:
                print(f"    {edge[0]} --> {edge[1]}")
    except Exception:
        print("    START -> llm -> (tools | END), tools -> llm")
    print()

    # 5. 选择运行模式
    #    - 如果有命令行参数 --demo，运行自动演示
    #    - 否则进入交互模式
    if "--demo" in sys.argv:
        run_demo(agent_app)
    else:
        # 先运行一次 demo 演示，再进入交互模式
        print("  [INFO] Running quick demo first...\n")
        run_demo(agent_app)
        print("-" * 60)
        print("  Demo finished. Entering interactive mode.")
        print("  Type your questions below (or 'quit' to exit).")
        print("-" * 60)
        print()
        run_interactive(agent_app)


if __name__ == "__main__":
    main()
