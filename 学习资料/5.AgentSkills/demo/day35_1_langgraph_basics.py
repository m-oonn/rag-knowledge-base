"""
Day 35 Demo 1: LangGraph 基础
运行方式: python day35_1_langgraph_basics.py

前置条件:
  pip install langgraph langchain-core

学习目标:
1. 构建最简单的两节点图
2. 理解 State（TypedDict）和节点函数签名
3. 三节点管道：分类 -> 生成 -> 格式化
4. 条件路由：根据意图走不同分支
5. 打印图结构（节点和边）
6. 实战：多分支聊天机器人图
"""

import sys
from typing import TypedDict, Annotated

# === 检查依赖 ===
try:
    from langgraph.graph import StateGraph, START, END
    from langgraph.graph.message import add_messages
except ImportError:
    print("[FAIL] langgraph 未安装")
    print("  请运行: pip install langgraph langchain-core")
    sys.exit(1)

print("=" * 60)
print("  Day 35: LangGraph 基础")
print("=" * 60)
print()


# === Part 1: 最简图 —— 两个节点 + 一条边 ===
print("=" * 60)
print("  Part 1: 最简图 (2 nodes, 1 edge)")
print("=" * 60)

# Step 1: 定义状态 —— 图中流转的数据结构
class SimpleState(TypedDict):
    text: str       # 输入文本
    result: str     # 处理结果


# Step 2: 定义节点函数
# 节点函数签名：接收完整 state，返回 partial state update（dict）
def greet_node(state: SimpleState) -> dict:
    """节点A：在文本前加问候语"""
    original = state["text"]
    return {"result": f"[OK] Hello! You said: {original}"}


def upper_node(state: SimpleState) -> dict:
    """节点B：把结果转大写"""
    current = state["result"]
    return {"result": current.upper()}


# Step 3: 构建图
simple_graph = StateGraph(SimpleState)

# 添加节点：graph.add_node("节点名", 节点函数)
simple_graph.add_node("greet", greet_node)
simple_graph.add_node("upper", upper_node)

# 添加边：START → greet → upper → END
simple_graph.add_edge(START, "greet")
simple_graph.add_edge("greet", "upper")
simple_graph.add_edge("upper", END)

# Step 4: 编译
simple_app = simple_graph.compile()

# Step 5: 执行
result = simple_app.invoke({"text": "LangGraph", "result": ""})

print(f"\n  输入: text='LangGraph'")
print(f"  输出: result='{result['result']}'")
print(f"\n  --> 数据流: START -> greet -> upper -> END")
print(f"  --> greet 节点加了前缀，upper 节点转了大写")
print()


# === Part 2: State 管理 —— 消息列表 + add_messages reducer ===
print("=" * 60)
print("  Part 2: State 管理 (messages + reducer)")
print("=" * 60)

# 使用 Annotated + add_messages：新消息追加而不是覆盖
class ChatState(TypedDict):
    messages: Annotated[list, add_messages]  # 消息列表，自动追加
    turn_count: int                          # 对话轮数，直接覆盖


def user_node(state: ChatState) -> dict:
    """模拟用户节点：添加一条用户消息"""
    # 返回新消息，add_messages reducer 会自动追加到 messages 列表
    return {
        "messages": [("user", "你好，LangGraph！")],
        "turn_count": state.get("turn_count", 0) + 1,
    }


def assistant_node(state: ChatState) -> dict:
    """模拟助手节点：根据消息列表回复"""
    msg_count = len(state["messages"])
    return {
        "messages": [("assistant", f"你好！这是第 {msg_count + 1} 条消息。")],
        "turn_count": state.get("turn_count", 0) + 1,
    }


chat_graph = StateGraph(ChatState)
chat_graph.add_node("user", user_node)
chat_graph.add_node("assistant", assistant_node)

chat_graph.add_edge(START, "user")
chat_graph.add_edge("user", "assistant")
chat_graph.add_edge("assistant", END)

chat_app = chat_graph.compile()

# 执行时传入初始状态
result = chat_app.invoke({"messages": [], "turn_count": 0})

print(f"\n  Messages after graph execution:")
for msg in result["messages"]:
    # msg 可能是 tuple 或 BaseMessage 对象
    if hasattr(msg, "content"):
        role = msg.type  # "human" or "ai"
        content = msg.content
    else:
        role, content = msg[0], msg[1]
    print(f"    [{role}] {content}")
print(f"  Turn count: {result['turn_count']}")
print(f"\n  --> add_messages reducer 让消息自动追加，不会覆盖")
print(f"  --> turn_count 是普通字段，直接覆盖（最终=2 因为两个节点各+1）")
print()


# === Part 3: 三节点管道 —— 分类 -> 生成回复 -> 格式化输出 ===
print("=" * 60)
print("  Part 3: 三节点管道 (classify -> respond -> format)")
print("=" * 60)


class PipelineState(TypedDict):
    user_input: str     # 用户输入
    intent: str         # 分类结果
    response: str       # 生成的回复
    formatted: str      # 格式化后的输出


def classify_intent(state: PipelineState) -> dict:
    """节点1：意图分类（用关键词模拟，实际项目用 LLM）"""
    text = state["user_input"].lower()

    # 简单关键词匹配
    if any(w in text for w in ["你好", "hi", "hello", "嗨"]):
        intent = "greeting"
    elif any(w in text for w in ["？", "?", "什么", "怎么", "为什么", "如何"]):
        intent = "question"
    elif any(w in text for w in ["再见", "bye", "拜拜"]):
        intent = "farewell"
    else:
        intent = "general"

    print(f"    [classify] '{state['user_input']}' -> intent={intent}")
    return {"intent": intent}


def generate_response(state: PipelineState) -> dict:
    """节点2：根据意图生成回复"""
    intent = state["intent"]
    responses = {
        "greeting": "你好！很高兴见到你。有什么我能帮到你的吗？",
        "question": "这是个好问题！让我帮你分析一下...",
        "farewell": "再见！祝你今天愉快！",
        "general":  "收到你的消息了。请问还有什么需要帮助的吗？",
    }
    resp = responses.get(intent, "我不太理解你的意思。")
    print(f"    [respond]  intent={intent} -> response='{resp[:30]}...'")
    return {"response": resp}


def format_output(state: PipelineState) -> dict:
    """节点3：格式化最终输出"""
    formatted = (
        f"[Intent: {state['intent']}]\n"
        f"[Response]: {state['response']}"
    )
    print(f"    [format]   formatted output ready")
    return {"formatted": formatted}


pipeline_graph = StateGraph(PipelineState)
pipeline_graph.add_node("classify", classify_intent)
pipeline_graph.add_node("respond", generate_response)
pipeline_graph.add_node("format", format_output)

# 线性管道：START → classify → respond → format → END
pipeline_graph.add_edge(START, "classify")
pipeline_graph.add_edge("classify", "respond")
pipeline_graph.add_edge("respond", "format")
pipeline_graph.add_edge("format", END)

pipeline_app = pipeline_graph.compile()

# 测试不同输入
test_inputs = ["你好呀！", "Python怎么学？", "拜拜~", "今天天气不错"]

for text in test_inputs:
    print(f"\n  Input: '{text}'")
    result = pipeline_app.invoke({
        "user_input": text,
        "intent": "",
        "response": "",
        "formatted": "",
    })
    print(f"  Output: {result['formatted']}")

print(f"\n  --> 三个节点按顺序执行，每个节点只更新自己负责的字段")
print()


# === Part 4: 条件路由 —— 根据意图走不同分支 ===
print("=" * 60)
print("  Part 4: 条件路由 (conditional edges)")
print("=" * 60)


class RouterState(TypedDict):
    user_input: str
    intent: str
    output: str


def classify_node(state: RouterState) -> dict:
    """分类节点：判断用户意图"""
    text = state["user_input"].lower()
    if any(w in text for w in ["你好", "hi", "hello"]):
        return {"intent": "greeting"}
    elif any(w in text for w in ["？", "?", "什么", "怎么"]):
        return {"intent": "question"}
    else:
        return {"intent": "other"}


def greeting_handler(state: RouterState) -> dict:
    """问候处理节点"""
    return {"output": "[greeting] 你好！欢迎来到 LangGraph 世界！"}


def question_handler(state: RouterState) -> dict:
    """问题处理节点"""
    return {"output": f"[question] 你问了: '{state['user_input']}' -- 让我想想..."}


def default_handler(state: RouterState) -> dict:
    """默认处理节点"""
    return {"output": f"[default] 收到: '{state['user_input']}'"}


# 路由函数：根据 state 返回下一步去哪
def route_by_intent(state: RouterState) -> str:
    """
    路由函数签名：接收 state，返回一个字符串（对应目标节点的 key）
    """
    intent = state.get("intent", "other")
    if intent == "greeting":
        return "go_greeting"
    elif intent == "question":
        return "go_question"
    else:
        return "go_default"


# 构建图
router_graph = StateGraph(RouterState)
router_graph.add_node("classify", classify_node)
router_graph.add_node("greeting_handler", greeting_handler)
router_graph.add_node("question_handler", question_handler)
router_graph.add_node("default_handler", default_handler)

# START → classify
router_graph.add_edge(START, "classify")

# classify → 条件路由 → 三个分支
router_graph.add_conditional_edges(
    "classify",             # 源节点
    route_by_intent,        # 路由函数
    {                       # 路由映射：路由函数返回值 → 目标节点名
        "go_greeting":  "greeting_handler",
        "go_question":  "question_handler",
        "go_default":   "default_handler",
    }
)

# 三个分支都汇合到 END
router_graph.add_edge("greeting_handler", END)
router_graph.add_edge("question_handler", END)
router_graph.add_edge("default_handler", END)

router_app = router_graph.compile()

# 测试条件路由
test_cases = [
    "Hello!",
    "Python怎么学？",
    "今天心情不错",
]

for text in test_cases:
    result = router_app.invoke({
        "user_input": text,
        "intent": "",
        "output": "",
    })
    print(f"\n  Input:  '{text}'")
    print(f"  Route:  intent='{result['intent']}'")
    print(f"  Output: {result['output']}")

print(f"\n  --> 条件边让图根据 state 动态选择下一步节点")
print()


# === Part 5: 打印图结构（可视化） ===
print("=" * 60)
print("  Part 5: 图结构可视化")
print("=" * 60)


def print_graph_structure(app, name: str):
    """打印编译后图的节点和边信息"""
    print(f"\n  Graph: {name}")
    print(f"  {'-' * 40}")

    try:
        graph_obj = app.get_graph()

        # 打印节点
        nodes = list(graph_obj.nodes)
        print(f"  Nodes ({len(nodes)}):")
        for node in nodes:
            print(f"    - {node}")

        # 打印边
        edges = list(graph_obj.edges)
        print(f"  Edges ({len(edges)}):")
        for edge in edges:
            # edge 是一个 (source, target) 元组或 namedtuple
            if hasattr(edge, "source") and hasattr(edge, "target"):
                src, tgt = edge.source, edge.target
            elif isinstance(edge, (tuple, list)) and len(edge) >= 2:
                src, tgt = edge[0], edge[1]
            else:
                print(f"    - {edge}")
                continue
            # 判断是否是条件边
            cond_marker = ""
            if hasattr(edge, "conditional") and edge.conditional:
                cond_marker = " (conditional)"
            elif hasattr(edge, "data") and edge.data:
                cond_marker = " (conditional)"
            print(f"    - {src} --> {tgt}{cond_marker}")

    except Exception as e:
        # 兜底：直接打印可用属性
        print(f"  (graph introspection limited: {e})")
        print(f"  Compiled graph type: {type(app)}")


# 打印前面构建的所有图
print_graph_structure(simple_app, "Simple (Part 1)")
print_graph_structure(chat_app, "Chat (Part 2)")
print_graph_structure(pipeline_app, "Pipeline (Part 3)")
print_graph_structure(router_app, "Router (Part 4)")

print(f"\n  --> 通过 app.get_graph() 可以检查图的结构，方便调试")
print()


# === Part 6: 实战 —— 多分支聊天机器人 ===
print("=" * 60)
print("  Part 6: 实战 - 多分支聊天机器人")
print("=" * 60)

# 完整的聊天机器人图：
#   START → check_input → check_intent
#                              |
#             ┌────────────────┼────────────────┐
#             v                v                v
#        greet_user      answer_question    say_goodbye
#             |                |                |
#             └────────────────┼────────────────┘
#                              v
#                        output_node → END


class ChatbotState(TypedDict):
    user_input: str         # 原始用户输入
    cleaned_input: str      # 清洗后的输入
    intent: str             # 意图：greeting / question / farewell / unknown
    response: str           # 生成的回复
    final_output: str       # 最终格式化输出


def check_input_node(state: ChatbotState) -> dict:
    """预处理节点：清洗用户输入"""
    raw = state["user_input"]
    cleaned = raw.strip()
    return {"cleaned_input": cleaned}


def check_intent_node(state: ChatbotState) -> dict:
    """意图识别节点"""
    text = state["cleaned_input"].lower()

    # 关键词规则（实际项目这里调 LLM）
    greeting_words = ["你好", "hi", "hello", "嗨", "早上好", "晚上好"]
    question_words = ["？", "?", "什么", "怎么", "为什么", "如何", "能不能", "可以"]
    farewell_words = ["再见", "bye", "拜拜", "下次见", "回头见"]

    if any(w in text for w in greeting_words):
        intent = "greeting"
    elif any(w in text for w in question_words):
        intent = "question"
    elif any(w in text for w in farewell_words):
        intent = "farewell"
    else:
        intent = "unknown"

    return {"intent": intent}


def greet_user_node(state: ChatbotState) -> dict:
    """问候回复节点"""
    return {"response": "你好！我是 LangGraph 聊天机器人。有什么可以帮你的吗？"}


def answer_question_node(state: ChatbotState) -> dict:
    """问题回复节点"""
    q = state["cleaned_input"]
    # 模拟回答（实际项目调 LLM）
    return {"response": f"关于你的问题 '{q}'，我的理解是：这是一个很好的问题，建议查阅相关文档。"}


def say_goodbye_node(state: ChatbotState) -> dict:
    """告别回复节点"""
    return {"response": "再见！感谢你的使用，希望对你有帮助。下次再见！"}


def unknown_handler_node(state: ChatbotState) -> dict:
    """未知意图回复节点"""
    return {"response": "我不太确定你的意思。你可以试着问一个问题，或者跟我打个招呼。"}


def output_node(state: ChatbotState) -> dict:
    """格式化输出节点"""
    final = (
        f"{'='*40}\n"
        f"  User:     {state['user_input']}\n"
        f"  Intent:   {state['intent']}\n"
        f"  Response: {state['response']}\n"
        f"{'='*40}"
    )
    return {"final_output": final}


# 路由函数
def chatbot_router(state: ChatbotState) -> str:
    """根据意图路由到不同处理节点"""
    intent = state["intent"]
    route_map = {
        "greeting": "go_greet",
        "question": "go_answer",
        "farewell": "go_goodbye",
    }
    return route_map.get(intent, "go_unknown")


# 构建聊天机器人图
chatbot_graph = StateGraph(ChatbotState)

# 添加所有节点
chatbot_graph.add_node("check_input", check_input_node)
chatbot_graph.add_node("check_intent", check_intent_node)
chatbot_graph.add_node("greet", greet_user_node)
chatbot_graph.add_node("answer", answer_question_node)
chatbot_graph.add_node("goodbye", say_goodbye_node)
chatbot_graph.add_node("unknown", unknown_handler_node)
chatbot_graph.add_node("output", output_node)

# 连接边
chatbot_graph.add_edge(START, "check_input")
chatbot_graph.add_edge("check_input", "check_intent")

# 条件路由：check_intent → 四个分支
chatbot_graph.add_conditional_edges(
    "check_intent",
    chatbot_router,
    {
        "go_greet":   "greet",
        "go_answer":  "answer",
        "go_goodbye": "goodbye",
        "go_unknown": "unknown",
    }
)

# 四个分支汇合到 output
chatbot_graph.add_edge("greet", "output")
chatbot_graph.add_edge("answer", "output")
chatbot_graph.add_edge("goodbye", "output")
chatbot_graph.add_edge("unknown", "output")

# output → END
chatbot_graph.add_edge("output", END)

# 编译
chatbot_app = chatbot_graph.compile()

# 测试聊天机器人
print("\n  Testing chatbot graph:\n")

test_messages = [
    "你好！",
    "LangGraph 怎么用？",
    "Python 是什么？",
    "再见！",
    "今天天气真好",
]

for msg in test_messages:
    result = chatbot_app.invoke({
        "user_input": msg,
        "cleaned_input": "",
        "intent": "",
        "response": "",
        "final_output": "",
    })
    print(result["final_output"])
    print()

# 打印聊天机器人图结构
print_graph_structure(chatbot_app, "Chatbot (Part 6)")

print()


# === 总结 ===
print("=" * 60)
print("  Day 35 Summary")
print("=" * 60)
print("""
  What you learned today:

  1. StateGraph + TypedDict = 图的基础
     - State 定义数据结构，Graph 定义执行流程

  2. Node function signature:
     - Input:  state (full TypedDict)
     - Output: dict (partial state update)

  3. Edge types:
     - Normal edge:      A -> B (always)
     - Conditional edge: A -> B or C (based on state)

  4. add_messages reducer:
     - Messages append instead of overwrite

  5. Graph building pattern:
     - Define State -> Create Graph -> Add Nodes -> Add Edges -> Compile

  6. Practical chatbot:
     - input -> classify intent -> route to handler -> format output

  Tomorrow (Day 36): LangGraph 进阶
    - Agent 循环（tool calling loop）
    - Human-in-the-loop
    - 错误处理
""")
