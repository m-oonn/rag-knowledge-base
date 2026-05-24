# Day 35：LangGraph 基础

> LangGraph 用"图"来编排 Agent。比 AgentExecutor 更灵活——你可以精确控制每一步的条件、分支、循环。

---

## 一、什么是 LangGraph

用**节点（Node）+ 边（Edge）+ 状态（State）**描述 Agent 的工作流。

```
AgentExecutor（LangChain）：
  "自动循环，你控制不了每一步的逻辑" → 简单但不够灵活

LangGraph：
  "你自己画流程图，每个节点你想干什么都行" → 灵活但要多写几行
```

**LangGraph = Agent 的流程图编辑器。**

---

## 二、三个核心概念

### State（状态）
一个共享的数据字典，在图的所有节点之间传递。每次经过一个节点，节点可以读取和修改它。

```python
from typing import TypedDict

class AgentState(TypedDict):
    messages: list        # 对话历史
    next_action: str      # 下一步做什么
    tool_results: list    # 工具执行结果
```

### Node（节点）
一个 Python 函数，接收 State，返回 State 的更新。

```python
def agent_node(state: AgentState) -> AgentState:
    """LLM 思考节点：决定下一步。"""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def tool_node(state: AgentState) -> AgentState:
    """工具执行节点：运行工具并记录结果。"""
    result = execute_tool(state["next_action"])
    return {"tool_results": [result]}
```

### Edge（边）
连接两个节点。有普通边（固定从 A 到 B）和条件边（根据状态选择去 B 还是 C）。

```python
# 普通边：A → B（无条件）
graph.add_edge("agent", "tool")

# 条件边：A → B 或 A → C（看状态决定）
graph.add_conditional_edges(
    "agent",
    route_function,        # 这个函数返回 next_node 的名字
    {"continue": "tool", "end": END}
)
```

---

## 三、完整 LangGraph Agent（核心源码）

```python
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

# ── Step 1: 定义 State ──
class AgentState(TypedDict):
    messages: list            # 对话历史（自动累积）
    next_step: str            # 路由决策：call_tool / end
    tool_name: str            # 要调用的工具名
    tool_input: str           # 工具参数

# ── Step 2: 定义节点 ──
def call_model(state: AgentState) -> AgentState:
    """调用 LLM，让它决定下一步。"""
    response = llm_with_tools.invoke(state["messages"])

    # LLM 返回了 tool_call 还是普通文本？
    if hasattr(response, "tool_calls") and response.tool_calls:
        tc = response.tool_calls[0]
        return {
            "messages": [response],
            "next_step": "call_tool",
            "tool_name": tc["name"],
            "tool_input": tc["args"],
        }
    else:
        return {
            "messages": [response],
            "next_step": "end",
        }

def execute_tools(state: AgentState) -> AgentState:
    """执行 LLM 请求的工具。"""
    tool = TOOLS[state["tool_name"]]
    result = tool.invoke(state["tool_input"])

    # 将工具结果作为新的消息追加
    tool_message = {"role": "tool", "content": result,
                    "name": state["tool_name"]}
    return {
        "messages": [tool_message],
        "next_step": "call_model",  # 回到 LLM 继续
    }

# ── Step 3: 路由函数 ──
def router(state: AgentState) -> Literal["call_tool", "end"]:
    """根据 LLM 的决策，选择去工具节点还是结束。"""
    if state["next_step"] == "call_tool":
        return "call_tool"
    return "end"

# ── Step 4: 建图 ──
graph = StateGraph(AgentState)

# 加节点
graph.add_node("agent", call_model)
graph.add_node("tools", execute_tools)

# 加边
graph.set_entry_point("agent")                         # 入口：从 agent 开始
graph.add_conditional_edges("agent", router, {         # agent → tools 或 END
    "call_tool": "tools",
    "end": END
})
graph.add_edge("tools", "agent")                       # tools → agent（循环）

# 编译
app = graph.compile()

# ── Step 5: 运行 ──
result = app.invoke({
    "messages": [HumanMessage(content="北京今天天气怎么样？")],
    "next_step": "call_model",
})
```

### 这个图的样子

```
    ┌──────────────────────────────────┐
    │          START                    │
    │            ↓                      │
    │         ┌──────┐                  │
    │    ┌──→ │agent │ (LLM思考+决策)    │
    │    │    └──┬───┘                  │
    │    │       │ router() 判断         │
    │    │    ┌──┴──────────┐           │
    │    │    ↓              ↓           │
    │    │ ┌──────┐       ┌────┐        │
    │    └─│tools │       │END │        │
    │      └──────┘       └────┘        │
    │   (执行工具)     (结束对话)         │
    └──────────────────────────────────┘
```

**关键：这是你自己画的循环。你可以加任意多的节点、任意复杂的条件——完全自由。**

---

## 四、State 中的 Reducer

默认情况下，节点返回的新值会**覆盖** State 中的旧值。但对于 `messages` 这种需要**累积**的字段，要用 reducer：

```python
from langgraph.graph import add_messages

class AgentState(TypedDict):
    messages: Annotated[list, add_messages]  # 用 add_messages 累积
    # add_messages 会把新消息追加到旧消息后面，而不是覆盖
```

---

## 五、LangGraph vs LangChain AgentExecutor

| | AgentExecutor | LangGraph |
|------|-------------|----------|
| 控制粒度 | 粗（自动循环） | 细（每步可控） |
| 自定义逻辑 | 困难 | 自由 |
| 条件分支 | 不支持 | 原生支持 |
| 适合场景 | 简单 Agent | 复杂工作流 |
| 学习曲线 | 低 | 中 |

---

## 六、动手练习

写一个只有 3 个节点的 LangGraph：`greet → ask_name → goodbye`，体验建图的过程。

---

## 七、面试速记

**Q1：LangGraph 是什么？**
用图（节点+边）编排 Agent 工作流的框架。比 AgentExecutor 更灵活，可以精确控制每一跳的条件分支。

**Q2：State、Node、Edge 分别是什么？**
State 是共享数据字典；Node 是处理函数（接收 State→返回更新）；Edge 连接节点，有普通边和条件边两种。

**Q3：条件边的作用？**
根据当前状态决定下一步去哪个节点。比如 LLM 输出 tool_call 就去工具节点，输出普通文本就结束。

**Q4：为什么用 LangGraph 而不是 AgentExecutor？**
AgentExecutor 只能用默认的 ReAct 循环，LangGraph 可以自定义任何工作流逻辑。

---

## 八、验收清单

- [ ] 能说出 State、Node、Edge 三个核心概念
- [ ] 能手写一个最简的 LangGraph（3 个节点+条件边）
- [ ] 能解释条件边和普通边的区别
- [ ] 能说出 LangGraph 相比 AgentExecutor 的优势
- [ ] 4 道面试速记全部能讲 1 分钟
