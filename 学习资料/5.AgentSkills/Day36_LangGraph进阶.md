# Day 36：LangGraph 进阶

> 昨天学会了建一个简单的 Agent 循环图。今天学高级功能：Human-in-the-loop、子图、流式输出。

---

## 一、Agent 循环 —— 昨天内容的回顾

昨天的核心代码本身就是一个完整的 Tool Calling Agent：

```
agent（LLM思考） → tools（执行工具） → agent（再思考） → ... → END
```

今天在这个基础上加三个高级能力。

---

## 二、Human-in-the-loop（人工审核）

**问题：** Agent 在关键操作（删除数据、发送邮件、大额交易）之前，应该停下来让人确认。

**实现：** 在图中加一个 `interrupt` 断点。

```python
# 定义一个需要审核的节点
def sensitive_action(state: AgentState) -> AgentState:
    """执行敏感操作前暂停，等待人工确认。"""
    print(f"⚠️ 即将执行: {state['pending_action']}")
    print("等待人工确认...")
    # 这里 LangGraph 会自动暂停，等待外部输入
    return {"status": "waiting_for_approval"}

# 添加节点时设置 interrupt
graph.add_node("sensitive_action", sensitive_action)

# 编译时指定哪些节点需要人工介入
app = graph.compile(interrupt_before=["sensitive_action"])
# 执行到 sensitive_action 节点前会暂停

# ── 运行时 ──
# 正常执行直到断点
result = app.invoke(initial_state)
# 此时 state 被冻结，等待确认

# 人工确认后继续
result = app.invoke(None, config={"thread_id": "123"})
# None 表示"继续"，不添加新的输入
```

**面试这么说：** "我在关键操作节点前设置了 interrupt，Agent 执行到那会暂停，等人工确认后再继续。"

---

## 三、子图（Subgraph）

**问题：** 一个节点内部的逻辑太复杂，想拆成更小的子流程。

**实现：** 把一张图作为另一张图的节点。

```python
# ── 子图：文档处理流程 ──
def parse(state): ...
def chunk(state): ...
def embed(state): ...

doc_subgraph = StateGraph(DocState)
doc_subgraph.add_node("parse", parse)
doc_subgraph.add_node("chunk", chunk)
doc_subgraph.add_node("embed", embed)
doc_subgraph.add_edge("parse", "chunk")
doc_subgraph.add_edge("chunk", "embed")
doc_subgraph.set_entry_point("parse")
doc_subgraph.set_finish_point("embed")
doc_processor = doc_subgraph.compile()

# ── 主图：把子图作为一个节点 ──
main_graph = StateGraph(MainState)
main_graph.add_node("think", agent_think)
main_graph.add_node("process_doc", doc_processor)  # ← 子图作为节点！
main_graph.add_node("answer", generate_answer)
main_graph.add_edge("think", "process_doc")
main_graph.add_edge("process_doc", "answer")
```

**好处：** 每个子图可以独立开发、独立测试、独立复用。

---

## 四、流式输出（Streaming）

LangGraph 支持多种流式模式：

```python
# 模式 1: values 模式 — 每步都输出完整 State
for event in app.stream(input_data, stream_mode="values"):
    last_message = event["messages"][-1]
    print(f"→ {last_message.content}")

# 模式 2: updates 模式 — 只输出这次改了什么
for event in app.stream(input_data, stream_mode="updates"):
    for node_name, update in event.items():
        print(f"节点 [{node_name}] 更新了 {list(update.keys())}")

# 模式 3: 混合模式 — 同时用
for event in app.stream(input_data, stream_mode=["updates", "values"]):
    if isinstance(event, tuple):
        mode, data = event
        if mode == "values":
            print(f"Current state: {data}")
        else:
            print(f"Update: {data}")

# 异步流式输出
async for event in app.astream(input_data, stream_mode="values"):
    print(event)
```

**面试这么说：** "LangGraph 支持流式输出，可以在 Agent 每一步执行后实时推送状态给前端，做类 ChatGPT 的逐字显示。"

---

## 五、错误处理

```python
# 方式 1：每个节点内部 try/except
def agent_node(state: AgentState) -> AgentState:
    try:
        response = llm.invoke(state["messages"])
        return {"messages": [response]}
    except Exception as e:
        return {"error": str(e), "next_step": "error_handler"}

# 方式 2：加一个专门的错误处理节点
def error_handler(state: AgentState) -> AgentState:
    error = state.get("error", "未知错误")
    # 尝试恢复：把错误信息告诉 LLM，让它重试
    state["messages"].append({
        "role": "system",
        "content": f"上一个操作失败: {error}。请换一种方式重试。"
    })
    return {"next_step": "call_model", "error": None}

graph.add_node("error_handler", error_handler)
graph.add_conditional_edges("agent", router, {
    "call_tool": "tools",
    "end": END,
    "error": "error_handler"
})
graph.add_edge("error_handler", "agent")  # 错误处理后回到 LLM 重试
```

---

## 六、在你的项目二中怎么用

```python
# 项目二的数据分析 Agent 可以用 LangGraph 这样设计：

# 节点设计：
#   understand_task → 理解用户要分析什么
#   read_data → 读取 CSV/Excel
#   analyze → pandas 计算
#   visualize → matplotlib 画图
#   report → 生成 Markdown 报告

# 条件路由：
#   分析完后 → 是否需要画图？→ 是 → visualize
#                         → 否 → report
#   画完后 → report

# Human-in-the-loop：
#   在"开始分析"前暂停，让用户确认"准备分析 XYZ, OK？"
```

---

## 七、LangGraph Agent vs LangChain AgentExecutor

| 维度 | AgentExecutor | LangGraph |
|------|-------------|----------|
| 工作流控制 | 自动 ReAct 循环 | 完全自定义 |
| 条件分支 | 不支持 | 原生支持 |
| 人工审核 | 不支持 | interrupt_before |
| 子流程 | 不支持 | 子图嵌套 |
| 流式输出 | 有限 | 多模式 |
| 学习曲线 | 低 | 中 |
| 适合 | 简单 Agent | 生产级复杂工作流 |

---

## 八、动手练习

写一个带 Human-in-the-loop 的 LangGraph：在执行 `delete_file` 节点前暂停，等用户输入 `approve` 才继续。

---

## 九、面试速记

**Q1：Human-in-the-loop 怎么实现？**
`graph.compile(interrupt_before=["敏感节点"])`，执行到该节点前自动暂停，等人工输入后再继续。

**Q2：子图有什么用？**
把复杂节点拆成更小的子流程，独立开发和测试。一张图可以作为另一张图的节点。

**Q3：LangGraph 的流式输出有哪几种模式？**
values（每步输出完整 State）、updates（只输出变更）、异步（astream）。

**Q4：LangGraph 和 AgentExecutor 的核心区别？**
AgentExecutor 只能用默认 ReAct 循环；LangGraph 可以自定义任意工作流，支持条件分支、人机交互、子图嵌套。

---

## 十、验收清单

- [ ] 能解释 interrupt_before 的作用
- [ ] 能手写一个带子图的简单 LangGraph
- [ ] 能说出两种流式模式的区别
- [ ] 能画出项目二数据分析 Agent 的节点图
- [ ] 4 道面试速记全部能讲 1 分钟
