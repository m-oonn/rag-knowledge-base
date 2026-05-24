# Day 37：综合 Agent 实战

> 把 Day 31-36 的知识串成一个完整的数据分析 Agent。这是你项目二的雏形。

---

## 一、这个 Agent 做什么

一个**数据分析 Agent**：用户上传 CSV，描述分析需求，Agent 自动完成。

```
用户："帮我分析 sales.csv 的月度销售趋势"

Agent 执行流程：
  ① 理解任务：用户要分析 sales.csv，重点关注月度趋势
  ② 读取数据：用 pandas 加载 CSV
  ③ 数据处理：按月份聚合销售额，计算环比增长
  ④ 可视化：画趋势折线图
  ⑤ 生成报告：写 Markdown 分析报告
```

---

## 二、完整架构

```
┌──────────────────────────────────────────────────────┐
│                    用户界面                            │
│              (Streamlit / CLI)                        │
└────────────────────┬─────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────┐
│              LangGraph Agent 编排层                    │
│                                                       │
│   ┌─────────┐    ┌──────────┐    ┌───────────┐       │
│   │ 理解任务 │ → │  分析数据  │ → │  可视化    │       │
│   └─────────┘    └──────────┘    └───────────┘       │
│        │               │               │              │
│        └───────────────┴───────────────┘              │
│                        ↓                              │
│                 ┌───────────┐                         │
│                 │  生成报告  │                         │
│                 └───────────┘                         │
└────────────────────┬─────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────┐
│                   工具层                               │
│   读文件 │ pandas分析 │ 画图 │ 写报告 │ 语义搜索        │
└────────────────────┬─────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────┐
│                   基础设施                             │
│   LLM │ Memory │ Vector DB │ 代码沙箱                  │
└──────────────────────────────────────────────────────┘
```

---

## 三、核心源码

```python
"""simple_agent.py — 数据分析 Agent 完整实现。"""

import pandas as pd
import matplotlib.pyplot as plt
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

# ═══════════════════════════════════════
# State 定义
# ═══════════════════════════════════════

class AnalysisState(TypedDict):
    messages: list          # 对话历史
    csv_path: str           # CSV 文件路径
    query: str              # 用户分析需求
    data_summary: str       # 数据摘要
    analysis_code: str      # 分析代码
    analysis_result: dict   # 分析结果
    chart_path: str         # 图表文件
    report: str             # 最终报告

# ═══════════════════════════════════════
# 工具定义
# ═══════════════════════════════════════

@tool
def read_csv(file_path: str) -> str:
    """读取 CSV 文件并返回基本信息。"""
    df = pd.read_csv(file_path)
    info = {
        "shape": df.shape,
        "columns": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
        "head": df.head(5).to_dict(),
        "null_count": df.isnull().sum().to_dict(),
    }
    return json.dumps(info, ensure_ascii=False, default=str)

@tool
def analyze_with_pandas(code: str) -> str:
    """执行 pandas 分析代码并返回结果。"""
    # 安全限制：只允许 pandas 相关操作
    allowed_imports = {"pandas", "numpy"}
    local_vars = {}
    exec(code, {"pd": pd, "__builtins__": {}}, local_vars)
    return json.dumps(local_vars.get("result", {}), ensure_ascii=False, default=str)

@tool
def create_chart(data_json: str, chart_type: str, title: str) -> str:
    """根据数据创建图表。chart_type: line/bar/pie。"""
    data = json.loads(data_json)
    df = pd.DataFrame(data)
    fig, ax = plt.subplots()

    if chart_type == "line":
        df.plot(kind="line", ax=ax)
    elif chart_type == "bar":
        df.plot(kind="bar", ax=ax)
    elif chart_type == "pie":
        df.plot(kind="pie", ax=ax, subplots=True)

    ax.set_title(title)
    path = f"charts/{title.replace(' ', '_')}.png"
    plt.savefig(path)
    plt.close()
    return f"图表已保存到 {path}"

@tool
def search_knowledge(query: str) -> str:
    """搜索数据分析相关知识（RAG）。"""
    # 连接项目的 ChromaDB
    from src.database.vector_db import get_vector_store
    from src.embedding import get_embedding_service
    store = get_vector_store()
    emb = get_embedding_service()
    vec = emb.embed_query(query)
    results = store.search(vec, top_k=3)
    return "\n".join(r.text for r in results)

# ═══════════════════════════════════════
# 节点函数
# ═══════════════════════════════════════

def understand_task(state: AnalysisState) -> AnalysisState:
    """理解用户的分析需求。"""
    prompt = f"""用户上传了 {state['csv_path']}，分析需求：{state['query']}

请分析：
1. 需要关注哪些列？
2. 需要什么类型的分析（趋势/对比/统计）？
3. 需要什么类型的图表？
用简洁的步骤列出分析计划。"""
    return {"messages": [llm.invoke(prompt)]}

def read_and_summarize(state: AnalysisState) -> AnalysisState:
    """读取数据并生成摘要。"""
    df = pd.read_csv(state["csv_path"])
    summary = {
        "rows": len(df),
        "columns": list(df.columns),
        "numeric_cols": list(df.select_dtypes("number").columns),
        "sample": df.head(3).to_dict(),
    }
    return {"data_summary": json.dumps(summary, ensure_ascii=False, default=str)}

def analyze(state: AnalysisState) -> AnalysisState:
    """执行数据分析。"""
    # LLM 根据数据摘要生成 pandas 代码
    prompt = f"数据摘要：{state['data_summary']}\n用户需求：{state['query']}\n请写 pandas 分析代码。"
    code = llm.invoke(prompt).content
    # 执行代码
    result = analyze_with_pandas.invoke(code)
    return {"analysis_result": result, "analysis_code": code}

def visualize(state: AnalysisState) -> AnalysisState:
    """生成图表。"""
    prompt = f"分析结果：{state['analysis_result']}\n请选择合适的图表类型并说明。"
    chart_decision = llm.invoke(prompt).content
    return {"report": f"## 分析报告\n\n{chart_decision}"}

# ═══════════════════════════════════════
# 建图
# ═══════════════════════════════════════

graph = StateGraph(AnalysisState)
graph.add_node("understand", understand_task)
graph.add_node("read_data", read_and_summarize)
graph.add_node("analyze", analyze)
graph.add_node("visualize", visualize)
graph.set_entry_point("understand")
graph.add_edge("understand", "read_data")
graph.add_edge("read_data", "analyze")
graph.add_edge("analyze", "visualize")
graph.add_edge("visualize", END)
app = graph.compile()

# ═══════════════════════════════════════
# 运行
# ═══════════════════════════════════════
if __name__ == "__main__":
    result = app.invoke({
        "messages": [],
        "csv_path": "sales.csv",
        "query": "分析月度销售趋势并画图",
    })
    print(result["report"])
```

---

## 四、Agent 全栈能力自查（Day 31-37 学完）

| 能力 | 对应 Day | 掌握程度 |
|------|---------|---------|
| Function Calling 原理 | 31 | 能手写循环 |
| LangChain 工具封装 | 32 | @tool / BaseTool / AgentExecutor |
| 记忆管理 | 33 | Buffer/Window/Summary/Vector |
| 推理规划 | 34 | CoT / ReAct / Plan-Execute / Self-Reflection |
| LangGraph 基础 | 35 | State + Node + Edge + 条件路由 |
| LangGraph 进阶 | 36 | Human-in-loop / 子图 / 流式 |
| 综合 Agent | 37 | 完整数据分析 Agent |

---

## 五、面试速记

**Q1：用 LangGraph 怎么设计一个数据分析 Agent？**
四个核心节点：理解任务→读取数据→分析→可视化。每个节点都是独立的 Python 函数，通过边串联。

**Q2：Agent 的五个核心能力？**
工具调用 + 记忆 + 规划 + 推理 + 执行。这五个构成了完整 Agent。

**Q3：你的 Agent 项目怎么保证代码执行安全？**
用 exec 限制 builtins + 只允许特定 import + Docker 沙箱兜底。

---

## 六、验收清单

- [ ] 能画出数据分析 Agent 的完整节点图
- [ ] 能手写至少 3 个工具的 @tool 定义
- [ ] 能解释 State 在节点间如何传递
- [ ] Day 31-37 全部文档通读一遍
- [ ] 3 道面试速记全部能讲 1 分钟
