# Day 32：LangChain 工具系统

> 昨天手写了 Function Calling 循环（50+ 行），今天用 LangChain 封装（10 行搞定）。学会三种创建工具的方式 + AgentExecutor。

---

## 一、手写 vs 框架

```
手写 Function Calling（Day 31）:
  自己定义 JSON Schema → 自己解析 tool_calls → 自己执行函数
  → 自己拼回 messages → 自己写循环 → 50+ 行

LangChain Tool + AgentExecutor（Day 32）:
  @tool 自动生成 Schema → AgentExecutor 自动管理循环
  → 内置错误重试 → 开箱即用的 ReAct 推理 → 10 行
```

**什么时候手写，什么时候用框架：**

| 场景 | 建议 |
|------|------|
| 学习原理 | 手写（Day 31） |
| 快速开发 Agent | LangChain |
| 需要精细控制 | 手写或 LangGraph |
| 生产环境 | LangChain + LangGraph |
| 面试 | 先说手写原理，再说框架封装 |

---

## 二、三种创建工具的方式

### 方式一：@tool 装饰器（90% 场景用这个）

```python
from langchain_core.tools import tool

@tool
def get_weather(city: str) -> str:
    """获取指定城市的天气信息。当用户询问天气时调用此工具。"""
    weather_data = {
        "北京": "晴天，28度",
        "上海": "多云，32度",
    }
    return weather_data.get(city, f"未找到{city}的天气")

# 自动生成！
print(get_weather.name)         # "get_weather"
print(get_weather.description)  # "获取指定城市的天气信息。当用户询问天气时调用此工具。"
print(get_weather.args)         # {"city": {"type": "string"}}

# 直接调用
get_weather.invoke("北京")       # "晴天，28度"
```

**关键：** 函数名 → 工具名。docstring → 工具描述。类型注解 → JSON Schema。所以 docstring 一定要写清楚。

### 方式二：StructuredTool.from_function（包装已有函数）

```python
from langchain_core.tools import StructuredTool

# 已有的普通函数，不改它
def legacy_search(query: str, limit: int = 5) -> str:
    return f"搜索'{query}'的结果(最多{limit}条)"

wrapped = StructuredTool.from_function(
    func=legacy_search,
    name="legacy_search",
    description="使用遗留系统搜索数据。当需要查旧系统记录时调用。",
)

wrapped.invoke({"query": "Python", "limit": 3})
```

**适用场景：** 第三方库的函数、不方便加装饰器的函数。

### 方式三：继承 BaseTool（最灵活，2% 场景）

```python
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from datetime import datetime

class TimeInput(BaseModel):
    """时间查询输入"""
    timezone: str = Field(default="Asia/Shanghai", description="时区")

class GetTimeTool(BaseTool):
    name: str = "get_current_time"
    description: str = "获取当前日期和时间。当用户问几点或日期时调用。"
    args_schema: type = TimeInput

    def _run(self, timezone: str = "Asia/Shanghai") -> str:
        now = datetime.now()
        return f"{now.strftime('%Y-%m-%d %H:%M:%S')}"

    async def _arun(self, timezone: str = "Asia/Shanghai") -> str:
        return self._run(timezone)  # 异步版本
```

**适用场景：** 需要初始化资源（数据库连接）、有状态、需要异步执行。

### 三种方式对比

| 特性 | @tool | StructuredTool | BaseTool |
|------|-------|---------------|----------|
| 代码量 | 3 行 | 6 行 | 10+ 行 |
| Schema 生成 | 自动 | 半自动 | 手动 |
| 异步支持 | 有限 | 有限 | 完整 |
| 使用频率 | 90% | 8% | 2% |

---

## 三、Pydantic 精确控制参数 Schema

`@tool` 会自动推断参数类型，但可以用 Pydantic 精确控制：

```python
from pydantic import BaseModel, Field

class SearchInput(BaseModel):
    """知识库搜索参数"""
    query: str = Field(description="搜索关键词")
    max_results: int = Field(default=5, ge=1, le=20, description="返回数量(1-20)")
    category: str = Field(default="all", description="类别: all/tech/news")

@tool(args_schema=SearchInput)
def search(query: str, max_results: int = 5, category: str = "all") -> str:
    """在知识库中搜索文档。当用户需要查资料时使用。"""
    return f"在{category}中搜索'{query}'，返回{max_results}条结果"

# LLM 能看到每个参数的描述和限制，传参更准确
```

---

## 四、完整源码内嵌

### 4.1 三种方式各写一个工具

```python
from langchain_core.tools import tool, BaseTool, StructuredTool
from pydantic import BaseModel, Field
from datetime import datetime
import json

# ── 方式一：@tool 装饰器（最简单）──
@tool
def get_weather(city: str) -> str:
    """获取指定城市的实时天气。当用户询问天气时调用。"""
    data = {"北京": "晴天28度", "上海": "多云32度", "广州": "雷阵雨35度"}
    return data.get(city, f"未找到{city}天气")

@tool
def calculator(expression: str) -> str:
    """计算数学表达式。当用户需要计算时调用。"""
    try:
        allowed = {"abs": abs, "round": round, "min": min, "max": max}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"{expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"

@tool
def search_docs(query: str) -> str:
    """在知识库中搜索技术文档。当用户问技术概念时调用。"""
    knowledge = {
        "RAG": "RAG通过检索外部知识增强LLM回答。",
        "Agent": "Agent能自主使用工具完成任务的AI系统。",
        "LangChain": "LangChain是LLM应用开发框架。",
    }
    results = [f"[{k}] {v}" for k,v in knowledge.items() if k.lower() in query.lower()]
    return "\n".join(results) if results else "未找到相关信息"

# ── 方式二：Pydantic 复杂输入 ──
class DataAnalysisInput(BaseModel):
    file_path: str = Field(description="CSV文件路径")
    analysis_type: str = Field(
        default="summary",
        description="分析类型: summary/trend/compare"
    )

@tool(args_schema=DataAnalysisInput)
def analyze_data(file_path: str, analysis_type: str = "summary") -> str:
    """分析CSV数据文件。当用户需要分析数据时调用。"""
    return json.dumps({
        "file": file_path, "type": analysis_type,
        "summary": "1000行5列，平均值42.5"
    }, ensure_ascii=False)

# ── 方式三：BaseTool 子类（有状态/需要初始化）──
class TimeInput(BaseModel):
    timezone: str = Field(default="Asia/Shanghai", description="时区")

class GetTimeTool(BaseTool):
    name: str = "get_current_time"
    description: str = "获取当前时间。当用户问几点或日期时调用。"
    args_schema: type = TimeInput

    def _run(self, timezone: str = "Asia/Shanghai") -> str:
        now = datetime.now()
        w = ["周一","周二","周三","周四","周五","周六","周日"]
        return f"{now.strftime('%Y-%m-%d %H:%M:%S')}, {w[now.weekday()]}"

# ── 方式四：StructuredTool 包装已有函数 ──
def existing_search(query: str, limit: int = 5) -> str:
    return f"搜索'{query}'(最多{limit}条): 3条记录"

legacy_tool = StructuredTool.from_function(
    func=existing_search,
    name="legacy_search",
    description="使用遗留系统搜索数据。",
)
```

### 4.2 构建 ReAct Agent（核心）

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

# 收集所有工具
tools = [get_weather, calculator, search_docs, GetTimeTool()]

# ReAct Prompt 模板——定义 Agent 怎么思考
react_prompt = PromptTemplate.from_template("""
Answer the following questions as best you can. You have access to:

{tools}

Use this format:
Question: the input question
Thought: what should I do?
Action: the tool name [{tool_names}]
Action Input: the input to the tool
Observation: the result from the tool
... (repeat Thought/Action/Observation as needed)
Thought: I now know the answer
Final Answer: the final answer in Chinese

Question: {input}
Thought:{agent_scratchpad}
""")

# 创建 Agent + Executor
agent = create_react_agent(llm=llm, tools=tools, prompt=react_prompt)

executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,              # 打印完整思考过程！
    max_iterations=5,          # 最多循环 5 次
    handle_parsing_errors=True, # 格式错误自动重试
)

# 运行
result = executor.invoke({"input": "北京天气怎么样？需要带伞吗？"})
print(result["output"])
```

### 4.3 verbose=True 的输出示例

```
> Entering new AgentExecutor chain...

Thought: 用户想知道北京天气和是否需要带伞
        我需要先获取天气信息
Action: get_weather
Action Input: {"city": "北京"}
Observation: 晴天28度

Thought: 晴天不需要带伞，可以给最终答案了
Final Answer: 北京今天晴天，28度。不需要带伞！

> Finished chain.
```

**verbose=True 是调试利器**——你能看到 Agent 的每一步推理过程。

### 4.4 AgentExecutor 底层做了什么

```
while iterations < max:
    action = agent.plan(input)     ← LLM 思考+决策（输出 Thought/Action）

    if action == "Final Answer":   ← 结束了
        return answer

    result = tool.run(action)      ← 执行工具
    observation = result           ← 拿到结果
    feed back to agent             ← 把结果告诉 LLM，继续循环
end while
```

**这就是 Day 31 手写循环的封装版。**

---

## 五、错误处理

```python
@tool
def risky_division(a: float, b: float) -> str:
    """执行除法。"""
    if b == 0:
        raise ValueError("除数不能为零!")
    return f"{a} / {b} = {a / b}"

# 正常调用
risky_division.invoke({"a": 10, "b": 3})  # "10 / 3 = 3.33..."

# 错误调用
risky_division.invoke({"a": 10, "b": 0})  # ValueError!
```

**AgentExecutor 的 `handle_parsing_errors=True` 会：**
1. 捕获工具异常
2. 把错误信息作为 Observation 反馈给 LLM
3. LLM 看到错误后可以换一种方式重试，或告知用户

---

## 六、工具设计三原则

**① 粒度适中**
```
太细: get_city(), get_temp(), get_humidity() → 调 3 次才拿到完整天气
太粗: do_everything() → LLM 不知道什么时候该调
刚好: get_weather(city) → 一次拿全
```

**② Description 写清楚**
```
差: "获取天气"
好: "获取指定城市的实时天气（温度+湿度+天气状况）。当用户询问天气或是否需要带伞时调用。"
```

**③ 返回值 LLM 能读懂**
```
差: return {"temp": 28}              # dict LLM 看不懂
好: return "北京晴天28度湿度45%"        # 自然语言 LLM 能直接用
```

---

## 七、动手练习

### 练习 1：用 @tool 装饰器写一个新工具

写一个 `get_stock_price(ticker: str)`，返回模拟股价，加入 tools 列表，用 AgentExecutor 测试。

### 练习 2：观察 verbose 输出

开 `verbose=True`，问"今天北京天气怎么样？帮我算 256*128 等于多少？"观察 Agent 先调哪个工具、推理链是什么样的。

### 练习 3：触发错误重试

写一个工具故意抛异常，设置 `handle_parsing_errors=True`，看 Agent 如何处理。

---

## 八、面试速记

**Q1：LangChain 创建 Tool 有哪三种方式？**
@tool 装饰器（最简单，90%），StructuredTool.from_function（包装已有函数），继承 BaseTool（最灵活）。

**Q2：@tool 怎么生成 Schema？**
从函数的类型注解自动推断：参数类型→JSON type，docstring→description。

**Q3：AgentExecutor 什么作用？**
管理 Agent 执行循环：LLM 思考→调用工具→结果反馈→继续思考，直到得出最终答案。就是对 Day 31 手写循环的封装。

**Q4：什么是 ReAct Agent？**
Reasoning + Acting 交替：先思考（Thought）→ 行动（Action）→ 观察结果（Observation）→ 循环直到答案。这是最经典的 Agent 推理策略。

**Q5：verbose=True 有什么用？**
打印 Agent 完整推理过程（Thought/Action/Observation），开发调试必备。

**Q6：工具出错 Agent 怎么办？**
handle_parsing_errors=True 捕获异常→作为 Observation 反馈给 LLM→LLM 可重试或告知用户。

---

## 九、验收清单

- [ ] @tool 装饰器、StructuredTool、BaseTool 三种方式各写过一个工具
- [ ] 能用 create_react_agent + AgentExecutor 创建一个完整的 Agent
- [ ] 能解释 AgentExecutor 的底层循环逻辑（对照 Day 31）
- [ ] 能解释 ReAct 的 Thought→Action→Observation 循环
- [ ] 练习 1、2、3 全部跑过
- [ ] 6 道面试速记题全部能讲 1 分钟
