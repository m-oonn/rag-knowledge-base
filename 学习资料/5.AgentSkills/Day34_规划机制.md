# Day 34：规划机制 (Planning)

> Agent 不是一步出答案的。复杂任务需要先想再干、干完再想。今天学四种推理/规划模式。

---

## 一、为什么需要规划

简单问题一步搞定，复杂问题需要拆解：

```
简单："今天天气怎么样？" → 直接调 get_weather → 回答
复杂："对比过去一周北京和上海的天气趋势" →
  ① 分别查两城市 7 天天气（7×2=14 次工具调用）
  ② 比较趋势
  ③ 生成对比报告
```

**没有规划能力的 Agent = 只会做一步操作的机器人。**

---

## 二、四种规划模式

| 模式 | 流程 | 适合 | 复杂度 |
|------|------|------|--------|
| Chain of Thought | 边想边说，一步到位 | 推理题、计算题 | 低 |
| ReAct | 想→做→看→再想→再做 | **通用 Agent（最常用）** | 中 |
| Plan-and-Execute | 先制定全计划，再逐步执行 | 多步复杂任务 | 高 |
| Self-Reflection | 做完后自我检查修正 | 需要高质量输出 | 高 |

---

## 三、Chain of Thought（思维链）

最简单的推理模式：**在 Prompt 里说"一步一步想"。**

```
无 CoT："127×365=？" → LLM 直接猜 → 可能答错

有 CoT："127×365=？请逐步推理。"
  → "127×365 = 127×(300+60+5)
     = 127×300 + 127×60 + 127×5
     = 38100 + 7620 + 635
     = 46355"
```

**关键：不需要改任何代码，只需要改 Prompt。**

---

## 四、ReAct（推理+行动）—— 最核心的模式

ReAct = **Reasoning（思考）+ Acting（行动）循环**。

```
Thought: 我需要知道北京天气 → 调天气工具
Action: get_weather("北京")
Observation: 晴天，28度

Thought: 还需要知道上海天气
Action: get_weather("上海")
Observation: 多云，32度

Thought: 已拿到两个城市数据，可以对比了
Final Answer: 北京28度晴天，上海32度多云。上海比北京热4度。
```

### ReAct Prompt 模板（最关键的代码）

```python
REACT_PROMPT = """
Answer the following questions using available tools.

Tools: {tools}
Tool Names: {tool_names}

Use this exact format:
Question: the input question
Thought: what should I do?
Action: the tool name [{tool_names}]
Action Input: the parameters for the tool
Observation: what the tool returned
... (repeat Thought/Action/Observation as needed)
Thought: I now have the answer
Final Answer: the final answer in Chinese

Question: {input}
Thought:{agent_scratchpad}
"""
```

### 用 LangChain 创建 ReAct Agent

```python
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate

prompt = PromptTemplate.from_template(REACT_PROMPT)

agent = create_react_agent(llm=llm, tools=tools, prompt=prompt)

executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,              # 打印思考过程
    max_iterations=5,          # 最多思考-行动循环 5 次
    handle_parsing_errors=True, # LLM 格式写错自动重试
)

result = executor.invoke({"input": "北京和上海今天天气对比"})
```

### ReAct 循环的本质

```
while not done and iterations < max:
    step = agent.think(messages)    # ① LLM 想：该调哪个工具？

    if step.is_final_answer:        # ② 有答案了？
        return step.answer

    result = tools[step.action].run(step.input)  # ③ 执行工具
    messages.append(result)          # ④ 把结果告诉 LLM，回到 ①
```

---

## 五、Plan-and-Execute（先规划后执行）

解决"ReAct 走一步看一步有时候缺乏全局视角"的问题。

```
Step 1: 制定计划（Planner）
  用户："帮我分析 sales.csv 的销售趋势并生成报告"

  LLM 制定计划：
    1. 读取 CSV 文件，查看数据结构
    2. 按月份聚合销售额
    3. 计算环比增长率
    4. 画趋势图
    5. 写分析报告

Step 2: 逐步执行（Executor）
  执行步骤 1 → 观察结果
  执行步骤 2 → 观察结果
  ...
  执行步骤 5 → 完成
```

```python
# 规划器：生成步骤列表
plan_prompt = f"将以下任务拆分为具体步骤：\n{user_task}\n用数字列表输出。"
plan_response = call_llm(plan_prompt)
steps = parse_steps(plan_response)  # ["1.读CSV", "2.聚合", "3.计算增长率", ...]

# 执行器：逐步执行
results = []
for step in steps:
    result = executor.invoke({"input": step})
    results.append(result)
    # 如果某步失败，可以调整计划
```

---

## 六、Self-Reflection（自我反思）

做完后让 LLM 自己检查修正。

```
用户："写一篇关于 AI 发展的文章"

第 1 轮：LLM 写出草稿
第 2 轮（反思）：
  Prompt: "检查这篇文章：① 有没有事实错误？② 有没有逻辑漏洞？③ 表达是否清晰？请给修改建议。"
  LLM 输出修改建议
第 3 轮（修正）：根据建议修改文章
第 4 轮（再反思）：再检查一遍
→ 最终高质量输出
```

---

## 七、四种模式选型指南

```
简单计算/推理     → Chain of Thought（只改 Prompt）
标准 Agent 任务   → ReAct（默认首选）
复杂多步任务      → Plan-and-Execute
需要高质量输出    → Plan-and-Execute + Self-Reflection
```

---

## 八、面试速记

**Q1：Agent 为什么需要规划？**
复杂任务不能一步完成。需要拆分成子步骤，想一步干一步检查一步。

**Q2：ReAct 是什么？**
Reasoning + Acting。Agent 思考该用什么工具→执行→观察结果→再思考→循环直到得出答案。最经典的 Agent 推理模式。

**Q3：ReAct vs Plan-and-Execute 区别？**
ReAct 走一步看一步（灵活），Plan-and-Execute 先制定完整计划再逐步执行（全局视角）。复杂多步任务优先 Plan-and-Execute。

**Q4：CoT 和 ReAct 区别？**
CoT 只是内部推理（不打工具），ReAct 是推理+行动（可以调工具）。

**Q5：Self-Reflection 的作用？**
让 LLM 检查自己的输出并修正。提高答案质量的最后一道防线。

---

## 九、验收清单

- [ ] 能写出 ReAct 的 Thought→Action→Observation 循环模板
- [ ] 能解释 ReAct 和 Plan-and-Execute 的区别和选型
- [ ] 能解释 AgentExecutor 的 max_iterations 参数是干什么的
- [ ] 知道 CoT 不需要改代码只改 Prompt
- [ ] 5 道面试速记全部能讲 1 分钟
