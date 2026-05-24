# Day 31：工具调用原理 (Function Calling)

> Agent 的基石。LLM 只能说话，Function Calling 让它能做事。

---

## 一、什么是 Function Calling

### 1.1 一句话理解

Function Calling = **让 LLM 自己决定：要不要调工具、调哪个、传什么参数。**

LLM 本身不会执行代码、不会访问网络、不会查数据库。但通过 Function Calling，LLM 可以输出一段 JSON 指令，告诉你的程序"帮我执行这个函数"。

### 1.2 为什么是 Agent 的基础

```
没有 Function Calling 的 LLM：
  用户："今天北京天气怎么样？"
  LLM："我没有实时数据，无法告诉你。"  ← 废物

有 Function Calling 的 LLM：
  用户："今天北京天气怎么样？"
  LLM（思考）：我需要调 get_weather(city="北京")
  程序：执行 get_weather("北京") → "晴天，25度"
  LLM："今天北京晴天，25度。"  ← 有用！
```

**没有工具调用的 Agent 只是一个聊天机器人。**

### 1.3 与你项目的关系

| 项目 | Function Calling 用在哪 |
|------|----------------------|
| RAG 知识库 | LLM 决定调用 search_knowledge_base 检索 |
| 数据分析 Agent | LLM 调用 read_csv、run_code、save_chart |
| 面试 | "请解释 Function Calling 原理" 必考 |

---

## 二、完整流程：五步循环

```
Step 1  用户输入："帮我查一下北京天气"
           ↓
Step 2  发送给 LLM（附带工具描述 JSON Schema）
           ↓
Step 3  LLM 返回工具调用指令（不是最终答案！）
        {
          "tool_calls": [{
            "function": {
              "name": "get_weather",
              "arguments": "{\"city\": \"北京\"}"
            }
          }]
        }
           ↓
Step 4  程序执行工具，拿到结果
        result = get_weather("北京")  →  "晴天，25度"
           ↓
Step 5  把结果发回 LLM，LLM 生成最终答案
        → "今天北京晴天，气温25度。"
```

**关键认知：LLM 不执行任何代码。** 它只是建议调哪个函数。你的程序负责解析建议→执行函数→把结果喂回去。

### LLM 什么时候调用工具 vs 直接回答

| 用户问题 | LLM 行为 | 原因 |
|---------|---------|------|
| "北京天气怎么样" | 调用 get_weather | 需要实时数据 |
| "Python 怎么读文件" | 直接回答 | LLM 自己知道 |
| "帮我算 127×365" | 调用 calculate | 计算交给工具更准 |
| "你好" | 直接回答 | 闲聊不需要工具 |

---

## 三、用 JSON Schema 定义工具

告诉 LLM 有什么工具可用、每个工具怎么用。

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的实时天气。当用户询问天气时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如'北京'、'上海'"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位，默认 celsius"
                    }
                },
                "required": ["city"]
            }
        }
    }
]
```

**description 写法原则（LLM 根据它决定是否调用）：**

- 差："获取天气"
- 好："获取指定城市的实时天气信息，包括温度和天气状况。当用户询问天气时使用。"

---

## 四、完整源码内嵌：手写 Function Calling 循环

以下是 `day31_1_function_calling.py` 的核心部分。

### 4.1 定义工具函数 + 注册表

```python
import json
from datetime import datetime

# ── 天气查询 ──
def get_weather(city: str) -> str:
    weather_data = {
        "北京": {"temp": 28, "condition": "晴天", "humidity": 45},
        "上海": {"temp": 32, "condition": "多云", "humidity": 72},
        "广州": {"temp": 35, "condition": "雷阵雨", "humidity": 88},
        "成都": {"temp": 26, "condition": "阴天", "humidity": 65},
    }
    if city in weather_data:
        d = weather_data[city]
        return json.dumps({"city": city, "temperature": d["temp"],
                          "condition": d["condition"], "humidity": d["humidity"]},
                         ensure_ascii=False)
    return json.dumps({"error": f"未找到{city}的天气"}, ensure_ascii=False)

# ── 计算器 ──
def calculate(expression: str) -> str:
    try:
        allowed = {"abs": abs, "round": round, "min": min, "max": max, "pow": pow}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return json.dumps({"expression": expression, "result": result}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)

# ── 知识搜索（模拟 RAG）──
def search_knowledge(query: str, top_k: int = 3) -> str:
    knowledge_base = [
        {"title": "Python", "content": "Python 由 Guido van Rossum 于1991年创建。"},
        {"title": "RAG", "content": "RAG 通过检索外部知识库增强 LLM 回答准确性。"},
        {"title": "Agent", "content": "Agent 能自主决策和执行任务，核心：工具调用+记忆+规划。"},
    ]
    scores = [(sum(1 for k in query if k in d["title"]+d["content"]), d)
              for d in knowledge_base]
    scores.sort(key=lambda x: x[0], reverse=True)
    results = [s[1] for s in scores[:top_k] if s[0] > 0]
    if results:
        return json.dumps({"query": query, "results": results}, ensure_ascii=False)
    return json.dumps({"query": query, "results": [], "message": "未找到"}, ensure_ascii=False)

# ── 当前时间 ──
def get_current_time() -> str:
    now = datetime.now()
    weekdays = ["周一","周二","周三","周四","周五","周六","周日"]
    return json.dumps({
        "date": now.strftime("%Y-%m-%d"), "time": now.strftime("%H:%M:%S"),
        "weekday": weekdays[now.weekday()]
    }, ensure_ascii=False)

# ── 工具注册表：名字 → 函数 ──
TOOL_REGISTRY = {
    "get_weather": get_weather,
    "calculate": calculate,
    "search_knowledge": search_knowledge,
    "get_current_time": get_current_time,
}
```

### 4.2 工具 JSON Schema 描述

```python
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的实时天气信息。当用户询问天气时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "城市名，如'北京'"}
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算数学表达式。当用户需要计算时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式，如'2+3*4'"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "在知识库中搜索文档。当用户提问需要查资料时使用。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"},
                    "top_k": {"type": "integer", "description": "返回数量，默认3", "default": 3}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前日期和时间。当用户问几点或今天日期时使用。",
            "parameters": {"type": "object", "properties": {}, "required": []}
        }
    }
]
```

### 4.3 核心：Function Calling 循环

```python
def function_calling_loop(user_query: str, max_rounds: int = 5) -> str:
    """手写 Function Calling 循环。这就是 Agent 的底层工作原理。"""
    
    # 初始化消息历史
    messages = [
        {
            "role": "system",
            "content": "你是一个智能助手，可以使用工具。需要实时数据时调用合适工具。用中文回答。"
        },
        {"role": "user", "content": user_query}
    ]

    for round_num in range(1, max_rounds + 1):
        # ① 调用 LLM（真实或模拟），传工具描述
        response = call_llm_with_tools(messages, TOOLS_SCHEMA)

        # ② LLM 请求调用工具？
        if "tool_calls" in response:
            tool_calls = response["tool_calls"]
            
            # 记录 assistant 消息（含 tool_calls）
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": tc.get("id"), "type": "function",
                     "function": tc["function"]}
                    for tc in tool_calls
                ]
            })

            # ③ 执行每个工具
            for tc in tool_calls:
                func_name = tc["function"]["name"]
                func_args = tc["function"]["arguments"]
                
                result = execute_tool(func_name, func_args)  # 真正执行
                
                # ④ 把结果加入消息历史
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id"),
                    "name": func_name,
                    "content": result
                })
            # 继续循环（LLM 可能还要调更多工具）
        else:
            # ⑤ LLM 直接回答 → 结束
            return response.get("content", "")

    return "达到最大调用轮数"


def execute_tool(name: str, arguments: str) -> str:
    """执行工具并返回结果。"""
    if name not in TOOL_REGISTRY:
        return json.dumps({"error": f"未知工具: {name}"})
    try:
        args = json.loads(arguments) if arguments else {}
        return TOOL_REGISTRY[name](**args)
    except Exception as e:
        return json.dumps({"error": str(e)})


def call_llm_with_tools(messages, tools):
    """调用 LLM。有真实 LLM 用真实，没有则用模拟。"""
    if USE_SIMULATION:
        return simulate_llm_decision(messages[-1]["content"], tools)
    else:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            temperature=0.1,
        )
        choice = response.choices[0]
        if choice.message.tool_calls:
            return {"tool_calls": [{
                "id": tc.id,
                "function": {"name": tc.function.name, "arguments": tc.function.arguments}
            } for tc in choice.message.tool_calls]}
        return {"content": choice.message.content}
```

### 4.4 模拟模式（无 LLM 时的备选）

```python
def simulate_llm_decision(user_message, tools, tool_results=None):
    """无 LLM 时用规则引擎模拟决策，理解流程原理。"""
    if tool_results:
        # 已有工具结果 → 生成最终回答
        parts = []
        for tr in tool_results:
            data = json.loads(tr["result"])
            if tr["name"] == "get_weather":
                parts.append(f"{data['city']}今天{data['condition']}，{data['temperature']}度")
            elif tr["name"] == "calculate":
                parts.append(f"计算结果：{data['expression']}={data['result']}")
            elif tr["name"] == "search_knowledge":
                if data.get("results"):
                    for d in data["results"]:
                        parts.append(f"{d['title']}: {d['content']}")
            elif tr["name"] == "get_current_time":
                parts.append(f"现在是{data['date']} {data['time']}，{data['weekday']}")
        return {"content": "\n".join(parts)}

    # 分析关键词 → 匹配工具
    msg = user_message.lower()
    weather_cities = ["北京","上海","广州","成都","深圳"]
    
    if "天气" in msg or "气温" in msg:
        for city in weather_cities:
            if city in msg:
                return {"tool_calls": [{"function": {
                    "name": "get_weather",
                    "arguments": json.dumps({"city": city}, ensure_ascii=False)
                }}]}
    
    if "计算" in msg or "算" in msg or "等于" in msg:
        expr = "2+2"  # fallback，真实情况会做更复杂的提取
        return {"tool_calls": [{"function": {
            "name": "calculate",
            "arguments": json.dumps({"expression": expr}, ensure_ascii=False)
        }}]}
    
    if "时间" in msg or "几点" in msg or "日期" in msg:
        return {"tool_calls": [{"function": {
            "name": "get_current_time", "arguments": "{}"
        }}]}
    
    # 知识检索（关键词匹配）
    if any(kw in msg for kw in ["什么是","RAG","Agent","Python","LangChain"]):
        return {"tool_calls": [{"function": {
            "name": "search_knowledge",
            "arguments": json.dumps({"query": user_message, "top_k": 3}, ensure_ascii=False)
        }}]}
    
    return {"content": f"关于「{user_message}」，我可以直接回答。"}
```

---

## 五、三种 API 格式对比

| 特性 | OpenAI | Claude | Ollama |
|------|--------|--------|--------|
| 工具定义字段 | `parameters` | `input_schema` | `parameters`（同 OpenAI） |
| 工具调用字段 | `tool_calls` | `tool_use` content block | `tool_calls`（同 OpenAI） |
| 参数格式 | JSON 字符串 | Python dict | JSON 字符串 |
| tool_choice | auto/required/none | auto/any/tool | auto |
| 需要联网 | 是 | 是 | 否（完全本地） |

---

## 六、Function Calling vs MCP vs LangChain Tools

```
┌─────────────────────────────────────────┐
│ LangChain Tools / Agent 框架 ← 框架封装层  │  Day 32
├─────────────────────────────────────────┤
│ MCP（Model Context Protocol）  ← 标准化协议 │  Day 35
├─────────────────────────────────────────┤
│ Function Calling  ← LLM API 原生能力      │  Day 31（今天）
└─────────────────────────────────────────┘
```

类比：
- Function Calling = 你会打电话（**能力**）
- MCP = 电话号码簿标准（**规范**）
- LangChain Tools = 秘书帮你打电话（**封装**）

---

## 七、Function Calling 本质总结

```
while True:
    response = llm(messages, tools)      # ① 问 LLM：要不要调工具？

    if response.has_tool_calls:          # ② LLM 要调工具
        for call in response.tool_calls:
            result = execute(call)       # ③ 你执行工具（LLM 不执行任何代码！）
            messages.append(result)      # ④ 把结果告诉 LLM
    else:
        return response.content          # ⑤ LLM 直接回答 → 结束
```

---

## 八、动手练习

### 练习 1：加一个新工具

在 `TOOL_REGISTRY` 中添加一个 `translate(text, target_lang)` 函数，在 `TOOLS_SCHEMA` 中添加它的 JSON Schema，跑一个翻译查询。

### 练习 2：观察错误处理

故意写一个错误的工具名，看 `execute_tool` 如何优雅处理而不崩溃。

### 练习 3：多步调用

跑 `get_weather("北京")` → 拿到温度 → `calculate("28*9/5+32")` 把摄氏度转华氏度。体会为什么需要"循环"而不是"一次调用"。

---

## 九、面试速记

**Q1：Function Calling 是什么？**
LLM 通过 JSON 格式输出工具调用指令，程序执行后把结果反馈给 LLM。LLM 本身不执行代码。

**Q2：完整的调用循环？**
用户输入 → LLM+工具描述 → 返回 tool_calls → 程序执行工具 → 结果拼回 messages → 再次调 LLM → 循环直到最终回答。

**Q3：Agent 和普通 LLM 的根本区别？**
普通 LLM 只能说话，Agent 有工具调用能力，能做事。

**Q4：怎么让 LLM 知道什么时候该用工具？**
通过 description 字段。描述越清楚，LLM 判断越准确。差："获取天气"；好："获取指定城市实时天气。当用户询问天气时使用此工具。"

---

## 十、验收清单

- [ ] 能画出 Function Calling 的五步循环图
- [ ] 能解释"LLM 不执行代码"这个关键点
- [ ] 能手写一个 JSON Schema 工具描述
- [ ] 能说出三种 API（OpenAI/Claude/Ollama）的差异
- [ ] 练习 1、2、3 全部跑过
- [ ] 4 道面试速记题全部能讲 1 分钟
