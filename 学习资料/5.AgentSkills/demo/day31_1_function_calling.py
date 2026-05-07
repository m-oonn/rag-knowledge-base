"""
Day 31 Demo 1: 工具调用原理 (Function Calling) 完整实战
运行方式: python day31_1_function_calling.py

前置条件（任选一个）:
  - Ollama 运行中，并拉取了支持 function calling 的模型（推荐 qwen2.5:7b）
  - 或 .env 中配置 DEEPSEEK_API_KEY
  - 或以上都没有 → 自动进入模拟模式（不调用任何 LLM，用规则模拟）

学习目标:
1. 理解 Function Calling 的完整循环（用户→LLM→工具→LLM→回答）
2. 学会用 JSON Schema 描述工具
3. 手动实现工具调用循环（不依赖任何 Agent 框架）
4. 处理需要工具 vs 不需要工具的情况
5. 处理多步串行工具调用
"""

import json
import time
import os
from datetime import datetime

# ============================================================
# Part 0: 初始化 - 检测 LLM 可用性
# ============================================================

print("=" * 60)
print("  Day 31: Function Calling - 工具调用原理")
print("=" * 60)

# 尝试加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # 没有 dotenv 也没关系，后面会自动检测

# 全局变量：标记是否使用模拟模式
USE_SIMULATION = False
client = None
model_name = None


def init_llm():
    """初始化 LLM 客户端，按优先级选择: Ollama > DeepSeek > 模拟模式"""
    global USE_SIMULATION, client, model_name

    # 尝试导入 openai 库
    try:
        from openai import OpenAI
    except ImportError:
        print("  [INFO] openai 库未安装，进入模拟模式")
        print("  (安装方法: pip install openai)")
        USE_SIMULATION = True
        return

    # 1. 尝试 Ollama（本地免费）
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        models = [m["name"] for m in resp.json().get("models", [])]
        if models:
            # 优先选择支持 function calling 的模型
            preferred = ["qwen2.5:7b", "qwen2.5:14b", "qwen2.5:3b", "llama3.1", "mistral"]
            model_name = models[0]  # 默认用第一个
            for p in preferred:
                for m in models:
                    if p in m:
                        model_name = m
                        break
            client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
            print(f"  [OK] 使用 Ollama 模型: {model_name}")
            return
    except Exception:
        pass

    # 2. 尝试 DeepSeek
    ds_key = os.getenv("DEEPSEEK_API_KEY")
    if ds_key:
        client = OpenAI(api_key=ds_key, base_url="https://api.deepseek.com")
        model_name = "deepseek-chat"
        print(f"  [OK] 使用 DeepSeek: {model_name}")
        return

    # 3. 进入模拟模式
    print("  [INFO] 未检测到 LLM，进入模拟模式")
    print("  (模拟模式下用规则引擎代替 LLM 的决策过程)")
    USE_SIMULATION = True


init_llm()
print()


# ============================================================
# Part 1: 定义工具函数（真正执行的代码）
# ============================================================

print("=" * 60)
print("  Part 1: 定义工具函数")
print("=" * 60)


def get_weather(city: str) -> str:
    """
    获取城市天气（模拟数据）。
    真实项目中这里会调用天气 API（如和风天气、OpenWeatherMap）。
    """
    # 模拟天气数据
    weather_data = {
        "北京": {"temp": 28, "condition": "晴天", "humidity": 45},
        "上海": {"temp": 32, "condition": "多云", "humidity": 72},
        "广州": {"temp": 35, "condition": "雷阵雨", "humidity": 88},
        "成都": {"temp": 26, "condition": "阴天", "humidity": 65},
        "深圳": {"temp": 33, "condition": "晴转多云", "humidity": 70},
    }
    if city in weather_data:
        d = weather_data[city]
        return json.dumps({
            "city": city,
            "temperature": d["temp"],
            "condition": d["condition"],
            "humidity": d["humidity"],
            "unit": "celsius"
        }, ensure_ascii=False)
    else:
        return json.dumps({"error": f"未找到城市 '{city}' 的天气数据"}, ensure_ascii=False)


def calculate(expression: str) -> str:
    """
    计算数学表达式。
    使用 Python eval 执行（生产环境应该用安全的表达式解析器）。
    """
    try:
        # 安全限制：只允许数学运算
        allowed_names = {"abs": abs, "round": round, "min": min, "max": max, "pow": pow}
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return json.dumps({"expression": expression, "result": result}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": f"计算错误: {str(e)}"}, ensure_ascii=False)


def search_knowledge(query: str, top_k: int = 3) -> str:
    """
    搜索知识库（模拟 RAG 检索）。
    真实项目中会连接 ChromaDB 做向量检索。
    """
    # 模拟知识库
    knowledge_base = [
        {"title": "Python 简介", "content": "Python 是一种解释型、面向对象的高级编程语言。由 Guido van Rossum 于 1991 年发布。"},
        {"title": "FastAPI 框架", "content": "FastAPI 是一个现代、快速的 Web 框架，支持异步编程，自动生成 API 文档。"},
        {"title": "RAG 技术", "content": "RAG（检索增强生成）通过检索外部知识库来增强 LLM 的回答准确性，避免幻觉。"},
        {"title": "LangChain", "content": "LangChain 是一个 LLM 应用开发框架，提供 Chain、Agent、Memory 等抽象。"},
        {"title": "向量数据库", "content": "向量数据库（如 Chroma、Pinecone）专门存储和检索高维向量，是 RAG 的核心组件。"},
        {"title": "Agent 智能体", "content": "Agent 是能自主决策和执行任务的 AI 系统，核心能力包括：工具调用、记忆、规划。"},
    ]

    # 简单的关键词匹配（真实项目用向量相似度）
    scores = []
    for doc in knowledge_base:
        score = sum(1 for keyword in query if keyword in doc["title"] + doc["content"])
        scores.append((score, doc))

    scores.sort(key=lambda x: x[0], reverse=True)
    results = [s[1] for s in scores[:top_k] if s[0] > 0]

    if results:
        return json.dumps({"query": query, "results": results}, ensure_ascii=False)
    else:
        return json.dumps({"query": query, "results": [], "message": "未找到相关知识"}, ensure_ascii=False)


def get_current_time() -> str:
    """获取当前日期和时间。"""
    now = datetime.now()
    return json.dumps({
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "weekday": ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][now.weekday()]
    }, ensure_ascii=False)


# 工具注册表：名字 → 函数的映射
TOOL_REGISTRY = {
    "get_weather": get_weather,
    "calculate": calculate,
    "search_knowledge": search_knowledge,
    "get_current_time": get_current_time,
}

print(f"\n  已注册 {len(TOOL_REGISTRY)} 个工具:")
for name in TOOL_REGISTRY:
    print(f"    - {name}")
print()


# ============================================================
# Part 2: 定义工具的 JSON Schema 描述
# ============================================================

print("=" * 60)
print("  Part 2: JSON Schema 工具描述")
print("=" * 60)

# 这是发送给 LLM 的工具描述，LLM 根据这些描述决定调用哪个工具
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的实时天气信息，包括温度、天气状况和湿度。当用户询问天气相关问题时使用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，如'北京'、'上海'、'广州'"
                    }
                },
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "计算数学表达式并返回结果。支持加减乘除、幂运算等。当用户需要进行数学计算时使用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "数学表达式，如 '2 + 3 * 4' 或 '(100 - 20) / 4'"
                    }
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "在知识库中搜索与查询相关的文档。当用户提问需要查找技术资料、概念解释时使用此工具。",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词或问题"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "返回最相关的文档数量，默认3",
                        "default": 3
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "获取当前的日期和时间。当用户询问现在几点、今天日期、星期几时使用此工具。",
            "parameters": {
                "type": "object",
                "properties": {},   # 无参数
                "required": []
            }
        }
    }
]

# 打印工具描述摘要
for tool in TOOLS_SCHEMA:
    fn = tool["function"]
    params = list(fn["parameters"].get("properties", {}).keys())
    print(f"\n  Tool: {fn['name']}")
    print(f"    Description: {fn['description'][:50]}...")
    print(f"    Parameters: {params}")
print()


# ============================================================
# Part 3: 实现 LLM 模拟器（无 LLM 时的备选方案）
# ============================================================

print("=" * 60)
print("  Part 3: 工具调用循环核心实现")
print("=" * 60)


def simulate_llm_decision(user_message: str, tools: list, tool_results: list = None):
    """
    模拟 LLM 的工具调用决策。

    当没有真实 LLM 可用时，用规则引擎模拟 LLM 的行为：
    1. 分析用户消息中的关键词
    2. 匹配到合适的工具
    3. 提取参数
    4. 返回模拟的 tool_call 或最终回答

    返回格式模拟 OpenAI API 的 response：
    - 如果需要工具调用: {"tool_calls": [{"function": {"name": ..., "arguments": ...}}]}
    - 如果直接回答: {"content": "回答内容"}
    """
    # 如果已经有工具结果了，生成最终回答
    if tool_results:
        # 把所有工具结果合并，生成一个综合回答
        parts = []
        for tr in tool_results:
            result_data = json.loads(tr["result"])
            if tr["name"] == "get_weather":
                if "error" in result_data:
                    parts.append(result_data["error"])
                else:
                    parts.append(
                        f"{result_data['city']}今天{result_data['condition']}，"
                        f"气温{result_data['temperature']}度，湿度{result_data['humidity']}%。"
                    )
            elif tr["name"] == "calculate":
                if "error" in result_data:
                    parts.append(f"计算出错: {result_data['error']}")
                else:
                    parts.append(
                        f"计算结果: {result_data['expression']} = {result_data['result']}"
                    )
            elif tr["name"] == "search_knowledge":
                if result_data.get("results"):
                    docs = result_data["results"]
                    parts.append(f"根据知识库查询到 {len(docs)} 条相关信息:")
                    for i, doc in enumerate(docs, 1):
                        parts.append(f"  {i}. {doc['title']}: {doc['content']}")
                else:
                    parts.append("知识库中未找到相关信息。")
            elif tr["name"] == "get_current_time":
                parts.append(
                    f"现在是{result_data['date']} {result_data['time']}，{result_data['weekday']}。"
                )
        return {"content": "\n".join(parts)}

    # 分析用户消息，决定调用哪个工具
    msg = user_message.lower()

    # 天气相关关键词
    weather_cities = ["北京", "上海", "广州", "成都", "深圳"]
    if "天气" in msg or "气温" in msg or "温度" in msg:
        for city in weather_cities:
            if city in msg:
                return {
                    "tool_calls": [{
                        "id": "call_sim_001",
                        "function": {
                            "name": "get_weather",
                            "arguments": json.dumps({"city": city}, ensure_ascii=False)
                        }
                    }]
                }
        # 没匹配到具体城市
        return {
            "tool_calls": [{
                "id": "call_sim_001",
                "function": {
                    "name": "get_weather",
                    "arguments": json.dumps({"city": "北京"}, ensure_ascii=False)
                }
            }]
        }

    # 计算相关
    if "计算" in msg or "算" in msg or "等于" in msg or any(op in msg for op in ["+", "-", "*", "/"]):
        # 尝试提取数学表达式
        import re
        # 匹配常见的数学表达式模式
        match = re.search(r'[\d]+[\s]*[\+\-\*\/\%\(\)\.]+[\s]*[\d\+\-\*\/\%\(\)\.\s]+', msg)
        if match:
            expr = match.group().strip()
        elif "127" in msg and "365" in msg:
            expr = "127 * 365"
        elif "平方" in msg:
            nums = re.findall(r'\d+', msg)
            expr = f"{nums[0]} ** 2" if nums else "2 ** 10"
        else:
            expr = "2 + 2"
        return {
            "tool_calls": [{
                "id": "call_sim_002",
                "function": {
                    "name": "calculate",
                    "arguments": json.dumps({"expression": expr}, ensure_ascii=False)
                }
            }]
        }

    # 时间相关
    if "时间" in msg or "几点" in msg or "日期" in msg or "星期" in msg or "今天" in msg:
        return {
            "tool_calls": [{
                "id": "call_sim_003",
                "function": {
                    "name": "get_current_time",
                    "arguments": "{}"
                }
            }]
        }

    # 知识检索相关
    knowledge_keywords = ["什么是", "解释", "介绍", "RAG", "LangChain", "Python", "Agent",
                          "FastAPI", "向量", "知识库"]
    if any(kw in msg for kw in knowledge_keywords):
        return {
            "tool_calls": [{
                "id": "call_sim_004",
                "function": {
                    "name": "search_knowledge",
                    "arguments": json.dumps({"query": user_message, "top_k": 3}, ensure_ascii=False)
                }
            }]
        }

    # 不需要工具，直接回答
    return {"content": f"你好！关于「{user_message}」，这是一个通用问题，我可以直接回答，不需要调用任何工具。"}


# ============================================================
# Part 4: 完整的工具调用循环
# ============================================================

def call_llm_with_tools(messages: list, tools: list):
    """
    调用 LLM（真实或模拟），返回统一格式的结果。

    返回: dict，包含以下两种情况之一:
      - {"tool_calls": [...]}   → LLM 请求调用工具
      - {"content": "..."}      → LLM 直接回答
    """
    if USE_SIMULATION:
        # 模拟模式：用规则引擎
        user_msg = messages[-1]["content"] if messages else ""
        # 检查是否有工具结果
        tool_results = []
        for msg in messages:
            if msg.get("role") == "tool":
                tool_results.append({
                    "name": msg.get("name", "unknown"),
                    "result": msg["content"]
                })
        return simulate_llm_decision(user_msg, tools, tool_results if tool_results else None)
    else:
        # 真实 LLM 模式
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                tools=tools,
                tool_choice="auto",   # LLM 自行决定是否调用工具
                temperature=0.1,      # 低温度让工具调用更确定
            )
            choice = response.choices[0]
            if choice.message.tool_calls:
                # LLM 请求调用工具
                return {
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in choice.message.tool_calls
                    ],
                    # 保存原始 message 以便后续拼接
                    "_raw_message": choice.message
                }
            else:
                # LLM 直接回答
                return {"content": choice.message.content}
        except Exception as e:
            # LLM 调用失败，回退到模拟模式
            print(f"  [WARN] LLM 调用失败 ({e})，回退到模拟模式")
            user_msg = messages[-1]["content"] if messages else ""
            return simulate_llm_decision(user_msg, tools)


def execute_tool(name: str, arguments: str) -> str:
    """
    执行工具函数并返回结果。

    name: 工具名称（对应 TOOL_REGISTRY 中的 key）
    arguments: JSON 字符串形式的参数

    返回: 工具执行结果（字符串）
    """
    if name not in TOOL_REGISTRY:
        return json.dumps({"error": f"未知工具: {name}"}, ensure_ascii=False)

    try:
        args = json.loads(arguments) if arguments else {}
        func = TOOL_REGISTRY[name]
        result = func(**args)
        return result
    except Exception as e:
        return json.dumps({"error": f"工具执行失败: {str(e)}"}, ensure_ascii=False)


def function_calling_loop(user_query: str, max_rounds: int = 5) -> str:
    """
    完整的 Function Calling 循环。

    这是核心函数！实现了:
    用户输入 → LLM 判断 → [调用工具 → 返回结果 → LLM 再判断] → 最终回答

    max_rounds: 最大工具调用轮数（防止无限循环）

    返回: LLM 的最终文本回答
    """
    print(f"\n  {'='*50}")
    print(f"  User: {user_query}")
    print(f"  {'='*50}")

    # 初始化消息列表
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个智能助手，可以使用工具来帮助用户。"
                "如果用户的问题需要实时数据或计算，请调用合适的工具。"
                "如果不需要工具就能回答，直接回答即可。"
                "回答使用中文。"
            )
        },
        {
            "role": "user",
            "content": user_query
        }
    ]

    for round_num in range(1, max_rounds + 1):
        print(f"\n  --- Round {round_num} ---")

        # Step 1: 调用 LLM
        response = call_llm_with_tools(messages, TOOLS_SCHEMA)

        # Step 2: 检查 LLM 是否请求工具调用
        if "tool_calls" in response:
            tool_calls = response["tool_calls"]
            print(f"  LLM 决策: 调用 {len(tool_calls)} 个工具")

            # 如果是真实 LLM，需要把 assistant 消息加入历史
            if not USE_SIMULATION and "_raw_message" in response:
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tc["id"],
                            "type": "function",
                            "function": tc["function"]
                        }
                        for tc in tool_calls
                    ]
                })
            else:
                # 模拟模式下也要保持消息结构
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": tc.get("id", f"call_{round_num}"),
                            "type": "function",
                            "function": tc["function"]
                        }
                        for tc in tool_calls
                    ]
                })

            # Step 3: 执行每个工具调用
            for tc in tool_calls:
                func_name = tc["function"]["name"]
                func_args = tc["function"]["arguments"]

                print(f"    -> 调用: {func_name}({func_args})")

                # 执行工具
                result = execute_tool(func_name, func_args)
                print(f"    <- 结果: {result[:100]}{'...' if len(result) > 100 else ''}")

                # Step 4: 把工具结果加入消息历史
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.get("id", f"call_{round_num}"),
                    "name": func_name,
                    "content": result
                })

            # 继续循环，让 LLM 根据工具结果生成回答或继续调用工具

        else:
            # LLM 直接回答（没有调用工具或者已经拿到工具结果后生成了最终回答）
            final_answer = response.get("content", "（未获取到回答）")
            print(f"  LLM 最终回答: {final_answer}")
            return final_answer

    # 超过最大轮数
    return "（达到最大工具调用轮数，停止循环）"


# ============================================================
# Part 5: 测试各种场景
# ============================================================

print("\n" + "=" * 60)
print("  Part 5: 测试工具调用")
print("=" * 60)

# 场景 1: 需要调用天气工具
print("\n" + "-" * 40)
print("  场景 1: 查询天气（需要工具）")
print("-" * 40)
result1 = function_calling_loop("北京今天天气怎么样？")

# 场景 2: 需要计算工具
print("\n" + "-" * 40)
print("  场景 2: 数学计算（需要工具）")
print("-" * 40)
result2 = function_calling_loop("帮我计算 127 * 365 等于多少")

# 场景 3: 需要知识库搜索
print("\n" + "-" * 40)
print("  场景 3: 知识检索（需要工具）")
print("-" * 40)
result3 = function_calling_loop("什么是 RAG 技术？")

# 场景 4: 不需要工具的简单问题
print("\n" + "-" * 40)
print("  场景 4: 简单闲聊（不需要工具）")
print("-" * 40)
result4 = function_calling_loop("你好，介绍一下你自己")

# 场景 5: 查询时间
print("\n" + "-" * 40)
print("  场景 5: 查询时间（需要工具）")
print("-" * 40)
result5 = function_calling_loop("现在几点了？今天星期几？")


# ============================================================
# Part 6: 模拟多步工具调用
# ============================================================

print("\n" + "=" * 60)
print("  Part 6: 多步工具调用演示")
print("=" * 60)

print("""
  多步工具调用示意（串行执行）:

  用户: "查一下北京天气，然后帮我算一下35度的华氏温度"

  Round 1: LLM 决定先查天气
    -> get_weather(city="北京") -> 28度

  Round 2: LLM 拿到28度后决定计算华氏温度
    -> calculate("28 * 9/5 + 32") -> 82.4

  Round 3: LLM 综合两个结果生成最终回答
    -> "北京今天28度（晴天），换算成华氏温度是82.4度"
""")


def demo_multi_step():
    """
    演示多步工具调用的执行过程。
    由于模拟模式难以完美模拟多步推理，这里用手动编排来展示流程。
    """
    print("  --- 手动编排多步调用演示 ---\n")

    # Step 1: 调用天气工具
    print("  Step 1: 调用 get_weather")
    weather_result = execute_tool("get_weather", '{"city": "北京"}')
    weather_data = json.loads(weather_result)
    print(f"    结果: {weather_result}")

    # Step 2: 根据温度计算华氏度
    temp_c = weather_data.get("temperature", 28)
    expression = f"{temp_c} * 9 / 5 + 32"
    print(f"\n  Step 2: 调用 calculate（基于 Step 1 的温度 {temp_c}度）")
    calc_result = execute_tool("calculate", json.dumps({"expression": expression}))
    calc_data = json.loads(calc_result)
    print(f"    结果: {calc_result}")

    # Step 3: 综合结果
    temp_f = calc_data.get("result", "N/A")
    condition = weather_data.get("condition", "未知")
    print(f"\n  Step 3: 综合生成最终回答")
    final = (
        f"    北京今天{condition}，气温{temp_c}摄氏度（华氏{temp_f}度），"
        f"湿度{weather_data.get('humidity', 'N/A')}%。"
    )
    print(final)

    print(f"\n  [OK] 多步调用完成: 2个工具调用 + 1个最终回答")


demo_multi_step()


# ============================================================
# Part 7: 工具调用错误处理
# ============================================================

print("\n" + "=" * 60)
print("  Part 7: 错误处理")
print("=" * 60)

print("\n  测试不存在的工具:")
result = execute_tool("nonexistent_tool", '{}')
print(f"    结果: {result}")

print("\n  测试错误的参数:")
result = execute_tool("calculate", '{"expression": "1/0"}')
print(f"    结果: {result}")

print("\n  测试未知城市:")
result = execute_tool("get_weather", '{"city": "亚特兰蒂斯"}')
print(f"    结果: {result}")

print("\n  测试无效的 JSON 参数:")
result = execute_tool("get_weather", 'not a json')
print(f"    结果: {result}")

print("\n  [OK] 所有错误都被优雅处理，不会导致程序崩溃")


# ============================================================
# Part 8: 理解 Function Calling 的本质
# ============================================================

print("\n" + "=" * 60)
print("  Part 8: Function Calling 本质总结")
print("=" * 60)

print("""
  Function Calling 的本质:

  1. LLM 不执行任何代码！
     它只是生成一段 JSON: {"name": "get_weather", "arguments": {"city": "北京"}}
     真正执行 get_weather() 的是你的 Python 程序

  2. 循环结构:
     while True:
       response = llm(messages, tools)     # 问 LLM
       if response.has_tool_calls:         # LLM 要调工具
         for call in response.tool_calls:
           result = execute(call)          # 你执行工具
           messages.append(result)         # 把结果告诉 LLM
       else:
         return response.content           # LLM 直接回答，结束

  3. 工具描述的质量 = 工具调用的准确率
     description 写得越清楚，LLM 越知道什么时候该用

  4. 这就是 Agent 的基石:
     Agent = LLM + Function Calling + Memory + Planning

  明天 (Day 32): 学习 LangChain 怎么把这个流程封装成更简洁的代码
""")

print("=" * 60)
print("  Day 31 完成！")
print("=" * 60)
