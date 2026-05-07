"""
Day 34 Demo 1: 规划机制 (Planning & Reasoning) 完整实战
运行方式: python day34_1_planning.py

前置条件（任选一个）:
  - Ollama 运行中（推荐 qwen2.5:7b）
  - 或 .env 中配置 DEEPSEEK_API_KEY
  - 或以上都没有 -> 自动进入模拟模式

学习目标:
1. 理解 Chain of Thought 对推理质量的影响
2. 手动实现 ReAct 循环（不用框架，从头写）
3. 用 ReAct 解析 Thought/Action/Observation 格式
4. 实现 Plan-and-Execute 模式
5. 实现 Self-Reflection（自我反思改进）
"""

import json
import re
import math
import time
import os
from datetime import datetime

# ============================================================
# Part 0: 环境初始化
# ============================================================

print("=" * 60)
print("  Day 34: Planning & Reasoning - 规划机制")
print("=" * 60)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

USE_SIMULATION = False
client = None
model_name = None


def init_llm():
    """初始化 LLM"""
    global USE_SIMULATION, client, model_name

    try:
        from openai import OpenAI
    except ImportError:
        USE_SIMULATION = True
        print("  [INFO] openai 库未安装，进入模拟模式")
        return

    # 尝试 Ollama
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        models = [m["name"] for m in resp.json().get("models", [])]
        if models:
            model_name = models[0]
            preferred = ["qwen2.5:7b", "qwen2.5:14b", "qwen2.5:3b"]
            for p in preferred:
                for m in models:
                    if p in m:
                        model_name = m
                        break
            client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
            print(f"  [OK] LLM: Ollama ({model_name})")
            return
    except Exception:
        pass

    # 尝试 DeepSeek
    ds_key = os.getenv("DEEPSEEK_API_KEY")
    if ds_key:
        client = OpenAI(api_key=ds_key, base_url="https://api.deepseek.com")
        model_name = "deepseek-chat"
        print(f"  [OK] LLM: DeepSeek")
        return

    USE_SIMULATION = True
    print("  [INFO] 未检测到 LLM，进入模拟模式")


init_llm()


def ask_llm(messages: list, temperature: float = 0.3) -> str:
    """统一的 LLM 调用接口"""
    if USE_SIMULATION:
        return None  # 返回 None 表示需要用模拟逻辑
    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=1024,
        )
        return resp.choices[0].message.content
    except Exception as e:
        print(f"  [WARN] LLM 调用失败: {e}")
        return None


print()


# ============================================================
# Part 1: Chain of Thought（思维链）对比演示
# ============================================================

print("=" * 60)
print("  Part 1: Chain of Thought (CoT) - 思维链")
print("=" * 60)
print("  对比：直接回答 vs 加上'请一步步思考'后回答")
print()

# 测试题目
COT_QUESTIONS = [
    {
        "question": "一个班50人，60%是女生，其中1/3参加了编程社团。编程社团有多少女生？",
        "answer": 10,
    },
    {
        "question": "小明用200元买了3件相同的T恤，每件打8折。T恤原价多少钱？",
        "answer": 83.33,
    },
    {
        "question": "一个项目有5个模块，每个模块需要3天开发+1天测试。2个人并行开发，至少需要多少天完成？",
        "answer": 12,  # ceil(5/2)*3 + 5*1 由于测试要等开发完..实际更复杂
    },
]

for i, q_data in enumerate(COT_QUESTIONS, 1):
    question = q_data["question"]
    correct = q_data["answer"]

    print(f"  --- 题目 {i} ---")
    print(f"  Q: {question}")
    print(f"  正确答案: {correct}")

    # 方式 A: 直接问（不加 CoT 提示）
    direct_messages = [
        {"role": "user", "content": f"{question}\n请直接给出数字答案，不要解释过程。"}
    ]
    direct_answer = ask_llm(direct_messages)

    if direct_answer:
        print(f"\n  [直接回答] {direct_answer[:100]}")
    else:
        # 模拟直接回答（容易出错）
        simulated_direct = {1: "15人", 2: "80元", 3: "20天"}
        print(f"\n  [直接回答-模拟] {simulated_direct.get(i, '?')}")
        print(f"    (模拟模式：展示LLM不经思考容易给出错误答案)")

    # 方式 B: 加 CoT 提示
    cot_messages = [
        {"role": "user", "content": f"{question}\n\n请一步一步思考，列出每一步的推理过程，最后给出答案。"}
    ]
    cot_answer = ask_llm(cot_messages, temperature=0.1)

    if cot_answer:
        print(f"\n  [CoT 回答] {cot_answer[:300]}")
    else:
        # 模拟 CoT 回答
        cot_simulations = {
            1: """让我一步步计算:
    1. 总人数: 50人
    2. 女生人数: 50 * 60% = 30人
    3. 参加编程社团的女生: 30 * 1/3 = 10人
    答案: 10名女生在编程社团""",
            2: """让我一步步计算:
    1. 3件T恤打8折共200元
    2. 打折后每件: 200 / 3 = 66.67元
    3. 原价 = 折后价 / 0.8 = 66.67 / 0.8 = 83.33元
    答案: T恤原价约83.33元""",
            3: """让我一步步分析:
    1. 共5个模块，每个需3天开发+1天测试
    2. 2人并行: 一次可开发2个模块
    3. 开发轮次: ceil(5/2) = 3轮, 共 3*3 = 9天开发
    4. 测试可以在开发完一个模块后就开始
    5. 最优安排约需12天
    答案: 至少需要约12天""",
        }
        print(f"\n  [CoT 回答-模拟]")
        print(f"    {cot_simulations.get(i, '...')}")

    print()

print("  [OK] CoT 总结:")
print("    - '请一步步思考' 让 LLM 展开推理过程")
print("    - 每步计算都写出来，减少跳跃性错误")
print("    - 简单一句话就能显著提升推理准确率")
print("    - 局限: 只能在一次调用内推理，不能调用工具")
print()


# ============================================================
# Part 2: 定义 ReAct 工具
# ============================================================

print("=" * 60)
print("  Part 2: 定义 ReAct Agent 的工具集")
print("=" * 60)

# 工具注册表
REACT_TOOLS = {}


def register_tool(name: str, description: str):
    """工具注册装饰器"""
    def decorator(func):
        REACT_TOOLS[name] = {
            "description": description,
            "func": func,
        }
        return func
    return decorator


@register_tool("calculator", "计算数学表达式。输入: 数学表达式字符串（如 '2+3*4'）")
def tool_calculator(expression: str) -> str:
    """安全的数学计算器"""
    try:
        allowed = {"abs": abs, "round": round, "min": min, "max": max,
                    "pow": pow, "math": math}
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {e}"


@register_tool("get_weather", "获取城市天气信息。输入: 城市名称（如 '北京'）")
def tool_weather(city: str) -> str:
    """模拟天气查询"""
    data = {
        "北京": "晴天, 28度, 湿度45%",
        "上海": "多云, 32度, 湿度72%",
        "广州": "雷阵雨, 35度, 湿度88%",
        "成都": "阴天, 26度, 湿度65%",
    }
    return data.get(city.strip(), f"未找到{city}的天气数据")


@register_tool("search_kb", "搜索知识库获取技术文档。输入: 搜索关键词")
def tool_search(query: str) -> str:
    """模拟知识库搜索"""
    kb = {
        "python": "Python是一种解释型高级编程语言，支持面向对象、函数式编程。",
        "rag": "RAG(检索增强生成)通过检索外部知识库来增强LLM回答，避免幻觉。",
        "fastapi": "FastAPI是现代Python Web框架，支持异步，自动生成API文档。",
        "agent": "Agent是能自主决策和执行任务的AI系统，核心包括工具调用、记忆和规划。",
        "langchain": "LangChain是LLM应用开发框架，提供Chain、Agent、Memory等核心组件。",
        "chroma": "ChromaDB是轻量级嵌入式向量数据库，适合中小规模RAG项目。",
    }
    query_lower = query.lower().strip()
    results = []
    for key, value in kb.items():
        if key in query_lower or any(c in query_lower for c in key if len(key) > 2):
            results.append(f"[{key}] {value}")
    return "\n".join(results) if results else f"未找到与'{query}'相关的知识"


@register_tool("get_time", "获取当前日期和时间。输入: 无（可传空字符串）")
def tool_time(_: str = "") -> str:
    """获取当前时间"""
    now = datetime.now()
    weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
    return f"当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}, {weekdays[now.weekday()]}"


# 打印工具列表
print(f"\n  已注册 {len(REACT_TOOLS)} 个工具:")
for name, info in REACT_TOOLS.items():
    print(f"    - {name}: {info['description']}")
print()


# ============================================================
# Part 3: 手动实现 ReAct 循环（核心！）
# ============================================================

print("=" * 60)
print("  Part 3: 手动 ReAct 循环实现")
print("=" * 60)
print("  不用任何框架，从头实现 Thought/Action/Observation 循环")
print()


def build_react_prompt(question: str, tools: dict) -> str:
    """
    构建 ReAct Prompt。
    这个 Prompt 告诉 LLM 要按 Thought/Action/Observation 格式输出。
    """
    tool_desc = "\n".join([f"  - {name}: {info['description']}" for name, info in tools.items()])

    return f"""你是一个能使用工具的智能助手。请按照以下格式回答问题。

可用工具:
{tool_desc}

输出格式（严格遵守！）:
Thought: [你的推理过程]
Action: [工具名称，必须是上面列出的工具之一]
Action Input: [传给工具的输入]

等待系统返回 Observation 后，你可以继续:
Thought: [基于观察结果的推理]
Action: [下一个工具]
Action Input: [输入]

当你有足够信息回答时:
Thought: 我已经有足够信息了
Final Answer: [最终回答，用中文]

注意: 如果问题不需要工具就能回答，直接给出 Final Answer。

问题: {question}"""


def parse_react_output(text: str):
    """
    解析 LLM 输出的 ReAct 格式。

    返回值:
    - ("action", action_name, action_input): LLM 要调用工具
    - ("final", answer): LLM 给出了最终答案
    - ("error", message): 解析失败
    """
    text = text.strip()

    # 检查是否有 Final Answer
    final_match = re.search(r'Final Answer:\s*(.*?)$', text, re.DOTALL | re.MULTILINE)
    # 也检查是否有 Action（如果两者都有，看哪个更后面）
    action_match = re.search(r'Action:\s*(.*?)$', text, re.MULTILINE)
    action_input_match = re.search(r'Action Input:\s*(.*?)$', text, re.MULTILINE)

    if final_match:
        # 如果有 Action 也有 Final Answer, 看位置
        if action_match and action_match.start() > (final_match.start() if final_match else 0):
            pass  # Action 在 Final Answer 后面，不太可能
        else:
            return ("final", final_match.group(1).strip())

    if action_match:
        action_name = action_match.group(1).strip()
        action_input = action_input_match.group(1).strip() if action_input_match else ""
        # 提取 Thought（如果有的话）
        thought_match = re.search(r'Thought:\s*(.*?)(?=\n|$)', text)
        if thought_match:
            thought = thought_match.group(1).strip()
        else:
            thought = ""
        return ("action", action_name, action_input, thought)

    # 都没匹配到，可能 LLM 直接给了文本回答
    if text:
        return ("final", text)

    return ("error", "无法解析 LLM 输出")


def simulate_react_response(question: str, observations: list) -> str:
    """
    模拟 LLM 的 ReAct 输出。
    当没有真实 LLM 时，用规则模拟 Thought/Action 决策。
    """
    q = question.lower()

    # 如果已经有观察结果，生成最终回答
    if observations:
        last_obs = observations[-1]
        return f"Thought: 我已经从工具获得了结果，可以回答用户了。\nFinal Answer: {last_obs}"

    # 根据问题类型决定调用什么工具
    if "天气" in q:
        for city in ["北京", "上海", "广州", "成都"]:
            if city in q:
                return f"Thought: 用户想知道{city}的天气，我需要调用天气工具。\nAction: get_weather\nAction Input: {city}"
        return "Thought: 用户问天气但没说城市，我猜是北京。\nAction: get_weather\nAction Input: 北京"

    if "计算" in q or any(op in q for op in ["+", "-", "*", "/"]):
        expr = re.findall(r'[\d\.\+\-\*\/\(\)\s]+', q)
        expression = expr[0].strip() if expr else "2+2"
        # 清理表达式
        expression = re.sub(r'[^\d\.\+\-\*\/\(\)\s]', '', expression).strip()
        if not expression:
            expression = "2+2"
        return f"Thought: 用户需要数学计算，我用计算器工具。\nAction: calculator\nAction Input: {expression}"

    if "什么是" in q or "解释" in q:
        keyword = q.replace("什么是", "").replace("解释", "").replace("？", "").replace("?", "").strip()
        return f"Thought: 用户想了解'{keyword}'的概念，我搜索知识库。\nAction: search_kb\nAction Input: {keyword}"

    if "时间" in q or "几点" in q or "日期" in q:
        return "Thought: 用户想知道当前时间。\nAction: get_time\nAction Input: "

    # 不需要工具
    return f"Thought: 这个问题我可以直接回答，不需要工具。\nFinal Answer: 你好！我是一个支持工具调用的智能助手，可以帮你查天气、做计算、搜知识库。"


def react_loop(question: str, max_iterations: int = 5) -> str:
    """
    完整的 ReAct 循环实现。

    这是本Demo的核心函数！实现了:
    1. 构建 ReAct Prompt
    2. 调用 LLM 获取 Thought/Action
    3. 解析输出
    4. 执行工具
    5. 把 Observation 反馈给 LLM
    6. 循环直到 Final Answer
    """
    print(f"\n  {'='*50}")
    print(f"  Question: {question}")
    print(f"  {'='*50}")

    # 构建初始 Prompt
    system_prompt = build_react_prompt(question, REACT_TOOLS)
    messages = [{"role": "user", "content": system_prompt}]

    observations = []  # 记录所有工具观察结果（模拟模式用）

    for iteration in range(1, max_iterations + 1):
        print(f"\n  --- Iteration {iteration} ---")

        # Step 1: 获取 LLM 的回复
        llm_output = ask_llm(messages, temperature=0.1)

        if llm_output is None:
            # 模拟模式
            llm_output = simulate_react_response(question, observations)

        print(f"  LLM 输出:\n    {llm_output.replace(chr(10), chr(10) + '    ')}")

        # Step 2: 解析输出
        parsed = parse_react_output(llm_output)

        if parsed[0] == "final":
            # LLM 给出了最终回答
            final_answer = parsed[1]
            print(f"\n  [Final Answer] {final_answer}")
            return final_answer

        elif parsed[0] == "action":
            _, action_name, action_input, thought = parsed
            print(f"\n  Thought: {thought}")
            print(f"  Action: {action_name}")
            print(f"  Action Input: {action_input}")

            # Step 3: 执行工具
            if action_name in REACT_TOOLS:
                observation = REACT_TOOLS[action_name]["func"](action_input)
            else:
                observation = f"错误: 未知工具 '{action_name}'。可用工具: {list(REACT_TOOLS.keys())}"

            print(f"  Observation: {observation}")
            observations.append(observation)

            # Step 4: 把 observation 反馈给 LLM
            # 拼接完整的对话历史
            messages.append({"role": "assistant", "content": llm_output})
            messages.append({"role": "user", "content": f"Observation: {observation}"})

        else:
            # 解析错误
            print(f"  [WARN] 解析错误: {parsed[1]}")
            return f"(解析错误: {parsed[1]})"

    # 超过最大迭代次数
    print(f"  [WARN] 达到最大迭代次数 ({max_iterations})")
    return "(达到最大迭代次数，停止循环)"


# ============================================================
# Part 4: 测试 ReAct Agent
# ============================================================

print("\n" + "=" * 60)
print("  Part 4: ReAct Agent 测试")
print("=" * 60)

# 测试查询 1: 需要天气工具
print("\n" + "-" * 40)
print("  测试 1: 天气查询")
print("-" * 40)
react_loop("北京今天天气怎么样？")

# 测试查询 2: 需要计算工具
print("\n" + "-" * 40)
print("  测试 2: 数学计算")
print("-" * 40)
react_loop("帮我计算 25 * 17 + 83")

# 测试查询 3: 需要知识库搜索
print("\n" + "-" * 40)
print("  测试 3: 知识库搜索")
print("-" * 40)
react_loop("什么是 RAG 技术？")

# 测试查询 4: 不需要工具
print("\n" + "-" * 40)
print("  测试 4: 直接回答（不需要工具）")
print("-" * 40)
react_loop("你好，请介绍一下你自己")

print()


# ============================================================
# Part 5: Plan-and-Execute 模式
# ============================================================

print("=" * 60)
print("  Part 5: Plan-and-Execute 模式")
print("=" * 60)
print("  先做完整计划，再逐步执行每个步骤")
print()


def plan_and_execute(task: str):
    """
    Plan-and-Execute 模式实现。

    Phase 1: Planner LLM 生成执行计划
    Phase 2: Executor 逐步执行计划中的每一步
    Phase 3: 汇总结果

    真实项目中 Plan 由 LLM 生成，这里用规则模拟。
    """
    print(f"  Task: {task}")

    # === Phase 1: 规划 ===
    print(f"\n  === Phase 1: Planning ===")

    # 构建规划请求（真实项目中发给 LLM）
    plan_prompt = f"""请为以下任务制定执行计划，用编号列表输出每个步骤:

任务: {task}

请输出 3-6 个具体的执行步骤。"""

    plan_response = ask_llm([{"role": "user", "content": plan_prompt}])

    if plan_response:
        print(f"  LLM 生成的计划:")
        print(f"    {plan_response[:500]}")
        # 尝试从 LLM 输出中提取步骤
        steps = re.findall(r'\d+\.\s*(.*?)(?=\n\d+\.|\n*$)', plan_response)
        if not steps:
            steps = [line.strip() for line in plan_response.split('\n') if line.strip() and line.strip()[0].isdigit()]
    else:
        # 模拟计划生成
        if "销售" in task or "数据分析" in task:
            steps = [
                "读取数据文件并检查数据结构",
                "计算各地区的销售总额",
                "对比同期数据，计算增长率",
                "找出下滑最严重的地区",
                "生成可视化图表",
                "撰写分析报告摘要",
            ]
        elif "天气" in task:
            steps = [
                "获取目标城市的当前天气",
                "分析温度和天气状况",
                "根据天气给出穿衣/出行建议",
            ]
        else:
            steps = [
                "理解用户需求",
                "收集所需信息",
                "分析和处理信息",
                "生成最终结果",
            ]

    # 打印计划
    print(f"\n  执行计划 ({len(steps)} 步):")
    for i, step in enumerate(steps, 1):
        step_text = step if isinstance(step, str) else str(step)
        step_text = step_text.lstrip('0123456789.、) ').strip()
        print(f"    Step {i}: {step_text}")

    # === Phase 2: 执行 ===
    print(f"\n  === Phase 2: Execution ===")

    results = []
    for i, step in enumerate(steps, 1):
        step_text = step if isinstance(step, str) else str(step)
        step_text = step_text.lstrip('0123456789.、) ').strip()

        print(f"\n    [Executing Step {i}/{len(steps)}] {step_text}")
        time.sleep(0.1)

        # 模拟每步执行结果
        if "读取" in step_text or "数据" in step_text:
            result = "已加载数据: 500行 x 6列 (日期, 地区, 产品, 数量, 金额, 类别)"
        elif "计算" in step_text or "销售" in step_text or "总额" in step_text:
            result = "地区销售额: 华北=120万, 华东=180万, 华南=95万, 西南=65万"
        elif "对比" in step_text or "增长" in step_text:
            result = "同比变化: 华北+5%, 华东+12%, 华南-8%, 西南-15%"
        elif "下滑" in step_text or "最" in step_text:
            result = "下滑最严重: 西南地区 (-15%), 其次华南地区 (-8%)"
        elif "图表" in step_text or "可视化" in step_text:
            result = "已生成: regional_comparison.png, trend_chart.png"
        elif "报告" in step_text or "摘要" in step_text:
            result = "报告: 华东表现最佳(+12%), 西南需重点关注(-15%), 建议增加西南营销投入"
        elif "天气" in step_text or "获取" in step_text:
            result = tool_weather("北京")
        elif "分析" in step_text:
            result = "分析完成: 基于收集到的信息进行了综合分析"
        else:
            result = f"已完成: {step_text}"

        print(f"      Result: {result}")
        results.append({"step": i, "description": step_text, "result": result})

    # === Phase 3: 汇总 ===
    print(f"\n  === Phase 3: Summary ===")
    print(f"\n  任务: {task}")
    print(f"  状态: {len(results)}/{len(steps)} 步全部完成")
    print(f"\n  执行摘要:")
    for r in results:
        print(f"    Step {r['step']}: {r['result'][:60]}")

    print(f"\n  [OK] Plan-and-Execute 完成!")
    return results


# 测试 Plan-and-Execute
print("-" * 40)
print("  测试: 数据分析任务")
print("-" * 40)
plan_and_execute("分析2024年各地区销售数据，找出下滑最严重的地区，生成报告")

print()


# ============================================================
# Part 6: Self-Reflection（自我反思）
# ============================================================

print("=" * 60)
print("  Part 6: Self-Reflection - 自我反思改进")
print("=" * 60)
print("  生成回答 -> 自我评估 -> 发现问题 -> 改进")
print()


def self_reflection(question: str, max_rounds: int = 3):
    """
    Self-Reflection 实现。

    流程:
    1. 初次生成回答
    2. 让 LLM（或规则）评估回答质量
    3. 如果有问题，根据反馈改进
    4. 重复直到满意或达到最大轮数
    """
    print(f"  Question: {question}")
    print(f"  {'='*45}")

    # Step 1: 初次生成
    generate_prompt = f"请回答以下问题（用中文，尽可能准确和完整）:\n\n{question}"
    answer = ask_llm([{"role": "user", "content": generate_prompt}])

    if answer is None:
        # 模拟初次回答（故意有缺陷）
        answer_map = {
            "FastAPI": "FastAPI是一个Python框架。",
            "RAG": "RAG是检索增强生成。",
            "Agent": "Agent就是一个AI助手。",
        }
        answer = answer_map.get(
            next((k for k in answer_map if k.lower() in question.lower()), ""),
            "这是一个关于AI的问题，答案是..."
        )

    print(f"\n  === Round 0: 初次回答 ===")
    print(f"    {answer}")

    for round_num in range(1, max_rounds + 1):
        # Step 2: 自我评估
        critique_prompt = f"""请评估以下回答的质量，严格评分并指出所有问题:

问题: {question}
回答: {answer}

请按以下格式输出:
评分: [1-10分]
问题列表:
- [问题1]
- [问题2]
改进建议:
- [建议1]
- [建议2]

如果评分 >= 8 分，在最后加上 "PASS"。"""

        critique = ask_llm([{"role": "user", "content": critique_prompt}])

        if critique is None:
            # 模拟评估
            issues = []
            if len(answer) < 50:
                issues.append("回答太简短，缺少详细解释")
            if "例如" not in answer and "比如" not in answer:
                issues.append("缺少具体示例")
            if "优点" not in answer and "适合" not in answer:
                issues.append("没有说明适用场景")

            if not issues:
                critique = "评分: 9/10\n回答质量很好。PASS"
            else:
                score = max(1, 8 - len(issues) * 2)
                issues_text = "\n".join(f"- {issue}" for issue in issues)
                critique = f"评分: {score}/10\n问题列表:\n{issues_text}"

        print(f"\n  === Round {round_num}: 自我评估 ===")
        print(f"    {critique[:200]}")

        # 检查是否通过
        if "PASS" in critique.upper():
            print(f"\n  [OK] 评估通过! 最终回答质量满意")
            break

        # Step 3: 根据反馈改进
        improve_prompt = f"""请根据反馈改进你的回答:

问题: {question}
原始回答: {answer}
反馈: {critique}

请生成一个更好的回答，解决所有被指出的问题。"""

        improved = ask_llm([{"role": "user", "content": improve_prompt}])

        if improved is None:
            # 模拟改进
            additions = []
            if "太简短" in critique:
                additions.append("它通过检索外部知识库的文档来增强LLM的回答质量，有效减少幻觉。")
            if "缺少" in critique and "示例" in critique:
                additions.append("例如，用户问'公司报销政策'，系统先从文档库检索相关规定，再让LLM基于文档回答。")
            if "适用场景" in critique or "没有说明" in critique:
                additions.append("适合企业知识库、客服问答等需要准确引用文档的场景。")

            if additions:
                improved = answer + " " + " ".join(additions)
            else:
                improved = answer + " 这是经过改进的更详细的回答。"

        answer = improved
        print(f"\n  === Round {round_num}: 改进后 ===")
        print(f"    {answer[:300]}")

    print(f"\n  最终回答: {answer[:300]}")
    return answer


# 测试自我反思
print("-" * 40)
print("  测试 1: 简短回答需要改进")
print("-" * 40)
self_reflection("什么是 RAG 技术？它有什么优势和适用场景？")

print()
print("-" * 40)
print("  测试 2: 另一个需要改进的回答")
print("-" * 40)
self_reflection("解释 FastAPI 框架的主要特点和使用方法")

print()


# ============================================================
# Part 7: 三种规划模式对比总结
# ============================================================

print("=" * 60)
print("  Part 7: 规划模式对比总结")
print("=" * 60)

print("""
  ┌──────────────────────────────────────────────────────────┐
  │              Agent 推理/规划模式对比                        │
  ├──────────────┬──────────────┬──────────────┬─────────────┤
  │              │   CoT        │   ReAct      │ Plan+Exec   │
  ├──────────────┼──────────────┼──────────────┼─────────────┤
  │ 核心理念      │ 逐步推理     │ 推理+行动    │ 先规划后执行 │
  │ 工具使用      │ 不用工具     │ 用工具       │ 用工具      │
  │ 规划方式      │ 一次性       │ 逐步决策     │ 预先规划    │
  │ 灵活性        │ 低          │ 高           │ 中等        │
  │ 适合任务      │ 推理/计算    │ 探索性任务   │ 流程化任务   │
  │ 典型场景      │ 数学题      │ 通用助手     │ 数据分析    │
  └──────────────┴──────────────┴──────────────┴─────────────┘

  Self-Reflection 可以和上面任何模式组合:
    CoT + Reflection = 计算后检查计算过程
    ReAct + Reflection = 工具调用后验证结果合理性
    Plan + Reflection = 执行后检查是否完成所有步骤

  你的项目应该用什么?
    RAG 知识库: 简单 Function Calling (流程固定)
    数据分析 Agent: Plan-and-Execute (步骤明确)
    通用助手: ReAct (灵活应对各种问题)

  Agent 能力总结 (Day 31-34):
    Day 31: Function Calling   - Agent 的"手" (调用工具)
    Day 32: LangChain Tools    - Agent 的"工具箱" (框架封装)
    Day 33: Memory             - Agent 的"大脑" (记住上下文)
    Day 34: Planning           - Agent 的"思考" (推理和规划)

    Agent = LLM + Tools + Memory + Planning
""")

print("=" * 60)
print("  Day 34 完成!")
print("=" * 60)
