"""
Day 32 Demo 1: LangChain 工具系统实战
运行方式: python day32_1_langchain_tools.py

前置条件:
  pip install langchain langchain-core langchain-community
  pip install openai pydantic

  LLM 后端（任选一个）:
  - Ollama 运行中（推荐 qwen2.5:7b）
  - 或 .env 中配置 DEEPSEEK_API_KEY
  - 或以上都没有 -> 自动进入模拟模式

学习目标:
1. 掌握 @tool 装饰器创建工具
2. 掌握 Pydantic 定义复杂输入 Schema
3. 理解 BaseTool 子类方式
4. 使用 AgentExecutor 运行 ReAct Agent
5. 理解 Agent 的思考过程（verbose 输出）
6. 工具错误处理机制
"""

import json
import os
from datetime import datetime

# ============================================================
# Part 0: 环境初始化
# ============================================================

print("=" * 60)
print("  Day 32: LangChain 工具系统")
print("=" * 60)

# 加载环境变量
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# 检测 LangChain 是否安装
LANGCHAIN_AVAILABLE = True
try:
    from langchain_core.tools import tool, BaseTool, StructuredTool
    from pydantic import BaseModel, Field
    print("  [OK] langchain-core 已安装")
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("  [WARN] langchain-core 未安装，将展示代码逻辑但使用模拟类")
    print("  (安装方法: pip install langchain-core pydantic)")

# 检测 LLM 可用性
USE_SIMULATION = False
llm = None


def init_llm():
    """初始化 LangChain 的 LLM 实例"""
    global USE_SIMULATION, llm

    try:
        from langchain_openai import ChatOpenAI
    except ImportError:
        print("  [INFO] langchain-openai 未安装，进入模拟模式")
        print("  (安装方法: pip install langchain-openai)")
        USE_SIMULATION = True
        return

    # 尝试 Ollama
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=3)
        models = [m["name"] for m in resp.json().get("models", [])]
        if models:
            # 选择一个支持工具调用的模型
            model_name = models[0]
            preferred = ["qwen2.5:7b", "qwen2.5:14b", "qwen2.5:3b"]
            for p in preferred:
                for m in models:
                    if p in m:
                        model_name = m
                        break
            llm = ChatOpenAI(
                api_key="ollama",
                base_url="http://localhost:11434/v1",
                model=model_name,
                temperature=0.1,
            )
            print(f"  [OK] LLM: Ollama ({model_name})")
            return
    except Exception:
        pass

    # 尝试 DeepSeek
    ds_key = os.getenv("DEEPSEEK_API_KEY")
    if ds_key:
        llm = ChatOpenAI(
            api_key=ds_key,
            base_url="https://api.deepseek.com",
            model="deepseek-chat",
            temperature=0.1,
        )
        print("  [OK] LLM: DeepSeek")
        return

    print("  [INFO] 未检测到 LLM，进入模拟模式")
    USE_SIMULATION = True


init_llm()
print()


# ============================================================
# Part 1: @tool 装饰器创建工具（最常用的方式）
# ============================================================

print("=" * 60)
print("  Part 1: @tool 装饰器创建工具")
print("=" * 60)

if LANGCHAIN_AVAILABLE:
    # --- 工具 1: 天气查询 ---
    @tool
    def get_weather(city: str) -> str:
        """获取指定城市的实时天气信息。当用户询问某个城市的天气、温度、是否需要带伞时调用此工具。"""
        weather_data = {
            "北京": "晴天，28度，湿度45%",
            "上海": "多云，32度，湿度72%",
            "广州": "雷阵雨，35度，湿度88%，建议带伞",
            "成都": "阴天，26度，湿度65%",
        }
        return weather_data.get(city, f"未找到{city}的天气数据")

    # --- 工具 2: 数学计算 ---
    @tool
    def calculator(expression: str) -> str:
        """计算数学表达式并返回结果。支持加减乘除和幂运算。当用户需要数学计算时调用此工具。"""
        try:
            allowed = {"abs": abs, "round": round, "min": min, "max": max, "pow": pow}
            result = eval(expression, {"__builtins__": {}}, allowed)
            return f"{expression} = {result}"
        except Exception as e:
            return f"计算错误: {e}"

    # --- 工具 3: 知识库搜索 ---
    @tool
    def search_docs(query: str) -> str:
        """在知识库中搜索与查询相关的技术文档。当用户询问技术概念、编程问题时调用此工具。"""
        knowledge = {
            "RAG": "RAG（检索增强生成）通过检索外部知识来增强LLM回答，包括索引和查询两个阶段。",
            "FastAPI": "FastAPI是Python的现代Web框架，支持异步，自动生成API文档。",
            "Agent": "Agent是能自主使用工具、记忆和规划来完成任务的AI系统。",
            "LangChain": "LangChain是LLM应用开发框架，提供Chain、Agent、Memory等核心抽象。",
        }
        # 简单关键词匹配
        results = []
        for key, value in knowledge.items():
            if key.lower() in query.lower() or any(c in query for c in key):
                results.append(f"[{key}] {value}")
        return "\n".join(results) if results else "知识库中未找到相关信息"

    # 查看工具的自动生成属性
    print(f"\n  --- 工具 1: get_weather ---")
    print(f"    name: {get_weather.name}")
    print(f"    description: {get_weather.description}")
    print(f"    args_schema: {get_weather.args}")
    print(f"    直接调用: {get_weather.invoke('北京')}")

    print(f"\n  --- 工具 2: calculator ---")
    print(f"    name: {calculator.name}")
    print(f"    description: {calculator.description}")
    print(f"    args_schema: {calculator.args}")
    print(f"    直接调用: {calculator.invoke('2 ** 10')}")

    print(f"\n  --- 工具 3: search_docs ---")
    print(f"    name: {search_docs.name}")
    print(f"    description: {search_docs.description}")
    print(f"    直接调用: {search_docs.invoke('RAG')}")

else:
    # LangChain 未安装时的模拟展示
    print("""
  [模拟模式] @tool 装饰器用法展示:

  @tool
  def get_weather(city: str) -> str:
      \"\"\"获取指定城市天气。当用户问天气时调用。\"\"\"
      return f"{city}今天晴天，28度"

  自动生成:
    name = "get_weather"
    description = "获取指定城市天气。当用户问天气时调用。"
    args_schema = {"city": {"type": "string"}}

  关键: docstring 自动变成 description, 类型注解变成 Schema
""")

print()


# ============================================================
# Part 2: Pydantic 模型定义复杂输入
# ============================================================

print("=" * 60)
print("  Part 2: Pydantic 定义复杂输入 Schema")
print("=" * 60)

if LANGCHAIN_AVAILABLE:
    # 定义复杂的输入 Schema
    class DataAnalysisInput(BaseModel):
        """数据分析工具的输入参数"""
        file_path: str = Field(description="CSV 文件路径")
        analysis_type: str = Field(
            description="分析类型: summary(统计摘要) / trend(趋势分析) / compare(对比分析)",
            default="summary"
        )
        columns: list[str] = Field(
            description="要分析的列名列表，为空则分析所有列",
            default=[]
        )

    @tool(args_schema=DataAnalysisInput)
    def analyze_data(file_path: str, analysis_type: str = "summary", columns: list[str] = []) -> str:
        """分析CSV数据文件。当用户需要对数据进行统计、趋势或对比分析时调用此工具。"""
        # 模拟数据分析（真实项目中会用 pandas）
        result = {
            "file": file_path,
            "type": analysis_type,
            "columns": columns if columns else ["all"],
            "result": f"已完成{analysis_type}分析",
            "summary": "数据包含1000行，5列，平均值为42.5"
        }
        return json.dumps(result, ensure_ascii=False)

    # 查看生成的 Schema
    print(f"\n  复杂工具: analyze_data")
    print(f"    name: {analyze_data.name}")
    print(f"    description: {analyze_data.description[:50]}...")
    print(f"    args_schema: {json.dumps(analyze_data.args, ensure_ascii=False, indent=4)}")

    # 测试调用
    result = analyze_data.invoke({
        "file_path": "sales.csv",
        "analysis_type": "trend",
        "columns": ["revenue", "month"]
    })
    print(f"\n    调用结果: {result}")

else:
    print("""
  [模拟模式] Pydantic Schema 定义:

  class DataAnalysisInput(BaseModel):
      file_path: str = Field(description="CSV 文件路径")
      analysis_type: str = Field(description="分析类型", default="summary")
      columns: list[str] = Field(description="列名列表", default=[])

  @tool(args_schema=DataAnalysisInput)
  def analyze_data(file_path, analysis_type, columns):
      ...

  优势: 每个参数有详细描述，LLM 能更准确地传参
""")

print()


# ============================================================
# Part 3: BaseTool 子类方式（最灵活）
# ============================================================

print("=" * 60)
print("  Part 3: BaseTool 子类创建工具")
print("=" * 60)

if LANGCHAIN_AVAILABLE:
    class TimeInput(BaseModel):
        """时间查询输入"""
        timezone: str = Field(default="Asia/Shanghai", description="时区，默认中国时区")

    class GetTimeTool(BaseTool):
        """
        通过继承 BaseTool 创建的工具。
        适用于需要初始化资源（数据库连接等）或复杂逻辑的场景。
        """
        name: str = "get_current_time"
        description: str = "获取当前日期和时间。当用户询问现在几点、今天日期时调用此工具。"
        args_schema: type = TimeInput

        def _run(self, timezone: str = "Asia/Shanghai") -> str:
            """同步执行 - 必须实现"""
            now = datetime.now()
            weekdays = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
            return f"当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}, {weekdays[now.weekday()]}"

        async def _arun(self, timezone: str = "Asia/Shanghai") -> str:
            """异步执行 - 可选实现"""
            return self._run(timezone)

    # 实例化并测试
    time_tool = GetTimeTool()
    print(f"\n  BaseTool 工具: {time_tool.name}")
    print(f"    description: {time_tool.description}")
    print(f"    调用结果: {time_tool.invoke({'timezone': 'Asia/Shanghai'})}")

else:
    print("""
  [模拟模式] BaseTool 子类:

  class GetTimeTool(BaseTool):
      name = "get_current_time"
      description = "获取当前时间"

      def _run(self, timezone="Asia/Shanghai") -> str:
          return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

      async def _arun(self, timezone="Asia/Shanghai") -> str:
          return self._run(timezone)

  优势: 可以有初始化逻辑(__init__)、异步支持、状态管理
""")

print()


# ============================================================
# Part 4: StructuredTool.from_function（包装已有函数）
# ============================================================

print("=" * 60)
print("  Part 4: StructuredTool.from_function")
print("=" * 60)


# 假设这是一个已存在的函数（不方便加装饰器）
def existing_search_function(query: str, limit: int = 5) -> str:
    """这是一个已存在的搜索函数，我们不想修改它"""
    return f"搜索 '{query}' 的结果 (最多{limit}条): 找到3条相关记录"


if LANGCHAIN_AVAILABLE:
    # 用 StructuredTool 包装
    wrapped_tool = StructuredTool.from_function(
        func=existing_search_function,
        name="legacy_search",
        description="使用遗留系统搜索数据。当需要在旧系统中查找记录时调用。",
    )

    print(f"\n  包装后的工具: {wrapped_tool.name}")
    print(f"    description: {wrapped_tool.description}")
    print(f"    args: {wrapped_tool.args}")
    print(f"    调用结果: {wrapped_tool.invoke({'query': 'Python', 'limit': 3})}")
else:
    print(f"\n  [模拟模式] StructuredTool.from_function 用法:")
    print(f"    wrapped_tool = StructuredTool.from_function(")
    print(f'        func=existing_search_function,')
    print(f'        name="legacy_search",')
    print(f'        description="搜索遗留系统",')
    print(f"    )")
    result = existing_search_function("Python", 3)
    print(f"    原始函数调用结果: {result}")

print()


# ============================================================
# Part 5: 构建 ReAct Agent（核心！）
# ============================================================

print("=" * 60)
print("  Part 5: 构建 ReAct Agent")
print("=" * 60)

if LANGCHAIN_AVAILABLE and not USE_SIMULATION:
    try:
        from langchain.agents import create_react_agent, AgentExecutor
        from langchain_core.prompts import PromptTemplate

        # 收集所有工具
        all_tools = [get_weather, calculator, search_docs, time_tool]

        # ReAct Prompt 模板
        # 这个模板定义了 Agent 的思考格式
        react_prompt = PromptTemplate.from_template(
            """Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin! Answer in Chinese.

Question: {input}
Thought:{agent_scratchpad}"""
        )

        # 创建 ReAct Agent
        agent = create_react_agent(
            llm=llm,
            tools=all_tools,
            prompt=react_prompt,
        )

        # 创建 AgentExecutor（管理执行循环）
        executor = AgentExecutor(
            agent=agent,
            tools=all_tools,
            verbose=True,           # 打印完整思考过程
            max_iterations=5,       # 最多循环5次
            handle_parsing_errors=True,  # 自动处理格式错误
        )

        print("\n  [OK] ReAct Agent 创建成功!")
        print(f"  工具列表: {[t.name for t in all_tools]}")

        # 测试查询
        test_queries = [
            "北京今天天气怎么样？",
            "帮我计算 256 * 128",
            "什么是 RAG 技术？",
        ]

        for query in test_queries:
            print(f"\n  {'='*50}")
            print(f"  Query: {query}")
            print(f"  {'='*50}")
            try:
                result = executor.invoke({"input": query})
                print(f"\n  [OK] Final Answer: {result['output']}")
            except Exception as e:
                print(f"  [FAIL] Agent 执行错误: {e}")

    except ImportError as e:
        print(f"  [WARN] 缺少依赖: {e}")
        print("  (安装方法: pip install langchain langchain-openai)")
        USE_SIMULATION = True

if USE_SIMULATION or not LANGCHAIN_AVAILABLE:
    print("""
  [模拟模式] ReAct Agent 执行流程演示:

  Query: "北京今天天气怎么样？需要带伞吗？"

  > Entering new AgentExecutor chain...

  Thought: 用户想知道北京天气和是否需要带伞
           我需要先调用天气工具获取实时天气信息
  Action: get_weather
  Action Input: {"city": "北京"}
  Observation: 晴天，28度，湿度45%

  Thought: 我已经获得了北京的天气信息。晴天不需要带伞。
           我可以给出最终回答了。
  Final Answer: 北京今天晴天，气温28度，湿度45%。天气不错，不需要带伞！

  > Finished chain.
  """)

    print("  -" * 25)
    print("""
  Query: "帮我算一下 2的10次方 再乘以 3"

  > Entering new AgentExecutor chain...

  Thought: 用户需要计算 2^10 * 3，我应该用计算器工具
  Action: calculator
  Action Input: {"expression": "2 ** 10 * 3"}
  Observation: 2 ** 10 * 3 = 3072

  Thought: 计算已完成，结果是 3072
  Final Answer: 2的10次方再乘以3等于3072。
                计算过程: 2^10 = 1024, 1024 * 3 = 3072

  > Finished chain.
  """)

    print("  -" * 25)
    print("""
  Query: "什么是 RAG？"

  > Entering new AgentExecutor chain...

  Thought: 用户询问技术概念，我应该在知识库中搜索
  Action: search_docs
  Action Input: {"query": "RAG"}
  Observation: [RAG] RAG（检索增强生成）通过检索外部知识来增强LLM回答...

  Thought: 我找到了相关信息，可以回答用户了
  Final Answer: RAG（Retrieval-Augmented Generation，检索增强生成）
                是通过检索外部知识库来增强大语言模型回答质量的技术。
                它包括离线索引和在线查询两个阶段。

  > Finished chain.
  """)

print()


# ============================================================
# Part 6: 工具错误处理
# ============================================================

print("=" * 60)
print("  Part 6: 工具错误处理")
print("=" * 60)

if LANGCHAIN_AVAILABLE:
    @tool
    def risky_division(a: float, b: float) -> str:
        """执行除法运算。将第一个数除以第二个数。"""
        if b == 0:
            raise ValueError("除数不能为零!")
        return f"{a} / {b} = {a / b}"

    # 测试正常调用
    print("\n  --- 正常调用 ---")
    try:
        result = risky_division.invoke({"a": 10.0, "b": 3.0})
        print(f"  [OK] 10 / 3 = {result}")
    except Exception as e:
        print(f"  [FAIL] {e}")

    # 测试错误调用
    print("\n  --- 错误调用（除以零）---")
    try:
        result = risky_division.invoke({"a": 10.0, "b": 0.0})
        print(f"  [OK] {result}")
    except Exception as e:
        print(f"  [FAIL] 捕获到错误: {e}")
        print("  -> AgentExecutor 会把这个错误作为 Observation 反馈给 LLM")
        print("  -> LLM 看到错误后会告诉用户 '除数不能为零'")

    # 测试类型错误
    print("\n  --- 类型错误（传入字符串）---")
    try:
        result = risky_division.invoke({"a": "hello", "b": 3.0})
        print(f"  [OK] {result}")
    except Exception as e:
        print(f"  [FAIL] 捕获到类型错误: {type(e).__name__}: {e}")
        print("  -> Pydantic 自动校验参数类型")
else:
    print("""
  [模拟模式] 错误处理演示:

  正常调用: risky_division(10, 3) -> "10 / 3 = 3.333..."
  除以零:   risky_division(10, 0) -> ValueError("除数不能为零!")
  类型错误: risky_division("hello", 3) -> ValidationError

  AgentExecutor 的 handle_parsing_errors=True 会:
  1. 捕获工具异常
  2. 把错误信息作为 Observation 反馈给 LLM
  3. LLM 可以选择重试或告知用户
""")

print()


# ============================================================
# Part 7: 三种创建方式对比总结
# ============================================================

print("=" * 60)
print("  Part 7: 三种创建方式对比总结")
print("=" * 60)

print("""
  ┌──────────────────────────────────────────────────────┐
  │           LangChain Tool 创建方式对比                   │
  ├──────────────┬──────────────┬──────────────────────────┤
  │   @tool      │ Structured   │ BaseTool                 │
  │   装饰器      │ Tool         │ 子类                     │
  ├──────────────┼──────────────┼──────────────────────────┤
  │ 代码量最少    │ 中等          │ 最多                     │
  │ 自动推断      │ 半自动        │ 完全手动                  │
  │ 适合简单工具  │ 包装已有函数   │ 复杂/有状态的工具          │
  │ 使用率 90%   │ 使用率 8%     │ 使用率 2%                │
  └──────────────┴──────────────┴──────────────────────────┘

  选择建议:
  1. 新写的简单工具 -> @tool（绝大多数情况）
  2. 包装第三方函数 -> StructuredTool.from_function
  3. 需要初始化/状态/异步 -> BaseTool 子类
""")

print()


# ============================================================
# Part 8: 与 Day 31 的对比 + 总结
# ============================================================

print("=" * 60)
print("  Part 8: Day 31 手写 vs Day 32 LangChain 对比")
print("=" * 60)

print("""
  Day 31 手写 Function Calling:
    - 自己写 JSON Schema        -> LangChain: @tool 自动生成
    - 自己解析 tool_calls       -> LangChain: AgentExecutor 自动处理
    - 自己写 while 循环         -> LangChain: AgentExecutor.invoke()
    - 自己处理错误              -> LangChain: handle_parsing_errors
    - 50+ 行编排代码            -> LangChain: 10 行搞定

  但 Day 31 学的原理很重要！面试时:
    面试官: "LangChain Agent 底层是怎么工作的？"
    你: "底层就是 Function Calling 循环:
         1. 把工具描述和用户消息发给 LLM
         2. LLM 返回 tool_calls
         3. 执行工具，把结果放回 messages
         4. 再次调用 LLM，循环直到得到最终回答
         LangChain 的 AgentExecutor 就是把这个循环封装了。"
    面试官: (满意地点头)

  明天 (Day 33): 记忆机制 - Agent 怎么记住之前的对话
  后天 (Day 34): 规划机制 - Agent 怎么分步完成复杂任务
""")

print("=" * 60)
print("  Day 32 完成!")
print("=" * 60)
