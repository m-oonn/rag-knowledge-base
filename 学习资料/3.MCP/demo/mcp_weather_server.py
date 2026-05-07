"""
MCP Demo: 自定义天气 MCP Server

这是一个完整的 MCP Server 示例，提供两个工具：
1. get_weather: 获取城市天气
2. get_time: 获取当前时间

运行方式：
  pip install mcp
  python mcp_weather_server.py

注意：这个 Server 需要通过 MCP Client 连接才能使用。
可以在 Claude Desktop 中配置使用，也可以用下面的测试脚本测试。

学习目标：
1. 理解 MCP Server 的结构
2. 理解 Tool 的定义方式（name + description + inputSchema）
3. 理解 Client 调用 Tool 的流程
"""

import asyncio
import json
from datetime import datetime

# ============================================================
# 方式一：不依赖 MCP SDK，手写一个"模拟 MCP Server"
# 让你理解 MCP 的核心逻辑，即使装不上 mcp 包也能跑
# ============================================================

class SimpleMCPServer:
    """
    简化版 MCP Server —— 不用 mcp SDK，纯 Python 实现核心逻辑。

    真正的 MCP Server 用 JSON-RPC 通过 stdio/SSE 通信，
    这里我们用普通函数调用模拟，让你理解原理。

    核心流程：
    1. 定义 tools（工具列表 + 参数 Schema）
    2. Client 问 "你有什么工具？" → list_tools()
    3. Client 说 "调用 get_weather，参数是 {city: 北京}" → call_tool()
    4. Server 执行逻辑，返回结果
    """

    def __init__(self, name: str):
        self.name = name
        self._tools = {}  # name -> {schema, handler}

    def tool(self, name: str, description: str, parameters: dict):
        """
        装饰器：注册一个工具。

        用法：
        @server.tool("get_weather", "Get weather", {...})
        def get_weather(city: str): ...

        这个 parameters 就是 JSON Schema，定义工具参数的类型：
        {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"}
            },
            "required": ["city"]
        }
        """
        def decorator(func):
            self._tools[name] = {
                "name": name,
                "description": description,
                "inputSchema": parameters,
                "handler": func,
            }
            return func
        return decorator

    def list_tools(self) -> list[dict]:
        """
        返回所有可用工具的描述。

        MCP Client 连接时首先调用这个，
        拿到工具列表后告诉 LLM "你可以用这些工具"。
        """
        return [
            {
                "name": t["name"],
                "description": t["description"],
                "inputSchema": t["inputSchema"],
            }
            for t in self._tools.values()
        ]

    async def call_tool(self, name: str, arguments: dict) -> str:
        """
        调用指定工具。

        LLM 决定要用某个工具后，Client 把工具名和参数发给 Server，
        Server 执行对应的 handler 并返回结果。
        """
        if name not in self._tools:
            return json.dumps({"error": f"Tool '{name}' not found"})

        handler = self._tools[name]["handler"]

        # 支持同步和异步 handler
        if asyncio.iscoroutinefunction(handler):
            result = await handler(**arguments)
        else:
            result = handler(**arguments)

        return result


# ============================================================
# 创建 Server 并注册工具
# ============================================================

server = SimpleMCPServer("weather-and-time")

# --- 工具 1：获取天气 ---
@server.tool(
    name="get_weather",
    description="Get the current weather for a city. Returns temperature, condition, and humidity.",
    parameters={
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name (e.g., Beijing, Shanghai, Tokyo)",
            },
            "unit": {
                "type": "string",
                "enum": ["celsius", "fahrenheit"],
                "description": "Temperature unit",
                "default": "celsius",
            },
        },
        "required": ["city"],
    },
)
def get_weather(city: str, unit: str = "celsius") -> str:
    """
    模拟天气 API（真实项目会调用实际的天气 API）。

    在 MCP 中，这个函数就是工具的实现逻辑。
    Client 调用 get_weather 时，Server 执行这个函数并返回结果。
    """
    # 模拟数据
    weather_data = {
        "beijing":  {"temp": 22, "condition": "sunny",  "humidity": 45},
        "shanghai": {"temp": 26, "condition": "cloudy", "humidity": 72},
        "tokyo":    {"temp": 20, "condition": "rainy",  "humidity": 85},
        "london":   {"temp": 15, "condition": "foggy",  "humidity": 90},
    }

    city_lower = city.lower().replace(" ", "")

    # 查找城市（模糊匹配）
    data = None
    for key, val in weather_data.items():
        if key in city_lower or city_lower in key:
            data = val
            break

    if not data:
        data = {"temp": 20, "condition": "unknown", "humidity": 50}

    temp = data["temp"]
    if unit == "fahrenheit":
        temp = temp * 9 / 5 + 32

    unit_symbol = "°C" if unit == "celsius" else "°F"

    return json.dumps({
        "city": city,
        "temperature": f"{temp}{unit_symbol}",
        "condition": data["condition"],
        "humidity": f"{data['humidity']}%",
    }, ensure_ascii=False)


# --- 工具 2：获取时间 ---
@server.tool(
    name="get_time",
    description="Get the current date and time.",
    parameters={
        "type": "object",
        "properties": {
            "format": {
                "type": "string",
                "description": "Time format: 'full' or 'short'",
                "default": "full",
            },
        },
    },
)
def get_time(format: str = "full") -> str:
    """获取当前时间"""
    now = datetime.now()
    if format == "short":
        return now.strftime("%Y-%m-%d %H:%M")
    return now.strftime("%Y-%m-%d %H:%M:%S (weekday: %A)")


# --- 工具 3：简单计算器 ---
@server.tool(
    name="calculate",
    description="Perform basic math calculation. Supports +, -, *, /, **.",
    parameters={
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "Math expression, e.g., '2 + 3 * 4'",
            },
        },
        "required": ["expression"],
    },
)
def calculate(expression: str) -> str:
    """
    安全的数学计算。

    注意：生产环境不能直接 eval 用户输入！
    这里做了基本的安全检查，只允许数字和运算符。
    """
    # 安全检查：只允许数字、运算符、空格、小数点、括号
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return json.dumps({"error": "Invalid characters in expression"})

    try:
        result = eval(expression)  # 简化处理，生产环境应该用 ast.literal_eval 或专门的解析器
        return json.dumps({"expression": expression, "result": result})
    except Exception as e:
        return json.dumps({"error": str(e)})


# ============================================================
# 模拟 MCP Client 调用流程
# ============================================================

async def simulate_mcp_interaction():
    """
    模拟完整的 MCP Client-Server 交互流程。

    真实场景：
    1. Claude Desktop 启动你的 MCP Server
    2. Claude 问 Server "你有什么工具？"
    3. 用户问 Claude "北京天气怎么样？"
    4. Claude 决定调用 get_weather 工具
    5. Claude Desktop 把请求发给 Server
    6. Server 执行并返回结果
    7. Claude 根据结果回答用户
    """

    print("=" * 55)
    print("  MCP Server-Client Interaction Demo")
    print("=" * 55)

    # Step 1: Client 获取工具列表
    print("\n  [Step 1] Client asks: 'What tools do you have?'\n")
    tools = server.list_tools()
    for tool in tools:
        print(f"    Tool: {tool['name']}")
        print(f"      Description: {tool['description']}")
        params = tool['inputSchema'].get('properties', {})
        print(f"      Parameters: {list(params.keys())}")
        print()

    # Step 2: 模拟 LLM 决定调用工具
    print("  [Step 2] User asks: 'What is the weather in Beijing?'")
    print("           LLM decides to call: get_weather(city='Beijing')\n")

    # Step 3: Client 调用工具
    print("  [Step 3] Client calls Server tool:\n")

    result = await server.call_tool("get_weather", {"city": "Beijing"})
    print(f"    get_weather(city='Beijing')")
    print(f"    Result: {result}")

    result = await server.call_tool("get_weather", {"city": "Tokyo", "unit": "fahrenheit"})
    print(f"\n    get_weather(city='Tokyo', unit='fahrenheit')")
    print(f"    Result: {result}")

    result = await server.call_tool("get_time", {"format": "short"})
    print(f"\n    get_time(format='short')")
    print(f"    Result: {result}")

    result = await server.call_tool("calculate", {"expression": "2 ** 10 + 3 * 7"})
    print(f"\n    calculate(expression='2 ** 10 + 3 * 7')")
    print(f"    Result: {result}")

    # Step 4: 模拟 LLM 用结果生成回答
    print("\n  [Step 4] LLM generates answer based on tool results:")
    print("           'Beijing is currently 22°C and sunny, humidity 45%.'")
    print()

    # 展示完整的 JSON-RPC 风格消息（MCP 实际通信格式）
    print("  [Bonus] What MCP messages look like (JSON-RPC):\n")

    # 工具列表请求
    print("    Client -> Server:")
    print(f'    {json.dumps({"jsonrpc": "2.0", "method": "tools/list", "id": 1})}')
    print()

    # 工具调用请求
    print("    Client -> Server:")
    call_msg = {
        "jsonrpc": "2.0",
        "method": "tools/call",
        "params": {"name": "get_weather", "arguments": {"city": "Beijing"}},
        "id": 2,
    }
    print(f"    {json.dumps(call_msg)}")
    print()

    # 工具调用响应
    print("    Server -> Client:")
    resp_msg = {
        "jsonrpc": "2.0",
        "result": {"content": [{"type": "text", "text": result}]},
        "id": 2,
    }
    print(f"    {json.dumps(resp_msg, ensure_ascii=False)}")
    print()


# ============================================================
# 展示 Claude Desktop 配置方式
# ============================================================

def show_claude_config():
    print("=" * 55)
    print("  How to use in Claude Desktop")
    print("=" * 55)
    print("""
  To use this MCP Server with Claude Desktop,
  add this to your claude_desktop_config.json:

  Windows: %APPDATA%\\Claude\\claude_desktop_config.json
  Mac: ~/Library/Application Support/Claude/claude_desktop_config.json

  {
    "mcpServers": {
      "weather": {
        "command": "python",
        "args": ["path/to/mcp_weather_server.py"]
      }
    }
  }

  Note: For a real MCP Server, you would use the official
  mcp SDK with stdio transport instead of our simplified version.

  Install the real SDK: pip install mcp
  Docs: https://modelcontextprotocol.io
""")


# ============================================================
# 主函数
# ============================================================

async def main():
    await simulate_mcp_interaction()
    show_claude_config()

    print("=" * 55)
    print("  MCP Summary")
    print("=" * 55)
    print("""
  MCP Core Concepts:

  1. Server defines tools (name + description + parameters)
  2. Client asks "what tools?" -> list_tools()
  3. LLM decides which tool to call based on user query
  4. Client calls tool with arguments -> call_tool()
  5. Server executes and returns result
  6. LLM uses result to answer user

  This is the SAME pattern as Agent tool calling!
  - MCP = protocol standard for tool communication
  - Function Calling = LLM's ability to decide which tool
  - LangChain Tools = framework wrapper around both

  You don't need MCP for your projects, but understanding
  the pattern helps you grasp Agent concepts (Day 31+).
""")


if __name__ == "__main__":
    asyncio.run(main())
