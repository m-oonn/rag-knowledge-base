"""
Day 6 Demo 2：Claude 流式输出 + 实用封装
运行方式：python day6_2_claude_stream.py

学习目标：
1. 掌握流式输出（像 ChatGPT 一样逐字显示）
2. 学会封装一个可复用的 LLM 调用类
3. 理解 token 统计和成本控制
4. 为后面 RAG 项目打基础
"""

import os
import time
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv()
api_key = os.getenv("CLAUDE_API_KEY")

if not api_key:
    print("❌ 未找到 CLAUDE_API_KEY，请配置 .env 文件")
    exit(1)

client = Anthropic(api_key=api_key)


# ============================================================
# 第一部分：流式输出
# ============================================================

print("=" * 50)
print("第一部分：流式输出 —— 逐字显示回答")
print("=" * 50)


def stream_chat(question: str, system: str = ""):
    """
    流式调用 Claude API。

    和普通调用的区别：
    - 普通: response = client.messages.create(...)  → 等全部生成完返回
    - 流式: client.messages.stream(...)             → 逐个 token 返回

    用 with 语句确保连接正确关闭（上下文管理器！Day 2 学的）
    """
    print(f"\n  👤 问: {question}")
    print(f"  🤖 答: ", end="", flush=True)

    full_response = ""
    start = time.time()

    # stream() 返回一个流式上下文管理器
    with client.messages.stream(
        model="claude-sonnet-4-20250514",
        max_tokens=512,
        system=system,
        messages=[{"role": "user", "content": question}],
    ) as stream:
        # 逐个 token 接收
        for text in stream.text_stream:
            print(text, end="", flush=True)  # 逐字打印（flush 确保立即显示）
            full_response += text

    elapsed = time.time() - start
    print()  # 换行

    # 获取最终的完整响应（包含 token 统计）
    final = stream.get_final_message()
    print(f"\n  📊 耗时: {elapsed:.1f}秒")
    print(f"     输入 tokens: {final.usage.input_tokens}")
    print(f"     输出 tokens: {final.usage.output_tokens}")

    return full_response


# 测试流式输出
stream_chat(
    "用Python写一个计算斐波那契数列的函数，加上中文注释。",
    system="你是Python编程助手。代码简洁实用。"
)
print()


# ============================================================
# 第二部分：封装可复用的 LLM 客户端
# ============================================================

print("=" * 50)
print("第二部分：封装 ChatClient 类（后面项目会用到）")
print("=" * 50)


class ChatClient:
    """
    可复用的聊天客户端。

    封装了：
    - 多轮对话（自动维护历史）
    - 流式/非流式输出
    - Token 统计
    - 对话重置

    后面做 RAG 项目时，会在这个基础上加上"上下文注入"功能。
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        system: str = "",
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ):
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.system = system
        self.max_tokens = max_tokens
        self.temperature = temperature

        # 对话历史
        self.messages: list[dict] = []

        # Token 统计（追踪总消耗）
        self.total_input_tokens = 0
        self.total_output_tokens = 0

    def chat(self, user_message: str) -> str:
        """
        发送消息并获取回复（非流式）。

        自动维护对话历史，支持多轮对话。
        """
        self.messages.append({"role": "user", "content": user_message})

        response = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=self.system,
            messages=self.messages,
        )

        assistant_reply = response.content[0].text
        self.messages.append({"role": "assistant", "content": assistant_reply})

        # 累计 token
        self.total_input_tokens += response.usage.input_tokens
        self.total_output_tokens += response.usage.output_tokens

        return assistant_reply

    def chat_stream(self, user_message: str) -> str:
        """
        发送消息并获取回复（流式，逐字打印）。
        """
        self.messages.append({"role": "user", "content": user_message})

        full_response = ""

        with self.client.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            system=self.system,
            messages=self.messages,
        ) as stream:
            for text in stream.text_stream:
                print(text, end="", flush=True)
                full_response += text

        print()  # 换行

        self.messages.append({"role": "assistant", "content": full_response})

        # 累计 token
        final = stream.get_final_message()
        self.total_input_tokens += final.usage.input_tokens
        self.total_output_tokens += final.usage.output_tokens

        return full_response

    def reset(self):
        """清空对话历史（开始新对话）"""
        self.messages = []

    def get_stats(self) -> dict:
        """获取使用统计"""
        return {
            "对话轮次": len(self.messages) // 2,
            "总输入tokens": self.total_input_tokens,
            "总输出tokens": self.total_output_tokens,
            "总tokens": self.total_input_tokens + self.total_output_tokens,
        }


# --- 使用封装好的客户端 ---

bot = ChatClient(
    api_key=api_key,
    system="你是一个AI学习助手，帮助大三学生理解AI开发概念。回答简洁，每次不超过3句话。",
    temperature=0.3,
)

print("\n--- 多轮对话演示 ---\n")

# 第一轮
print("👤 问: 什么是RAG？")
print("🤖 答: ", end="")
bot.chat_stream("什么是RAG？")

# 第二轮（AI 记得上文）
print("\n👤 问: 它和直接问大模型有什么区别？")
print("🤖 答: ", end="")
bot.chat_stream("它和直接问大模型有什么区别？")

# 第三轮
print("\n👤 问: 我的项目用到哪些技术？")
print("🤖 答: ", end="")
bot.chat_stream("如果我要做一个RAG知识库问答系统，需要哪些核心技术？")

# 查看统计
print(f"\n📊 使用统计: {bot.get_stats()}")
print()


# ============================================================
# 第三部分：错误处理
# ============================================================

print("=" * 50)
print("第三部分：错误处理（API 调用必须有）")
print("=" * 50)

from anthropic import APIError, APIConnectionError, RateLimitError


def safe_chat(client_instance: Anthropic, question: str) -> str:
    """
    带错误处理的 API 调用。

    在生产项目中，API 调用一定要有错误处理：
    - 网络可能断开
    - API Key 可能过期
    - 可能触发限流
    - 服务器可能故障
    """
    try:
        response = client_instance.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=256,
            messages=[{"role": "user", "content": question}],
        )
        return response.content[0].text

    except RateLimitError:
        # 触发限流（请求太频繁）
        print("⚠️ 请求太频繁，请稍等后重试")
        return "请求频率过高，请稍后再试"

    except APIConnectionError:
        # 网络连接失败
        print("⚠️ 无法连接到 Claude API，请检查网络")
        return "网络连接失败"

    except APIError as e:
        # 其他 API 错误
        print(f"⚠️ API 错误: {e.status_code} - {e.message}")
        return f"API 调用失败: {e.message}"

    except Exception as e:
        # 未知错误
        print(f"⚠️ 未知错误: {e}")
        return "发生未知错误"


# 测试正常调用
result = safe_chat(client, "1+1等于几？")
print(f"\n  正常调用结果: {result}")
print()


# ============================================================
# 总结
# ============================================================

print("=" * 50)
print("📝 Day 6 完整总结")
print("=" * 50)
print("""
今天学了 Claude API 的三大核心：

【基础调用】
  - client.messages.create() 发送请求
  - messages 列表维护对话历史
  - system prompt 定义 AI 人设
  - temperature 控制创造性

【流式输出】
  - client.messages.stream() 流式调用
  - for text in stream.text_stream 逐字接收
  - 用户体验好，后面项目必用

【工程实践】
  - 封装 ChatClient 类（可复用）
  - Token 统计（成本控制）
  - 错误处理（必须有）

明天学国内大模型 API（DeepSeek/月之暗面），
写法几乎一样，只是换个 SDK。
""")
