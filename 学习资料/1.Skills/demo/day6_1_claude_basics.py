"""
Day 6 Demo 1：Claude API 基础调用
运行方式：python day6_1_claude_basics.py

前置条件：
  1. pip install anthropic python-dotenv
  2. 在项目根目录 .env 文件中配置：CLAUDE_API_KEY=sk-ant-api03-xxx

学习目标：
1. 学会用 Anthropic SDK 调用 Claude
2. 理解 messages 格式和参数
3. 掌握多轮对话的实现方式
4. 理解 token 计费
"""

import os
from dotenv import load_dotenv
from anthropic import Anthropic

# ============================================================
# 第零步：加载环境变量
# ============================================================

# 加载项目根目录的 .env 文件
# .env 文件内容示例：CLAUDE_API_KEY=sk-ant-api03-xxxxx
load_dotenv()

# 从环境变量获取 API Key
api_key = os.getenv("CLAUDE_API_KEY")

if not api_key:
    print("❌ 未找到 CLAUDE_API_KEY 环境变量！")
    print("   请在项目根目录创建 .env 文件，添加：")
    print("   CLAUDE_API_KEY=sk-ant-api03-你的key")
    print()
    print("   如果没有 API Key，可以先跳过这个 Demo，")
    print("   后面 Day 7 会用国内免费 API 做同样的事情。")
    exit(1)

# ============================================================
# 第一部分：创建客户端
# ============================================================

print("=" * 50)
print("第一部分：创建 Claude 客户端")
print("=" * 50)

# 创建 Anthropic 客户端
# 它会自动从环境变量 ANTHROPIC_API_KEY 读取 Key
# 也可以手动传入：Anthropic(api_key="xxx")
client = Anthropic(api_key=api_key)

print("✅ Claude 客户端创建成功")
print()


# ============================================================
# 第二部分：最简单的调用
# ============================================================

print("=" * 50)
print("第二部分：最简单的 API 调用")
print("=" * 50)

response = client.messages.create(
    model="claude-sonnet-4-20250514",  # 模型版本（Sonnet 性价比最高）
    max_tokens=256,                     # 最大输出 token 数
    messages=[
        {"role": "user", "content": "用一句话介绍Python。"}
    ],
)

# 解析响应
# response.content 是一个列表，通常只有一个元素
# 每个元素的 .text 是实际文本内容
answer = response.content[0].text

print(f"\n  问: 用一句话介绍Python。")
print(f"  答: {answer}")

# 查看 token 使用量
print(f"\n  📊 Token 使用:")
print(f"     输入 tokens: {response.usage.input_tokens}")
print(f"     输出 tokens: {response.usage.output_tokens}")
print(f"     模型: {response.model}")
print(f"     停止原因: {response.stop_reason}")
# stop_reason: "end_turn" = 正常结束, "max_tokens" = 到达上限被截断
print()


# ============================================================
# 第三部分：System Prompt（系统提示词）
# ============================================================

print("=" * 50)
print("第三部分：System Prompt —— 给 AI 设定人设")
print("=" * 50)

# 不加 system prompt
response1 = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=256,
    messages=[{"role": "user", "content": "什么是装饰器？"}],
)
print(f"\n  [无 system prompt]")
print(f"  答: {response1.content[0].text[:100]}...")

# 加 system prompt
response2 = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=256,
    system="你是一个Python编程老师，面对零基础学生。回答要通俗易懂，用生活中的例子类比，不超过3句话。",
    messages=[{"role": "user", "content": "什么是装饰器？"}],
)
print(f"\n  [有 system prompt: Python老师，面对零基础学生]")
print(f"  答: {response2.content[0].text[:150]}...")

print("\n  → system prompt 让同样的问题得到完全不同风格的回答")
print()


# ============================================================
# 第四部分：Temperature —— 控制创造性
# ============================================================

print("=" * 50)
print("第四部分：Temperature 参数")
print("=" * 50)

prompt = "给我的AI学习项目起一个名字。"

# temperature=0: 最确定性（每次结果几乎一样）
response_low = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=100,
    temperature=0,
    messages=[{"role": "user", "content": prompt}],
)

# temperature=1: 最创造性（每次结果可能不同）
response_high = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=100,
    temperature=1,
    messages=[{"role": "user", "content": prompt}],
)

print(f"\n  问: {prompt}")
print(f"  temperature=0（确定性）: {response_low.content[0].text[:80]}...")
print(f"  temperature=1（创造性）: {response_high.content[0].text[:80]}...")
print()
print("  → 写代码/分析用低温度(0-0.3)，创意/对话用高温度(0.7-1)")
print()


# ============================================================
# 第五部分：多轮对话
# ============================================================

print("=" * 50)
print("第五部分：多轮对话 —— 让 AI 记住上下文")
print("=" * 50)

# 多轮对话的关键：把历史消息全部传给 API
# API 本身是无状态的，它不会"记住"你之前说了什么
# 要靠你在代码里维护 messages 列表

messages = []  # 对话历史

# 第一轮
messages.append({"role": "user", "content": "我叫张三，是AI专业大三学生。"})

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=256,
    system="你是一个友好的学习顾问。",
    messages=messages,
)
assistant_reply = response.content[0].text
messages.append({"role": "assistant", "content": assistant_reply})

print(f"\n  👤 我: 我叫张三，是AI专业大三学生。")
print(f"  🤖 AI: {assistant_reply[:100]}...")

# 第二轮 —— AI 应该记得你叫张三
messages.append({"role": "user", "content": "我叫什么名字？我是什么专业的？"})

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=256,
    system="你是一个友好的学习顾问。",
    messages=messages,
)
assistant_reply = response.content[0].text
messages.append({"role": "assistant", "content": assistant_reply})

print(f"\n  👤 我: 我叫什么名字？我是什么专业的？")
print(f"  🤖 AI: {assistant_reply[:100]}...")

# 第三轮
messages.append({"role": "user", "content": "给我推荐一个适合我背景的AI项目。"})

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=256,
    system="你是一个友好的学习顾问。",
    messages=messages,
)
assistant_reply = response.content[0].text

print(f"\n  👤 我: 给我推荐一个适合我背景的AI项目。")
print(f"  🤖 AI: {assistant_reply[:150]}...")

print(f"\n  📊 对话历史共 {len(messages) + 1} 条消息")
print(f"     本次 token: 输入={response.usage.input_tokens}, 输出={response.usage.output_tokens}")
print("  → 注意：随着对话轮次增加，输入 token 越来越多（因为要带上所有历史）")
print()


# ============================================================
# 总结
# ============================================================

print("=" * 50)
print("📝 Claude API 基础总结")
print("=" * 50)
print("""
1. Anthropic(api_key=...) 创建客户端
2. client.messages.create() 发送请求
3. messages 列表是对话核心（user/assistant 交替）
4. system prompt 设定 AI 人设
5. temperature 控制创造性（0=确定, 1=创意）
6. response.content[0].text 获取回答文本
7. response.usage 查看 token 消耗
8. 多轮对话 = 维护完整 messages 列表

下一个 Demo：流式输出（边生成边返回）
""")
