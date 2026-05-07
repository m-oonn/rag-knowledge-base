"""
Day 7 Demo 1：国内大模型 API 调用
运行方式：python day7_1_domestic_llm.py

前置条件：
  1. pip install openai python-dotenv
  2. 在 .env 中配置至少一个 Key：
     DEEPSEEK_API_KEY=sk-xxx
     MOONSHOT_API_KEY=sk-xxx

学习目标：
1. 掌握 OpenAI 兼容格式的调用方式
2. 学会切换不同模型（只改 base_url）
3. 理解多模型统一调用的封装思路
4. 实现流式输出
"""

import os
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


# ============================================================
# 第一部分：DeepSeek 调用
# ============================================================

print("=" * 50)
print("第一部分：DeepSeek API 调用")
print("=" * 50)

deepseek_key = os.getenv("DEEPSEEK_API_KEY")

if deepseek_key:
    # 创建 DeepSeek 客户端
    # 关键：用 OpenAI SDK，但 base_url 指向 DeepSeek
    ds_client = OpenAI(
        api_key=deepseek_key,
        base_url="https://api.deepseek.com",
    )

    response = ds_client.chat.completions.create(
        model="deepseek-chat",
        max_tokens=256,
        temperature=0.3,
        messages=[
            # 注意：OpenAI 格式中 system 放在 messages 列表里
            {"role": "system", "content": "你是一个简洁的编程助手，回答不超过3句话。"},
            {"role": "user", "content": "Python的async/await有什么用？"},
        ],
    )

    # 解析响应 —— 和 Claude 格式不同！
    # Claude: response.content[0].text
    # OpenAI: response.choices[0].message.content
    answer = response.choices[0].message.content

    print(f"\n  🤖 DeepSeek: {answer}")
    print(f"  📊 tokens: 输入={response.usage.prompt_tokens}, 输出={response.usage.completion_tokens}")
else:
    print("\n  ⏭ 跳过 DeepSeek（未配置 DEEPSEEK_API_KEY）")

print()


# ============================================================
# 第二部分：Moonshot（月之暗面）调用
# ============================================================

print("=" * 50)
print("第二部分：Moonshot API 调用")
print("=" * 50)

moonshot_key = os.getenv("MOONSHOT_API_KEY")

if moonshot_key:
    # 同样的 SDK，只是换了 base_url
    ms_client = OpenAI(
        api_key=moonshot_key,
        base_url="https://api.moonshot.cn/v1",
    )

    response = ms_client.chat.completions.create(
        model="moonshot-v1-8k",
        max_tokens=256,
        messages=[
            {"role": "system", "content": "你是一个简洁的编程助手，回答不超过3句话。"},
            {"role": "user", "content": "Python的async/await有什么用？"},
        ],
    )

    answer = response.choices[0].message.content
    print(f"\n  🤖 Moonshot: {answer}")
    print(f"  📊 tokens: 输入={response.usage.prompt_tokens}, 输出={response.usage.completion_tokens}")
else:
    print("\n  ⏭ 跳过 Moonshot（未配置 MOONSHOT_API_KEY）")

print()


# ============================================================
# 第三部分：统一封装 —— 一套代码调用多个模型
# ============================================================

print("=" * 50)
print("第三部分：统一封装 LLMClient")
print("=" * 50)


class LLMClient:
    """
    统一的大模型客户端 —— 一套接口调用所有模型。

    这是你后面项目中会用到的核心封装：
    - 开发时用 DeepSeek（便宜）
    - 测试时用 Ollama（免费本地）
    - 生产环境用 Claude（最强）

    只需要改一个参数就能切换模型，业务代码完全不用动。
    """

    # 模型配置表
    PROVIDERS = {
        "deepseek": {
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-chat",
            "env_key": "DEEPSEEK_API_KEY",
        },
        "moonshot": {
            "base_url": "https://api.moonshot.cn/v1",
            "model": "moonshot-v1-8k",
            "env_key": "MOONSHOT_API_KEY",
        },
        "ollama": {
            "base_url": "http://localhost:11434/v1",
            "model": "qwen2.5:7b",
            "env_key": None,  # Ollama 不需要 Key
        },
    }

    def __init__(self, provider: str = "deepseek"):
        if provider not in self.PROVIDERS:
            raise ValueError(f"不支持的模型: {provider}，可选: {list(self.PROVIDERS.keys())}")

        config = self.PROVIDERS[provider]
        api_key = os.getenv(config["env_key"]) if config["env_key"] else "ollama"

        self.client = OpenAI(api_key=api_key, base_url=config["base_url"])
        self.model = config["model"]
        self.provider = provider

        # 对话历史
        self.messages: list[dict] = []
        self.system_prompt = ""

    def set_system(self, prompt: str):
        """设置系统提示词"""
        self.system_prompt = prompt

    def chat(self, user_message: str) -> str:
        """发送消息，返回回答"""
        # 构建完整的 messages（system + 历史 + 新消息）
        full_messages = []
        if self.system_prompt:
            full_messages.append({"role": "system", "content": self.system_prompt})
        full_messages.extend(self.messages)
        full_messages.append({"role": "user", "content": user_message})

        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=1024,
            temperature=0.3,
            messages=full_messages,
        )

        answer = response.choices[0].message.content

        # 记录历史
        self.messages.append({"role": "user", "content": user_message})
        self.messages.append({"role": "assistant", "content": answer})

        return answer

    def chat_stream(self, user_message: str) -> str:
        """流式输出"""
        full_messages = []
        if self.system_prompt:
            full_messages.append({"role": "system", "content": self.system_prompt})
        full_messages.extend(self.messages)
        full_messages.append({"role": "user", "content": user_message})

        stream = self.client.chat.completions.create(
            model=self.model,
            max_tokens=1024,
            temperature=0.3,
            messages=full_messages,
            stream=True,  # 开启流式！
        )

        full_response = ""
        for chunk in stream:
            # 流式响应的每个 chunk 结构
            if chunk.choices[0].delta.content:
                text = chunk.choices[0].delta.content
                print(text, end="", flush=True)
                full_response += text

        print()  # 换行

        self.messages.append({"role": "user", "content": user_message})
        self.messages.append({"role": "assistant", "content": full_response})

        return full_response

    def reset(self):
        """清空对话历史"""
        self.messages = []


# --- 测试统一客户端 ---

# 找到一个可用的 provider
available = None
for provider in ["deepseek", "moonshot"]:
    config = LLMClient.PROVIDERS[provider]
    if os.getenv(config["env_key"]):
        available = provider
        break

if available:
    print(f"\n  使用模型: {available}")
    llm = LLMClient(provider=available)
    llm.set_system("你是Python编程助手，回答简洁，不超过3句话。")

    # 非流式
    answer = llm.chat("FastAPI和Flask哪个好？")
    print(f"\n  非流式回答: {answer}")

    # 流式
    print(f"\n  流式回答: ", end="")
    llm.chat_stream("为什么FastAPI更适合AI项目？")

    print(f"\n  ✅ 统一客户端测试成功！")
    print(f"     切换模型只需改 LLMClient('deepseek') → LLMClient('moonshot')")
else:
    print("\n  ⏭ 没有可用的 API Key，跳过测试")
    print("     请在 .env 中配置 DEEPSEEK_API_KEY 或 MOONSHOT_API_KEY")

print()


# ============================================================
# 第四部分：多模型对比
# ============================================================

print("=" * 50)
print("第四部分：多模型对比（如果有多个 Key）")
print("=" * 50)

question = "用一句话解释什么是向量数据库。"
results = {}

for provider in ["deepseek", "moonshot"]:
    config = LLMClient.PROVIDERS[provider]
    key = os.getenv(config["env_key"]) if config["env_key"] else None
    if not key:
        continue

    try:
        llm = LLMClient(provider=provider)
        start = time.time()
        answer = llm.chat(question)
        elapsed = time.time() - start
        results[provider] = {"answer": answer, "time": elapsed}
        print(f"\n  [{provider}] ({elapsed:.1f}s): {answer[:100]}")
    except Exception as e:
        print(f"\n  [{provider}] 调用失败: {e}")

if not results:
    print("\n  ⏭ 未配置任何 API Key，跳过对比")
    print("     这部分等你注册了 API 再回来跑")

print()


# ============================================================
# 总结
# ============================================================

print("=" * 50)
print("📝 Day 7 总结")
print("=" * 50)
print("""
1. 国内大模型几乎都兼容 OpenAI 格式
2. 用 openai SDK + 改 base_url 就能切换模型
3. Claude 格式 vs OpenAI 格式的区别：
   - system 位置不同（单独参数 vs messages里）
   - 回答位置不同（content[0].text vs choices[0].message.content）
4. 封装统一 LLMClient 可以一套代码切换多模型
5. 流式输出: stream=True + 遍历 chunk

明天学 Ollama 本地模型 —— 同样兼容 OpenAI 格式，
也就是说你今天写的 LLMClient 类，明天不用改就能调用本地模型！
""")
