"""
Day 8 Demo 1：Ollama 本地模型调用
运行方式：python day8_1_ollama_basics.py

前置条件：
  1. 安装 Ollama: https://ollama.com/download
  2. 拉取模型: ollama pull qwen2.5:7b
  3. 确认 Ollama 在运行: ollama ps 或 curl http://localhost:11434/api/tags

学习目标：
1. 学会通过原生 API 和 OpenAI 兼容模式调用 Ollama
2. 理解本地模型和云端 API 的用法一致性
3. 掌握流式输出在本地模型中的使用
4. 验证 Day 7 的 LLMClient 可以无缝切换到本地模型
"""

import sys
import time
import requests

# ============================================================
# 第零步：检查 Ollama 是否在运行
# ============================================================

OLLAMA_BASE = "http://localhost:11434"
MODEL = "qwen2.5:7b"  # 如果你拉的是其他模型，改这里

print("=" * 50)
print("Check: Ollama is running?")
print("=" * 50)

try:
    resp = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
    models = [m["name"] for m in resp.json().get("models", [])]
    print(f"\n  [OK] Ollama is running")
    print(f"  Installed models: {models}")

    if not any(MODEL.split(":")[0] in m for m in models):
        print(f"\n  [WARN] Model '{MODEL}' not found!")
        print(f"  Run: ollama pull {MODEL}")
        print(f"  Available models: {models}")
        if models:
            MODEL = models[0]
            print(f"  Using '{MODEL}' instead")
        else:
            print("  No models installed. Please run: ollama pull qwen2.5:7b")
            sys.exit(1)
    else:
        print(f"  Using model: {MODEL}")

except requests.ConnectionError:
    print("\n  [FAIL] Cannot connect to Ollama!")
    print("  Please make sure Ollama is installed and running:")
    print("    1. Download from https://ollama.com/download")
    print("    2. Install and start Ollama")
    print("    3. Run: ollama pull qwen2.5:7b")
    sys.exit(1)

print()


# ============================================================
# 第一部分：Ollama 原生 API 调用
# ============================================================

print("=" * 50)
print("Part 1: Ollama Native API")
print("=" * 50)


def ollama_generate(prompt: str, model: str = MODEL) -> str:
    """
    Ollama 原生 API 调用。

    端点: POST /api/generate
    这是 Ollama 自己的 API 格式，不是 OpenAI 格式。
    简单直接，适合快速测试。
    """
    response = requests.post(
        f"{OLLAMA_BASE}/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,  # 非流式，等全部生成完返回
            # 可选参数：
            # "temperature": 0.3,
            # "num_predict": 256,  # 最大生成 token 数
        },
        timeout=60,
    )
    data = response.json()
    return data["response"]


print(f"\n  Calling {MODEL}...")
start = time.time()
answer = ollama_generate("What is Python? Answer in Chinese, max 2 sentences.")
elapsed = time.time() - start

print(f"  Q: What is Python?")
print(f"  A: {answer.strip()}")
print(f"  Time: {elapsed:.1f}s")
print()


# ============================================================
# 第二部分：Ollama Chat API（多轮对话）
# ============================================================

print("=" * 50)
print("Part 2: Ollama Chat API (multi-turn)")
print("=" * 50)


def ollama_chat(messages: list, model: str = MODEL) -> str:
    """
    Ollama 的 Chat API（支持多轮对话）。

    端点: POST /api/chat
    messages 格式和 OpenAI 一样：
    [{"role": "system/user/assistant", "content": "..."}]
    """
    response = requests.post(
        f"{OLLAMA_BASE}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
        },
        timeout=60,
    )
    return response.json()["message"]["content"]


# 多轮对话
messages = [
    {"role": "system", "content": "You are a Python teacher. Answer in Chinese, max 2 sentences."},
    {"role": "user", "content": "What is a decorator?"},
]

print(f"\n  Q: What is a decorator?")
answer1 = ollama_chat(messages)
print(f"  A: {answer1.strip()}")

# 追问（带上历史）
messages.append({"role": "assistant", "content": answer1})
messages.append({"role": "user", "content": "Give me a simple example."})

print(f"\n  Q: Give me a simple example.")
answer2 = ollama_chat(messages)
print(f"  A: {answer2.strip()[:200]}")
print()


# ============================================================
# 第三部分：OpenAI 兼容模式（重点！）
# ============================================================

print("=" * 50)
print("Part 3: OpenAI Compatible Mode")
print("=" * 50)

from openai import OpenAI

# 关键：用 OpenAI SDK 调用 Ollama！
# 只需要改 base_url 和 api_key
ollama_client = OpenAI(
    api_key="ollama",                         # Ollama 不检查 key，随便填
    base_url=f"{OLLAMA_BASE}/v1",            # 指向 Ollama 的 OpenAI 兼容端点
)

print(f"\n  Using OpenAI SDK to call Ollama...")

response = ollama_client.chat.completions.create(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a helpful assistant. Answer in Chinese."},
        {"role": "user", "content": "Explain async/await in one sentence."},
    ],
    temperature=0.3,
    max_tokens=256,
)

answer = response.choices[0].message.content
print(f"  Q: Explain async/await in one sentence.")
print(f"  A: {answer.strip()}")
print()
print("  --> Same code as calling DeepSeek/Moonshot, only base_url changed!")
print()


# ============================================================
# 第四部分：流式输出
# ============================================================

print("=" * 50)
print("Part 4: Streaming Output")
print("=" * 50)


def ollama_stream_native(prompt: str, model: str = MODEL):
    """
    Ollama 原生流式输出。

    stream=True 时，返回的是一行一行的 JSON（NDJSON 格式），
    每行包含一个 token。
    """
    response = requests.post(
        f"{OLLAMA_BASE}/api/generate",
        json={"model": model, "prompt": prompt, "stream": True},
        stream=True,  # requests 也要开启流式
        timeout=60,
    )

    full = ""
    for line in response.iter_lines():
        if line:
            import json
            data = json.loads(line)
            token = data.get("response", "")
            print(token, end="", flush=True)
            full += token
            if data.get("done"):
                break
    print()
    return full


print(f"\n  Native streaming:")
print(f"  A: ", end="")
ollama_stream_native("List 3 benefits of FastAPI. Answer in Chinese, be brief.")
print()


# OpenAI SDK 流式
print(f"  OpenAI SDK streaming:")
print(f"  A: ", end="")

stream = ollama_client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "List 3 benefits of async programming. Answer in Chinese, be brief."}],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
print()


# ============================================================
# 第五部分：验证 Day 7 的 LLMClient 兼容性
# ============================================================

print("=" * 50)
print("Part 5: LLMClient Compatibility Test")
print("=" * 50)

import os

# 直接用 Day 7 封装的统一客户端结构
# （这里重新简化实现一个，证明同一套代码可以切换模型）

class UnifiedLLM:
    """
    统一 LLM 客户端 —— 证明同一套代码调用云端和本地。
    """
    CONFIGS = {
        "ollama": {"base_url": f"{OLLAMA_BASE}/v1", "api_key": "ollama", "model": MODEL},
        "deepseek": {"base_url": "https://api.deepseek.com", "api_key": os.getenv("DEEPSEEK_API_KEY", ""), "model": "deepseek-chat"},
    }

    def __init__(self, provider="ollama"):
        cfg = self.CONFIGS[provider]
        self.client = OpenAI(api_key=cfg["api_key"], base_url=cfg["base_url"])
        self.model = cfg["model"]
        self.provider = provider

    def ask(self, question: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": question}],
            max_tokens=128,
        )
        return resp.choices[0].message.content


# 用本地模型
print(f"\n  --- Ollama (local, free) ---")
local_llm = UnifiedLLM("ollama")
start = time.time()
answer = local_llm.ask("1+1=? Just give the number.")
print(f"  A: {answer.strip()}")
print(f"  Time: {time.time()-start:.1f}s")

# 如果有 DeepSeek key，也测一下
ds_key = os.getenv("DEEPSEEK_API_KEY")
if ds_key:
    print(f"\n  --- DeepSeek (cloud, paid) ---")
    cloud_llm = UnifiedLLM("deepseek")
    start = time.time()
    answer = cloud_llm.ask("1+1=? Just give the number.")
    print(f"  A: {answer.strip()}")
    print(f"  Time: {time.time()-start:.1f}s")
else:
    print(f"\n  [SKIP] DeepSeek (no API key)")

print()
print("  --> Same code, same interface, different providers!")
print("  --> Switch with one parameter: UnifiedLLM('ollama') vs UnifiedLLM('deepseek')")
print()


# ============================================================
# 第六部分：性能测试
# ============================================================

print("=" * 50)
print("Part 6: Speed Benchmark")
print("=" * 50)

print(f"\n  Model: {MODEL}")
print(f"  Testing generation speed...\n")

start = time.time()
response = requests.post(
    f"{OLLAMA_BASE}/api/generate",
    json={
        "model": MODEL,
        "prompt": "Write a Python function to calculate fibonacci numbers. Include comments in Chinese.",
        "stream": False,
    },
    timeout=120,
)
elapsed = time.time() - start
data = response.json()

output_text = data.get("response", "")
# Ollama 返回的 eval_count 是生成的 token 数
eval_count = data.get("eval_count", 0)
eval_duration_ns = data.get("eval_duration", 1)  # 纳秒

tokens_per_second = eval_count / (eval_duration_ns / 1e9) if eval_duration_ns > 0 else 0

print(f"  Output length: {len(output_text)} chars, {eval_count} tokens")
print(f"  Total time: {elapsed:.1f}s")
print(f"  Generation speed: {tokens_per_second:.1f} tokens/sec")
print()

if tokens_per_second > 20:
    print("  --> Great speed! GPU acceleration is likely active.")
elif tokens_per_second > 5:
    print("  --> Decent speed. Good enough for development.")
else:
    print("  --> Slow. Consider using a smaller model (qwen2.5:3b)")
    print("  --> Or check if GPU is being used: ollama ps")

print()


# ============================================================
# 总结
# ============================================================

print("=" * 50)
print("Day 8 Summary")
print("=" * 50)
print("""
1. Ollama = local LLM runner, free & unlimited
2. Two ways to call:
   - Native API: POST /api/generate or /api/chat
   - OpenAI compatible: POST /v1/chat/completions (RECOMMENDED)
3. OpenAI SDK works with Ollama by changing base_url
4. Day 7's LLMClient works with Ollama without code changes
5. Streaming works the same way (stream=True)
6. Dev strategy: Ollama for debugging, Cloud API for production

Tomorrow: Prompt Engineering - how to write prompts
that make LLMs give you exactly what you want.
""")
