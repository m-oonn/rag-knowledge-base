"""
Day 9 Demo 1：Prompt 工程基础
运行方式：python day9_1_prompt_basics.py

前置条件（任选一个）：
  - Ollama 运行中（免费，推荐调试用）
  - 或 .env 中配置 DEEPSEEK_API_KEY

学习目标：
1. 掌握 System Prompt 的写法
2. 掌握 Few-Shot（少样本学习）
3. 掌握 Chain of Thought（思维链）
4. 掌握结构化 JSON 输出
5. 预览 RAG / Agent 的 Prompt 模板
"""

import os
import json
import time
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ============================================================
# 初始化：自动选择可用的模型
# ============================================================

def get_client():
    """按优先级选择可用的 LLM：Ollama > DeepSeek > Moonshot"""
    # 1. 尝试 Ollama（本地免费）
    try:
        import requests
        resp = requests.get("http://localhost:11434/api/tags", timeout=2)
        models = [m["name"] for m in resp.json().get("models", [])]
        if models:
            model = models[0]
            client = OpenAI(api_key="ollama", base_url="http://localhost:11434/v1")
            print(f"  Using: Ollama ({model})")
            return client, model
    except Exception:
        pass

    # 2. 尝试 DeepSeek
    ds_key = os.getenv("DEEPSEEK_API_KEY")
    if ds_key:
        client = OpenAI(api_key=ds_key, base_url="https://api.deepseek.com")
        print("  Using: DeepSeek")
        return client, "deepseek-chat"

    # 3. 尝试 Moonshot
    ms_key = os.getenv("MOONSHOT_API_KEY")
    if ms_key:
        client = OpenAI(api_key=ms_key, base_url="https://api.moonshot.cn/v1")
        print("  Using: Moonshot")
        return client, "moonshot-v1-8k"

    print("  [FAIL] No LLM available!")
    print("  Please either:")
    print("    1. Start Ollama (ollama serve)")
    print("    2. Set DEEPSEEK_API_KEY in .env")
    exit(1)


def ask(messages: list, temperature: float = 0.3) -> str:
    """统一调用接口"""
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=1024,
    )
    return resp.choices[0].message.content


print("=" * 55)
print("  Day 9: Prompt Engineering")
print("=" * 55)
client, model = get_client()
print()


# ============================================================
# 第一部分：System Prompt 对比
# ============================================================

print("=" * 55)
print("Part 1: System Prompt Effect")
print("=" * 55)

question = "What is RAG?"

# 无 System Prompt
print("\n  --- No system prompt ---")
answer1 = ask([{"role": "user", "content": question}])
print(f"  A: {answer1.strip()[:150]}...")

# 有 System Prompt
print("\n  --- With system prompt ---")
system = (
    "You are a Python AI tutor for a junior university student. "
    "Answer in Chinese. Use simple analogies. Max 3 sentences."
)
answer2 = ask([
    {"role": "system", "content": system},
    {"role": "user", "content": question},
])
print(f"  A: {answer2.strip()[:200]}")

print("\n  --> System prompt controls style, length, language, and depth")
print()


# ============================================================
# 第二部分：Few-Shot（少样本学习）
# ============================================================

print("=" * 55)
print("Part 2: Few-Shot Learning")
print("=" * 55)

# 任务：把用户问题分类
few_shot_prompt = """Classify the user's question into one of these categories:
- tech_question (programming/technical)
- chitchat (casual conversation)
- help_request (asking for help with a task)

Examples:

Q: How to read a file in Python?
Category: tech_question

Q: What's the weather like today?
Category: chitchat

Q: Can you help me write my resume?
Category: help_request

Now classify:
Q: {question}
Category:"""

test_questions = [
    "How does FastAPI dependency injection work?",
    "Hi, how are you doing?",
    "Please help me debug this error in my code",
]

print()
for q in test_questions:
    prompt = few_shot_prompt.format(question=q)
    result = ask([{"role": "user", "content": prompt}], temperature=0)
    print(f"  Q: {q}")
    print(f"  -> {result.strip()}")
    print()

print("  --> Few-shot examples teach the model your desired format")
print()


# ============================================================
# 第三部分：Chain of Thought（思维链）
# ============================================================

print("=" * 55)
print("Part 3: Chain of Thought (CoT)")
print("=" * 55)

problem = "A store has 150 items. Monday sold 30%, Tuesday sold 25% of remaining. How many left?"

# 直接问
print("\n  --- Direct question ---")
answer_direct = ask([
    {"role": "user", "content": f"{problem} Give only the final number."}
], temperature=0)
print(f"  A: {answer_direct.strip()[:100]}")

# 加思维链
print("\n  --- With Chain of Thought ---")
answer_cot = ask([
    {"role": "system", "content": "Solve step by step. Show each calculation. Then give final answer."},
    {"role": "user", "content": problem},
], temperature=0)
print(f"  A: {answer_cot.strip()[:300]}")

print("\n  --> CoT reduces errors on reasoning tasks")
print()


# ============================================================
# 第四部分：结构化 JSON 输出（重点！）
# ============================================================

print("=" * 55)
print("Part 4: Structured JSON Output")
print("=" * 55)

json_prompt = """Analyze the user's data analysis request and output JSON.

Output ONLY valid JSON, no other text:
{{
  "intent": "trend_analysis / comparison / summary / chart",
  "time_range": "mentioned time range or null",
  "metrics": ["list of metrics mentioned"],
  "needs_chart": true or false,
  "chart_type": "line / bar / pie or null"
}}

User request: {request}"""

test_requests = [
    "Show me the monthly sales trend for the last 6 months",
    "Compare revenue between product A and product B",
    "Give me a summary of this quarter's performance",
]

print()
for req in test_requests:
    prompt = json_prompt.format(request=req)
    result = ask([{"role": "user", "content": prompt}], temperature=0)

    print(f"  Request: {req}")

    # 尝试解析 JSON
    try:
        # 清理：有些模型会加 ```json ... ```
        clean = result.strip()
        if clean.startswith("```"):
            clean = clean.split("\n", 1)[1]
            clean = clean.rsplit("```", 1)[0]
        parsed = json.loads(clean)
        print(f"  JSON:    {json.dumps(parsed, ensure_ascii=False)}")
        print(f"  Parsed OK!")
    except (json.JSONDecodeError, IndexError):
        print(f"  Raw: {result.strip()[:150]}")
        print(f"  (JSON parse failed - may need prompt tuning)")
    print()

print("  --> Key: give JSON template + say 'only JSON' + low temperature")
print()


# ============================================================
# 第五部分：RAG Prompt 模板实战
# ============================================================

print("=" * 55)
print("Part 5: RAG Prompt Template (Project 1 Preview)")
print("=" * 55)

# 模拟检索到的文档片段
mock_documents = """
[Document 1: FastAPI Introduction]
FastAPI is a modern, fast web framework for building APIs with Python 3.7+.
It is based on standard Python type hints and uses Pydantic for data validation.
FastAPI automatically generates interactive API documentation (Swagger UI).

[Document 2: FastAPI vs Flask]
FastAPI supports async/await natively, while Flask requires extensions.
FastAPI's performance is comparable to NodeJS and Go frameworks.
Flask has a larger ecosystem but FastAPI is growing rapidly.
"""

rag_prompt = """You are a knowledge base Q&A assistant.

## Reference Documents
---
{context}
---

## Rules
1. Answer ONLY based on the reference documents above
2. Cite sources in format: [Source: Document Name]
3. If the documents don't contain relevant info, say "Not found in knowledge base"
4. Answer in Chinese
5. Be concise

## User Question
{question}"""

test_qs = [
    "What is FastAPI?",
    "How does FastAPI compare to Flask in terms of performance?",
    "How to deploy FastAPI with Kubernetes?",  # 文档里没有的信息
]

print()
for q in test_qs:
    prompt = rag_prompt.format(context=mock_documents, question=q)
    answer = ask([{"role": "user", "content": prompt}], temperature=0.1)
    print(f"  Q: {q}")
    print(f"  A: {answer.strip()[:200]}")
    print()

print("  --> This is exactly how your RAG project will work!")
print("  --> Replace mock_documents with real vector DB search results")
print()


# ============================================================
# 第六部分：Agent Prompt 模板实战
# ============================================================

print("=" * 55)
print("Part 6: Agent Prompt Template (Project 2 Preview)")
print("=" * 55)

agent_prompt = """You are a data analysis Agent.

## Available Data
File: sales_2024.csv
Columns: date, product, quantity, revenue, region
Sample rows:
  2024-01-15, Product A, 100, 5000, East
  2024-01-15, Product B, 80, 4800, West
  2024-02-15, Product A, 120, 6000, East

## Task
Based on the user's request, generate a Python analysis plan.

Output JSON only:
{{
  "understanding": "what the user wants (1 sentence)",
  "steps": ["step1", "step2", ...],
  "code": "pandas/matplotlib code to execute",
  "output_type": "table / chart / both"
}}

## User Request
{request}"""

user_request = "Analyze the monthly revenue trend and show me a line chart"
prompt = agent_prompt.format(request=user_request)

print(f"\n  User: {user_request}")
print(f"  Agent thinking...\n")

result = ask([{"role": "user", "content": prompt}], temperature=0.1)
print(f"  {result.strip()[:500]}")

print()
print("  --> This is the core pattern of your Agent project!")
print("  --> Agent receives user intent, generates code, executes it")
print()


# ============================================================
# 总结
# ============================================================

print("=" * 55)
print("Day 9 Summary")
print("=" * 55)
print("""
Core Prompt Techniques:

1. System Prompt  - Define role, rules, output format
2. Few-Shot       - Give examples, model follows the pattern
3. Chain of Thought - Step-by-step reasoning, fewer errors
4. JSON Output    - Template + "only JSON" + low temperature
5. Separators     - ---doc--- to prevent confusion/injection
6. Guardrails     - "Don't make up info" + fallback responses

Project Applications:
  RAG:   context + question -> answer with citations
  Agent: user intent -> analysis plan -> code -> results

This is the foundation for everything you'll build next.
Stage 1 (AI basics) complete! Next: Stage 2 (RAG).
""")
