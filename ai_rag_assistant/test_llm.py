import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 1. 加载环境变量
load_dotenv()

# 2. 初始化大模型 (以DeepSeek为例)
llm = ChatOpenAI(
    model="moonshot-v1-8k",  # 或 "moonshot-v1-32k" 根据需求
    api_key=os.getenv("MOONSHOT_API_KEY"),
    base_url="https://api.moonshot.cn/v1",  # Moonshot 端点
    temperature=0.1,
    timeout=30,
    streaming=True
)

# 3. 测试调用
print("=== 测试1：简单调用 ===")
try:
    response = llm.invoke("请用中文，极其简洁地回答：什么是RAG？")
    print(f"结果：{response.content}\n")
except Exception as e:
    print(f"简单调用失败：{e}\n")

print("=== 测试2：流式调用 ===")
try:
    for chunk in llm.stream("请用一句话介绍Python："):
        print(chunk.content, end="", flush=True)
    print("\n流式调用成功。")
except Exception as e:
    print(f"流式调用失败：{e}")