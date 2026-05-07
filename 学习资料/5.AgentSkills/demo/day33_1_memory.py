"""
Day 33 Demo 1: 记忆机制 (Memory) 完整实战
运行方式: python day33_1_memory.py

前置条件:
  pip install langchain langchain-core langchain-openai
  pip install chromadb      (Part 5 向量记忆需要)

  LLM 后端（任选一个）:
  - Ollama 运行中（推荐 qwen2.5:7b）
  - 或 .env 中配置 DEEPSEEK_API_KEY
  - 或以上都没有 -> 自动进入模拟模式（仍能展示记忆数据结构）

学习目标:
1. 手动管理对话历史（最原始的记忆方式）
2. 使用 LangChain ConversationBufferMemory
3. 使用 ConversationBufferWindowMemory（滑动窗口）
4. 使用 ConversationSummaryMemory（摘要压缩）
5. 用向量数据库实现长期记忆
6. 实战：一个能记住用户偏好的聊天机器人
"""

import json
import os
from datetime import datetime

# ============================================================
# Part 0: 环境初始化
# ============================================================

print("=" * 60)
print("  Day 33: Memory - 记忆机制")
print("=" * 60)

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# 检测依赖
LANGCHAIN_AVAILABLE = False
try:
    from langchain.memory import (
        ConversationBufferMemory,
        ConversationBufferWindowMemory,
        ConversationSummaryMemory,
    )
    from langchain_core.messages import HumanMessage, AIMessage
    LANGCHAIN_AVAILABLE = True
    print("  [OK] langchain memory 模块已导入")
except ImportError:
    print("  [INFO] langchain 未安装，使用纯 Python 演示记忆原理")
    print("  (安装方法: pip install langchain langchain-core)")

# 检测 Chroma
CHROMA_AVAILABLE = False
try:
    import chromadb
    CHROMA_AVAILABLE = True
    print("  [OK] chromadb 已安装")
except ImportError:
    print("  [INFO] chromadb 未安装，Part 5 将使用简化实现")

# 初始化 LLM
USE_SIMULATION = False
client = None
model_name = None


def init_llm():
    """初始化 LLM（用于对话和摘要生成）"""
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


def chat_with_llm(messages: list, temperature: float = 0.3) -> str:
    """调用 LLM 进行对话"""
    if USE_SIMULATION:
        # 模拟 LLM 回复
        last_msg = messages[-1]["content"] if messages else ""
        # 根据上下文生成模拟回复
        if "名字" in last_msg or "叫什么" in last_msg:
            # 检查历史中是否有名字
            for msg in messages:
                if msg["role"] == "user" and "叫" in msg["content"]:
                    name = msg["content"].split("叫")[-1].split("，")[0].split(",")[0].strip()
                    return f"根据我们之前的对话，你叫{name}。"
            return "你还没告诉我你的名字呢。"
        elif "推荐" in last_msg:
            return "根据你的偏好，我推荐学习 FastAPI + LangChain 的组合。"
        elif "记得" in last_msg or "记住" in last_msg:
            return "我记得你之前告诉我的信息。作为AI助手，我会在对话中保持上下文。"
        else:
            return f"好的，我了解了。关于「{last_msg[:20]}」，你还有什么问题吗？"

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=512,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"(LLM 调用失败: {e})"


print()


# ============================================================
# Part 1: 手动管理对话历史（最原始的记忆方式）
# ============================================================

print("=" * 60)
print("  Part 1: 手动管理对话历史")
print("=" * 60)
print("  这是最简单的记忆方式：用一个 list 存所有 messages")
print()


class ManualMemoryChat:
    """
    手动管理对话记忆。
    就是一个 messages 列表 + 每次调用 LLM 时带上完整历史。
    """

    def __init__(self, system_prompt: str = "你是一个友好的中文AI助手。"):
        # 对话历史列表，第一条是系统提示
        self.messages = [
            {"role": "system", "content": system_prompt}
        ]

    def chat(self, user_input: str) -> str:
        """发送消息并获取回复"""
        # 1. 把用户消息加入历史
        self.messages.append({"role": "user", "content": user_input})

        # 2. 把完整历史发给 LLM
        response = chat_with_llm(self.messages)

        # 3. 把 LLM 回复也加入历史
        self.messages.append({"role": "assistant", "content": response})

        return response

    def get_history_length(self) -> int:
        """返回对话轮数（不含system消息）"""
        return (len(self.messages) - 1) // 2

    def show_history(self):
        """打印完整对话历史"""
        for msg in self.messages:
            role = msg["role"]
            content = msg["content"][:80]
            print(f"    [{role:10s}] {content}")


# 测试手动记忆
bot = ManualMemoryChat()

conversations = [
    "你好，我叫小明，我是大三学生",
    "我在学习 Python 和 AI 开发",
    "你还记得我叫什么名字吗？",
]

for user_msg in conversations:
    print(f"  User: {user_msg}")
    reply = bot.chat(user_msg)
    print(f"  Bot:  {reply}")
    print(f"  (历史轮数: {bot.get_history_length()}, messages 数: {len(bot.messages)})")
    print()

print("  完整对话历史:")
bot.show_history()
print()
print("  [OK] 手动记忆的问题: messages 越来越长，最终会超出 token 限制")
print()


# ============================================================
# Part 2: ConversationBufferMemory（LangChain 封装）
# ============================================================

print("=" * 60)
print("  Part 2: ConversationBufferMemory")
print("=" * 60)
print("  LangChain 封装的缓冲记忆：存储所有对话，等价于 Part 1 的手动方式")
print()

if LANGCHAIN_AVAILABLE:
    # 创建 Buffer Memory
    buffer_memory = ConversationBufferMemory(
        return_messages=True,       # 返回 Message 对象列表（而不是纯文本）
        memory_key="history"        # 在 Prompt 模板中用 {history} 引用
    )

    # 模拟几轮对话
    test_conversations = [
        ("我叫小明，是AI专业的学生", "你好小明！AI专业很有前景。"),
        ("我在做一个RAG知识库项目", "RAG项目很好！需要什么帮助？"),
        ("用的是 LangChain + ChromaDB", "不错的技术选型！"),
    ]

    for human_msg, ai_msg in test_conversations:
        buffer_memory.save_context(
            {"input": human_msg},
            {"output": ai_msg}
        )

    # 查看记忆内容
    memory_vars = buffer_memory.load_memory_variables({})
    print(f"  Buffer Memory 内容 (memory_key='history'):")
    for msg in memory_vars["history"]:
        role = "Human" if isinstance(msg, HumanMessage) else "AI"
        print(f"    [{role:6s}] {msg.content}")

    print(f"\n  总消息数: {len(memory_vars['history'])}")
    print(f"  [OK] Buffer Memory 存了所有 {len(test_conversations)} 轮对话")

else:
    print("  [模拟] ConversationBufferMemory 效果:")
    print("    存储所有对话历史，等价于手动管理 messages 列表")
    print("    buffer_memory.save_context({'input': '...'}, {'output': '...'})")
    print("    buffer_memory.load_memory_variables({}) -> 返回所有历史")

print()


# ============================================================
# Part 3: ConversationBufferWindowMemory（滑动窗口）
# ============================================================

print("=" * 60)
print("  Part 3: ConversationBufferWindowMemory (k=3)")
print("=" * 60)
print("  只保留最近 k 轮对话，更早的自动丢弃")
print()

if LANGCHAIN_AVAILABLE:
    # 创建窗口记忆，k=3 表示只保留最近3轮
    window_memory = ConversationBufferWindowMemory(
        k=3,
        return_messages=True,
        memory_key="history"
    )

    # 模拟5轮对话
    all_conversations = [
        ("Round 1: 我叫小明", "你好小明！"),
        ("Round 2: 我学AI专业", "AI很有前景！"),
        ("Round 3: 我在做RAG项目", "RAG项目很棒！"),
        ("Round 4: 用LangChain框架", "好的技术选型！"),
        ("Round 5: 遇到了Memory问题", "我来帮你解决！"),
    ]

    for i, (human_msg, ai_msg) in enumerate(all_conversations, 1):
        window_memory.save_context(
            {"input": human_msg},
            {"output": ai_msg}
        )

        # 每轮都查看当前记忆中有什么
        current = window_memory.load_memory_variables({})
        msgs = current["history"]
        kept_rounds = len(msgs) // 2
        print(f"  添加第 {i} 轮后，记忆中保留 {kept_rounds} 轮:")
        for msg in msgs:
            role = "H" if isinstance(msg, HumanMessage) else "A"
            print(f"    [{role}] {msg.content}")
        print()

    print("  [OK] 可以看到:")
    print("    - 添加第4轮后，Round 1 被丢弃了")
    print("    - 添加第5轮后，Round 2 也被丢弃了")
    print("    - 始终只保留最近3轮 (k=3)")
    print("    - 缺点: LLM 不再知道用户叫小明（Round 1 的信息丢了）")

else:
    print("  [模拟] WindowMemory(k=3) 效果:")
    print()
    conversations_sim = [
        "Round 1: 我叫小明",
        "Round 2: 我学AI",
        "Round 3: 做RAG项目",
        "Round 4: 用LangChain",
        "Round 5: Memory问题",
    ]
    # 模拟窗口记忆
    window = []  # 简单的列表模拟
    k = 3
    for i, msg in enumerate(conversations_sim, 1):
        window.append(msg)
        if len(window) > k:
            dropped = window.pop(0)
            print(f"  Round {i}: 添加 '{msg}', 丢弃 '{dropped}'")
        else:
            print(f"  Round {i}: 添加 '{msg}', 窗口未满")
        print(f"    当前窗口: {window}")
        print()
    print("  [OK] 窗口大小始终不超过 k=3")

print()


# ============================================================
# Part 4: ConversationSummaryMemory（摘要记忆）
# ============================================================

print("=" * 60)
print("  Part 4: ConversationSummaryMemory")
print("=" * 60)
print("  用 LLM 把旧对话压缩成摘要，大幅节省 token")
print()

if LANGCHAIN_AVAILABLE and not USE_SIMULATION:
    try:
        from langchain_openai import ChatOpenAI

        # 创建用于生成摘要的 LLM
        if client and "ollama" in str(getattr(client, '_base_url', '')):
            summary_llm = ChatOpenAI(
                api_key="ollama",
                base_url="http://localhost:11434/v1",
                model=model_name,
                temperature=0.1,
            )
        else:
            ds_key = os.getenv("DEEPSEEK_API_KEY", "")
            summary_llm = ChatOpenAI(
                api_key=ds_key,
                base_url="https://api.deepseek.com",
                model="deepseek-chat",
                temperature=0.1,
            )

        summary_memory = ConversationSummaryMemory(
            llm=summary_llm,
            return_messages=True,
            memory_key="history"
        )

        # 添加多轮对话
        summary_conversations = [
            ("我叫小明，大三AI专业学生，准备暑期实习", "你好小明！暑期实习加油！"),
            ("我在做RAG知识库项目，用LangChain和ChromaDB", "很好的技术栈选择！"),
            ("遇到了分块策略的问题，不知道chunk_size设多大", "建议500-1000字符，overlap 50-200"),
            ("我还想加一个数据分析Agent功能", "可以用LangChain的Agent + pandas工具"),
        ]

        for human_msg, ai_msg in summary_conversations:
            summary_memory.save_context(
                {"input": human_msg},
                {"output": ai_msg}
            )

        # 查看生成的摘要
        memory_vars = summary_memory.load_memory_variables({})
        print(f"  4轮对话后的摘要:")
        if memory_vars.get("history"):
            for msg in memory_vars["history"]:
                content = msg.content if hasattr(msg, 'content') else str(msg)
                print(f"    {content[:200]}")
        else:
            print(f"    {memory_vars}")

        # 获取内部摘要文本
        if hasattr(summary_memory, 'buffer'):
            print(f"\n  内部摘要 buffer:")
            print(f"    {summary_memory.buffer[:300]}")

        print(f"\n  [OK] 4轮对话被压缩成了一段摘要，大幅减少 token 消耗")

    except Exception as e:
        print(f"  [WARN] Summary Memory 创建失败: {e}")
        USE_SIMULATION = True

if USE_SIMULATION or not LANGCHAIN_AVAILABLE:
    # 手动演示摘要记忆的原理
    print("  [模拟] Summary Memory 原理演示:")
    print()

    original_conversations = [
        "User: 我叫小明，大三AI专业学生，准备暑期实习",
        "AI: 你好小明！暑期实习加油！",
        "User: 我在做RAG知识库项目，用LangChain和ChromaDB",
        "AI: 很好的技术栈选择！",
        "User: 遇到了chunk_size的问题",
        "AI: 建议500-1000字符，overlap 50-200",
        "User: 还想加数据分析Agent功能",
        "AI: 可以用LangChain Agent + pandas",
    ]

    total_chars = sum(len(c) for c in original_conversations)
    print(f"  原始对话 ({len(original_conversations)} 条消息, ~{total_chars} 字符):")
    for conv in original_conversations:
        print(f"    {conv}")

    # 模拟摘要
    summary = (
        "用户小明是大三AI专业学生，正在准备暑期实习。"
        "他在做RAG知识库项目，使用LangChain和ChromaDB。"
        "遇到了文本分块策略问题（建议chunk_size 500-1000, overlap 50-200）。"
        "还计划添加数据分析Agent功能（可用LangChain Agent + pandas）。"
    )
    print(f"\n  LLM 生成的摘要 (~{len(summary)} 字符):")
    print(f"    {summary}")
    print(f"\n  压缩率: {total_chars} -> {len(summary)} 字符 ({len(summary)/total_chars*100:.0f}%)")
    print(f"  [OK] 摘要保留了关键信息，但字符数减少了约 {100-len(summary)/total_chars*100:.0f}%")

print()


# ============================================================
# Part 5: 向量记忆（长期记忆）
# ============================================================

print("=" * 60)
print("  Part 5: Vector Memory - 向量记忆（长期记忆）")
print("=" * 60)
print("  用向量数据库存储对话历史，按相似度检索相关记忆")
print()

if CHROMA_AVAILABLE:
    # 使用 ChromaDB 实现长期记忆
    chroma_client = chromadb.Client()  # 内存模式，不持久化

    # 创建记忆集合
    memory_collection = chroma_client.get_or_create_collection(
        name="chat_memory",
        metadata={"description": "聊天记忆存储"}
    )

    # 存储一些历史对话（模拟用户的长期使用历史）
    historical_conversations = [
        {"id": "mem_001", "text": "用户名叫小明，是大三AI专业学生", "metadata": {"type": "user_info"}},
        {"id": "mem_002", "text": "用户喜欢用Python编程，最擅长FastAPI", "metadata": {"type": "preference"}},
        {"id": "mem_003", "text": "用户正在做RAG知识库项目，使用LangChain和ChromaDB", "metadata": {"type": "project"}},
        {"id": "mem_004", "text": "用户遇到文本分块问题，建议chunk_size为500-1000", "metadata": {"type": "problem"}},
        {"id": "mem_005", "text": "用户计划添加数据分析Agent功能", "metadata": {"type": "plan"}},
        {"id": "mem_006", "text": "用户偏好用Ollama在本地跑模型，不喜欢付费API", "metadata": {"type": "preference"}},
        {"id": "mem_007", "text": "用户的目标是找一份AI相关的暑期实习", "metadata": {"type": "goal"}},
        {"id": "mem_008", "text": "用户学过Python基础、FastAPI、异步编程、Prompt工程", "metadata": {"type": "skill"}},
    ]

    # 批量添加
    memory_collection.add(
        ids=[h["id"] for h in historical_conversations],
        documents=[h["text"] for h in historical_conversations],
        metadatas=[h["metadata"] for h in historical_conversations],
    )

    print(f"  已存储 {len(historical_conversations)} 条长期记忆")
    print()

    # 测试按相似度检索记忆
    test_queries = [
        "用户叫什么名字？",
        "用户在做什么项目？",
        "用户擅长什么技术？",
        "用户有什么目标？",
    ]

    for query in test_queries:
        results = memory_collection.query(
            query_texts=[query],
            n_results=2,    # 返回最相关的2条
        )
        print(f"  Query: '{query}'")
        for i, (doc, dist) in enumerate(zip(results["documents"][0], results["distances"][0])):
            # Chroma 返回的是距离（越小越相似）
            similarity = 1 / (1 + dist)  # 转为相似度
            print(f"    [{i+1}] (dist={dist:.3f}) {doc}")
        print()

    print("  [OK] 向量记忆可以精确地检索到与问题最相关的历史信息")
    print("  这就是'长期记忆'的实现: 存储到向量库, 需要时按语义检索")

else:
    print("  [模拟] 向量记忆原理:")
    print()
    print("  存储阶段:")
    print("    '用户叫小明' -> Embedding -> [0.1, 0.3, ...] -> 存入 ChromaDB")
    print("    '用户做RAG项目' -> Embedding -> [0.4, 0.2, ...] -> 存入 ChromaDB")
    print()
    print("  检索阶段:")
    print("    '用户叫什么?' -> Embedding -> [0.1, 0.3, ...]")
    print("    -> 和 '用户叫小明' 的向量最接近 -> 返回这条记忆")
    print()
    print("  安装 chromadb 体验完整效果: pip install chromadb")

print()


# ============================================================
# Part 6: 实战 - 带记忆的聊天机器人
# ============================================================

print("=" * 60)
print("  Part 6: 实战 - 带记忆的智能聊天机器人")
print("=" * 60)
print("  组合短期记忆(Window) + 长期记忆(Vector) 的完整聊天机器人")
print()


class SmartChatBot:
    """
    智能聊天机器人，组合了两种记忆:
    1. 短期记忆: 最近 k 轮完整对话（Window Memory）
    2. 长期记忆: 存入向量库的关键信息（Vector Memory）

    每次对话的 Prompt 结构:
      [系统提示] + [相关长期记忆] + [最近k轮对话] + [用户新消息]
    """

    def __init__(self, window_size: int = 3, system_prompt: str = None):
        # 短期记忆: 最近 k 轮对话
        self.window_size = window_size
        self.short_term: list[dict] = []  # [(user_msg, ai_msg), ...]

        # 长期记忆: 用列表模拟（有 Chroma 时用向量库）
        self.long_term: list[str] = []
        self.long_term_collection = None

        if CHROMA_AVAILABLE:
            chroma_client_local = chromadb.Client()
            self.long_term_collection = chroma_client_local.get_or_create_collection("bot_memory")

        # 系统提示
        self.system_prompt = system_prompt or (
            "你是一个友好的中文AI助手。你会记住用户告诉你的信息。"
            "如果用户告诉你他的名字、偏好、项目等信息，请记住并在后续对话中使用。"
        )

        # 统计
        self.total_rounds = 0

    def _save_to_long_term(self, info: str):
        """保存关键信息到长期记忆"""
        if self.long_term_collection:
            mem_id = f"ltm_{len(self.long_term) + 1:04d}"
            self.long_term_collection.add(
                ids=[mem_id],
                documents=[info],
            )
        self.long_term.append(info)

    def _search_long_term(self, query: str, top_k: int = 3) -> list[str]:
        """从长期记忆中检索相关信息"""
        if self.long_term_collection and self.long_term:
            try:
                results = self.long_term_collection.query(
                    query_texts=[query],
                    n_results=min(top_k, len(self.long_term)),
                )
                return results["documents"][0] if results["documents"] else []
            except Exception:
                pass
        # 回退: 返回所有长期记忆
        return self.long_term[-top_k:]

    def _extract_key_info(self, user_msg: str, ai_msg: str):
        """
        从对话中提取关键信息存入长期记忆。
        真实项目中可以用 LLM 来做这个提取，这里用规则简化。
        """
        # 简单的关键词提取规则
        key_phrases = ["叫", "名字", "专业", "学", "做", "喜欢", "不喜欢",
                       "目标", "项目", "使用", "偏好", "擅长"]
        for phrase in key_phrases:
            if phrase in user_msg:
                self._save_to_long_term(f"用户说: {user_msg}")
                return  # 只保存一次

    def chat(self, user_input: str) -> str:
        """发送消息并获取回复"""
        self.total_rounds += 1

        # 1. 从长期记忆中检索相关信息
        relevant_memories = self._search_long_term(user_input)

        # 2. 构建 messages
        messages = [{"role": "system", "content": self.system_prompt}]

        # 2a. 添加长期记忆上下文
        if relevant_memories:
            memory_context = "\n".join(f"- {m}" for m in relevant_memories)
            messages.append({
                "role": "system",
                "content": f"以下是关于用户的已知信息（长期记忆）:\n{memory_context}"
            })

        # 2b. 添加短期记忆（最近 k 轮）
        for user_msg, ai_msg in self.short_term[-self.window_size:]:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": ai_msg})

        # 2c. 添加当前用户消息
        messages.append({"role": "user", "content": user_input})

        # 3. 调用 LLM
        response = chat_with_llm(messages)

        # 4. 更新短期记忆
        self.short_term.append((user_input, response))

        # 5. 提取关键信息到长期记忆
        self._extract_key_info(user_input, response)

        return response

    def show_memory_status(self):
        """显示当前记忆状态"""
        print(f"    [记忆状态]")
        print(f"    短期记忆: {len(self.short_term)} 轮 (窗口: 最近 {self.window_size} 轮)")
        print(f"    长期记忆: {len(self.long_term)} 条")
        if self.long_term:
            for i, mem in enumerate(self.long_term[-3:], 1):
                print(f"      最近{i}: {mem[:60]}")
        print(f"    总对话轮数: {self.total_rounds}")


# 演示多轮对话
print("  --- 开始多轮对话演示 ---\n")
bot = SmartChatBot(window_size=3)

demo_conversations = [
    "你好！我叫小明，是一个大三学生",
    "我的专业是人工智能，正在学习 Python",
    "我在做一个 RAG 知识库项目",
    "我喜欢用本地模型，不喜欢付费API",
    "你还记得我叫什么名字吗？",
    "我之前说我在做什么项目来着？",
]

for i, user_msg in enumerate(demo_conversations, 1):
    print(f"  [{i}/{len(demo_conversations)}] User: {user_msg}")
    reply = bot.chat(user_msg)
    print(f"      Bot: {reply}")
    bot.show_memory_status()
    print()

print("  --- 多轮对话演示结束 ---\n")
print("  [OK] 机器人能够:")
print("    1. 通过短期记忆维持最近几轮的对话连贯性")
print("    2. 通过长期记忆记住用户的关键信息（名字、项目等）")
print("    3. 即使短期窗口滑过去了，长期记忆仍然保留")
print()


# ============================================================
# Part 7: 记忆策略对比总结
# ============================================================

print("=" * 60)
print("  Part 7: 记忆策略对比总结")
print("=" * 60)

print("""
  ┌────────────────────────────────────────────────────────────┐
  │                     记忆策略对比                             │
  ├──────────────┬──────────────┬──────────────┬───────────────┤
  │              │   Buffer     │   Window     │   Summary     │
  ├──────────────┼──────────────┼──────────────┼───────────────┤
  │ 存储内容      │ 所有对话     │ 最近 k 轮    │ 旧对话的摘要   │
  │ Token 消耗   │ 线性增长     │ 固定 (k轮)   │ 固定 (摘要)    │
  │ 信息完整性    │ 完整         │ 丢失旧对话    │ 摘要可能丢细节  │
  │ 额外 LLM 调用│ 无           │ 无           │ 每轮需摘要调用  │
  │ 适合场景      │ 短对话       │ 中等对话     │ 长对话          │
  └──────────────┴──────────────┴──────────────┴───────────────┘

  ┌────────────────────────────────────────────────────────────┐
  │  Vector Memory (向量记忆)                                   │
  ├──────────────┬─────────────────────────────────────────────┤
  │ 存储          │ 向量数据库 (Chroma/FAISS/Pinecone)          │
  │ 检索          │ 按语义相似度检索最相关的记忆                    │
  │ 生命周期      │ 永久（可持久化）                              │
  │ 适合          │ 跨对话记忆、用户画像、长期偏好                  │
  │ 和 RAG 关系   │ 本质相同！只是"文档"换成了"对话历史"            │
  └──────────────┴─────────────────────────────────────────────┘

  推荐组合: Window(k=5) + Vector Memory
    短期: 最近5轮完整对话（保持上下文连贯）
    长期: 关键信息存向量库（跨对话记忆）

  明天 (Day 34): 规划机制 - Agent 怎么把复杂任务分解为步骤
""")

print("=" * 60)
print("  Day 33 完成!")
print("=" * 60)
