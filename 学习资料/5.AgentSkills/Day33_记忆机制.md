# Day 33：记忆机制 (Memory)

> LLM 天生健忘——每次调用都是独立的。今天学怎么给 Agent 加上记忆。

---

## 一、为什么 LLM 健忘

LLM 的每次 API 调用都是**无状态**的——模型不保存任何对话历史。

```
第1轮: 用户"我叫小明，是大三学生" → LLM"你好小明！"
第2轮: 用户"我学什么专业来着？"  → LLM"不知道。" ← 完全不记得！
```

**解决方案：把历史对话塞进下一次的 messages 里。**

```
第2轮的 messages = [
    上一轮的 user+assistant 对话,    ← 历史上下文
    新的用户消息                      ← 当前问题
]
→ LLM 看到"我叫小明，大三" → 能正确回答
```

**记忆的本质 = 每次调用时把相关历史拼进 messages。不是 LLM 真记住了，是你每次都在提醒它。**

---

## 二、五种记忆策略

| 类型 | 原理 | 优点 | 缺点 | 场景 |
|------|------|------|------|------|
| Buffer | 存全部对话 | 信息完整 | token 爆炸 | 短对话(<10轮) |
| Window | 只存最近 N 轮 | 控制 token | 丢失早期信息 | 客服问答 |
| Summary | LLM 摘要旧对话 | 省 token 又保要点 | 需额外 LLM 调用 | 长对话 |
| Token Buffer | 按 token 数截断 | 精确控制长度 | 可能切断对话 | token 敏感 |
| Vector | 向量库按需检索 | 海量跨对话 | 检索延迟 | 长期记忆 |

---

## 三、手写记忆实现

### Buffer Memory（最原始）

```python
# 手动管理对话历史
messages = []

while True:
    user_input = input("你: ")

    # 新消息追加到历史
    messages.append({"role": "user", "content": user_input})

    # 全部历史发给 LLM
    response = call_llm(messages)

    # LLM 回复也追加
    messages.append({"role": "assistant", "content": response})
    print(f"AI: {response}")
```

问题：对话 20 轮后 messages 几千 token，LLM 上下文窗口撑不住。

### Window Memory（滑动窗口）

```python
messages = []
MAX_HISTORY = 6  # 只保留最近 6 条消息

while True:
    user_input = input("你: ")
    messages.append({"role": "user", "content": user_input})

    # 只取最近 N 条
    recent = messages[-MAX_HISTORY:]

    response = call_llm(recent)
    messages.append({"role": "assistant", "content": response})
    print(f"AI: {response}")
```

问题：用户第 1 轮说"我叫小明"，第 15 轮问"我叫什么"，消息已经被窗口丢掉了。

### Summary Memory（摘要压缩）

```python
# 定期总结旧对话
def compress_history(messages):
    """用 LLM 把历史对话压缩成一段摘要。"""
    if len(messages) <= 10:
        return messages  # 对话短，不压缩

    old = messages[:-8]           # 前一半压缩
    recent = messages[-8:]         # 后一半保留原文

    summary_prompt = "将以下对话总结为一段简短摘要：\n" + \
        "\n".join(f"{m['role']}: {m['content']}" for m in old)

    summary = call_llm(summary_prompt)  # 用 LLM 生成摘要

    # 返回：[摘要] + [最近对话原文]
    return [{"role": "system", "content": f"之前的对话摘要：{summary}"}] + recent
```

---

## 四、LangChain 三种 Memory 实现

### ConversationBufferMemory

```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()

memory.save_context(
    {"input": "我叫小明"},
    {"output": "你好小明！"}
)
memory.save_context(
    {"input": "我在学 Python"},
    {"output": "Python 是一门好语言！"}
)

# 获取记忆变量
vars = memory.load_memory_variables({})
# {'history': 'Human: 我叫小明\nAI: 你好小明！\nHuman: 我在学 Python\nAI: Python...'}

print(vars["history"])
```

### ConversationBufferWindowMemory

```python
from langchain.memory import ConversationBufferWindowMemory

memory = ConversationBufferWindowMemory(k=2)  # 只记最近 2 轮

memory.save_context({"input": "我叫小明"}, {"output": "你好"})
memory.save_context({"input": "我学 Python"}, {"output": "Python 不错"})
memory.save_context({"input": "推荐书"}, {"output": "《Python编程》"})

vars = memory.load_memory_variables({})
# 只保留最近 2 轮：最后两条输入/输出，最早那条被丢弃
```

### ConversationSummaryMemory

```python
from langchain.memory import ConversationSummaryMemory
from langchain_openai import ChatOpenAI

memory = ConversationSummaryMemory(llm=ChatOpenAI(model="gpt-4"))

memory.save_context(
    {"input": "你好，我叫小明，我在北京上大学，专业是计算机"},
    {"output": "你好小明！计算机专业很好啊"}
)
memory.save_context(
    {"input": "我对 AI 很感兴趣"},
    {"output": "AI 是计算机最热门的方向之一"}
)

# 加载记忆 → LLM 自动生成的摘要
vars = memory.load_memory_variables({})
# {'history': '小明是北京某大学计算机专业学生，对AI感兴趣。'}
```

---

## 五、长期记忆：向量库存储

跨对话的记忆需要持久化存储。

```python
import json
import chromadb

client = chromadb.PersistentClient(path="./memory_db")
collection = client.get_or_create_collection("user_memory")

# ── 存入记忆 ──
def save_memory(user_id: str, content: str, embedding: list):
    collection.add(
        ids=[f"{user_id}_{len(collection.get()['ids'])}"],
        documents=[content],
        embeddings=[embedding],
        metadatas=[{"user_id": user_id, "timestamp": str(datetime.now())}]
    )

# ── 检索记忆 ──
def recall_memory(user_id: str, query_embedding: list, top_k: int = 5):
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"user_id": user_id}
    )
    return results["documents"][0] if results["documents"] else []

# ── 使用时：把长期记忆注入 Prompt ──
long_term = recall_memory("user_123", embed_query("用户偏好"), top_k=3)
if long_term:
    messages.insert(0, {
        "role": "system",
        "content": f"关于该用户的历史信息：\n" + "\n".join(f"- {m}" for m in long_term)
    })
```

---

## 六、在我的 RAG 项目中的应用

`src/services/memory_service.py` 做了双模式记忆：

```python
class MemorySystem:
    def __init__(self):
        # 短期记忆：优先 Redis，降级到内存字典
        self._redis = None
        try:
            import redis
            self._redis = redis.from_url(settings.REDIS_URL)
        except Exception:
            self._local_store = {}  # 降级：普通 Python 字典

    def get_short_term_memory(self, session_id: str, limit: int = 10):
        """获取这个会话的历史消息。"""
        if self._redis:
            data = self._redis.lrange(f"session:{session_id}:history", 0, limit-1)
            messages = [json.loads(d) for d in data]
            messages.reverse()
            return messages
        # 降级：从内存字典取
        return self._local_store.get(session_id, [])[-limit:]

    def add_short_term_memory(self, session_id: str, role: str, content: str):
        """记录一条消息。"""
        msg = {"role": role, "content": content}
        if self._redis:
            self._redis.lpush(f"session:{session_id}:history", json.dumps(msg))
        else:
            self._local_store.setdefault(session_id, []).append(msg)
```

---

## 七、面试速记

**Q1：LLM 为什么需要记忆？**
LLM 每次调用都是无状态的，不记得之前的对话。需要程序手动管理历史并拼进 messages。

**Q2：五种记忆策略和各自适用场景？**
Buffer（短对话）、Window（客服）、Summary（长对话）、Token Buffer（精确控制）、Vector（长期跨对话）。

**Q3：Buffer vs Window 区别？**
Buffer 存全部（完整但不控制 token），Window 只存最近 N 轮（省 token 但丢失早期信息）。

**Q4：长期记忆怎么实现？**
向量库（Chroma/Milvus）存历史信息，查询时按用户 ID + 语义检索相关记忆，拼进 Prompt。

---

## 八、验收清单

- [ ] 能手写 Buffer Memory 的消息管理循环
- [ ] 能解释 Window Memory 的 k 参数含义
- [ ] 能解释 Summary Memory 为什么要额外调 LLM
- [ ] 能说出五种策略各适合什么场景
- [ ] 知道项目中 memory_service.py 的双模式设计
- [ ] 4 道面试速记全部能讲 1 分钟
