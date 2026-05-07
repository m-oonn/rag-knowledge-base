"""
Day 14 Demo 1: 文档解析与分割实战
运行方式: python day14_1_doc_parsing.py

前置条件:
  pip install langchain langchain-text-splitters

学习目标:
1. 掌握纯文本和 Markdown 文件的解析方法
2. 理解 RecursiveCharacterTextSplitter 的工作原理和参数调节
3. 对比不同 chunk_size 对分割结果的影响
4. 学会给 chunk 添加元数据（source、chunk_index、page）
5. 走通完整管线：读取文件 -> 解析 -> 分割 -> 添加元数据 -> 准备 Embedding
"""

import os
import tempfile

# ============================================================
# === Part 1 === 创建样例文档（供后续解析和分割使用）
# ============================================================

print("=" * 60)
print("Part 1: 创建样例文档")
print("=" * 60)

# 创建临时目录存放样例文档
SAMPLE_DIR = os.path.join(tempfile.gettempdir(), "day14_sample_docs")
os.makedirs(SAMPLE_DIR, exist_ok=True)

# --- 样例 1：纯文本文件 ---
txt_content = """Python 是一种广泛使用的高级编程语言，由 Guido van Rossum 在 1989 年底发明。Python 的设计哲学强调代码的可读性和简洁性。

Python 支持多种编程范式，包括面向对象编程、函数式编程和过程式编程。它拥有丰富的标准库和第三方库生态系统，在 Web 开发、数据科学、人工智能等领域被广泛应用。

在 Web 开发方面，Python 有 Django 和 Flask 两大框架。Django 是全功能框架，内置 ORM、模板引擎和管理后台。Flask 是微框架，轻量灵活，适合小型项目和 API 开发。近年来 FastAPI 凭借其高性能和类型提示支持迅速崛起。

在人工智能领域，Python 是事实上的标准语言。TensorFlow、PyTorch、scikit-learn 等主流框架都以 Python 为主要接口。LangChain 框架让开发者可以方便地构建基于大语言模型的应用。

Python 的包管理工具 pip 让安装第三方库变得非常简单。虚拟环境工具 venv 可以为每个项目创建隔离的 Python 环境，避免依赖冲突。conda 则提供了更强大的环境管理能力，特别适合数据科学项目。

Python 3.10 引入了结构化模式匹配（match-case），3.12 进一步优化了性能。类型提示（Type Hints）从 3.5 版本开始支持，配合 mypy 等工具可以在开发阶段发现类型错误。
"""

txt_path = os.path.join(SAMPLE_DIR, "python_intro.txt")
with open(txt_path, "w", encoding="utf-8") as f:
    f.write(txt_content)
print(f"\n  [OK] 创建纯文本文件: {txt_path}")
print(f"       内容长度: {len(txt_content)} 字符")

# --- 样例 2：Markdown 文件 ---
md_content = """# FastAPI 开发指南

FastAPI 是一个现代、高性能的 Python Web 框架，基于 Starlette 和 Pydantic 构建。

## 安装与快速开始

使用 pip 安装 FastAPI 和 ASGI 服务器 uvicorn：

```bash
pip install fastapi uvicorn
```

创建 main.py 文件，编写第一个 API：

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello World"}
```

启动命令：`uvicorn main:app --reload`

## 路由与请求处理

FastAPI 使用装饰器定义路由。支持所有 HTTP 方法：GET、POST、PUT、DELETE 等。路径参数通过花括号定义，查询参数通过函数参数定义。

### 路径参数

路径参数用于标识特定资源。FastAPI 会自动进行类型转换和验证：

```python
@app.get("/users/{user_id}")
def get_user(user_id: int):
    return {"user_id": user_id}
```

### 查询参数

查询参数是 URL 中 ? 后面的键值对。设置默认值可以让参数变为可选：

```python
@app.get("/items")
def list_items(skip: int = 0, limit: int = 10):
    return {"skip": skip, "limit": limit}
```

## 数据验证与序列化

Pydantic 模型是 FastAPI 的核心。定义数据模型后，FastAPI 自动完成请求验证、数据转换和 JSON 序列化。验证失败时返回详细的 422 错误。

```python
from pydantic import BaseModel

class Item(BaseModel):
    name: str
    price: float
    description: str | None = None
```

## 依赖注入

依赖注入是 FastAPI 最强大的特性之一。通过 Depends 声明依赖关系，FastAPI 自动解析依赖树并注入所需对象。常用于数据库连接管理、用户认证、权限校验等场景。

## 异步支持

FastAPI 原生支持 async/await 异步编程。异步路由函数可以高效处理 IO 密集型请求，如数据库查询、外部 API 调用等。性能可媲美 Node.js 和 Go。

## 自动文档

FastAPI 自动生成两种交互式 API 文档：Swagger UI（/docs）和 ReDoc（/redoc）。无需额外配置，开发者可以直接在浏览器中测试 API。
"""

md_path = os.path.join(SAMPLE_DIR, "fastapi_guide.md")
with open(md_path, "w", encoding="utf-8") as f:
    f.write(md_content)
print(f"  [OK] 创建 Markdown 文件: {md_path}")
print(f"       内容长度: {len(md_content)} 字符")

print(f"\n  样例文档目录: {SAMPLE_DIR}")


# ============================================================
# === Part 2 === 解析纯文本和 Markdown 文件
# ============================================================

print("\n" + "=" * 60)
print("Part 2: 解析纯文本和 Markdown 文件")
print("=" * 60)

# --- 2.1 读取纯文本 ---
print("\n--- 2.1 解析纯文本 (.txt) ---")
with open(txt_path, "r", encoding="utf-8") as f:
    parsed_txt = f.read()

# 基本统计
lines = parsed_txt.strip().split("\n")
paragraphs = [p.strip() for p in parsed_txt.split("\n\n") if p.strip()]
print(f"  文件: python_intro.txt")
print(f"  总字符数: {len(parsed_txt)}")
print(f"  总行数: {len(lines)}")
print(f"  段落数: {len(paragraphs)}")
print(f"  前100字符: {parsed_txt[:100]}...")

# --- 2.2 读取并解析 Markdown ---
print("\n--- 2.2 解析 Markdown (.md) ---")
with open(md_path, "r", encoding="utf-8") as f:
    parsed_md = f.read()

# 提取 Markdown 结构信息
md_lines = parsed_md.split("\n")
headings = [line for line in md_lines if line.startswith("#")]
code_block_count = parsed_md.count("```") // 2  # 每个代码块有开头和结尾

print(f"  文件: fastapi_guide.md")
print(f"  总字符数: {len(parsed_md)}")
print(f"  标题数量: {len(headings)}")
for h in headings:
    level = len(h.split(" ")[0])  # 几个 # 号
    indent = "  " * (level - 1)
    print(f"    {indent}{h}")
print(f"  代码块数量: {code_block_count}")

# --- 2.3 简单的 Markdown 清洗 ---
print("\n--- 2.3 Markdown 文本清洗 ---")

def clean_markdown(text: str) -> str:
    """
    清洗 Markdown 文本，去掉格式标记，保留纯文本内容。
    实际项目中也常用 unstructured 库来做这件事。
    """
    import re
    cleaned = text
    # 去掉代码块（保留代码块内的内容描述会更好，这里简单演示）
    cleaned = re.sub(r"```[\s\S]*?```", "[代码块]", cleaned)
    # 去掉标题的 # 号
    cleaned = re.sub(r"^#{1,6}\s+", "", cleaned, flags=re.MULTILINE)
    # 去掉加粗/斜体标记
    cleaned = re.sub(r"\*{1,2}(.*?)\*{1,2}", r"\1", cleaned)
    # 去掉行内代码标记
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    return cleaned.strip()

cleaned = clean_markdown(parsed_md)
print(f"  清洗前长度: {len(parsed_md)} 字符")
print(f"  清洗后长度: {len(cleaned)} 字符")
print(f"  清洗后前200字符:\n    {cleaned[:200]}...")


# ============================================================
# === Part 3 === RecursiveCharacterTextSplitter 参数实验
# ============================================================

print("\n" + "=" * 60)
print("Part 3: RecursiveCharacterTextSplitter 参数实验")
print("=" * 60)

# 先尝试导入 LangChain，如果没有就用自己实现的版本
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    print("\n  [OK] 使用 langchain_text_splitters")
    USE_LANGCHAIN = True
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        print("\n  [OK] 使用 langchain.text_splitter")
        USE_LANGCHAIN = True
    except ImportError:
        print("\n  [INFO] LangChain 未安装，使用自定义实现")
        USE_LANGCHAIN = False

        # 自己实现一个简化版的 RecursiveCharacterTextSplitter
        class RecursiveCharacterTextSplitter:
            """
            简化版递归字符文本分割器。
            原理与 LangChain 版本相同：
            1. 按优先级尝试不同分隔符
            2. 如果分割后的段落仍然太长，用下一级分隔符继续分割
            3. 添加 overlap 保证上下文连续性
            """
            def __init__(self, chunk_size=500, chunk_overlap=100,
                         separators=None, length_function=len):
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap
                self.separators = separators or ["\n\n", "\n", "。", ".", " ", ""]
                self.length_function = length_function

            def split_text(self, text: str) -> list:
                """递归分割文本"""
                return self._split(text, self.separators)

            def _split(self, text: str, separators: list) -> list:
                """用当前分隔符分割，太长的部分用下一级分隔符继续分"""
                if self.length_function(text) <= self.chunk_size:
                    return [text.strip()] if text.strip() else []

                # 找到当前能用的分隔符
                separator = separators[0] if separators else ""
                remaining_separators = separators[1:] if len(separators) > 1 else [""]

                if separator == "":
                    # 最后手段：按字符切割
                    chunks = []
                    start = 0
                    while start < len(text):
                        end = start + self.chunk_size
                        chunk = text[start:end].strip()
                        if chunk:
                            chunks.append(chunk)
                        start = end - self.chunk_overlap
                    return chunks

                # 用分隔符切割
                parts = text.split(separator)
                chunks = []
                current = ""

                for part in parts:
                    test_chunk = current + separator + part if current else part
                    if self.length_function(test_chunk) <= self.chunk_size:
                        current = test_chunk
                    else:
                        if current.strip():
                            chunks.append(current.strip())
                        # 如果单个 part 就超过 chunk_size，递归用更细的分隔符
                        if self.length_function(part) > self.chunk_size:
                            sub_chunks = self._split(part, remaining_separators)
                            chunks.extend(sub_chunks)
                            current = ""
                        else:
                            current = part

                if current.strip():
                    chunks.append(current.strip())

                # 添加 overlap（简化实现）
                if self.chunk_overlap > 0 and len(chunks) > 1:
                    overlapped = [chunks[0]]
                    for i in range(1, len(chunks)):
                        prev = chunks[i - 1]
                        overlap_text = prev[-self.chunk_overlap:] if len(prev) > self.chunk_overlap else prev
                        overlapped.append(overlap_text + " " + chunks[i])
                    # 注意：简化版 overlap 可能导致 chunk 稍微超过 chunk_size
                    chunks = overlapped

                return chunks

# --- 3.1 默认参数分割 ---
print("\n--- 3.1 默认参数: chunk_size=500, overlap=100 ---")
splitter_default = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)
chunks_default = splitter_default.split_text(parsed_md)

print(f"  原文长度: {len(parsed_md)} 字符")
print(f"  分割结果: {len(chunks_default)} 个 chunk")
for i, chunk in enumerate(chunks_default):
    # 显示前50字符作为预览
    preview = chunk[:50].replace("\n", " ")
    print(f"    Chunk {i+1}: [{len(chunk):4d} 字符] {preview}...")

# --- 3.2 小 chunk ---
print("\n--- 3.2 小 chunk: chunk_size=200, overlap=50 ---")
splitter_small = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=50,
)
chunks_small = splitter_small.split_text(parsed_md)

print(f"  分割结果: {len(chunks_small)} 个 chunk")
for i, chunk in enumerate(chunks_small[:5]):  # 只展示前5个
    preview = chunk[:50].replace("\n", " ")
    print(f"    Chunk {i+1}: [{len(chunk):4d} 字符] {preview}...")
if len(chunks_small) > 5:
    print(f"    ... 还有 {len(chunks_small) - 5} 个 chunk")

# --- 3.3 大 chunk ---
print("\n--- 3.3 大 chunk: chunk_size=1000, overlap=200 ---")
splitter_large = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks_large = splitter_large.split_text(parsed_md)

print(f"  分割结果: {len(chunks_large)} 个 chunk")
for i, chunk in enumerate(chunks_large):
    preview = chunk[:50].replace("\n", " ")
    print(f"    Chunk {i+1}: [{len(chunk):4d} 字符] {preview}...")

# --- 3.4 中文优化分隔符 ---
print("\n--- 3.4 中文优化分隔符 ---")
splitter_cn = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n\n", "\n", "。", "；", "，", " ", ""],  # 中文标点优先
)
chunks_cn = splitter_cn.split_text(txt_content)

print(f"  使用中文分隔符分割纯文本:")
print(f"  分割结果: {len(chunks_cn)} 个 chunk")
for i, chunk in enumerate(chunks_cn):
    preview = chunk[:60].replace("\n", " ")
    print(f"    Chunk {i+1}: [{len(chunk):4d} 字符] {preview}...")


# ============================================================
# === Part 4 === 对比实验：小 chunk vs 大 chunk 的检索影响
# ============================================================

print("\n" + "=" * 60)
print("Part 4: 小 chunk vs 大 chunk 对比")
print("=" * 60)

print(f"""
  分割对比结果:
  +------------------+--------+----------+-------------------+
  | 参数             | chunk数 | 平均长度 | 适用场景          |
  +------------------+--------+----------+-------------------+
  | size=200, ol=50  | {len(chunks_small):6d} | {sum(len(c) for c in chunks_small)//max(len(chunks_small),1):8d} | 精确检索、FAQ     |
  | size=500, ol=100 | {len(chunks_default):6d} | {sum(len(c) for c in chunks_default)//max(len(chunks_default),1):8d} | 通用场景（推荐）  |
  | size=1000,ol=200 | {len(chunks_large):6d} | {sum(len(c) for c in chunks_large)//max(len(chunks_large),1):8d} | 长文本、技术文档  |
  +------------------+--------+----------+-------------------+

  分析:
  - 小 chunk: 数量多，检索精准但可能丢失上下文
  - 中 chunk: 平衡精准度和上下文完整性
  - 大 chunk: 数量少，上下文完整但检索粒度粗

  实际建议: 先用 chunk_size=500 跑通，再根据检索效果调参
""")

# 具体对比：同一段内容在不同 chunk_size 下的呈现
print("  示例: '依赖注入' 相关内容在不同 chunk 中的位置")
keyword = "依赖注入"
for label, chunks in [("小chunk(200)", chunks_small),
                       ("中chunk(500)", chunks_default),
                       ("大chunk(1000)", chunks_large)]:
    found = [(i, c) for i, c in enumerate(chunks) if keyword in c]
    if found:
        idx, content = found[0]
        print(f"\n  [{label}] 出现在 Chunk {idx+1}/{len(chunks)}, 长度={len(content)}")
        # 找到关键词周围的上下文
        kw_pos = content.find(keyword)
        start = max(0, kw_pos - 30)
        end = min(len(content), kw_pos + 50)
        print(f"    上下文: ...{content[start:end].replace(chr(10), ' ')}...")
    else:
        print(f"\n  [{label}] 未找到 '{keyword}'")


# ============================================================
# === Part 5 === 给 Chunk 添加元数据
# ============================================================

print("\n" + "=" * 60)
print("Part 5: 给 Chunk 添加元数据")
print("=" * 60)

print("""
  元数据的作用:
  1. source   - 记录来源文件，回答时可以引用出处
  2. chunk_id - 唯一标识，方便更新和删除
  3. page     - PDF 页码，方便用户定位原文
  4. heading  - 所属章节标题，提供上下文
""")

def add_metadata(chunks: list, source: str, extra_meta: dict = None) -> list:
    """
    给每个 chunk 添加元数据。
    这是存入向量数据库前的关键步骤。

    参数:
        chunks: 分割后的文本块列表
        source: 来源文件名
        extra_meta: 额外的元数据字典

    返回:
        带元数据的 chunk 字典列表
    """
    result = []
    for i, chunk in enumerate(chunks):
        meta = {
            "source": source,              # 来源文件
            "chunk_index": i,              # 在该文件中的序号
            "total_chunks": len(chunks),   # 该文件总 chunk 数
            "char_count": len(chunk),      # 字符数
        }
        # 合并额外元数据
        if extra_meta:
            meta.update(extra_meta)

        result.append({
            "content": chunk,
            "metadata": meta,
        })
    return result


# 给 Markdown 文件的 chunks 添加元数据
chunks_with_meta = add_metadata(
    chunks_default,
    source="fastapi_guide.md",
    extra_meta={"category": "web_framework", "language": "zh"}
)

print(f"  共 {len(chunks_with_meta)} 个带元数据的 chunk\n")
for item in chunks_with_meta[:3]:  # 展示前3个
    print(f"  --- Chunk ---")
    print(f"  元数据: {item['metadata']}")
    preview = item["content"][:80].replace("\n", " ")
    print(f"  内容预览: {preview}...")
    print()


# --- 5.2 提取 Markdown 标题作为元数据 ---
print("--- 5.2 利用 Markdown 标题增强元数据 ---")

def split_md_with_headings(text: str, chunk_size: int = 500) -> list:
    """
    按 Markdown 标题分割，并把标题作为元数据保留。
    这样每个 chunk 都知道自己属于哪个章节。
    """
    chunks = []
    current_h1 = ""
    current_h2 = ""
    current_text = ""

    for line in text.split("\n"):
        if line.startswith("# ") and not line.startswith("## "):
            # 保存上一段
            if current_text.strip():
                chunks.append({
                    "content": current_text.strip(),
                    "metadata": {"h1": current_h1, "h2": current_h2}
                })
            current_h1 = line.lstrip("# ").strip()
            current_h2 = ""
            current_text = ""
        elif line.startswith("## "):
            # 保存上一段
            if current_text.strip():
                chunks.append({
                    "content": current_text.strip(),
                    "metadata": {"h1": current_h1, "h2": current_h2}
                })
            current_h2 = line.lstrip("# ").strip()
            current_text = ""
        else:
            current_text += line + "\n"

    # 最后一段
    if current_text.strip():
        chunks.append({
            "content": current_text.strip(),
            "metadata": {"h1": current_h1, "h2": current_h2}
        })

    return chunks

md_heading_chunks = split_md_with_headings(parsed_md)
print(f"  按标题分割得到 {len(md_heading_chunks)} 个 chunk\n")
for item in md_heading_chunks:
    h1 = item["metadata"]["h1"] or "(none)"
    h2 = item["metadata"]["h2"] or "(none)"
    preview = item["content"][:50].replace("\n", " ")
    print(f"  [h1={h1}] [h2={h2}]")
    print(f"    {preview}...")
    print()


# ============================================================
# === Part 6 === 完整管线：读文件 -> 解析 -> 分割 -> 元数据 -> 就绪
# ============================================================

print("=" * 60)
print("Part 6: 完整管线 - 从文件到 RAG 就绪的 Chunks")
print("=" * 60)

def full_pipeline(doc_dir: str, chunk_size: int = 500,
                  chunk_overlap: int = 100) -> list:
    """
    完整的文档处理管线。

    流程: 扫描目录 -> 解析每个文件 -> 分割 -> 添加元数据 -> 返回

    参数:
        doc_dir: 文档目录路径
        chunk_size: 每个 chunk 的最大字符数
        chunk_overlap: 相邻 chunk 的重叠字符数

    返回:
        list[dict]: 每个元素包含 content 和 metadata
    """
    # 支持的文件类型
    SUPPORTED_EXTENSIONS = {".txt", ".md"}

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    all_chunks = []
    files_processed = 0
    files_skipped = 0

    print(f"\n  扫描目录: {doc_dir}")

    for filename in sorted(os.listdir(doc_dir)):
        filepath = os.path.join(doc_dir, filename)

        # 跳过目录
        if os.path.isdir(filepath):
            continue

        # 检查文件类型
        ext = os.path.splitext(filename)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            files_skipped += 1
            print(f"  [SKIP] {filename} (不支持的格式: {ext})")
            continue

        # Step 1: 读取文件
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                raw_text = f.read()
        except UnicodeDecodeError:
            # 尝试 GBK 编码（Windows 常见）
            try:
                with open(filepath, "r", encoding="gbk") as f:
                    raw_text = f.read()
                print(f"  [INFO] {filename} 使用 GBK 编码读取")
            except Exception as e:
                print(f"  [FAIL] {filename} 读取失败: {e}")
                files_skipped += 1
                continue

        # Step 2: 基本清洗
        # 去掉多余空行
        import re
        raw_text = re.sub(r"\n{3,}", "\n\n", raw_text)

        # Step 3: 分割
        chunks = splitter.split_text(raw_text)

        # Step 4: 添加元数据
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "content": chunk,
                "metadata": {
                    "source": filename,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "char_count": len(chunk),
                    "file_type": ext,
                }
            })

        files_processed += 1
        print(f"  [OK] {filename}: {len(raw_text)} 字符 -> {len(chunks)} 个 chunks")

    print(f"\n  管线完成!")
    print(f"  处理文件: {files_processed} 个")
    print(f"  跳过文件: {files_skipped} 个")
    print(f"  总 chunk 数: {len(all_chunks)} 个")

    return all_chunks


# 运行完整管线
all_chunks = full_pipeline(SAMPLE_DIR, chunk_size=500, chunk_overlap=100)

# 展示结果
print(f"\n  === 管线输出样例 ===")
for item in all_chunks[:3]:
    print(f"\n  来源: {item['metadata']['source']}, "
          f"chunk {item['metadata']['chunk_index']+1}/{item['metadata']['total_chunks']}")
    print(f"  类型: {item['metadata']['file_type']}, "
          f"长度: {item['metadata']['char_count']} 字符")
    preview = item["content"][:100].replace("\n", " ")
    print(f"  内容: {preview}...")

# 展示最终数据结构
print(f"""
  === 最终数据结构 ===

  每个 chunk 的格式:
  {{
      "content": "文本内容...",
      "metadata": {{
          "source": "fastapi_guide.md",
          "chunk_index": 0,
          "total_chunks": 8,
          "char_count": 456,
          "file_type": ".md"
      }}
  }}

  下一步: 把这些 chunks 送入 Embedding 模型，生成向量，存入向量数据库
  这就是 Day 11-12 学的 Chroma + Embedding 的输入数据!
""")


# ============================================================
# 总结
# ============================================================

print("=" * 60)
print("Day 14 Summary")
print("=" * 60)
print(f"""
  今天学到了:

  1. 文档解析
     - TXT: 直接读取
     - MD:  读取 + 提取结构信息（标题、代码块）
     - PDF/DOCX: 需要专用库（PyPDF2, pdfplumber, python-docx）

  2. RecursiveCharacterTextSplitter
     - 多级分隔符: 段落 -> 换行 -> 句号 -> 空格 -> 字符
     - chunk_size: 推荐 500-1000（中文 500，英文 1000）
     - chunk_overlap: 推荐 50-200（防止切断上下文）

  3. Chunk 大小的影响
     - 小 chunk(200): {len(chunks_small)} 个，精准但上下文少
     - 中 chunk(500): {len(chunks_default)} 个，平衡（推荐）
     - 大 chunk(1000): {len(chunks_large)} 个，完整但粗粒度

  4. 元数据很重要
     - source: 引用出处
     - chunk_index: 定位原文
     - heading: 章节上下文

  5. 完整管线
     扫描目录 -> 读取文件 -> 清洗 -> 分割 -> 添加元数据 -> 就绪

  明天: Day 15 - RAG 优化（混合检索、重排序、查询改写）
""")
