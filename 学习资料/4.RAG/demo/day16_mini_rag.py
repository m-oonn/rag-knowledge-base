"""
Day 16: Mini RAG 系统 -- 完整的单文件 RAG 问答原型
运行方式: python day16_mini_rag.py

前置条件:
  必须:  pip install chromadb numpy scikit-learn
  推荐:  pip install sentence-transformers   (更好的 Embedding)
  可选:  pip install requests                (调用 LLM API)

  可选 LLM (有则用，无则仅展示检索结果):
  - Ollama 本地运行 (http://localhost:11434)
  - DeepSeek API (需设置环境变量 DEEPSEEK_API_KEY)

学习目标:
1. 把 Day 10-15 学到的所有 RAG 技术串成一个完整系统
2. 理解 RAG 从文档加载到问答输出的全流程
3. 体验一个可运行的 RAG 原型，为项目一打基础
4. 学会处理各种 fallback（无 LLM、无 sentence-transformers 等）
"""

import os
import re
import sys
import hashlib
import math
from collections import Counter

# ============================================================
# 配置常量（可以修改这些参数做实验）
# ============================================================

# 文档和数据目录（相对于脚本所在目录）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DOCS_DIR = os.path.join(SCRIPT_DIR, "sample_docs")
CHROMA_DB_DIR = os.path.join(SCRIPT_DIR, "chroma_db")

# 文本分割参数
CHUNK_SIZE = 500          # 每个 chunk 的最大字符数
CHUNK_OVERLAP = 100       # 相邻 chunk 的重叠字符数

# 检索参数
TOP_K = 3                 # 返回最相关的 K 个文档

# LLM 配置
OLLAMA_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen2.5:7b"  # 可改为你安装的模型

# ============================================================
# 工具函数
# ============================================================

def print_step(step: str, detail: str = ""):
    """打印带格式的步骤信息"""
    print(f"\n  [STEP] {step}")
    if detail:
        print(f"         {detail}")


def print_ok(msg: str):
    print(f"  [OK] {msg}")


def print_fail(msg: str):
    print(f"  [FAIL] {msg}")


def print_info(msg: str):
    print(f"  [INFO] {msg}")


def compute_doc_hash(content: str) -> str:
    """计算文档内容的 hash，用于检测文档是否变更"""
    return hashlib.md5(content.encode("utf-8")).hexdigest()[:12]


# ============================================================
# 第一部分：样例文档创建
# ============================================================

SAMPLE_DOCUMENTS = {
    "python_basics.md": """# Python 基础知识

## 变量与数据类型

Python 是动态类型语言，不需要声明变量类型。常见数据类型包括：int（整数）、float（浮点数）、str（字符串）、bool（布尔值）、list（列表）、dict（字典）、tuple（元组）。

Python 使用缩进来表示代码块，而不是花括号。推荐使用 4 个空格作为缩进。

## 函数与装饰器

函数用 def 关键字定义。Python 支持默认参数、可变参数（*args）和关键字参数（**kwargs）。

装饰器是 Python 的高级特性，本质上是一个接受函数作为参数并返回新函数的函数。常用于日志、权限检查、缓存等场景。使用 @decorator 语法糖来应用装饰器。

## 类型注解

Python 3.5+ 支持类型注解（Type Hints）。类型注解不影响运行，但能帮助 IDE 提供补全和类型检查。配合 mypy 工具可以在开发阶段发现类型错误。

常见类型注解：str, int, float, bool, list[str], dict[str, int], Optional[str], Union[str, int]。

## 异步编程

async/await 是 Python 异步编程的核心。用 async def 定义异步函数，用 await 调用异步操作。asyncio 模块提供事件循环和异步原语。

异步编程适合 IO 密集型场景：网络请求、数据库查询、文件读写。不适合 CPU 密集型计算。

## 虚拟环境

venv 模块创建隔离的 Python 环境。每个项目使用独立的虚拟环境，避免依赖冲突。

创建：python -m venv myenv
激活：myenv\\Scripts\\activate (Windows) 或 source myenv/bin/activate (Linux/Mac)
安装依赖：pip install -r requirements.txt
""",

    "fastapi_guide.md": """# FastAPI 开发指南

## 什么是 FastAPI

FastAPI 是一个现代、高性能的 Python Web 框架。基于 Starlette（异步支持）和 Pydantic（数据验证）构建。性能可媲美 Node.js 和 Go 框架。

FastAPI 的核心优势：自动数据验证、自动 API 文档（Swagger UI）、原生异步支持、类型安全。

## 安装与启动

安装 FastAPI 和 ASGI 服务器 uvicorn：pip install fastapi uvicorn

启动命令：uvicorn main:app --reload --host 0.0.0.0 --port 8000

--reload 参数在开发时自动重载代码变更，生产环境不要使用。

## 路由与请求处理

使用装饰器定义路由：@app.get("/path")、@app.post("/path") 等。

路径参数用花括号定义：@app.get("/users/{user_id}")，FastAPI 自动进行类型转换和验证。

查询参数通过函数参数定义，设置默认值使其可选：def list_items(skip: int = 0, limit: int = 10)。

请求体通过 Pydantic BaseModel 定义。POST 请求会自动解析 JSON 并验证类型。

## 依赖注入

FastAPI 的依赖注入通过 Depends 实现。定义依赖函数，在路由参数中声明 param: Type = Depends(func)。

常用场景：数据库连接管理、用户认证、权限校验、公共参数提取。

## 中间件与错误处理

中间件在每个请求前后执行。常用于 CORS 配置、日志记录、性能监控。

HTTPException 用于返回特定的 HTTP 错误响应。常见错误码：400（请求错误）、401（未认证）、403（无权限）、404（不存在）、422（验证失败）。

## 部署

生产部署推荐使用 Gunicorn + Uvicorn workers：gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker

也可以使用 Docker 容器化部署。配合 Nginx 做反向代理和负载均衡。
""",

    "rag_knowledge.md": """# RAG 技术详解

## 什么是 RAG

RAG 全称 Retrieval-Augmented Generation，检索增强生成。核心思想：先从知识库检索相关文档，再把文档作为上下文交给 LLM 生成回答。

RAG 解决两个关键问题：1）LLM 知识过时（训练数据有截止日期）；2）LLM 幻觉（编造不存在的内容）。

类比：RAG 就是"开卷考试"，让 LLM 带着参考资料回答问题。

## Embedding 向量化

Embedding 把文本转成固定维度的数字向量。语义相似的文本，向量在空间中距离近。

常用 Embedding 模型：sentence-transformers（本地免费）、OpenAI text-embedding-ada-002（云端收费）。中文推荐 BAAI/bge-base-zh-v1.5。

向量维度通常是 384、768 或 1536。维度越高，表达能力越强，但计算成本也越高。

## 向量数据库

向量数据库专门存储和检索向量。核心操作：存入向量 + 按相似度查找最近邻。

常用数据库：Chroma（轻量嵌入式，适合学习和小项目）、Pinecone（云服务）、Milvus（分布式高性能）、FAISS（纯向量检索库）。

Chroma 的优势：Python 原生、不需单独部署、支持持久化、API 简洁。

## 文本分割

长文档需要分割成小段（chunk）。原因：1）Embedding 对短文本效果更好；2）检索时需要定位到具体段落；3）LLM 上下文有长度限制。

RecursiveCharacterTextSplitter 是最推荐的分割器。按优先级递归尝试不同分隔符：段落 > 换行 > 句号 > 空格。

关键参数：chunk_size（通常500-1000字符）、chunk_overlap（通常50-200字符，防止切断上下文）。

## RAG 优化

基础 RAG 的问题：纯向量检索对精确关键词匹配不好、粗排不够精准、用户表达不规范。

混合检索：BM25（关键词）+ 向量（语义）结合，用 RRF（倒数排名融合）合并结果。

重排序（Reranking）：粗排后用 Cross-encoder 对 Top-K 结果精排。Cross-encoder 比 Bi-encoder 精度高但速度慢。

查询改写：把口语化问题转为适合检索的规范查询。可用 LLM 改写或规则改写。

## 评估指标

Recall@K：检索到的相关文档数 / 所有相关文档数（找全了多少）。
Precision@K：检索到的相关文档数 / K（找到的有多少是对的）。
MRR：1 / 第一个相关文档的排名（正确答案排多前）。

评估流程：准备测试集（问题+标注答案） -> 执行检索 -> 计算指标 -> 调优 -> 重新评估。
""",

    "llm_basics.txt": """大语言模型（LLM）基础知识

什么是大语言模型？
大语言模型是基于 Transformer 架构训练的深度学习模型，参数量通常在数十亿到数千亿之间。通过大规模文本数据预训练，模型学会了语言的统计规律和世界知识。

常见的大语言模型：
- GPT 系列（OpenAI）：GPT-3.5、GPT-4、GPT-4o
- Claude 系列（Anthropic）：Claude 3 Haiku/Sonnet/Opus
- 国内模型：通义千问（阿里）、DeepSeek、文心一言（百度）、GLM（智谱）

模型调用方式：
1. API 调用：通过 HTTP 请求调用云端模型（如 OpenAI API、DeepSeek API）
2. 本地部署：使用 Ollama、vLLM 等工具在本地运行开源模型

Ollama 本地部署：
安装 Ollama 后，使用 ollama pull qwen2.5:7b 下载模型。
启动后通过 http://localhost:11434/api/generate 调用。
适合开发测试，不需要 API Key，不产生费用。

Prompt 工程：
好的 Prompt 是 LLM 应用的关键。基本原则：
1. 角色设定：告诉模型它是谁
2. 任务描述：清晰说明要做什么
3. 输出格式：指定期望的输出格式
4. 示例（Few-shot）：给几个输入输出的例子
5. 约束条件：限制回答范围，减少幻觉

RAG Prompt 模板：
"根据以下文档内容回答用户问题。如果文档中没有相关信息，请明确说明。"
然后把检索到的文档和用户问题拼接在一起。
""",
}


def create_sample_docs():
    """创建样例文档（如果不存在）"""
    os.makedirs(SAMPLE_DOCS_DIR, exist_ok=True)

    existing = set(os.listdir(SAMPLE_DOCS_DIR))
    created_count = 0

    for filename, content in SAMPLE_DOCUMENTS.items():
        filepath = os.path.join(SAMPLE_DOCS_DIR, filename)
        if filename not in existing:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)
            created_count += 1
            print_ok(f"创建样例文档: {filename}")

    if created_count == 0:
        print_info(f"样例文档已存在 ({len(existing)} 个文件)")
    else:
        print_ok(f"共创建 {created_count} 个样例文档")

    return SAMPLE_DOCS_DIR


# ============================================================
# 第二部分：文本分割器
# ============================================================

class TextSplitter:
    """
    递归字符文本分割器（简化版 RecursiveCharacterTextSplitter）。
    不依赖 LangChain，直接实现核心逻辑。
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        # 中文优化的分隔符优先级
        self.separators = ["\n\n", "\n", "。", ".", "；", ";", " "]

    def split_text(self, text: str) -> list:
        """将文本分割为 chunks"""
        # 先清洗：去掉多余空行
        text = re.sub(r"\n{3,}", "\n\n", text).strip()

        if len(text) <= self.chunk_size:
            return [text] if text else []

        return self._recursive_split(text, self.separators)

    def _recursive_split(self, text: str, separators: list) -> list:
        """递归分割：先用大粒度分隔符，不够再用小粒度"""
        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        if not separators:
            # 最后手段：硬切
            chunks = []
            for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
                chunk = text[i:i + self.chunk_size].strip()
                if chunk:
                    chunks.append(chunk)
            return chunks

        sep = separators[0]
        next_seps = separators[1:]

        parts = text.split(sep)
        chunks = []
        current = ""

        for part in parts:
            candidate = (current + sep + part) if current else part

            if len(candidate) <= self.chunk_size:
                current = candidate
            else:
                # 保存当前累积的内容
                if current.strip():
                    chunks.append(current.strip())
                # 如果单个 part 超长，递归用更细的分隔符
                if len(part) > self.chunk_size:
                    sub_chunks = self._recursive_split(part, next_seps)
                    chunks.extend(sub_chunks)
                    current = ""
                else:
                    current = part

        if current.strip():
            chunks.append(current.strip())

        return chunks


# ============================================================
# 第三部分：Embedding 管理器（自动选择最佳方案）
# ============================================================

class EmbeddingManager:
    """
    Embedding 管理器。
    优先使用 sentence-transformers（效果好），
    自动 fallback 到 TF-IDF（零额外依赖）。
    """

    def __init__(self):
        self.method = None
        self.model = None
        self._init_embedding()

    def _init_embedding(self):
        """尝试加载最佳可用的 Embedding 方案"""
        # 方案 1：sentence-transformers（推荐）
        try:
            from sentence_transformers import SentenceTransformer
            model_name = "all-MiniLM-L6-v2"
            print_info(f"正在加载 sentence-transformers 模型: {model_name}")
            print_info("(首次加载需要下载模型，约 80MB，请稍候...)")
            self.model = SentenceTransformer(model_name)
            self.method = "sentence-transformers"
            print_ok(f"Embedding 方案: sentence-transformers ({model_name})")
            return
        except ImportError:
            pass
        except Exception as e:
            print_info(f"sentence-transformers 加载失败: {e}")

        # 方案 2：Chroma 内置 Embedding
        try:
            import chromadb
            self.method = "chroma-default"
            print_ok("Embedding 方案: Chroma 内置默认 (all-MiniLM-L6-v2)")
            return
        except Exception:
            pass

        # 方案 3：TF-IDF fallback
        self.method = "tfidf"
        print_info("Embedding 方案: TF-IDF (fallback, 效果有限但可用)")

    def get_chroma_embedding_function(self):
        """返回适用于 Chroma 的 embedding function"""
        if self.method == "sentence-transformers":
            try:
                from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
                return SentenceTransformerEmbeddingFunction(
                    model_name="all-MiniLM-L6-v2"
                )
            except Exception:
                return None
        elif self.method == "chroma-default":
            return None  # Chroma 使用默认
        else:
            # TF-IDF fallback
            return TfidfEmbeddingFunction()


class TfidfEmbeddingFunction:
    """
    用 TF-IDF 实现的 Embedding Function，作为 Chroma 的 fallback。
    效果不如 sentence-transformers，但零额外依赖（仅需 scikit-learn）。
    """

    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(max_features=384)
        self.is_fitted = False
        self._corpus = []

    def __call__(self, input_texts: list) -> list:
        """Chroma 要求的接口：接受文本列表，返回向量列表"""
        # 将新文本加入语料库并重新 fit
        self._corpus.extend(input_texts)
        self.vectorizer.fit(self._corpus)
        self.is_fitted = True

        vectors = self.vectorizer.transform(input_texts).toarray()

        # 补零到固定维度 384（Chroma 要求向量维度一致）
        result = []
        for vec in vectors:
            if len(vec) < 384:
                padded = list(vec) + [0.0] * (384 - len(vec))
            else:
                padded = list(vec[:384])
            result.append(padded)

        return result


# ============================================================
# 第四部分：LLM 调用器（自动检测可用的 LLM）
# ============================================================

class LLMCaller:
    """
    LLM 调用器。自动检测可用的 LLM：
    1. Ollama 本地模型（优先）
    2. DeepSeek API
    3. 无 LLM（仅展示检索结果）
    """

    def __init__(self):
        self.method = None
        self.available = False
        self._ollama_model = OLLAMA_MODEL
        self._detect_llm()

    def _detect_llm(self):
        """检测可用的 LLM"""
        try:
            import requests
        except ImportError:
            print_info("requests 未安装，无法调用 LLM (pip install requests)")
            print_info("将仅展示检索到的文档，不生成 LLM 回答")
            return

        # 检测 Ollama
        try:
            resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=3)
            if resp.status_code == 200:
                models = resp.json().get("models", [])
                model_names = [m.get("name", "") for m in models]

                # 优先使用配置的模型
                if any(OLLAMA_MODEL.split(":")[0] in m for m in model_names):
                    self.method = "ollama"
                    self.available = True
                    self._ollama_model = OLLAMA_MODEL
                    print_ok(f"LLM 方案: Ollama ({OLLAMA_MODEL})")
                    return
                elif model_names:
                    # 使用第一个可用模型
                    self.method = "ollama"
                    self.available = True
                    self._ollama_model = model_names[0]
                    print_ok(f"LLM 方案: Ollama ({self._ollama_model})")
                    return
                else:
                    print_info("Ollama 运行中但没有已下载的模型")
        except Exception:
            pass

        # 检测 DeepSeek API
        api_key = os.environ.get("DEEPSEEK_API_KEY", "")
        if api_key:
            self.method = "deepseek"
            self.available = True
            print_ok("LLM 方案: DeepSeek API")
            return

        print_info("未检测到可用 LLM (Ollama 未运行 / DeepSeek API Key 未设置)")
        print_info("将仅展示检索到的文档，不生成 LLM 回答")

    def generate(self, prompt: str) -> str:
        """调用 LLM 生成回答"""
        if not self.available:
            return ""

        import requests

        if self.method == "ollama":
            try:
                resp = requests.post(
                    f"{OLLAMA_URL}/api/generate",
                    json={
                        "model": self._ollama_model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.3, "num_predict": 512},
                    },
                    timeout=60,
                )
                if resp.status_code == 200:
                    return resp.json().get("response", "")
                else:
                    return f"[Ollama 错误: HTTP {resp.status_code}]"
            except requests.exceptions.Timeout:
                return "[Ollama 超时，模型可能正在加载，请重试]"
            except Exception as e:
                return f"[Ollama 调用失败: {e}]"

        elif self.method == "deepseek":
            api_key = os.environ.get("DEEPSEEK_API_KEY", "")
            try:
                resp = requests.post(
                    "https://api.deepseek.com/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json={
                        "model": "deepseek-chat",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.3,
                        "max_tokens": 512,
                    },
                    timeout=30,
                )
                if resp.status_code == 200:
                    return resp.json()["choices"][0]["message"]["content"]
                else:
                    return f"[DeepSeek 错误: HTTP {resp.status_code}]"
            except Exception as e:
                return f"[DeepSeek 调用失败: {e}]"

        return ""


# ============================================================
# 第五部分：Mini RAG 核心系统
# ============================================================

class MiniRAG:
    """
    完整的 Mini RAG 系统。

    功能:
    1. 文档加载与解析（.md, .txt）
    2. 文本分割（递归字符分割）
    3. Embedding + Chroma 向量存储
    4. 相似度检索 Top-K
    5. RAG Prompt 组装
    6. LLM 调用生成回答（自动检测 Ollama/DeepSeek）
    7. 引用来源标注
    8. 交互式问答 + 文档管理命令
    """

    def __init__(self):
        self.splitter = TextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        self.embedding_mgr = EmbeddingManager()
        self.llm = LLMCaller()
        self.collection = None
        self.client = None
        self.chunk_count = 0
        self._init_chroma()

    def _init_chroma(self):
        """初始化 Chroma 向量数据库"""
        try:
            import chromadb

            os.makedirs(CHROMA_DB_DIR, exist_ok=True)

            # 使用持久化客户端，数据保存到磁盘
            self.client = chromadb.PersistentClient(path=CHROMA_DB_DIR)

            # 获取 embedding function
            ef = self.embedding_mgr.get_chroma_embedding_function()

            # 获取或创建 collection
            if ef is not None:
                self.collection = self.client.get_or_create_collection(
                    name="mini_rag_docs",
                    embedding_function=ef,
                    metadata={"hnsw:space": "cosine"},
                )
            else:
                self.collection = self.client.get_or_create_collection(
                    name="mini_rag_docs",
                    metadata={"hnsw:space": "cosine"},
                )

            existing_count = self.collection.count()
            if existing_count > 0:
                print_ok(f"Chroma 已加载，已有 {existing_count} 个 chunks")
                self.chunk_count = existing_count
            else:
                print_ok("Chroma 数据库初始化完成（空库）")

        except ImportError:
            print_fail("chromadb 未安装! 请运行: pip install chromadb")
            print_info("无法使用向量数据库，程序将退出")
            sys.exit(1)
        except Exception as e:
            print_fail(f"Chroma 初始化失败: {e}")
            sys.exit(1)

    def load_documents(self, doc_dir: str):
        """
        加载目录下所有 .md 和 .txt 文件。

        流程: 扫描文件 -> 读取内容 -> 分割 -> 存入 Chroma
        """
        print_step("加载文档", f"目录: {doc_dir}")

        supported_ext = {".md", ".txt"}
        files_loaded = 0
        total_new_chunks = 0

        for filename in sorted(os.listdir(doc_dir)):
            ext = os.path.splitext(filename)[1].lower()
            if ext not in supported_ext:
                continue

            filepath = os.path.join(doc_dir, filename)

            # 读取文件
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError:
                try:
                    with open(filepath, "r", encoding="gbk") as f:
                        content = f.read()
                except Exception as e:
                    print_fail(f"读取失败: {filename} ({e})")
                    continue

            if not content.strip():
                continue

            # 检查是否已经加载过（通过 hash 去重）
            doc_hash = compute_doc_hash(content)
            try:
                existing = self.collection.get(
                    where={"doc_hash": doc_hash},
                    limit=1,
                )
                if existing and existing["ids"]:
                    print_info(f"跳过（已存在）: {filename}")
                    files_loaded += 1
                    continue
            except Exception:
                pass  # where 过滤不支持时忽略，直接加载

            # 分割文本
            chunks = self.splitter.split_text(content)

            if not chunks:
                continue

            # 准备存入 Chroma 的数据
            ids = []
            documents = []
            metadatas = []

            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_hash}_{i}"
                ids.append(chunk_id)
                documents.append(chunk)
                metadatas.append({
                    "source": filename,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "doc_hash": doc_hash,
                    "char_count": len(chunk),
                })

            # 分批存入 Chroma（每批最多 40 个，避免超限）
            batch_size = 40
            for start in range(0, len(ids), batch_size):
                end = start + batch_size
                try:
                    self.collection.add(
                        ids=ids[start:end],
                        documents=documents[start:end],
                        metadatas=metadatas[start:end],
                    )
                except Exception as e:
                    print_fail(f"存入 Chroma 失败: {filename} ({e})")

            files_loaded += 1
            total_new_chunks += len(chunks)
            print_ok(f"已加载: {filename} "
                     f"({len(content)} 字符 -> {len(chunks)} chunks)")

        self.chunk_count = self.collection.count()
        print_ok(f"加载完成: {files_loaded} 个文件, "
                 f"知识库共 {self.chunk_count} 个 chunks")

    def search(self, query: str, top_k: int = TOP_K) -> list:
        """
        检索最相关的文档块。

        参数:
            query: 用户查询
            top_k: 返回数量

        返回:
            list[dict]: 每个包含 content, source, similarity, chunk_index
        """
        if self.collection.count() == 0:
            return []

        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=min(top_k, self.collection.count()),
                include=["documents", "metadatas", "distances"],
            )
        except Exception as e:
            print_fail(f"检索失败: {e}")
            return []

        # 整理结果
        search_results = []
        if results and results["documents"] and results["documents"][0]:
            for i in range(len(results["documents"][0])):
                doc_text = results["documents"][0][i]
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0

                # Chroma cosine 距离转相似度（距离越小越相似）
                similarity = max(0, 1 - distance)

                search_results.append({
                    "content": doc_text,
                    "source": metadata.get("source", "unknown"),
                    "chunk_index": metadata.get("chunk_index", -1),
                    "total_chunks": metadata.get("total_chunks", -1),
                    "similarity": round(similarity, 4),
                })

        return search_results

    def build_rag_prompt(self, query: str, docs: list) -> str:
        """
        组装 RAG Prompt。
        将检索到的文档和用户问题拼接成结构化的 Prompt。
        """
        context_parts = []
        for i, doc in enumerate(docs, 1):
            source_info = (f"{doc['source']} "
                           f"(chunk {doc['chunk_index']+1}/{doc['total_chunks']})")
            context_parts.append(
                f"[文档{i}] 来源: {source_info}\n{doc['content']}"
            )
        context = "\n\n---\n\n".join(context_parts)

        prompt = f"""你是一个知识问答助手。请根据以下参考文档回答用户的问题。

要求：
1. 只根据文档内容回答，不要编造信息
2. 如果文档中没有相关信息，明确说"根据已有文档，没有找到相关信息"
3. 回答后标注引用来源（来自哪个文档）
4. 用中文回答，简洁准确

=== 参考文档 ===
{context}
=== 文档结束 ===

用户问题: {query}

请回答:"""
        return prompt

    def ask(self, query: str) -> dict:
        """
        完整的 RAG 问答流程。

        流程: 检索 -> 组装 Prompt -> LLM 生成 -> 格式化输出

        返回:
            dict: 包含 answer, sources, retrieved_docs
        """
        # Step 1: 检索
        docs = self.search(query, top_k=TOP_K)

        if not docs:
            return {
                "answer": "知识库为空或没有找到相关文档。请先加载文档。",
                "sources": [],
                "retrieved_docs": [],
            }

        # Step 2: 组装 Prompt
        prompt = self.build_rag_prompt(query, docs)

        # Step 3: 调用 LLM（如果可用）
        if self.llm.available:
            answer = self.llm.generate(prompt)
        else:
            answer = ""

        # Step 4: 提取来源信息
        sources = []
        for doc in docs:
            source = (f"{doc['source']} "
                      f"(chunk {doc['chunk_index']+1}/{doc['total_chunks']})")
            if source not in sources:
                sources.append(source)

        return {
            "answer": answer,
            "sources": sources,
            "retrieved_docs": docs,
        }

    def add_document_file(self, filepath: str):
        """添加单个文档文件到知识库"""
        if not os.path.isfile(filepath):
            print_fail(f"文件不存在: {filepath}")
            return

        ext = os.path.splitext(filepath)[1].lower()
        if ext not in {".md", ".txt"}:
            print_fail(f"不支持的文件类型: {ext} (仅支持 .md, .txt)")
            return

        # 读取文件
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(filepath, "r", encoding="gbk") as f:
                content = f.read()

        filename = os.path.basename(filepath)
        doc_hash = compute_doc_hash(content)

        # 分割
        chunks = self.splitter.split_text(content)
        if not chunks:
            print_info("文件内容为空或太短，无需分割")
            return

        # 存入 Chroma
        ids = [f"{doc_hash}_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "source": filename,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "doc_hash": doc_hash,
                "char_count": len(chunk),
            }
            for i, chunk in enumerate(chunks)
        ]

        try:
            self.collection.add(
                ids=ids, documents=chunks, metadatas=metadatas,
            )
            self.chunk_count = self.collection.count()
            print_ok(f"已添加: {filename} ({len(chunks)} chunks), "
                     f"知识库共 {self.chunk_count} 个 chunks")
        except Exception as e:
            print_fail(f"添加失败: {e}")

    def get_status(self) -> str:
        """获取系统状态"""
        chunk_count = self.collection.count() if self.collection else 0

        # 统计各文件的 chunk 数
        source_stats = {}
        if chunk_count > 0:
            try:
                all_meta = self.collection.get(include=["metadatas"])
                if all_meta and all_meta["metadatas"]:
                    for meta in all_meta["metadatas"]:
                        src = meta.get("source", "unknown")
                        source_stats[src] = source_stats.get(src, 0) + 1
            except Exception:
                source_stats = {"(无法获取详情)": chunk_count}

        lines = [
            f"  知识库状态:",
            f"  - 总 chunk 数: {chunk_count}",
            f"  - Embedding:   {self.embedding_mgr.method}",
            f"  - LLM:         {self.llm.method or '未检测到'}",
            f"  - Chroma 路径: {CHROMA_DB_DIR}",
            f"  - 文档目录:    {SAMPLE_DOCS_DIR}",
            f"  - 分割参数:    chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}",
            f"  - 检索参数:    top_k={TOP_K}",
            f"",
            f"  文档统计:",
        ]
        for src, count in sorted(source_stats.items()):
            lines.append(f"    {src}: {count} chunks")

        if not source_stats:
            lines.append("    (空)")

        return "\n".join(lines)

    def rebuild_index(self):
        """重建向量索引（删除旧数据，重新加载全部文档）"""
        print_step("重建索引")

        try:
            self.client.delete_collection("mini_rag_docs")
            print_ok("已删除旧 collection")
        except Exception:
            pass

        # 重新初始化 Chroma
        self._init_chroma()

        # 重新加载文档
        if os.path.isdir(SAMPLE_DOCS_DIR):
            self.load_documents(SAMPLE_DOCS_DIR)
        else:
            print_info("文档目录不存在，请先添加文档")


# ============================================================
# 第六部分：交互式问答循环
# ============================================================

def print_welcome():
    """打印欢迎信息"""
    print("\n" + "=" * 60)
    print("  Mini RAG 问答系统 (Day 16 Demo)")
    print("=" * 60)
    print("""
  这是一个完整的 RAG 问答原型，整合了:
  - 文档加载与解析 (Day 14)
  - 文本分割 RecursiveCharacterTextSplitter (Day 14)
  - Embedding 向量化 (Day 12)
  - Chroma 向量数据库 (Day 11)
  - LLM 生成回答 (Day 8-9)
  - 引用来源标注

  命令:
    直接输入问题     - 进行 RAG 问答
    /add <文件路径>  - 添加新文档到知识库
    /status          - 查看系统状态
    /rebuild         - 重建向量索引
    quit 或 exit     - 退出
""")


def format_response(result: dict) -> str:
    """格式化 RAG 响应输出"""
    lines = []

    # LLM 回答
    if result["answer"]:
        lines.append("\n  --- 回答 ---")
        for line in result["answer"].strip().split("\n"):
            lines.append(f"  {line}")
    else:
        lines.append(
            "\n  --- 检索结果 (无 LLM，仅展示检索到的文档) ---"
        )

    # 检索到的文档
    lines.append("\n  --- 检索到的相关文档 ---")
    for i, doc in enumerate(result["retrieved_docs"], 1):
        lines.append(f"  [{i}] 相似度: {doc['similarity']:.4f}")
        lines.append(f"      来源: {doc['source']} "
                     f"(chunk {doc['chunk_index']+1}/{doc['total_chunks']})")
        # 截取前 120 字符作为预览
        preview = doc["content"][:120].replace("\n", " ")
        lines.append(f"      内容: {preview}...")

    # 引用来源汇总
    if result["sources"]:
        lines.append("\n  --- 引用来源 ---")
        for src in result["sources"]:
            lines.append(f"  - {src}")

    return "\n".join(lines)


def interactive_loop(rag: MiniRAG):
    """交互式问答主循环"""
    print_welcome()

    while True:
        try:
            user_input = input("\n  问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  再见!")
            break

        if not user_input:
            continue

        # 退出命令
        if user_input.lower() in ("quit", "exit", "q"):
            print("\n  再见!")
            break

        # /status 命令
        if user_input.lower() == "/status":
            print(rag.get_status())
            continue

        # /rebuild 命令
        if user_input.lower() == "/rebuild":
            rag.rebuild_index()
            continue

        # /add 命令：添加新文档
        if user_input.lower().startswith("/add "):
            filepath = user_input[5:].strip().strip('"').strip("'")
            # 支持相对路径
            if not os.path.isabs(filepath):
                filepath = os.path.join(os.getcwd(), filepath)
            rag.add_document_file(filepath)
            continue

        # 未知命令提示
        if user_input.startswith("/"):
            print("  未知命令。可用: /status, /rebuild, /add <路径>, quit")
            continue

        # 普通问题 -> RAG 问答
        print("  (检索中...)")
        result = rag.ask(user_input)
        print(format_response(result))


# ============================================================
# 主入口
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("  初始化 Mini RAG 系统")
    print("=" * 60)

    # Step 1: 创建样例文档
    print_step("1/4 准备样例文档")
    doc_dir = create_sample_docs()

    # Step 2: 初始化 RAG 系统（含 Embedding、Chroma、LLM 检测）
    print_step("2/4 初始化 RAG 组件")
    rag = MiniRAG()

    # Step 3: 加载文档到向量库
    print_step("3/4 加载文档到向量数据库")
    if rag.chunk_count == 0:
        rag.load_documents(doc_dir)
    else:
        print_info(f"向量库已有 {rag.chunk_count} 个 chunks，跳过重复加载")
        print_info("如需重新加载，请使用 /rebuild 命令")

    # Step 4: 快速测试
    print_step("4/4 快速测试检索")
    test_query = "什么是RAG？"
    test_results = rag.search(test_query, top_k=2)
    if test_results:
        print_ok(f"测试查询 '{test_query}' 返回 {len(test_results)} 个结果")
        for r in test_results:
            print(f"         {r['similarity']:.4f} - {r['source']} "
                  f"(chunk {r['chunk_index']+1})")
    else:
        print_fail("测试检索无结果，请检查文档是否正确加载")

    # 进入交互循环
    interactive_loop(rag)


if __name__ == "__main__":
    main()
