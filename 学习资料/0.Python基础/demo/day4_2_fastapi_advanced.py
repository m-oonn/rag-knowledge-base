"""
Day 4 Demo 2：FastAPI 进阶
运行方式：uvicorn day4_2_fastapi_advanced:app --reload --port 8001

学习目标：
1. 掌握依赖注入（Depends）
2. 掌握中间件（Middleware）
3. 掌握异常处理
4. 掌握文件上传
5. 掌握路由分组（APIRouter）
"""

import time
import asyncio
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Request, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from contextlib import asynccontextmanager


# ============================================================
# 应用生命周期（启动和关闭时执行）
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理：
    - yield 之前 = 应用启动时执行（初始化数据库、加载模型等）
    - yield 之后 = 应用关闭时执行（清理资源）

    在 AI 项目中：
    - 启动时加载 Embedding 模型、初始化向量数据库连接
    - 关闭时释放模型内存、关闭连接
    """
    print("🚀 应用启动：初始化资源...")
    # 模拟初始化
    app.state.db = {"connected": True}
    app.state.start_time = time.time()
    print("✅ 资源初始化完成")

    yield  # 应用运行中...

    print("🛑 应用关闭：清理资源...")
    app.state.db["connected"] = False
    print("✅ 资源清理完成")


app = FastAPI(
    title="Day 4 Demo 2: FastAPI 进阶",
    lifespan=lifespan,
)


# ============================================================
# 第一部分：中间件（Middleware）
# ============================================================

# --- CORS 中间件（前后端分离必需）---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 开发时允许所有来源，生产环境要限制
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- 自定义中间件：请求计时 + 日志 ---
@app.middleware("http")
async def log_and_time_middleware(request: Request, call_next):
    """
    中间件：每个请求都会经过。

    作用：
    1. 记录每个请求的方法、路径
    2. 测量处理时间
    3. 在响应头中添加处理时间

    在 AI 项目中：
    - 监控哪些接口慢（大模型调用通常最慢）
    - 记录请求日志用于调试
    """
    start = time.time()

    # 执行实际的路由处理
    response = await call_next(request)

    elapsed = time.time() - start

    # 添加自定义响应头
    response.headers["X-Process-Time"] = f"{elapsed:.4f}"

    # 打印日志
    print(f"  [{request.method}] {request.url.path} → {response.status_code} ({elapsed:.3f}s)")

    return response


# ============================================================
# 第二部分：依赖注入（Depends）
# ============================================================

# --- 依赖1：模拟数据库连接 ---
class FakeDB:
    """模拟数据库"""
    def __init__(self):
        self.connected = True
        self.data = {
            "users": [
                {"id": 1, "name": "张三"},
                {"id": 2, "name": "李四"},
            ]
        }

    def query(self, table: str):
        return self.data.get(table, [])

    def close(self):
        self.connected = False


def get_db():
    """
    数据库连接依赖 —— FastAPI 最经典的依赖注入模式。

    用 yield 的函数就是上下文管理器（Day 2 学过）：
    - yield 之前：创建连接
    - yield 返回连接给路由使用
    - yield 之后（finally）：关闭连接

    无论路由是否出错，连接都会被关闭！
    """
    db = FakeDB()
    print("  [DB] 创建数据库连接")
    try:
        yield db
    finally:
        db.close()
        print("  [DB] 关闭数据库连接")


# --- 依赖2：用户认证 ---
VALID_TOKENS = {"token-zhang": "张三", "token-li": "李四"}


def get_current_user(authorization: str | None = None):
    """
    认证依赖：从 Header 中获取 token 并验证。

    如果 token 无效，直接返回 401，路由函数不会执行。
    这样每个需要认证的接口只需要加一个 Depends 就行。
    """
    if not authorization or authorization not in VALID_TOKENS:
        raise HTTPException(
            status_code=401,
            detail="请在 Query 参数中传入有效的 authorization",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return {"name": VALID_TOKENS[authorization], "token": authorization}


# --- 依赖3：分页参数（复用）---
class Pagination:
    """分页参数依赖，所有列表接口都能复用"""
    def __init__(self, skip: int = 0, limit: int = 10):
        self.skip = skip
        self.limit = max(1, min(limit, 100))  # 限制 1-100


# --- 使用依赖注入的路由 ---

@app.get("/db/users")
def get_users(db: FakeDB = Depends(get_db), page: Pagination = Depends()):
    """
    演示多个依赖注入组合：
    - db: 自动创建和关闭数据库连接
    - page: 自动解析分页参数

    注意 Depends(get_db) vs Depends()：
    - Depends(get_db): 调用 get_db 函数
    - Depends(): 自动实例化 Pagination 类（类也能做依赖）
    """
    users = db.query("users")
    return {
        "users": users[page.skip: page.skip + page.limit],
        "pagination": {"skip": page.skip, "limit": page.limit},
    }


@app.get("/profile")
def get_profile(user=Depends(get_current_user)):
    """
    需要认证的接口。

    测试：
    - GET /profile?authorization=token-zhang → 成功
    - GET /profile → 401 未认证
    - GET /profile?authorization=wrong → 401 无效 token
    """
    return {"message": f"欢迎，{user['name']}！", "user": user}


# ============================================================
# 第三部分：异常处理
# ============================================================

# --- 自定义异常类 ---
class AIServiceError(Exception):
    """AI 服务异常"""
    def __init__(self, service_name: str, detail: str):
        self.service_name = service_name
        self.detail = detail


# --- 全局异常处理器 ---
@app.exception_handler(AIServiceError)
async def ai_service_error_handler(request: Request, exc: AIServiceError):
    """
    全局异常处理器：捕获 AIServiceError 并返回统一格式。

    好处：
    - 所有 AI 相关错误统一格式
    - 路由函数里只管 raise，不需要自己构造响应
    """
    return JSONResponse(
        status_code=503,
        content={
            "error": "AI 服务不可用",
            "service": exc.service_name,
            "detail": exc.detail,
            "tip": "请稍后重试",
        }
    )


@app.get("/ai/chat")
async def ai_chat(question: str):
    """
    模拟 AI 聊天接口。

    演示自定义异常的使用：
    - 当问题为空时抛出 HTTPException
    - 模拟 AI 服务不可用时抛出 AIServiceError

    测试：
    - GET /ai/chat?question=你好 → 正常返回
    - GET /ai/chat?question=error → 模拟 AI 服务错误
    """
    if question == "error":
        raise AIServiceError("Claude API", "模型响应超时")

    await asyncio.sleep(0.5)  # 模拟 AI 思考时间
    return {"question": question, "answer": f"你问了：{question}，这是AI的回答。"}


# ============================================================
# 第四部分：文件上传
# ============================================================

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """
    单文件上传。

    在 RAG 项目中：
    - 用户上传 PDF/Word/Markdown 文档
    - 后端接收文件 → 解析 → 分割 → 向量化 → 存入 Chroma

    测试：在 Swagger 文档中可以直接上传文件
    """
    # 读取文件内容
    content = await file.read()

    # 获取文件信息
    file_info = {
        "filename": file.filename,
        "content_type": file.content_type,
        "size_bytes": len(content),
        "size_kb": f"{len(content) / 1024:.1f} KB",
    }

    # 检查文件类型
    allowed_types = [
        "application/pdf",
        "text/plain",
        "text/markdown",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ]

    # content_type 可能是 None
    if file.content_type and file.content_type not in allowed_types:
        # 不严格限制，只是提示
        file_info["warning"] = f"类型 {file.content_type} 可能不受支持"

    # 预览前 200 个字符（如果是文本文件）
    try:
        preview = content[:200].decode("utf-8")
        file_info["preview"] = preview
    except UnicodeDecodeError:
        file_info["preview"] = "(二进制文件，无法预览)"

    return file_info


@app.post("/upload-multiple")
async def upload_multiple(files: list[UploadFile] = File(...)):
    """
    多文件上传。

    场景：用户一次性上传多个文档到知识库。
    """
    results = []
    for file in files:
        content = await file.read()
        results.append({
            "filename": file.filename,
            "size_kb": f"{len(content) / 1024:.1f} KB",
        })
    return {
        "count": len(results),
        "files": results,
    }


# ============================================================
# 第五部分：路由分组（APIRouter）
# ============================================================

# --- 知识库管理路由组 ---
kb_router = APIRouter(
    prefix="/knowledge-base",
    tags=["知识库管理"],  # Swagger 文档中的分组标签
)

# 模拟知识库数据
knowledge_bases = [
    {"id": 1, "name": "Python教程库", "doc_count": 15},
    {"id": 2, "name": "AI论文库", "doc_count": 42},
]


@kb_router.get("/")
def list_knowledge_bases():
    """获取所有知识库列表"""
    return knowledge_bases


@kb_router.get("/{kb_id}")
def get_knowledge_base(kb_id: int):
    """获取单个知识库详情"""
    for kb in knowledge_bases:
        if kb["id"] == kb_id:
            return kb
    raise HTTPException(status_code=404, detail=f"知识库 {kb_id} 不存在")


class KBCreate(BaseModel):
    name: str
    description: str | None = None


@kb_router.post("/", status_code=201)
def create_knowledge_base(kb: KBCreate):
    """创建新知识库"""
    new_kb = {"id": len(knowledge_bases) + 1, "name": kb.name, "doc_count": 0}
    knowledge_bases.append(new_kb)
    return new_kb


# 把路由组注册到 app
app.include_router(kb_router)


# --- 系统信息路由组 ---
system_router = APIRouter(prefix="/system", tags=["系统信息"])


@system_router.get("/health")
def health_check(request: Request):
    """系统健康检查"""
    uptime = time.time() - request.app.state.start_time
    return {
        "status": "healthy",
        "uptime_seconds": round(uptime, 1),
        "db_connected": request.app.state.db.get("connected", False),
    }


@system_router.get("/info")
def system_info():
    """系统信息"""
    return {
        "app": "RAG Knowledge Base",
        "version": "1.0.0",
        "framework": "FastAPI",
        "python": "3.11",
    }


app.include_router(system_router)


# ============================================================
# 第六部分：异步路由实战
# ============================================================

@app.get("/async-demo", tags=["异步演示"])
async def async_demo():
    """
    异步路由演示 —— 同时执行多个耗时操作。

    模拟 RAG 场景：同时查询缓存、Embedding、数据库。
    """
    start = time.time()

    async def check_cache():
        await asyncio.sleep(0.2)
        return {"cache": "miss"}

    async def get_embedding():
        await asyncio.sleep(0.3)
        return {"vector": [0.1, 0.2, 0.3]}

    async def query_db():
        await asyncio.sleep(0.4)
        return {"docs": ["doc1", "doc2"]}

    # 三个操作同时执行
    cache, embedding, db_result = await asyncio.gather(
        check_cache(),
        get_embedding(),
        query_db(),
    )

    elapsed = time.time() - start

    return {
        "results": {**cache, **embedding, **db_result},
        "time": f"{elapsed:.2f}s (同步需要 0.9s，异步只需 ~0.4s)",
    }


# ============================================================
# 启动提示
# ============================================================

print("""
╔══════════════════════════════════════════════════╗
║          Day 4 Demo 2: FastAPI 进阶              ║
╠══════════════════════════════════════════════════╣
║                                                  ║
║  启动命令:                                        ║
║  uvicorn day4_2_fastapi_advanced:app --reload    ║
║       --port 8001                                ║
║                                                  ║
║  Swagger 文档:                                    ║
║  http://127.0.0.1:8001/docs                      ║
║                                                  ║
║  重点测试:                                        ║
║  1. /db/users → 依赖注入（看控制台日志）            ║
║  2. /profile?authorization=token-zhang → 认证      ║
║  3. /ai/chat?question=error → 异常处理             ║
║  4. /upload → 文件上传                             ║
║  5. /knowledge-base/ → 路由分组                    ║
║  6. /async-demo → 异步并发                         ║
╚══════════════════════════════════════════════════╝
""")
