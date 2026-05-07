"""
Day 5 实战：完整的 TodoList API

启动方式：
  cd day5_todo_project
  uvicorn main:app --reload

打开浏览器：
  http://127.0.0.1:8000/docs  → Swagger 自动文档

这个项目整合了前 4 天学的所有知识：
  - 装饰器 → @app.get() 路由
  - 类型注解 → 参数和返回值类型
  - Pydantic → 数据验证模型
  - 上下文管理器 → 数据库连接 yield
  - async/await → 异步中间件
  - FastAPI → 路由、参数、依赖注入、中间件、异常处理、路由分组

项目结构：
  main.py        ← 你现在看的这个文件（应用入口）
  models.py      ← Pydantic 数据模型
  database.py    ← 模拟数据库
  routers/
    todos.py     ← Todo 路由（CRUD 接口）
"""

import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from routers.todos import router as todos_router


# ============================================================
# 创建 FastAPI 应用
# ============================================================

app = FastAPI(
    title="TodoList API",
    description=(
        "Day 5 实战项目 —— 完整的待办事项管理 API\n\n"
        "功能：增删改查 + 搜索过滤 + 分页 + 统计\n\n"
        "**测试步骤：**\n"
        "1. 先用 `GET /todos` 看看初始数据\n"
        "2. 用 `POST /todos` 创建新待办\n"
        "3. 用 `PATCH /todos/{id}` 修改状态为 completed\n"
        "4. 用 `GET /todos?status=completed` 过滤已完成的\n"
        "5. 用 `GET /todos/stats` 查看统计信息\n"
        "6. 用 `DELETE /todos/{id}` 删除一个\n"
    ),
    version="1.0.0",
)


# ============================================================
# 中间件
# ============================================================

# CORS 中间件（前后端分离时需要）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# 请求日志 + 计时中间件
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """
    每个请求都经过这个中间件：
    1. 记录请求开始
    2. 执行路由处理
    3. 记录耗时
    4. 在响应头添加处理时间
    """
    start = time.time()

    response = await call_next(request)

    elapsed = time.time() - start
    response.headers["X-Process-Time"] = f"{elapsed:.4f}"

    # 控制台日志（生产环境应该用 logging 模块）
    method = request.method
    path = request.url.path
    status_code = response.status_code
    print(f"  {method:6s} {path:30s} → {status_code} ({elapsed:.3f}s)")

    return response


# ============================================================
# 全局异常处理
# ============================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    兜底异常处理：捕获所有未处理的异常。

    好处：
    - 防止内部错误信息泄露给前端
    - 统一错误响应格式
    - 方便调试（打印完整错误）
    """
    print(f"  ❌ 未处理的异常: {type(exc).__name__}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "服务器内部错误",
            "detail": "请稍后重试或联系管理员",
        },
    )


# ============================================================
# 注册路由
# ============================================================

# 注册 Todo 路由组
app.include_router(todos_router)


# ============================================================
# 根路径
# ============================================================

@app.get("/", tags=["系统"])
def root():
    """API 入口 —— 返回欢迎信息和可用端点"""
    return {
        "message": "TodoList API 运行中",
        "docs": "/docs",
        "endpoints": {
            "list": "GET /todos",
            "detail": "GET /todos/{id}",
            "create": "POST /todos",
            "update": "PATCH /todos/{id}",
            "delete": "DELETE /todos/{id}",
            "stats": "GET /todos/stats",
        },
    }


@app.get("/health", tags=["系统"])
def health():
    """健康检查"""
    return {"status": "ok"}


# ============================================================
# 启动提示
# ============================================================

print("""
╔══════════════════════════════════════════════════════╗
║          Day 5 实战: TodoList API                    ║
╠══════════════════════════════════════════════════════╣
║                                                      ║
║  启动命令（在 day5_todo_project 目录下）:              ║
║  uvicorn main:app --reload                           ║
║                                                      ║
║  Swagger 文档:                                        ║
║  http://127.0.0.1:8000/docs                          ║
║                                                      ║
║  测试顺序:                                            ║
║  1. GET  /todos          → 查看初始数据               ║
║  2. POST /todos          → 创建新待办                 ║
║  3. GET  /todos?q=Python → 搜索                      ║
║  4. PATCH /todos/4       → 修改状态                   ║
║  5. GET  /todos/stats    → 查看统计                   ║
║  6. DELETE /todos/5      → 删除                       ║
║                                                      ║
║  练习: 自己尝试加一个新功能!                            ║
╚══════════════════════════════════════════════════════╝
""")
