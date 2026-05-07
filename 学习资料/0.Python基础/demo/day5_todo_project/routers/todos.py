"""
Todo 路由模块

所有 /todos 相关的接口都在这里。
用 APIRouter 分组，然后在 main.py 中注册。

这个文件是整个项目的核心，包含完整的 CRUD 操作。
"""

from fastapi import APIRouter, Depends, HTTPException, status, Query
from models import (
    TodoCreate,
    TodoUpdate,
    TodoResponse,
    TodoListResponse,
    TodoStatsResponse,
    TodoStatus,
    Priority,
)
from database import TodoDatabase, get_database

# ============================================================
# 创建路由器
# ============================================================

router = APIRouter(
    prefix="/todos",
    tags=["待办事项"],  # Swagger 文档中的分组名
)


# ============================================================
# 依赖注入：获取数据库连接
# ============================================================

def get_db():
    """
    数据库依赖。

    用 yield 实现上下文管理：
    - yield 之前：获取连接
    - yield：把连接交给路由使用
    - yield 之后：可以做清理（这里用内存字典不需要）

    后面做 RAG 项目时，这里会换成真正的数据库连接：
        db = SessionLocal()
        try:
            yield db
        finally:
            db.close()
    """
    db = get_database()
    yield db


# ============================================================
# GET /todos/stats —— 统计信息
# 注意：这个路由必须放在 /todos/{todo_id} 之前！
# 否则 FastAPI 会把 "stats" 当成 todo_id 去匹配
# ============================================================

@router.get(
    "/stats",
    response_model=TodoStatsResponse,
    summary="获取统计信息",
)
def get_stats(db: TodoDatabase = Depends(get_db)):
    """
    返回待办事项的统计数据。

    包括：总数、各状态数量、完成率。
    """
    return db.get_stats()


# ============================================================
# GET /todos —— 获取列表（支持搜索、过滤、分页）
# ============================================================

@router.get(
    "/",
    response_model=TodoListResponse,
    summary="获取待办列表",
)
def list_todos(
    skip: int = Query(default=0, ge=0, description="跳过前N条"),
    limit: int = Query(default=10, ge=1, le=100, description="每页数量(1-100)"),
    status: TodoStatus | None = Query(default=None, description="按状态过滤"),
    priority: Priority | None = Query(default=None, description="按优先级过滤"),
    q: str | None = Query(default=None, min_length=1, description="搜索关键词"),
    db: TodoDatabase = Depends(get_db),
):
    """
    获取待办事项列表。

    支持多种查询方式，可以组合使用：
    - **分页**: skip=0&limit=10
    - **按状态过滤**: status=pending / in_progress / completed
    - **按优先级过滤**: priority=low / medium / high
    - **搜索**: q=关键词（在标题和描述中搜索）

    示例：
    - `GET /todos` → 全部
    - `GET /todos?status=pending` → 未完成的
    - `GET /todos?priority=high&status=pending` → 高优先级未完成
    - `GET /todos?q=Python` → 搜索包含Python的
    """
    items, total = db.get_all(
        skip=skip,
        limit=limit,
        status=status,
        priority=priority,
        q=q,
    )
    return TodoListResponse(
        total=total,
        skip=skip,
        limit=limit,
        items=items,
    )


# ============================================================
# GET /todos/{todo_id} —— 获取单个
# ============================================================

@router.get(
    "/{todo_id}",
    response_model=TodoResponse,
    summary="获取单个待办",
)
def get_todo(
    todo_id: int,
    db: TodoDatabase = Depends(get_db),
):
    """
    根据 ID 获取待办详情。

    - 路径参数 todo_id 自动转成 int
    - 不存在时返回 404
    """
    todo = db.get_by_id(todo_id)
    if not todo:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"待办 {todo_id} 不存在",
        )
    return todo


# ============================================================
# POST /todos —— 创建
# ============================================================

@router.post(
    "/",
    response_model=TodoResponse,
    status_code=status.HTTP_201_CREATED,
    summary="创建待办",
)
def create_todo(
    todo: TodoCreate,
    db: TodoDatabase = Depends(get_db),
):
    """
    创建新的待办事项。

    请求体示例：
    ```json
    {
        "title": "学习LangChain",
        "description": "RAG核心框架",
        "priority": "high"
    }
    ```

    - title 必填（1-100字符）
    - description 可选
    - priority 可选，默认 medium
    - status 自动设为 pending
    - id 和时间自动生成
    """
    new_todo = db.create(todo.model_dump())
    return new_todo


# ============================================================
# PATCH /todos/{todo_id} —— 部分更新
# ============================================================

@router.patch(
    "/{todo_id}",
    response_model=TodoResponse,
    summary="更新待办",
)
def update_todo(
    todo_id: int,
    todo: TodoUpdate,
    db: TodoDatabase = Depends(get_db),
):
    """
    部分更新待办事项（PATCH 语义：只传需要改的字段）。

    示例 —— 只改状态：
    ```json
    {"status": "completed"}
    ```

    示例 —— 改标题和优先级：
    ```json
    {"title": "新标题", "priority": "high"}
    ```
    """
    # 检查是否存在
    existing = db.get_by_id(todo_id)
    if not existing:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"待办 {todo_id} 不存在",
        )

    # exclude_unset=True: 只取用户传了的字段，没传的不算
    update_data = todo.model_dump(exclude_unset=True)

    if not update_data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="至少需要传入一个要更新的字段",
        )

    updated = db.update(todo_id, update_data)
    return updated


# ============================================================
# DELETE /todos/{todo_id} —— 删除
# ============================================================

@router.delete(
    "/{todo_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="删除待办",
)
def delete_todo(
    todo_id: int,
    db: TodoDatabase = Depends(get_db),
):
    """
    删除待办事项。

    - 成功返回 204 No Content（无响应体）
    - 不存在返回 404
    """
    success = db.delete(todo_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"待办 {todo_id} 不存在",
        )
    # 204 不返回内容
