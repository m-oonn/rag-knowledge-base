"""
数据模型定义（Pydantic）

这个文件定义了所有的数据结构：
- TodoCreate: 创建时前端传什么数据
- TodoUpdate: 更新时前端传什么数据
- TodoResponse: 返回给前端什么数据
- TodoInDB: 数据库里存什么数据

为什么要分这么多模型？
- 创建时不需要传 id（后端自动生成）
- 更新时所有字段都是可选的（只传要改的）
- 返回时要包含 id 和创建时间
- 数据库里可能还有其他内部字段

这种分层在正式项目中非常常见，面试可能会问。
"""

from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum


# ============================================================
# 枚举类型：优先级和状态
# ============================================================

class Priority(str, Enum):
    """
    优先级枚举。

    继承 str 和 Enum，这样：
    - 在 JSON 中序列化为字符串（"high" 而不是 1）
    - FastAPI 自动在文档中显示可选值
    """
    low = "low"
    medium = "medium"
    high = "high"


class TodoStatus(str, Enum):
    """待办状态"""
    pending = "pending"
    in_progress = "in_progress"
    completed = "completed"


# ============================================================
# 请求模型
# ============================================================

class TodoCreate(BaseModel):
    """
    创建 Todo 时的请求体。

    注意：
    - 没有 id（后端自动生成）
    - 没有 created_at（后端自动记录）
    - 没有 status（默认 pending）
    """
    title: str = Field(
        min_length=1,
        max_length=100,
        description="待办标题",
        examples=["学习FastAPI"],
    )
    description: str | None = Field(
        default=None,
        max_length=500,
        description="详细描述",
    )
    priority: Priority = Field(
        default=Priority.medium,
        description="优先级: low/medium/high",
    )


class TodoUpdate(BaseModel):
    """
    更新 Todo 时的请求体。

    所有字段都是可选的 —— PATCH 语义：只传需要修改的字段。
    """
    title: str | None = Field(default=None, min_length=1, max_length=100)
    description: str | None = None
    priority: Priority | None = None
    status: TodoStatus | None = None


# ============================================================
# 响应模型
# ============================================================

class TodoResponse(BaseModel):
    """
    返回给前端的 Todo 数据。

    比 TodoCreate 多了：
    - id: 唯一标识
    - status: 当前状态
    - created_at: 创建时间
    - updated_at: 最后更新时间
    """
    id: int
    title: str
    description: str | None = None
    priority: Priority
    status: TodoStatus
    created_at: datetime
    updated_at: datetime


class TodoListResponse(BaseModel):
    """列表响应：包含分页信息"""
    total: int
    skip: int
    limit: int
    items: list[TodoResponse]


class TodoStatsResponse(BaseModel):
    """统计信息响应"""
    total: int
    pending: int
    in_progress: int
    completed: int
    completion_rate: str  # 如 "42.9%"
