"""
模拟数据库（内存字典）

正式项目会用 SQLite / PostgreSQL + SQLAlchemy，
但现在用字典模拟，专注学 FastAPI 本身。

后面做 RAG 项目时会换成真正的数据库。
"""

from datetime import datetime
from models import Priority, TodoStatus


class TodoDatabase:
    """
    模拟数据库。

    用法和真实数据库类似：
    - get_all(): 查询所有
    - get_by_id(id): 按 ID 查询
    - create(data): 插入
    - update(id, data): 更新
    - delete(id): 删除
    """

    def __init__(self):
        self._id_counter = 0
        self._data: dict[int, dict] = {}
        self._init_sample_data()

    def _next_id(self) -> int:
        """自增 ID"""
        self._id_counter += 1
        return self._id_counter

    def _init_sample_data(self):
        """初始化一些示例数据，方便测试"""
        samples = [
            {"title": "学习Python装饰器", "description": "理解装饰器原理和用法", "priority": Priority.high, "status": TodoStatus.completed},
            {"title": "学习异步编程", "description": "掌握async/await", "priority": Priority.high, "status": TodoStatus.completed},
            {"title": "学习FastAPI", "description": "路由、参数、依赖注入", "priority": Priority.high, "status": TodoStatus.in_progress},
            {"title": "完成TodoList项目", "description": "今天的实战任务", "priority": Priority.medium, "status": TodoStatus.pending},
            {"title": "复习Pydantic", "description": None, "priority": Priority.low, "status": TodoStatus.pending},
            {"title": "学习Claude API调用", "description": "明天的任务", "priority": Priority.medium, "status": TodoStatus.pending},
        ]
        for sample in samples:
            self.create(sample)

    def get_all(
        self,
        skip: int = 0,
        limit: int = 10,
        status: TodoStatus | None = None,
        priority: Priority | None = None,
        q: str | None = None,
    ) -> tuple[list[dict], int]:
        """
        查询所有 Todo，支持过滤和分页。

        返回: (结果列表, 总数)
        """
        results = list(self._data.values())

        # 按状态过滤
        if status:
            results = [t for t in results if t["status"] == status]

        # 按优先级过滤
        if priority:
            results = [t for t in results if t["priority"] == priority]

        # 关键词搜索（在标题和描述中搜索）
        if q:
            q_lower = q.lower()
            results = [
                t for t in results
                if q_lower in t["title"].lower()
                or (t["description"] and q_lower in t["description"].lower())
            ]

        total = len(results)

        # 按创建时间倒序（最新的在前面）
        results.sort(key=lambda x: x["created_at"], reverse=True)

        # 分页
        results = results[skip: skip + limit]

        return results, total

    def get_by_id(self, todo_id: int) -> dict | None:
        """按 ID 查询，找不到返回 None"""
        return self._data.get(todo_id)

    def create(self, data: dict) -> dict:
        """创建新 Todo"""
        todo_id = self._next_id()
        now = datetime.now()
        todo = {
            "id": todo_id,
            "title": data["title"],
            "description": data.get("description"),
            "priority": data.get("priority", Priority.medium),
            "status": data.get("status", TodoStatus.pending),
            "created_at": now,
            "updated_at": now,
        }
        self._data[todo_id] = todo
        return todo

    def update(self, todo_id: int, data: dict) -> dict | None:
        """
        更新 Todo。

        只更新 data 中不为 None 的字段（PATCH 语义）。
        """
        todo = self._data.get(todo_id)
        if not todo:
            return None

        for key, value in data.items():
            if value is not None:
                todo[key] = value

        todo["updated_at"] = datetime.now()
        return todo

    def delete(self, todo_id: int) -> bool:
        """删除 Todo，成功返回 True，不存在返回 False"""
        if todo_id in self._data:
            del self._data[todo_id]
            return True
        return False

    def get_stats(self) -> dict:
        """统计信息"""
        todos = list(self._data.values())
        total = len(todos)
        pending = sum(1 for t in todos if t["status"] == TodoStatus.pending)
        in_progress = sum(1 for t in todos if t["status"] == TodoStatus.in_progress)
        completed = sum(1 for t in todos if t["status"] == TodoStatus.completed)
        rate = f"{completed / total * 100:.1f}%" if total > 0 else "0%"

        return {
            "total": total,
            "pending": pending,
            "in_progress": in_progress,
            "completed": completed,
            "completion_rate": rate,
        }


# 全局单例（模拟数据库连接池）
_db_instance: TodoDatabase | None = None


def get_database() -> TodoDatabase:
    """
    获取数据库实例（单例模式）。

    在 FastAPI 中通过 Depends 注入：
        def get_db():
            yield get_database()
    """
    global _db_instance
    if _db_instance is None:
        _db_instance = TodoDatabase()
    return _db_instance
