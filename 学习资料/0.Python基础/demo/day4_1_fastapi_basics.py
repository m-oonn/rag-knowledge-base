"""
Day 4 Demo 1：FastAPI 基础
运行方式：
  1. 确保虚拟环境激活
  2. 运行: uvicorn day4_1_fastapi_basics:app --reload --port 8000
  3. 打开浏览器: http://127.0.0.1:8000/docs（Swagger 自动文档）
  4. 在 Swagger 页面直接测试每个接口

学习目标：
1. 掌握 FastAPI 基本路由定义
2. 掌握 4 种请求参数（路径/查询/请求体/Header）
3. 理解 Pydantic 自动验证
4. 学会用 Swagger 文档测试接口
"""

from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field

# ============================================================
# 创建 FastAPI 应用实例
# ============================================================

app = FastAPI(
    title="Day 4 学习 API",           # 显示在 Swagger 文档标题
    description="FastAPI 基础学习Demo", # 文档描述
    version="1.0.0",
)


# ============================================================
# 第一部分：最简单的路由
# ============================================================

@app.get("/")
def root():
    """
    最简单的接口：访问根路径返回 Hello World。

    测试方法：
    - 浏览器直接访问 http://127.0.0.1:8000/
    - 或在 Swagger 文档页面点击 "Try it out"
    """
    return {"message": "Hello World", "status": "FastAPI 运行中！"}


@app.get("/ping")
def ping():
    """健康检查接口 —— 生产环境中常用，确认服务是否在线"""
    return {"ping": "pong"}


# ============================================================
# 第二部分：路径参数
# ============================================================

# 模拟数据库
USERS_DB = {
    1: {"id": 1, "name": "张三", "age": 21, "major": "人工智能"},
    2: {"id": 2, "name": "李四", "age": 22, "major": "计算机科学"},
    3: {"id": 3, "name": "王五", "age": 20, "major": "软件工程"},
}


@app.get("/users/{user_id}")
def get_user(user_id: int):
    """
    路径参数示例。

    - {user_id} 是路径参数，FastAPI 自动转成 int 类型
    - 如果传入非数字（如 /users/abc），FastAPI 自动返回 422 错误
    - 如果用户不存在，手动返回 404 错误

    测试：
    - GET /users/1 → 返回张三
    - GET /users/99 → 返回 404
    - GET /users/abc → 返回 422（类型错误）
    """
    if user_id not in USERS_DB:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"用户 {user_id} 不存在"
        )
    return USERS_DB[user_id]


# ============================================================
# 第三部分：查询参数
# ============================================================

ITEMS_DB = [
    {"id": 1, "name": "Python编程", "category": "书籍", "price": 59.9},
    {"id": 2, "name": "机械键盘", "category": "电子", "price": 299.0},
    {"id": 3, "name": "AI入门", "category": "书籍", "price": 45.0},
    {"id": 4, "name": "显示器", "category": "电子", "price": 1299.0},
    {"id": 5, "name": "LangChain实战", "category": "书籍", "price": 69.9},
    {"id": 6, "name": "鼠标", "category": "电子", "price": 99.0},
]


@app.get("/items")
def list_items(
    skip: int = 0,
    limit: int = 10,
    category: str | None = None,
    min_price: float | None = None,
    q: str | None = None,
):
    """
    查询参数示例 —— 实现搜索、过滤、分页。

    所有参数都是可选的（有默认值或为 None）。

    参数说明：
    - skip: 跳过前 N 条（分页用）
    - limit: 最多返回 N 条
    - category: 按分类过滤
    - min_price: 最低价格过滤
    - q: 搜索关键词（在名称中搜索）

    测试：
    - GET /items → 返回全部
    - GET /items?category=书籍 → 只返回书籍
    - GET /items?min_price=100 → 价格≥100的商品
    - GET /items?q=Python → 搜索包含"Python"的商品
    - GET /items?category=书籍&min_price=50 → 组合过滤
    - GET /items?skip=2&limit=2 → 分页
    """
    results = ITEMS_DB.copy()

    # 按分类过滤
    if category:
        results = [item for item in results if item["category"] == category]

    # 按最低价格过滤
    if min_price is not None:
        results = [item for item in results if item["price"] >= min_price]

    # 搜索关键词
    if q:
        results = [item for item in results if q.lower() in item["name"].lower()]

    # 分页
    total = len(results)
    results = results[skip: skip + limit]

    return {
        "total": total,
        "skip": skip,
        "limit": limit,
        "items": results,
    }


# ============================================================
# 第四部分：请求体（POST + Pydantic）
# ============================================================

class ItemCreate(BaseModel):
    """
    Pydantic 模型：定义创建商品时需要的数据。

    FastAPI 自动做的事：
    1. 验证 JSON 格式是否正确
    2. 验证每个字段的类型
    3. 验证 Field 里的规则（长度、范围等）
    4. 类型自动转换（如字符串 "59.9" → float 59.9）
    5. 验证失败自动返回 422 错误 + 详细错误信息
    """
    name: str = Field(min_length=1, max_length=50, description="商品名称")
    category: str = Field(description="商品分类")
    price: float = Field(gt=0, description="价格，必须大于0")
    description: str | None = Field(default=None, max_length=200, description="商品描述")

    # 提供示例数据（会显示在 Swagger 文档中）
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "name": "FastAPI入门",
                    "category": "书籍",
                    "price": 49.9,
                    "description": "一本很好的FastAPI教程"
                }
            ]
        }
    }


# 自增 ID
next_id = len(ITEMS_DB) + 1


@app.post("/items", status_code=status.HTTP_201_CREATED)
def create_item(item: ItemCreate):
    """
    请求体示例 —— 创建新商品。

    - 请求体是 JSON 格式，自动用 ItemCreate 模型验证
    - 验证通过才会执行函数体
    - 返回 201 状态码表示创建成功

    测试（在 Swagger 中）：
    1. 点击 "Try it out"
    2. 修改 JSON 数据
    3. 点击 "Execute"

    试试这些情况：
    - 正常数据 → 201 成功
    - price 传 -10 → 422 错误
    - name 传空字符串 → 422 错误
    - 不传 description → 正常（可选字段）
    """
    global next_id
    new_item = {"id": next_id, **item.model_dump()}
    ITEMS_DB.append(new_item)
    next_id += 1
    return new_item


class ItemUpdate(BaseModel):
    """部分更新模型：所有字段都是可选的"""
    name: str | None = None
    category: str | None = None
    price: float | None = Field(default=None, gt=0)
    description: str | None = None


@app.patch("/items/{item_id}")
def update_item(item_id: int, item: ItemUpdate):
    """
    路径参数 + 请求体 的组合。

    PATCH 方法用于部分更新：只传需要修改的字段。
    """
    # 找到要更新的商品
    target = None
    for i, existing in enumerate(ITEMS_DB):
        if existing["id"] == item_id:
            target = i
            break

    if target is None:
        raise HTTPException(status_code=404, detail=f"商品 {item_id} 不存在")

    # 只更新传入的字段（不为 None 的）
    update_data = item.model_dump(exclude_unset=True)
    ITEMS_DB[target].update(update_data)

    return ITEMS_DB[target]


@app.delete("/items/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_item(item_id: int):
    """
    删除商品 —— 返回 204 No Content（删除成功无需返回数据）。
    """
    for i, item in enumerate(ITEMS_DB):
        if item["id"] == item_id:
            ITEMS_DB.pop(i)
            return  # 204 不返回内容

    raise HTTPException(status_code=404, detail=f"商品 {item_id} 不存在")


# ============================================================
# 第五部分：Header 参数 + 简单认证
# ============================================================

API_KEYS = {"sk-abc123": "张三", "sk-def456": "李四"}


@app.get("/me")
def get_current_user(x_api_key: str = Header()):
    """
    Header 参数示例 —— 简单的 API Key 认证。

    注意：Header 参数名中的下划线 _ 会自动转成连字符 -
    所以 x_api_key 对应 Header 中的 X-Api-Key

    测试：
    - 在 Swagger 中填入 x-api-key: sk-abc123 → 成功
    - 填入 x-api-key: wrong → 401 错误
    """
    if x_api_key not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的 API Key",
        )
    return {"user": API_KEYS[x_api_key], "api_key": x_api_key[:6] + "***"}


# ============================================================
# 第六部分：响应模型（过滤敏感信息）
# ============================================================

class UserIn(BaseModel):
    """注册时的输入（包含密码）"""
    username: str
    email: str
    password: str


class UserOut(BaseModel):
    """返回给前端的数据（不包含密码！）"""
    username: str
    email: str


@app.post("/register", response_model=UserOut)
def register(user: UserIn):
    """
    response_model 示例 —— 自动过滤敏感字段。

    - 用户注册时传入密码
    - 返回时 response_model=UserOut 自动去掉 password
    - 保护敏感信息不被返回给前端

    测试：传入 {"username": "test", "email": "t@t.com", "password": "secret123"}
    返回中不会有 password 字段
    """
    # 实际项目中这里要 hash 密码并存入数据库
    print(f"注册用户: {user.username}, 密码已加密存储")
    return user  # 虽然 user 有 password，但 response_model 会自动过滤


# ============================================================
# 启动提示
# ============================================================

print("""
╔══════════════════════════════════════════════════╗
║          Day 4 Demo 1: FastAPI 基础              ║
╠══════════════════════════════════════════════════╣
║                                                  ║
║  启动命令:                                        ║
║  uvicorn day4_1_fastapi_basics:app --reload      ║
║                                                  ║
║  打开浏览器访问:                                   ║
║  http://127.0.0.1:8000/docs  (Swagger 文档)      ║
║  http://127.0.0.1:8000/redoc (ReDoc 文档)        ║
║                                                  ║
║  在 Swagger 中逐个测试每个接口！                    ║
╚══════════════════════════════════════════════════╝
""")
