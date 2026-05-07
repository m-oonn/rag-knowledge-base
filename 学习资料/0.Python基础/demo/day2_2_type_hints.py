"""
Day 2 Demo 2：类型注解（Type Hints）
运行方式：python day2_2_type_hints.py

学习目标：
1. 掌握基础类型注解写法
2. 掌握复合类型（list, dict, Optional 等）
3. 理解 Pydantic BaseModel（FastAPI 核心）
4. 知道类型注解在 FastAPI 中怎么用
"""

from typing import Optional, Union
from pydantic import BaseModel, Field, ValidationError


# ============================================================
# 第一部分：基础类型注解
# ============================================================

print("=" * 50)
print("第一部分：基础类型注解")
print("=" * 50)

# 变量类型注解
# 注意：类型注解不会强制检查，只是"标签"
name: str = "张三"
age: int = 21
score: float = 95.5
is_student: bool = True

print(f"name: {name} (类型: {type(name).__name__})")
print(f"age: {age} (类型: {type(age).__name__})")
print(f"score: {score} (类型: {type(score).__name__})")
print(f"is_student: {is_student} (类型: {type(is_student).__name__})")

# 类型注解不强制检查的证明：
# 下面这行不会报错，Python 不会阻止你
wrong_type: int = "这其实是字符串"
print(f"\nwrong_type 标注为 int，实际是: {type(wrong_type).__name__}")
print("→ Python 不会报错，类型注解只是提示，不是强制的")
print()


# ============================================================
# 第二部分：函数类型注解
# ============================================================

print("=" * 50)
print("第二部分：函数类型注解")
print("=" * 50)


def add(a: int, b: int) -> int:
    """
    类型注解写法：
    - 参数后面用 : 标注类型
    - 返回值用 -> 标注类型
    """
    return a + b


def greet(name: str, greeting: str = "你好") -> str:
    """带默认值的参数"""
    return f"{greeting}，{name}！"


def process_data(items: list[str]) -> dict[str, int]:
    """
    复合类型注解：
    - list[str]: 字符串列表
    - dict[str, int]: 键是字符串、值是整数的字典
    """
    return {item: len(item) for item in items}


print(f"add(3, 5) = {add(3, 5)}")
print(f"greet('李四') = {greet('李四')}")
print(f"greet('李四', '早上好') = {greet('李四', '早上好')}")
print(f"process_data(['hello', 'world']) = {process_data(['hello', 'world'])}")
print()


# ============================================================
# 第三部分：常用复合类型
# ============================================================

print("=" * 50)
print("第三部分：常用复合类型")
print("=" * 50)

# --- Optional：可以为 None ---
# Optional[str] 等价于 str | None
def find_user(user_id: int) -> str | None:
    """
    Optional 类型：返回值可能是 str，也可能是 None。
    在查询数据库时非常常见 —— 查到了返回数据，查不到返回 None。

    Python 3.10+ 可以用 str | None
    旧版本用 Optional[str]（需要从 typing 导入）
    """
    users = {1: "张三", 2: "李四"}
    return users.get(user_id)  # 找不到返回 None


result1 = find_user(1)
result2 = find_user(99)
print(f"find_user(1) = {result1}")
print(f"find_user(99) = {result2}")

# --- Union：多种类型 ---
def format_id(user_id: int | str) -> str:
    """
    Union 类型：参数可以是 int 或 str。
    API 接口经常遇到 —— 有时传数字 ID，有时传字符串 ID。

    Python 3.10+ 用 int | str
    旧版本用 Union[int, str]
    """
    return f"USER-{user_id}"


print(f"\nformat_id(123) = {format_id(123)}")
print(f"format_id('abc') = {format_id('abc')}")

# --- 嵌套复合类型 ---
def get_class_scores() -> dict[str, list[float]]:
    """
    嵌套类型：字典的值是浮点数列表。
    表示每个学生有多门成绩。
    """
    return {
        "张三": [90.5, 85.0, 92.0],
        "李四": [78.0, 88.5, 95.0],
    }


scores = get_class_scores()
print(f"\n班级成绩: {scores}")
print()


# ============================================================
# 第四部分：Pydantic BaseModel（重点！）
# ============================================================

print("=" * 50)
print("第四部分：Pydantic BaseModel（FastAPI 核心）")
print("=" * 50)
print()


# --- 定义数据模型 ---
class User(BaseModel):
    """
    Pydantic 模型 = 类型注解 + 自动验证。

    和普通 class 的区别：
    - 自动验证数据类型（传错类型会报错）
    - 自动类型转换（传 "21" 会自动转成 int 21）
    - 自动生成 JSON Schema（FastAPI 用这个生成 API 文档）
    """
    name: str                           # 必填字段
    age: int                            # 必填字段
    email: str | None = None            # 可选字段，默认 None
    tags: list[str] = []                # 可选字段，默认空列表


# 正常创建
user1 = User(name="张三", age=21, email="zhangsan@example.com")
print(f"用户1: {user1}")
print(f"  name: {user1.name}")
print(f"  age: {user1.age}")
print(f"  email: {user1.email}")
print(f"  tags: {user1.tags}")

# 只传必填字段，可选字段用默认值
user2 = User(name="李四", age=22)
print(f"\n用户2: {user2}")
print(f"  email 默认值: {user2.email}")

# 自动类型转换：字符串 "21" → 整数 21
user3 = User(name="王五", age="25")  # 传字符串也行！
print(f"\n用户3: age 传入 '25'(str)，自动转为 {user3.age}({type(user3.age).__name__})")

# 验证失败：传入不合法的数据
print("\n--- 验证失败的例子 ---")
try:
    bad_user = User(name="赵六", age="不是数字")
except ValidationError as e:
    print(f"创建失败！错误信息:")
    print(f"  {e.errors()[0]['msg']}")

print()


# --- 带验证规则的模型 ---
class Item(BaseModel):
    """
    Field() 可以添加额外的验证规则。

    在 FastAPI 中，这些验证规则会自动出现在 API 文档里，
    前端开发看到文档就知道该传什么数据。
    """
    name: str = Field(min_length=1, max_length=50, description="商品名称")
    price: float = Field(gt=0, description="价格，必须大于0")
    quantity: int = Field(ge=0, default=0, description="库存数量，不能为负")
    description: str | None = Field(default=None, max_length=200)


print("=" * 50)
print("带验证规则的 Pydantic 模型")
print("=" * 50)

# 正常创建
item = Item(name="Python编程书", price=59.9, quantity=100)
print(f"商品: {item}")

# 验证规则生效：价格不能为负
try:
    bad_item = Item(name="测试", price=-10)
except ValidationError as e:
    print(f"\n价格为负数 → 验证失败: {e.errors()[0]['msg']}")

# 验证规则生效：名称不能为空
try:
    bad_item = Item(name="", price=10)
except ValidationError as e:
    print(f"名称为空 → 验证失败: {e.errors()[0]['msg']}")

print()


# --- 模型转字典和 JSON ---
print("--- 模型转字典/JSON ---")
item_dict = item.model_dump()        # 转成 Python 字典
item_json = item.model_json_schema() # 获取 JSON Schema

print(f"转字典: {item_dict}")
print(f"JSON Schema 的字段: {list(item_json.get('properties', {}).keys())}")
print()


# ============================================================
# 第五部分：嵌套模型
# ============================================================

class Address(BaseModel):
    city: str
    street: str


class Employee(BaseModel):
    """
    嵌套模型：Employee 里面包含 Address。
    在实际项目中，数据结构经常是嵌套的。

    FastAPI 会自动处理嵌套的 JSON 请求体：
    {
        "name": "张三",
        "department": "AI研发部",
        "address": {
            "city": "北京",
            "street": "中关村大街1号"
        }
    }
    """
    name: str
    department: str
    address: Address                    # 嵌套另一个模型
    skills: list[str] = []


print("=" * 50)
print("第五部分：嵌套模型")
print("=" * 50)

# 可以直接传嵌套字典，Pydantic 自动解析
employee = Employee(
    name="张三",
    department="AI研发部",
    address={"city": "北京", "street": "中关村大街1号"},  # 自动转成 Address 对象
    skills=["Python", "LangChain", "FastAPI"]
)

print(f"员工: {employee.name}")
print(f"部门: {employee.department}")
print(f"城市: {employee.address.city}")      # 访问嵌套属性
print(f"技能: {', '.join(employee.skills)}")
print()


# ============================================================
# 总结
# ============================================================

print("=" * 50)
print("📝 类型注解总结")
print("=" * 50)
print("""
1. 类型注解只是"标签"，Python 运行时不强制检查
2. 但 Pydantic/FastAPI 会利用类型注解做自动验证
3. 基础类型: int, str, float, bool
4. 复合类型: list[str], dict[str, int], str | None
5. Pydantic BaseModel = 类型注解 + 自动验证 + JSON Schema
6. Field() 可以加额外验证规则（范围、长度等）
7. 嵌套模型可以表示复杂的数据结构

在 FastAPI 中，几乎所有请求/响应数据都用 Pydantic 模型定义！
""")
