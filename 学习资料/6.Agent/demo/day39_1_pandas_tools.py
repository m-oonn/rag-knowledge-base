"""
Day 39 Demo：pandas 数据分析工具
运行方式：python day39_1_pandas_tools.py
前置条件：pip install pandas numpy

学习目标：
1. pandas 核心操作（Agent 常用的那些）
2. 创建示例数据集
3. 封装为 Agent 工具函数
4. 模拟 Agent 调用分析工具
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# ============================================================
# Part 1：创建示例销售数据
# ============================================================

print("=" * 55)
print("Part 1: Create Sample Dataset")
print("=" * 55)

np.random.seed(42)
n_rows = 200

# 生成日期范围
start_date = datetime(2024, 1, 1)
dates = [start_date + timedelta(days=np.random.randint(0, 365)) for _ in range(n_rows)]

df = pd.DataFrame({
    "date": dates,
    "product": np.random.choice(["Product A", "Product B", "Product C"], n_rows),
    "region": np.random.choice(["East", "West", "North", "South"], n_rows),
    "quantity": np.random.randint(10, 200, n_rows),
    "revenue": np.random.randint(1000, 20000, n_rows),
    "cost": np.random.randint(500, 10000, n_rows),
})
df["date"] = pd.to_datetime(df["date"])
df["profit"] = df["revenue"] - df["cost"]
df = df.sort_values("date").reset_index(drop=True)

# 保存为 CSV（Agent 会读这个文件）
csv_path = "sample_sales.csv"
df.to_csv(csv_path, index=False)

print(f"\n  Created: {csv_path} ({len(df)} rows, {len(df.columns)} columns)")
print(f"  Columns: {list(df.columns)}")
print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")

# ============================================================
# Part 2：数据概览（Agent 首先需要了解数据）
# ============================================================

print("\n" + "=" * 55)
print("Part 2: Data Overview (Agent's first step)")
print("=" * 55)


def get_data_overview(df: pd.DataFrame) -> str:
    """Agent 工具：获取数据概览"""
    info = []
    info.append(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
    info.append(f"Columns: {list(df.columns)}")
    info.append(f"\nColumn types:")
    for col in df.columns:
        info.append(f"  {col}: {df[col].dtype}")
    info.append(f"\nFirst 5 rows:\n{df.head().to_string()}")
    info.append(f"\nStatistics:\n{df.describe().to_string()}")
    return "\n".join(info)


overview = get_data_overview(df)
print(f"\n{overview[:500]}...")

# ============================================================
# Part 3：常用分析操作
# ============================================================

print("\n" + "=" * 55)
print("Part 3: Common Analysis Operations")
print("=" * 55)

# 按产品分组统计
print("\n  --- Revenue by Product ---")
product_stats = df.groupby("product").agg({
    "revenue": ["sum", "mean", "count"],
    "profit": "sum",
}).round(0)
print(product_stats.to_string())

# 按区域分组
print("\n  --- Revenue by Region ---")
region_stats = df.groupby("region")["revenue"].agg(["sum", "mean", "count"])
print(region_stats.to_string())

# 月度趋势
print("\n  --- Monthly Revenue Trend ---")
df["month"] = df["date"].dt.to_period("M")
monthly = df.groupby("month")["revenue"].sum()
print(monthly.to_string())

# Top 5 最高收入日
print("\n  --- Top 5 Highest Revenue Days ---")
top5 = df.nlargest(5, "revenue")[["date", "product", "revenue", "profit"]]
print(top5.to_string(index=False))

# ============================================================
# Part 4：封装为 Agent 工具
# ============================================================

print("\n" + "=" * 55)
print("Part 4: Agent Tool Functions")
print("=" * 55)


def tool_load_data(file_path: str) -> str:
    """Agent工具：加载CSV数据"""
    try:
        df = pd.read_csv(file_path)
        return f"[OK] Loaded {file_path}: {df.shape[0]} rows, {df.shape[1]} columns\nColumns: {list(df.columns)}"
    except Exception as e:
        return f"[FAIL] {e}"


def tool_describe(df: pd.DataFrame, column: str = None) -> str:
    """Agent工具：描述性统计"""
    if column:
        return df[column].describe().to_string()
    return df.describe().to_string()


def tool_group_analysis(df: pd.DataFrame, group_col: str, value_col: str, agg: str = "sum") -> str:
    """Agent工具：分组聚合分析"""
    result = df.groupby(group_col)[value_col].agg(agg)
    return result.to_string()


def tool_filter(df: pd.DataFrame, column: str, operator: str, value) -> str:
    """Agent工具：条件过滤"""
    ops = {">=": "ge", "<=": "le", ">": "gt", "<": "lt", "==": "eq", "!=": "ne"}
    if operator in ops:
        filtered = df[getattr(df[column], ops[operator])(value)]
        return f"Filtered: {len(filtered)} rows\n{filtered.head(10).to_string()}"
    return "Invalid operator"


def tool_trend(df: pd.DataFrame, date_col: str, value_col: str, freq: str = "M") -> str:
    """Agent工具：时间趋势分析"""
    df_copy = df.copy()
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])
    trend = df_copy.set_index(date_col)[value_col].resample(freq).sum()
    return trend.to_string()


# 测试工具
print(f"\n  tool_load_data('{csv_path}'):")
print(f"    {tool_load_data(csv_path)}")

print(f"\n  tool_group_analysis(df, 'product', 'revenue', 'sum'):")
print(f"    {tool_group_analysis(df, 'product', 'revenue', 'sum')}")

print(f"\n  tool_filter(df, 'revenue', '>=', 15000):")
result = tool_filter(df, 'revenue', '>=', 15000)
print(f"    {result[:150]}...")

# ============================================================
# Part 5：模拟 Agent 分析流程
# ============================================================

print("\n" + "=" * 55)
print("Part 5: Simulated Agent Analysis Flow")
print("=" * 55)


def simulate_agent_analysis(user_request: str, df: pd.DataFrame):
    """模拟 Agent 接到分析需求后的完整流程"""
    print(f"\n  User: {user_request}")
    print(f"  {'─' * 45}")

    # Step 1: 理解需求（LLM 决策）
    print(f"  [Agent] Step 1: Understanding request...")

    # Step 2: 获取数据概览
    print(f"  [Agent] Step 2: Getting data overview...")
    print(f"    Shape: {df.shape}, Columns: {list(df.columns)[:5]}...")

    # Step 3: 执行分析
    print(f"  [Agent] Step 3: Analyzing...")
    if "product" in user_request.lower() or "产品" in user_request:
        result = df.groupby("product").agg({"revenue": "sum", "profit": "sum"}).round(0)
        print(f"    {result.to_string()}")
    elif "trend" in user_request.lower() or "趋势" in user_request:
        monthly = df.groupby(df["date"].dt.to_period("M"))["revenue"].sum()
        print(f"    {monthly.tail(6).to_string()}")
    elif "region" in user_request.lower() or "区域" in user_request:
        result = df.groupby("region")["revenue"].sum().sort_values(ascending=False)
        print(f"    {result.to_string()}")
    else:
        print(f"    {df.describe().to_string()[:200]}...")

    # Step 4: 生成报告
    print(f"  [Agent] Step 4: Generating report...")
    total = df["revenue"].sum()
    avg = df["revenue"].mean()
    best_product = df.groupby("product")["revenue"].sum().idxmax()
    print(f"    Total revenue: {total:,.0f}")
    print(f"    Average per transaction: {avg:,.0f}")
    print(f"    Best product: {best_product}")
    print(f"  [Agent] Analysis complete!")


simulate_agent_analysis("Compare revenue by product", df)
simulate_agent_analysis("Show monthly revenue trend", df)
simulate_agent_analysis("Analyze revenue by region", df)

# 清理
import os
if os.path.exists(csv_path):
    os.remove(csv_path)

print("\n" + "=" * 55)
print("Day 39 Summary")
print("=" * 55)
print("""
  pandas for Agent:
  1. Data loading: pd.read_csv()
  2. Overview: df.head(), df.describe(), df.dtypes
  3. Group analysis: df.groupby().agg()
  4. Filtering: df[df['col'] >= value]
  5. Time trends: df.resample().sum()
  6. Wrap as tool functions for Agent to call

  Tomorrow: Day 40 - Chart generation (matplotlib)
""")
