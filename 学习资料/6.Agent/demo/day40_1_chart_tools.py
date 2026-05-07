"""
Day 40 Demo：图表生成工具
运行方式：python day40_1_chart_tools.py
前置条件：pip install matplotlib pandas numpy

学习目标：
1. matplotlib 基础图表（折线/柱状/饼图）
2. 中文字体配置
3. 封装为 Agent 工具函数
4. 从数据到图表的完整流程
"""

import os
import pandas as pd
import numpy as np

# ============================================================
# Part 1：matplotlib 基础 + 中文配置
# ============================================================

print("=" * 55)
print("Part 1: matplotlib Setup")
print("=" * 55)

import matplotlib
matplotlib.use('Agg')  # 非交互后端，不弹窗
import matplotlib.pyplot as plt

# 中文字体配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

print("\n  [OK] matplotlib configured (Agg backend, Chinese fonts)")

# ============================================================
# Part 2：创建示例数据 + 各种图表
# ============================================================

print("\n" + "=" * 55)
print("Part 2: Generate Charts")
print("=" * 55)

# 示例数据
np.random.seed(42)
months = ['1月', '2月', '3月', '4月', '5月', '6月']
revenue = [45000, 52000, 48000, 61000, 58000, 65000]
costs = [32000, 38000, 35000, 42000, 40000, 44000]
products = ['Product A', 'Product B', 'Product C', 'Product D']
product_revenue = [120000, 95000, 78000, 55000]
regions = ['East', 'West', 'North', 'South']
region_share = [35, 28, 22, 15]

output_dir = "agent_charts"
os.makedirs(output_dir, exist_ok=True)

# --- 折线图：月度趋势 ---
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(months, revenue, 'b-o', label='Revenue', linewidth=2)
ax.plot(months, costs, 'r--s', label='Cost', linewidth=2)
ax.fill_between(range(len(months)), costs, revenue, alpha=0.1, color='green')
ax.set_title('Monthly Revenue vs Cost Trend', fontsize=14)
ax.set_xlabel('Month')
ax.set_ylabel('Amount')
ax.legend()
ax.grid(True, alpha=0.3)
path1 = os.path.join(output_dir, "line_trend.png")
plt.savefig(path1, dpi=100, bbox_inches='tight')
plt.close()
print(f"\n  [OK] Line chart: {path1}")

# --- 柱状图：产品对比 ---
fig, ax = plt.subplots(figsize=(8, 5))
colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336']
bars = ax.bar(products, product_revenue, color=colors)
ax.set_title('Revenue by Product', fontsize=14)
ax.set_ylabel('Revenue')
for bar, val in zip(bars, product_revenue):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
            f'{val:,}', ha='center', fontsize=10)
path2 = os.path.join(output_dir, "bar_comparison.png")
plt.savefig(path2, dpi=100, bbox_inches='tight')
plt.close()
print(f"  [OK] Bar chart: {path2}")

# --- 饼图：区域占比 ---
fig, ax = plt.subplots(figsize=(8, 8))
ax.pie(region_share, labels=regions, autopct='%1.1f%%',
       colors=['#2196F3', '#4CAF50', '#FF9800', '#F44336'],
       startangle=90, explode=(0.05, 0, 0, 0))
ax.set_title('Revenue Distribution by Region', fontsize=14)
path3 = os.path.join(output_dir, "pie_distribution.png")
plt.savefig(path3, dpi=100, bbox_inches='tight')
plt.close()
print(f"  [OK] Pie chart: {path3}")

# --- 组合图：双Y轴 ---
fig, ax1 = plt.subplots(figsize=(10, 6))
ax2 = ax1.twinx()
ax1.bar(months, revenue, alpha=0.7, color='steelblue', label='Revenue')
profit = [r - c for r, c in zip(revenue, costs)]
ax2.plot(months, profit, 'r-o', linewidth=2, label='Profit')
ax1.set_xlabel('Month')
ax1.set_ylabel('Revenue', color='steelblue')
ax2.set_ylabel('Profit', color='red')
ax1.legend(loc='upper left')
ax2.legend(loc='upper right')
ax1.set_title('Revenue and Profit Overview', fontsize=14)
path4 = os.path.join(output_dir, "combo_chart.png")
plt.savefig(path4, dpi=100, bbox_inches='tight')
plt.close()
print(f"  [OK] Combo chart: {path4}")

# ============================================================
# Part 3：封装为 Agent 图表工具
# ============================================================

print("\n" + "=" * 55)
print("Part 3: Agent Chart Tool")
print("=" * 55)


def create_chart(
    data: dict,
    chart_type: str,
    title: str,
    x_label: str = "",
    y_label: str = "",
    save_path: str = "chart.png",
) -> str:
    """
    Agent 图表工具：根据参数自动生成图表。

    参数：
      data: {"labels": [...], "values": [...], "values2": [...](可选)}
      chart_type: "line" / "bar" / "pie" / "scatter"
      title: 图表标题
      save_path: 保存路径

    返回：保存的文件路径
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    labels = data.get("labels", [])
    values = data.get("values", [])
    values2 = data.get("values2", None)

    if chart_type == "line":
        ax.plot(labels, values, 'b-o', linewidth=2, label="Series 1")
        if values2:
            ax.plot(labels, values2, 'r--s', linewidth=2, label="Series 2")
            ax.legend()
    elif chart_type == "bar":
        ax.bar(labels, values, color='steelblue')
    elif chart_type == "pie":
        ax.pie(values, labels=labels, autopct='%1.1f%%', startangle=90)
    elif chart_type == "scatter":
        ax.scatter(values, values2 or values, alpha=0.6)

    ax.set_title(title, fontsize=14)
    if x_label:
        ax.set_xlabel(x_label)
    if y_label:
        ax.set_ylabel(y_label)
    if chart_type != "pie":
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()
    return save_path


# 测试 Agent 工具
print("\n  Testing Agent chart tool:")

path = create_chart(
    data={"labels": months, "values": revenue, "values2": costs},
    chart_type="line",
    title="Agent Generated: Revenue Trend",
    x_label="Month", y_label="Amount",
    save_path=os.path.join(output_dir, "agent_line.png"),
)
print(f"  [OK] Agent created: {path}")

path = create_chart(
    data={"labels": products, "values": product_revenue},
    chart_type="bar",
    title="Agent Generated: Product Comparison",
    save_path=os.path.join(output_dir, "agent_bar.png"),
)
print(f"  [OK] Agent created: {path}")

# ============================================================
# Part 4：从 DataFrame 自动生成图表
# ============================================================

print("\n" + "=" * 55)
print("Part 4: Auto-chart from DataFrame")
print("=" * 55)


def auto_chart(df: pd.DataFrame, x_col: str, y_col: str, chart_type: str = "auto", save_dir: str = ".") -> str:
    """
    自动分析数据类型并选择最佳图表。

    Agent 调用这个函数时只需要指定列名，
    函数自动判断用什么图表类型。
    """
    if chart_type == "auto":
        if df[x_col].dtype in ["object", "category"]:
            unique = df[x_col].nunique()
            chart_type = "pie" if unique <= 6 else "bar"
        elif pd.api.types.is_datetime64_any_dtype(df[x_col]):
            chart_type = "line"
        else:
            chart_type = "scatter"

    agg_data = df.groupby(x_col)[y_col].sum()

    path = os.path.join(save_dir, f"auto_{chart_type}_{y_col}.png")
    create_chart(
        data={"labels": [str(x) for x in agg_data.index], "values": agg_data.values.tolist()},
        chart_type=chart_type,
        title=f"{y_col} by {x_col}",
        save_path=path,
    )
    return path


# 测试
np.random.seed(42)
df = pd.DataFrame({
    "product": np.random.choice(["A", "B", "C", "D"], 100),
    "region": np.random.choice(["East", "West", "North"], 100),
    "revenue": np.random.randint(1000, 10000, 100),
})

path = auto_chart(df, "product", "revenue", save_dir=output_dir)
print(f"\n  auto_chart(product, revenue) -> {path}")

path = auto_chart(df, "region", "revenue", save_dir=output_dir)
print(f"  auto_chart(region, revenue) -> {path}")

# 清理
import shutil
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
    print(f"\n  [OK] Cleaned up {output_dir}/")

print("\n" + "=" * 55)
print("Day 40 Summary")
print("=" * 55)
print("""
  Chart tools for Agent:
  1. matplotlib Agg backend (no GUI, save to file)
  2. Chinese font config (SimHei/YaHei)
  3. 4 chart types: line, bar, pie, combo
  4. create_chart() tool function
  5. auto_chart() smart type selection from DataFrame

  Tomorrow: Day 41 - File tools (read CSV/Excel/JSON)
""")
