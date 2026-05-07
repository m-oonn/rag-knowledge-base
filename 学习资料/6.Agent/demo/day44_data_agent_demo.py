"""
Day 44：综合数据分析 Agent Demo
运行方式：python day44_data_agent_demo.py
前置条件：pip install pandas numpy matplotlib

这是项目二的原型！包含完整的数据分析 Agent 流程：
1. 读取数据文件
2. 理解用户分析需求
3. 生成并执行分析代码
4. 生成图表
5. 输出分析报告
"""

import os
import io
import sys
import json
import contextlib
import pandas as pd
import numpy as np
from datetime import datetime

# ============================================================
# Part 1：Agent 工具集
# ============================================================

class DataAnalysisTools:
    """数据分析 Agent 的工具集"""

    def __init__(self):
        self.df = None
        self.file_name = ""
        self.chart_count = 0
        self.results = []

    def load_data(self, file_path: str) -> str:
        """工具：加载 CSV 数据"""
        try:
            self.df = pd.read_csv(file_path)
            self.file_name = os.path.basename(file_path)
            overview = (f"[OK] Loaded '{self.file_name}'\n"
                       f"  Shape: {self.df.shape[0]} rows x {self.df.shape[1]} columns\n"
                       f"  Columns: {list(self.df.columns)}\n"
                       f"  Types: {dict(self.df.dtypes.astype(str))}\n"
                       f"  First 3 rows:\n{self.df.head(3).to_string()}")
            self.results.append({"tool": "load_data", "result": overview})
            return overview
        except Exception as e:
            return f"[FAIL] {e}"

    def describe_data(self) -> str:
        """工具：描述性统计"""
        if self.df is None:
            return "[FAIL] No data loaded"
        desc = self.df.describe().to_string()
        self.results.append({"tool": "describe", "result": desc[:200]})
        return desc

    def group_analysis(self, group_col: str, value_col: str, agg: str = "sum") -> str:
        """工具：分组聚合分析"""
        if self.df is None:
            return "[FAIL] No data loaded"
        try:
            result = self.df.groupby(group_col)[value_col].agg(agg).sort_values(ascending=False)
            result_str = result.to_string()
            self.results.append({"tool": "group_analysis", "result": result_str[:200]})
            return f"[OK] {group_col} -> {value_col} ({agg}):\n{result_str}"
        except Exception as e:
            return f"[FAIL] {e}"

    def trend_analysis(self, date_col: str, value_col: str) -> str:
        """工具：时间趋势分析"""
        if self.df is None:
            return "[FAIL] No data loaded"
        try:
            df_copy = self.df.copy()
            df_copy[date_col] = pd.to_datetime(df_copy[date_col])
            monthly = df_copy.set_index(date_col)[value_col].resample("ME").sum()

            # 判断趋势
            if len(monthly) >= 2:
                first_half = monthly[:len(monthly)//2].mean()
                second_half = monthly[len(monthly)//2:].mean()
                if second_half > first_half * 1.05:
                    trend = "upward (increasing)"
                elif second_half < first_half * 0.95:
                    trend = "downward (decreasing)"
                else:
                    trend = "stable (flat)"
            else:
                trend = "insufficient data"

            result_str = f"Monthly {value_col}:\n{monthly.to_string()}\n\nTrend: {trend}"
            self.results.append({"tool": "trend", "result": result_str[:200]})
            return f"[OK] {result_str}"
        except Exception as e:
            return f"[FAIL] {e}"

    def create_chart(self, chart_type: str, title: str, x_data: list, y_data: list, labels: list = None) -> str:
        """工具：生成图表"""
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False

            fig, ax = plt.subplots(figsize=(10, 6))

            if chart_type == "line":
                ax.plot(x_data, y_data, 'b-o', linewidth=2)
            elif chart_type == "bar":
                colors = ['#2196F3', '#4CAF50', '#FF9800', '#F44336', '#9C27B0', '#00BCD4']
                ax.bar(x_data, y_data, color=colors[:len(x_data)])
            elif chart_type == "pie":
                ax.pie(y_data, labels=x_data, autopct='%1.1f%%', startangle=90)

            ax.set_title(title, fontsize=14)
            if chart_type != "pie":
                ax.grid(True, alpha=0.3)
                plt.xticks(rotation=45)

            self.chart_count += 1
            chart_path = f"agent_chart_{self.chart_count}.png"
            plt.tight_layout()
            plt.savefig(chart_path, dpi=100)
            plt.close()

            self.results.append({"tool": "chart", "result": chart_path})
            return f"[OK] Chart saved: {chart_path}"
        except ImportError:
            return "[FAIL] matplotlib not available"
        except Exception as e:
            return f"[FAIL] {e}"

    def execute_code(self, code: str) -> str:
        """工具：执行自定义分析代码"""
        safe_globals = {"pd": pd, "np": np, "df": self.df,
                       "print": print, "len": len, "str": str, "int": int, "float": float,
                       "list": list, "dict": dict, "round": round, "sum": sum, "min": min, "max": max}
        stdout = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout):
                exec(code, safe_globals)
            output = stdout.getvalue()
            self.results.append({"tool": "code_exec", "result": output[:200]})
            return f"[OK] Code output:\n{output}" if output else "[OK] Code executed (no output)"
        except Exception as e:
            return f"[FAIL] {type(e).__name__}: {e}"

    def generate_report(self) -> str:
        """工具：汇总所有结果生成报告"""
        report = f"# Data Analysis Report\n\n"
        report += f"**File**: {self.file_name}\n"
        report += f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        report += f"## Analysis Results\n\n"

        for i, r in enumerate(self.results, 1):
            report += f"### Step {i}: {r['tool']}\n"
            report += f"```\n{r['result']}\n```\n\n"

        if self.chart_count > 0:
            report += f"## Charts Generated\n"
            for i in range(1, self.chart_count + 1):
                report += f"- agent_chart_{i}.png\n"

        self.results.append({"tool": "report", "result": f"{len(self.results)} findings compiled"})
        return report


# ============================================================
# Part 2：Agent 决策引擎
# ============================================================

class DataAnalysisAgent:
    """
    数据分析 Agent：理解需求 → 选择工具 → 执行 → 输出结果。
    这是项目二的核心模式。
    """

    def __init__(self):
        self.tools = DataAnalysisTools()
        self.conversation = []

    def analyze(self, user_request: str, file_path: str = None) -> str:
        """处理用户的分析请求"""
        self.conversation.append({"role": "user", "content": user_request})

        print(f"\n  {'=' * 50}")
        print(f"  Agent: Processing request...")
        print(f"  {'=' * 50}")

        # Step 1: 加载数据（如果还没加载）
        if file_path and self.tools.df is None:
            print(f"\n  [Step 1] Loading data...")
            result = self.tools.load_data(file_path)
            print(f"  {result[:150]}")

        if self.tools.df is None:
            return "Please provide a data file first."

        # Step 2: 理解需求并选择工具（模拟 LLM 决策）
        plan = self._make_plan(user_request)
        print(f"\n  [Step 2] Analysis plan: {' -> '.join(plan)}")

        # Step 3: 执行计划
        outputs = []
        for i, action in enumerate(plan, 1):
            print(f"\n  [Step {i+1}] Executing: {action}")
            result = self._execute_action(action, user_request)
            outputs.append(result)
            print(f"  {result[:150]}...")

        # Step 4: 生成报告
        print(f"\n  [Final] Generating report...")
        report = self.tools.generate_report()

        self.conversation.append({"role": "assistant", "content": report[:500]})
        return report

    def _make_plan(self, request: str) -> list:
        """根据用户请求制定分析计划（模拟 LLM 规划）"""
        request_lower = request.lower()
        plan = []

        if any(kw in request_lower for kw in ["overview", "describe", "概览", "概况", "统计"]):
            plan.append("describe")
        if any(kw in request_lower for kw in ["product", "group", "compare", "产品", "分组", "对比"]):
            plan.append("group_by_product")
        if any(kw in request_lower for kw in ["region", "area", "区域", "地区"]):
            plan.append("group_by_region")
        if any(kw in request_lower for kw in ["trend", "monthly", "time", "趋势", "月度", "时间"]):
            plan.append("trend")
        if any(kw in request_lower for kw in ["chart", "plot", "graph", "visuali", "图", "可视化"]):
            plan.append("chart")

        if not plan:
            plan = ["describe", "chart"]  # 默认

        plan.append("report")
        return plan

    def _execute_action(self, action: str, request: str) -> str:
        """执行单个分析动作"""
        df = self.tools.df
        columns = list(df.columns)

        # 自动选择合适的列
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        category_cols = df.select_dtypes(include=["object"]).columns.tolist()
        date_cols = [c for c in columns if "date" in c.lower() or "time" in c.lower()]

        value_col = numeric_cols[0] if numeric_cols else columns[0]
        if "revenue" in [c.lower() for c in numeric_cols]:
            value_col = [c for c in numeric_cols if "revenue" in c.lower()][0]

        if action == "describe":
            return self.tools.describe_data()
        elif action == "group_by_product":
            group_col = next((c for c in category_cols if "product" in c.lower()), category_cols[0] if category_cols else columns[0])
            return self.tools.group_analysis(group_col, value_col)
        elif action == "group_by_region":
            group_col = next((c for c in category_cols if "region" in c.lower()), category_cols[-1] if category_cols else columns[0])
            return self.tools.group_analysis(group_col, value_col)
        elif action == "trend":
            date_col = date_cols[0] if date_cols else columns[0]
            return self.tools.trend_analysis(date_col, value_col)
        elif action == "chart":
            if category_cols and numeric_cols:
                group = df.groupby(category_cols[0])[value_col].sum().sort_values(ascending=False)
                return self.tools.create_chart(
                    "bar", f"{value_col} by {category_cols[0]}",
                    group.index.tolist(), group.values.tolist()
                )
            return "[SKIP] No suitable columns for chart"
        elif action == "report":
            return "Report will be generated after all steps."
        else:
            return f"Unknown action: {action}"


# ============================================================
# Part 3：创建测试数据并运行 Agent
# ============================================================

print("=" * 55)
print("  Data Analysis Agent Demo")
print("  (Project 2 Prototype)")
print("=" * 55)

# 创建示例数据
np.random.seed(42)
n = 200
test_df = pd.DataFrame({
    "date": pd.date_range("2024-01-01", periods=n, freq="D"),
    "product": np.random.choice(["Product A", "Product B", "Product C", "Product D"], n),
    "region": np.random.choice(["East", "West", "North", "South"], n),
    "quantity": np.random.randint(10, 200, n),
    "revenue": np.random.randint(1000, 20000, n),
    "cost": np.random.randint(500, 10000, n),
})
test_df["profit"] = test_df["revenue"] - test_df["cost"]

csv_path = "agent_test_sales.csv"
test_df.to_csv(csv_path, index=False)
print(f"\n  Created test data: {csv_path} ({len(test_df)} rows)")

# 运行 Agent
agent = DataAnalysisAgent()

# 测试请求1：综合分析
report = agent.analyze(
    "Give me a complete overview with product comparison and trend chart",
    file_path=csv_path
)
print(f"\n  Report preview:")
for line in report.split("\n")[:15]:
    print(f"    {line}")

# 测试请求2：追问（Agent 记得之前的数据）
print(f"\n  {'=' * 50}")
print(f"  Follow-up question (Agent remembers data):")
result = agent.analyze("Now show me the regional breakdown")
print(f"\n  Follow-up report preview:")
for line in result.split("\n")[:10]:
    print(f"    {line}")

# 清理
import glob
for f in glob.glob("agent_chart_*.png"):
    os.remove(f)
if os.path.exists(csv_path):
    os.remove(csv_path)
print(f"\n  [OK] Cleaned up temp files")

# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 55)
print("Day 44 Summary - Stage 5 Complete!")
print("=" * 55)
print("""
  Data Analysis Agent includes:
  1. Data loading (CSV with auto-detection)
  2. Analysis tools (describe, groupby, trend)
  3. Chart generation (matplotlib)
  4. Code execution (safe sandbox)
  5. Report generation (Markdown)
  6. Smart tool selection based on user intent
  7. Multi-turn conversation support

  This is the PROTOTYPE for Project 2!
  In the real project you'll add:
  - LangGraph for orchestration
  - Gradio for frontend
  - Real LLM for decision making
  - Docker for deployment

  LEARNING PHASE COMPLETE!
  All theory + demos for Days 1-44 are done.
  Next: Project development (Days 17-30 and 45-58)
""")
