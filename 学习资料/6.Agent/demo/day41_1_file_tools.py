"""
Day 41 Demo：文件工具
运行方式：python day41_1_file_tools.py

学习目标：
1. 读取 CSV/Excel/JSON 文件
2. 自动编码检测
3. 文件信息预览工具
4. 写分析报告
"""

import os
import json
import tempfile
import pandas as pd
import numpy as np

# ============================================================
# Part 1：创建测试文件
# ============================================================

print("=" * 55)
print("Part 1: Create Test Files")
print("=" * 55)

test_dir = tempfile.mkdtemp(prefix="agent_files_")

# CSV
np.random.seed(42)
df = pd.DataFrame({
    "date": pd.date_range("2024-01-01", periods=50, freq="W"),
    "product": np.random.choice(["A", "B", "C"], 50),
    "revenue": np.random.randint(1000, 10000, 50),
    "quantity": np.random.randint(10, 200, 50),
})
csv_path = os.path.join(test_dir, "sales.csv")
df.to_csv(csv_path, index=False)

# JSON
json_data = {"config": {"model": "qwen2.5:7b", "temperature": 0.3}, "results": [{"query": "test", "score": 0.95}]}
json_path = os.path.join(test_dir, "config.json")
with open(json_path, "w") as f:
    json.dump(json_data, f, indent=2)

# Excel
excel_path = os.path.join(test_dir, "report.xlsx")
try:
    with pd.ExcelWriter(excel_path) as writer:
        df.head(20).to_excel(writer, sheet_name="Sales", index=False)
        df.groupby("product")["revenue"].sum().reset_index().to_excel(writer, sheet_name="Summary", index=False)
    has_excel = True
except Exception:
    has_excel = False

print(f"\n  Created test files in: {test_dir}")
print(f"    CSV: sales.csv ({len(df)} rows)")
print(f"    JSON: config.json")
print(f"    Excel: {'report.xlsx (2 sheets)' if has_excel else 'skipped (openpyxl not installed)'}")


# ============================================================
# Part 2：Agent 文件工具函数
# ============================================================

print("\n" + "=" * 55)
print("Part 2: Agent File Tools")
print("=" * 55)


def tool_read_csv(file_path: str, encoding: str = None) -> str:
    """Agent工具：读取CSV文件，自动检测编码"""
    encodings = [encoding] if encoding else ["utf-8", "gbk", "gb2312", "latin-1"]
    for enc in encodings:
        try:
            df = pd.read_csv(file_path, encoding=enc)
            return (f"[OK] Loaded CSV: {df.shape[0]} rows, {df.shape[1]} cols\n"
                    f"Columns: {list(df.columns)}\n"
                    f"Types:\n{df.dtypes.to_string()}\n\n"
                    f"First 5 rows:\n{df.head().to_string()}")
        except UnicodeDecodeError:
            continue
        except Exception as e:
            return f"[FAIL] {e}"
    return "[FAIL] Could not decode file with any encoding"


def tool_read_json(file_path: str) -> str:
    """Agent工具：读取JSON文件"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        preview = json.dumps(data, ensure_ascii=False, indent=2)
        if len(preview) > 500:
            preview = preview[:500] + "\n... (truncated)"
        return f"[OK] JSON loaded\n{preview}"
    except Exception as e:
        return f"[FAIL] {e}"


def tool_read_excel(file_path: str, sheet_name: str = None) -> str:
    """Agent工具：读取Excel文件"""
    try:
        xls = pd.ExcelFile(file_path)
        sheets = xls.sheet_names
        result = f"[OK] Excel loaded, sheets: {sheets}\n"
        target = sheet_name if sheet_name else sheets[0]
        df = pd.read_excel(file_path, sheet_name=target)
        result += f"\nSheet '{target}': {df.shape[0]} rows, {df.shape[1]} cols\n"
        result += f"Columns: {list(df.columns)}\n"
        result += f"First 5 rows:\n{df.head().to_string()}"
        return result
    except Exception as e:
        return f"[FAIL] {e}"


def tool_file_info(file_path: str) -> str:
    """Agent工具：获取文件基本信息"""
    if not os.path.exists(file_path):
        return f"[FAIL] File not found: {file_path}"
    size = os.path.getsize(file_path)
    ext = os.path.splitext(file_path)[1].lower()
    info = f"File: {os.path.basename(file_path)}\n"
    info += f"Size: {size:,} bytes ({size/1024:.1f} KB)\n"
    info += f"Type: {ext}\n"
    if ext == ".csv":
        df = pd.read_csv(file_path, nrows=0)
        info += f"Columns: {list(df.columns)}"
    return info


def tool_write_report(content: str, file_path: str) -> str:
    """Agent工具：写分析报告（Markdown格式）"""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"[OK] Report saved: {file_path} ({len(content)} chars)"
    except Exception as e:
        return f"[FAIL] {e}"


# 测试工具
print(f"\n  --- tool_file_info ---")
print(f"  {tool_file_info(csv_path)}")

print(f"\n  --- tool_read_csv ---")
result = tool_read_csv(csv_path)
print(f"  {result[:300]}...")

print(f"\n  --- tool_read_json ---")
print(f"  {tool_read_json(json_path)}")

if has_excel:
    print(f"\n  --- tool_read_excel ---")
    print(f"  {tool_read_excel(excel_path)[:300]}...")

# ============================================================
# Part 3：生成分析报告
# ============================================================

print("\n" + "=" * 55)
print("Part 3: Generate Analysis Report")
print("=" * 55)

df = pd.read_csv(csv_path)
total_rev = df["revenue"].sum()
avg_rev = df["revenue"].mean()
best_product = df.groupby("product")["revenue"].sum().idxmax()

report = f"""# Sales Analysis Report

## Data Overview
- Records: {len(df)}
- Date range: {df['date'].min()} to {df['date'].max()}
- Products: {', '.join(df['product'].unique())}

## Key Findings
- Total revenue: {total_rev:,}
- Average weekly revenue: {avg_rev:,.0f}
- Best performing product: {best_product}

## Revenue by Product
{df.groupby('product')['revenue'].agg(['sum', 'mean', 'count']).to_string()}

## Recommendation
Focus on {best_product} which generates the highest revenue.
"""

report_path = os.path.join(test_dir, "analysis_report.md")
print(f"  {tool_write_report(report, report_path)}")
print(f"\n  Report preview:")
for line in report.split("\n")[:10]:
    print(f"    {line}")

# 清理
import shutil
shutil.rmtree(test_dir)
print(f"\n  [OK] Cleaned up test files")

print("\n" + "=" * 55)
print("Day 41 Summary")
print("=" * 55)
print("""
  File tools for Agent:
  1. tool_read_csv: auto-detect encoding, preview data
  2. tool_read_json: parse nested structures
  3. tool_read_excel: multi-sheet support
  4. tool_file_info: quick file overview
  5. tool_write_report: save Markdown reports

  Tomorrow: Day 42 - Multi-tool coordination
""")
