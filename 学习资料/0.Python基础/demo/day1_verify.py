"""
Day 1 验证脚本：检查环境是否搭建成功
运行方式：python day1_verify.py

所有检查项都显示 [OK] 就说明环境没问题，可以开始 Day 2 的学习。
如果有 [FAIL]，按照提示修复后重新运行。
"""

import sys
import os
import io

# Windows 终端默认 GBK 编码，遇到 emoji 会报错
# 强制 stdout 使用 utf-8（解决 Windows 中文编码问题）
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def check(label: str, condition: bool, fix_hint: str = ""):
    """打印检查结果"""
    icon = "[OK]  " if condition else "[FAIL]"
    print(f"  {icon} {label}")
    if not condition and fix_hint:
        print(f"         -> {fix_hint}")
    return condition


def find_project_root():
    """
    向上查找项目根目录（包含 .git 或 .env 的目录）。
    比硬编码层数更可靠。
    """
    current = os.path.dirname(os.path.abspath(__file__))
    for _ in range(10):  # 最多往上找 10 层
        if os.path.exists(os.path.join(current, ".git")) or os.path.exists(os.path.join(current, ".env")):
            return current
        parent = os.path.dirname(current)
        if parent == current:  # 到达根目录
            break
        current = parent
    # 兜底：假设在 学习资料/0.Python基础/demo/ 下，往上4层
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def main():
    print()
    print("=" * 55)
    print("  Day 1 - Environment Verification")
    print("=" * 55)
    print()

    passed = 0
    total = 0

    # ============================================================
    # 1. Python 版本
    # ============================================================
    print("[Python]")
    ver = sys.version_info
    total += 1
    if check(
        f"Python {ver.major}.{ver.minor}.{ver.micro}",
        ver.major == 3 and ver.minor >= 10,
        "Need Python 3.10+, download from python.org"
    ):
        passed += 1

    # 虚拟环境
    total += 1
    in_venv = hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    )
    if check(
        "venv activated" if in_venv else "venv NOT activated",
        in_venv,
        r"Run: venv\Scripts\activate (Windows) or source venv/bin/activate (Linux/Mac)"
    ):
        passed += 1

    print()

    # ============================================================
    # 2. 核心依赖
    # ============================================================
    print("[Core packages]")

    packages = {
        "fastapi": "pip install fastapi",
        "uvicorn": "pip install uvicorn",
        "pydantic": "pip install pydantic",
        "python-dotenv": "pip install python-dotenv",
        "requests": "pip install requests",
        "httpx": "pip install httpx",
    }

    # 特殊处理导入名和包名不同的情况
    import_map = {
        "python-dotenv": "dotenv",
    }

    for name, install_cmd in packages.items():
        total += 1
        import_name = import_map.get(name, name)
        try:
            __import__(import_name)
            if check(f"{name}", True):
                passed += 1
        except ImportError:
            check(f"{name}", False, install_cmd)

    print()

    # ============================================================
    # 3. AI 相关依赖
    # ============================================================
    print("[AI packages]")

    ai_packages = {
        "anthropic": "pip install anthropic",
        "openai": "pip install openai",
        "langchain": "pip install langchain",
    }

    for name, install_cmd in ai_packages.items():
        total += 1
        try:
            __import__(name)
            if check(f"{name}", True):
                passed += 1
        except ImportError:
            check(f"{name}", False, install_cmd)

    print()

    # ============================================================
    # 4. 数据处理依赖
    # ============================================================
    print("[Data packages]")

    data_packages = {
        "numpy": "pip install numpy",
        "pandas": "pip install pandas --only-binary=:all:",
    }

    for name, install_cmd in data_packages.items():
        total += 1
        try:
            __import__(name)
            if check(f"{name}", True):
                passed += 1
        except ImportError:
            check(f"{name}", False, install_cmd)

    print()

    # ============================================================
    # 5. 向量数据库
    # ============================================================
    print("[Vector DB]")

    total += 1
    try:
        import chromadb
        if check("chromadb", True):
            passed += 1
    except ImportError:
        check("chromadb", False, "pip install chromadb")

    print()

    # ============================================================
    # 6. 配置文件
    # ============================================================
    print("[Config files]")

    # 自动查找项目根目录
    project_root = find_project_root()
    print(f"  Project root: {project_root}")

    env_path = os.path.join(project_root, ".env")
    total += 1
    if check(
        ".env exists" if os.path.exists(env_path) else ".env NOT found",
        os.path.exists(env_path),
        f"Create .env in {project_root}"
    ):
        passed += 1

    gitignore_path = os.path.join(project_root, ".gitignore")
    total += 1
    if check(
        ".gitignore exists",
        os.path.exists(gitignore_path),
        f"Create .gitignore in {project_root}"
    ):
        passed += 1

    # 检查 .gitignore 是否包含 .env
    if os.path.exists(gitignore_path):
        total += 1
        with open(gitignore_path, "r", encoding="utf-8", errors="replace") as f:
            gitignore_content = f.read()
        if check(
            ".gitignore contains .env (prevents key leak)",
            ".env" in gitignore_content,
            "Add '.env' line to .gitignore"
        ):
            passed += 1

    print()

    # ============================================================
    # 7. Git
    # ============================================================
    print("[Git]")

    git_dir = os.path.join(project_root, ".git")
    total += 1
    if check(
        "Git repo initialized",
        os.path.isdir(git_dir),
        f"Run: git init (in {project_root})"
    ):
        passed += 1

    print()

    # ============================================================
    # 8. 目录结构
    # ============================================================
    print("[Directories]")

    expected_dirs = [
        os.path.join("project1_rag_knowledge_base"),
        os.path.join("project2_data_analysis_agent"),
    ]

    for dir_path in expected_dirs:
        full_path = os.path.join(project_root, dir_path)
        total += 1
        if check(
            f"{dir_path}/",
            os.path.isdir(full_path),
            f"mkdir {full_path}"
        ):
            passed += 1

    print()

    # ============================================================
    # 结果汇总
    # ============================================================
    print("=" * 55)
    if passed == total:
        print(f"  ALL PASSED! ({passed}/{total})")
        print(f"  Environment is ready. Start Day 2 tomorrow!")
    else:
        print(f"  PASSED {passed}/{total}, {total - passed} items need fixing")
        print(f"  Fix and re-run: python day1_verify.py")
    print("=" * 55)
    print()


if __name__ == "__main__":
    main()
