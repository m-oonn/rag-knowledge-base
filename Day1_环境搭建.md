# Day 1（4月15日）：环境搭建

> 时间: 4小时
> 目标: 搭建完整开发环境，为后续75天学习做好准备
> 工作目录: `E:\projects\`

---

## 任务清单

### 1. Python环境检查（10分钟）

```bash
# 检查Python版本（需要3.10+）
python --version

# 如果没装Python，去 python.org 下载 Python 3.11（官方推荐版本）
```

**✅ 验收标准**: 终端输出 `Python 3.10` 或以上

---

### 2. 创建项目目录结构（15分钟）

在 `E:\projects\` 内执行：

```bash
# 进入工作目录
cd E:\projects

# 创建主目录
mkdir 学习资料
mkdir project1_rag_knowledge_base
mkdir project2_data_analysis_agent

# 创建学习资料子目录
mkdir 学习资料\0.Python基础\demo
mkdir 学习资料\1.Skills\demo
mkdir 学习资料\2.Ollama\demo
mkdir 学习资料\3.MCP\demo
mkdir 学习资料\4.RAG\demo
mkdir 学习资料\5.AgentSkills\demo
mkdir 学习资料\6.Agent\demo

# 创建项目模板目录
mkdir project1_rag_knowledge_base\app
mkdir project1_rag_knowledge_base\config
mkdir project1_rag_knowledge_base\logs

mkdir project2_data_analysis_agent\app
mkdir project2_data_analysis_agent\config
mkdir project2_data_analysis_agent\logs
```

**目录结构应该是这样**:
```
E:\projects\
├── 学习资料/
│   ├── 0.Python基础/
│   │   ├── 1.装饰器.md
│   │   ├── 2.异步编程.md
│   │   ├── 3.FastAPI基础.md
│   │   ├── 4.FastAPI进阶.md
│   │   └── demo/
│   ├── 1.Skills/
│   ├── 2.Ollama/
│   └── ...
├── project1_rag_knowledge_base/
│   ├── app/
│   ├── config/
│   └── logs/
├── project2_data_analysis_agent/
│   ├── app/
│   ├── config/
│   └── logs/
└── 学习进度.md
```

**✅ 验收标准**: 所有目录创建成功

---

### 3. 虚拟环境设置（15分钟）

```bash
# 进入工作目录
cd E:\projects

# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows 用户执行：
venv\Scripts\activate

# 验证激活成功（终端应该显示 (venv) 前缀）
python --version
pip --version
```

**✅ 验收标准**: 终端左边显示 `(venv)` 前缀

---

### 4. Git初始化（10分钟）

```bash
# 确保在工作目录，虚拟环境已激活
cd E:\projects
# venv\Scripts\activate  （如果没激活，先激活）

# 初始化git
git init

# 配置git用户（如果是首次使用）
git config --global user.name "Your Name"
git config --global user.email "your.email@example.com"

# 查看git状态
git status
```

**✅ 验收标准**: `git status` 显示 `On branch master (or main)`

---

### 5. 创建关键配置文件（20分钟）

#### 5.1 创建 `.env` 文件（存放API密钥）

在 `E:\projects\` 创建 `.env` 文件（用VS Code或任意文本编辑器创建）：

**文件名**: `.env` （注意前面有个点）

**内容**:
```
# Claude API
CLAUDE_API_KEY=your_claude_api_key_here

# DeepSeek API（OpenAI兼容）
DEEPSEEK_API_KEY=your_deepseek_api_key_here
DEEPSEEK_BASE_URL=https://api.deepseek.com/v1

# 通义千问 API（可选）
QWEN_API_KEY=your_qwen_api_key_here
QWEN_BASE_URL=https://dashscope.aliyuncs.com/api/v1

# Ollama本地模型
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen2.5:7b

# 项目配置
LOG_LEVEL=INFO
DEBUG=False
```

**⚠️ 重要**: 
- `.env` 文件包含API密钥，**永远不要上传到GitHub**
- 使用时用 `python-dotenv` 库读取

#### 5.2 创建 `.gitignore` 文件

在 `E:\projects\` 创建 `.gitignore` 文件，内容如下：

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# IDE
.vscode/
.idea/
*.swp
*.swo
*~
.DS_Store

# 环境变量
.env
.env.local
.env.*.local

# 数据和日志
logs/
*.log
data/
*.db
*.sqlite
chroma_data/

# 项目特定
project1_rag_knowledge_base/uploaded_files/
project1_rag_knowledge_base/vector_store/
project2_data_analysis_agent/uploaded_files/
project2_data_analysis_agent/analysis_results/
```

**✅ 验收标准**: `.env` 和 `.gitignore` 文件都创建成功

---

### 6. 创建 `requirements.txt`（初始化）（10分钟）

在 `E:\projects\` 创建 `requirements-base.txt` 文件，内容如下：

```
# 核心依赖
python-dotenv==1.0.0
requests==2.31.0

# AI / LLM框架（阶段1-2会扩展）
openai==1.3.0
langchain==0.1.0
langgraph==0.0.1

# Web框架
fastapi==0.104.0
uvicorn==0.24.0
pydantic==2.5.0

# 向量数据库（阶段3用）
chroma-db==0.4.0
sentence-transformers==2.2.2

# 数据处理（项目2用）
pandas==2.1.0
numpy==1.24.0
matplotlib==3.8.0
plotly==5.17.0
openpyxl==3.11.0

# 前端（项目1-2用）
streamlit==1.28.0
gradio==4.8.0

# 工具
jupyter==1.0.0
pytest==7.4.0
black==23.11.0
```

然后在终端执行：

```bash
# 确保虚拟环境已激活
cd E:\projects
venv\Scripts\activate

# 安装基础依赖
pip install -r requirements-base.txt

# 验证安装成功
python -c "import fastapi; import langchain; print('✅ 安装成功')"
```

**✅ 验收标准**: 所有包都安装成功，没有错误

---

### 7. 创建 `学习进度.md` 文件（10分钟）

在 `E:\projects\` 创建 `学习进度.md` 文件：

```markdown
# 学习进度追踪

## 基本信息
- **开始日期**: 2026/04/15
- **目标完成日期**: 2026/06/28
- **当前Day**: Day 1
- **工作目录**: E:\projects\

## 阶段0：Python后端必备基础（Days 1-7）

### Day 1 - 2026/04/15（环境搭建）
- [x] Python环境检查
- [x] 项目目录结构创建
- [x] 虚拟环境配置
- [x] Git初始化
- [x] .env和.gitignore配置
- [x] requirements.txt创建和包安装
- **今日收获**: 搭建好了完整的Python开发环境，所有依赖都安装成功
- **遇到的问题**: 无
- **输出文件**: 
  - E:\projects\venv\（虚拟环境）
  - .env（API配置）
  - .gitignore（Git忽略文件）
  - requirements-base.txt

### Day 2 - 2026/04/16（Python装饰器）
- [ ] 生成装饰器.md理论文档
- [ ] 完成3个decorator_demo.py
- **今日收获**: 
- **遇到的问题**: 
- **输出文件**: 

### Day 3 - 2026/04/17（Python异步编程）
- [ ] ...

---

## 阶段1：AI基础技能（Days 8-11）
...

---

## 关键日期提醒
- **5月9日（Day 25）**: 开始项目一 + 准备简历
- **5月22日（Day 38）**: 项目一完成 + 投简历
- **6月6日（Day 53）**: 开始项目二
- **6月19日（Day 66）**: 项目二完成 + 更新简历
- **6月28日（Day 75）**: 全部完成！
```

**✅ 验收标准**: 学习进度.md创建成功，能够跟踪学习

---

### 8. 第一次Git提交（10分钟）

```bash
# 确保在工作目录，虚拟环境已激活
cd E:\projects

# 添加所有文件到git
git add .

# 查看将要提交的文件
git status

# 第一次提交
git commit -m "初始化：环境搭建完成

- Python 3.11虚拟环境
- 项目目录结构
- .env配置文件
- 基础依赖安装（FastAPI + LangChain + Chroma等）
- Git初始化"

# 查看git日志
git log
```

**✅ 验收标准**: `git log` 显示第一次提交记录

---

## 最后检查清单（60分钟总耗时）

完成Day 1前，确保以下都打✅：

- [ ] Python 3.10+ 已安装
- [ ] 虚拟环境已创建并激活（`(venv)` 显示在终端）
- [ ] 项目目录结构已创建在 `E:\projects\`
- [ ] `.env` 文件已创建（准备好API Key位置）
- [ ] `.gitignore` 文件已创建
- [ ] `requirements-base.txt` 已创建，所有包都装好了
- [ ] `学习进度.md` 已创建
- [ ] Git初始化完成，第一次提交成功
- [ ] 能在终端执行 `python -c "import fastapi; import langchain; print('✅')"` 不报错

---

## 🎉 Day 1完成！

**明天的任务** (Day 2, 4月16日)：
- 学习Python装饰器（decorator）
- 让AI生成装饰器理论讲解文档（放在 `E:\projects\学习资料\0.Python基础\1.装饰器.md`）
- 跑通3个装饰器Demo（放在 `E:\projects\学习资料\0.Python基础\demo\` 目录）
- 告诉AI "更新学习进度"

**每天早上激活虚拟环境**：

```bash
cd E:\projects
venv\Scripts\activate
```

准备好了吗？明天开始学Python装饰器！ 💪

---

## 问题排查

### 问题1：pip install 太慢或超时
```bash
# 使用阿里云镜像加速
pip install -r requirements-base.txt -i https://mirrors.aliyun.com/pypi/simple/
```

### 问题2：某个包装不上
```bash
# 可以先跳过，继续装其他包，后面再装
pip install [package-name] --ignore-installed
```

### 问题3：虚拟环境激活不了
```bash
# 确保在正确的目录，重新创建虚拟环境
cd E:\projects
rmdir venv /s  # 删除旧的
python -m venv venv  # 重新创建
venv\Scripts\activate  # 激活
```

有任何问题，直接问我！
