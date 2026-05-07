# RAG Knowledge Base Q&A System

企业级 RAG 知识库问答系统 — 双模式部署，多模型切换，多格式文档解析，智能检索增强生成。

**当前状态**: ✅ 全链路跑通 (DeepSeek + Chroma + sentence-transformers)

## Architecture

```
User → FastAPI → RAG Service → LLM (Ollama/DeepSeek/Qwen)
                    │
                    ├── Vector Retriever → Chroma / Milvus
                    ├── Reranker → Local CrossEncoder / DashScope
                    ├── Embedding → Sentence-Transformers / DashScope
                    └── Document Pipeline → PDF/DOCX/MD/TXT → Chunks → Vectors
```

## Tech Stack

| Layer | Lightweight (default) | Enterprise |
|-------|----------------------|------------|
| LLM | Ollama / DeepSeek | Qwen-Max |
| Vector DB | ChromaDB | Milvus |
| Embedding | sentence-transformers | DashScope |
| Rerank | CrossEncoder | gte-rerank |
| Storage | Local filesystem | MinIO |
| Queue | Sync (in-process) | Celery + RabbitMQ |
| Cache | In-memory dict | Redis |
| Frontend | Streamlit | Vue 3 |

## Quick Start

### 1. Install

```bash
pip install -r requirements.txt
```

### 2. Configure `.env`

```env
# Pick your LLM
LLM_PROVIDER=deepseek       # or ollama, dashscope
DEEPSEEK_API_KEY=sk-xxx     # for DeepSeek

# HF mirror for China users
HF_ENDPOINT=https://hf-mirror.com
```

### 3. One-click demo (no server needed)

```bash
python demo.py
```

### 4. Start server

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
# API docs: http://localhost:8000/docs
# Streamlit: streamlit run streamlit_app.py
```

### 5. Docker (single container)

```bash
docker-compose -f docker/docker-compose.light.yml up -d
```

## API

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/v1/auth/register` | Register |
| POST | `/api/v1/auth/login/access-token` | Login |
| POST | `/api/v1/knowledge-bases/` | Create KB |
| POST | `/api/v1/knowledge-bases/{id}/upload` | Upload doc |
| POST | `/api/v1/chat/` | RAG query |
| GET | `/api/v1/chat/sessions` | History |
| POST | `/api/v1/evaluations/` | Evaluate |
| GET | `/api/v1/health/` | Health |

## Supported Formats

PDF / DOCX / Markdown / TXT / Excel / PowerPoint (enterprise)

## Docker (Enterprise)

```bash
docker-compose -f docker/docker-compose.yml up -d
```

## Features

- **Dual-mode**: Chroma + Ollama for dev, Milvus + Qwen-Max for prod
- **Multi-LLM**: Switch by changing one config line
- **Hybrid search**: BM25 (keyword) + Vector (semantic) + RRF fusion, CrossEncoder reranking
- **Multi-hop**: Auto question decomposition for complex queries
- **Memory**: Short-term conversation + long-term vector memory
- **Evaluation**: RAGAS evaluation with auto QA pair generation

## Project Structure

```
ragPdfSystem/
├── src/
│   ├── main.py              # FastAPI entry
│   ├── settings.py          # Config (from .env)
│   ├── api/routers/         # 12 routers
│   ├── services/            # RAG, Evaluation, Memory
│   ├── database/            # SQL + Chroma/Milvus
│   ├── retrieval/           # Retriever + Reranker
│   ├── embedding/           # Multi-provider
│   ├── llm/                 # Multi-model
│   └── processors/          # PDF, DOCX parsers
├── streamlit_app.py         # Streamlit UI
├── frontend/                # Vue 3 UI
├── docker/                  # Docker
└── requirements.txt
```
