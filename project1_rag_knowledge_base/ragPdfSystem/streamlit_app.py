"""Streamlit frontend for RAG Knowledge Base Q&A System.

Run: streamlit run streamlit_app.py
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import streamlit as st
import requests

API_BASE = "http://localhost:8000/api/v1"

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Knowledge Base",
    page_icon="📚",
    layout="wide",
)

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.title("📚 RAG Knowledge Base")
st.sidebar.caption("企业级 RAG 知识库问答系统")

menu = st.sidebar.radio(
    "导航",
    ["💬 问答", "📁 文档管理", "📊 系统状态"],
)

# ── Helpers ──────────────────────────────────────────────────

def api_get(path: str, **kwargs):
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=10, **kwargs)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None

def api_post(path: str, **kwargs):
    try:
        r = requests.post(f"{API_BASE}{path}", timeout=30, **kwargs)
        return r.json() if r.status_code in (200, 201) else r.text
    except Exception as e:
        return f"Error: {e}"

def api_upload(path: str, files):
    try:
        r = requests.post(f"{API_BASE}{path}", files=files, timeout=60)
        return r.json() if r.status_code in (200, 201) else r.text
    except Exception as e:
        return f"Error: {e}"

# ── Q&A Page ─────────────────────────────────────────────────

if menu == "💬 问答":
    st.title("💬 RAG 智能问答")

    col1, col2 = st.columns([3, 1])
    with col2:
        top_k = st.slider("检索数量", 1, 20, 5)
        kb_id = st.number_input("知识库 ID (可选)", min_value=0, value=0, step=1)

    with col1:
        query = st.text_input("输入你的问题", placeholder="例如：李清照的意向岗位是什么？")

    if query:
        with st.spinner("检索中..."):
            payload = {"query": query, "top_k": top_k}
            if kb_id > 0:
                payload["kb_ids"] = [kb_id]

            try:
                resp = requests.post(
                    f"{API_BASE}/chat/query",
                    json=payload,
                    timeout=60,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    st.markdown("### 回答")
                    st.markdown(data.get("answer", "(无回答)"))

                    with st.expander("📎 引用来源"):
                        for doc in data.get("source_documents", []):
                            st.markdown(f"""
                            **相似度**: {doc.get('score', 0):.4f} | **ID**: {doc.get('id', '')}
                            > {doc.get('text', '')[:300]}...
                            ---
                            """)
                else:
                    st.error(f"请求失败: {resp.text}")
            except Exception as e:
                st.error(f"连接失败 - 请确认后端已启动: {e}")

# ── Document Management ─────────────────────────────────────

elif menu == "📁 文档管理":
    st.title("📁 文档管理")

    tab1, tab2 = st.tabs(["📤 上传文档", "📋 文档列表"])

    with tab1:
        uploaded_file = st.file_uploader(
            "选择文件", type=["pdf", "docx", "txt", "md"], help="支持 PDF、Word、TXT、Markdown"
        )
        kb_name = st.text_input("知识库名称", "default")

        if uploaded_file and st.button("上传并处理"):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            with st.spinner("上传中..."):
                result = api_upload(f"/knowledge-bases/upload?kb_name={kb_name}", files)
            if isinstance(result, dict):
                st.success(f"✅ 上传成功! 文档 ID: {result.get('doc_id', 'N/A')}")
            else:
                st.error(result)

    with tab2:
        if st.button("刷新列表"):
            st.rerun()
        kb_list = api_get("/knowledge-bases/")
        if kb_list:
            for kb in kb_list:
                st.markdown(f"**{kb.get('name', 'N/A')}** — {kb.get('description', '')}")
                docs = api_get(f"/knowledge-bases/{kb.get('id')}/documents")
                if docs:
                    for doc in docs:
                        st.text(f"  {doc.get('filename')} | chunks: {doc.get('chunk_count', 0)} | status: {doc.get('status')}")
        else:
            st.info("暂无文档，请先上传")

# ── System Status ────────────────────────────────────────────

elif menu == "📊 系统状态":
    st.title("📊 系统状态")

    import subprocess
    from src.settings import settings

    st.markdown("### 配置信息")
    st.json({
        "Vector Store": settings.VECTOR_STORE,
        "LLM Provider": settings.LLM_PROVIDER,
        "Embedding": settings.EMBEDDING_PROVIDER,
        "Rerank": settings.RERANK_PROVIDER,
        "Chunk Size": settings.CHUNK_SIZE,
        "Top K": settings.TOP_K,
    })

    health = api_get("/health/")
    if health:
        st.success(f"✅ 后端服务正常 — {health}")
    else:
        st.error("❌ 后端服务未启动")

    # LLM check
    try:
        if settings.LLM_PROVIDER == "ollama":
            r = requests.get(f"{settings.OLLAMA_HOST}/api/tags", timeout=5)
            if r.status_code == 200:
                models = [m["name"] for m in r.json().get("models", [])]
                st.success(f"🤖 Ollama 已连接，可用模型: {models}")
            else:
                st.warning("⚠️ Ollama 未响应")
    except Exception:
        st.warning("⚠️ Ollama 未连接")
