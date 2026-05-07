import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.settings import settings
from src.database.sql_session import engine, Base
from src.utils.logger import logger

def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        version="1.0.0",
        description="RAG Knowledge Base Q&A System",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Ensure tables are created after all models are imported
    from src.database.models import (  # noqa: F811
        User, KnowledgeBase, KnowledgeDocument, DocumentChunk,
        GeneratedQAPair, ChatSession, ChatInteraction,
        EvaluationTask, EvaluationResult, EvaluationDatasetItem,
        Assistant, Agent,
    )
    Base.metadata.create_all(bind=engine)

    # ── Core routers (always available) ──────────────────────
    from src.api.routers import health, auth, knowledge_base, chat, evaluation

    app.include_router(health.router, prefix=settings.API_PREFIX + "/health", tags=["Health"])
    app.include_router(auth.router, prefix=settings.API_PREFIX + "/auth", tags=["Auth"])
    app.include_router(knowledge_base.router, prefix=settings.API_PREFIX + "/knowledge-bases", tags=["Knowledge Base"])
    app.include_router(chat.router, prefix=settings.API_PREFIX + "/chat", tags=["RAG Chat"])
    app.include_router(evaluation.router, prefix=settings.API_PREFIX + "/evaluations", tags=["Evaluation"])

    # ── Extended routers ─────────────────────────────────────
    for router_name, module_path, prefix_suffix, tag in [
        ("assistant", "src.api.routers.assistant", "/assistants", "Assistant"),
        ("agent", "src.api.routers.agent", "/agents", "Agent"),
        ("monitor", "src.api.routers.monitor", "/monitor", "Monitor"),
        ("query", "src.api.routers.query", "/query", "Query"),
        ("loadfile", "src.api.routers.loadfile", "/upload", "File Management"),
    ]:
        try:
            mod = __import__(module_path, fromlist=["router"])
            app.include_router(mod.router, prefix=settings.API_PREFIX + prefix_suffix, tags=[tag])
        except Exception as e:
            logger.warning(f"Router '{router_name}' skipped: {e}")

    # ── Optional: MinIO storage (only when enabled) ──────────
    if settings.ENABLE_MINIO:
        try:
            from src.api.routers import storage
            app.include_router(storage.router, prefix=settings.API_PREFIX + "/storage", tags=["MinIO Storage"])
        except Exception as e:
            logger.warning(f"MinIO storage router skipped: {e}")
    else:
        logger.info("MinIO disabled (ENABLE_MINIO=False)")

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Vector store: {settings.VECTOR_STORE}")
    logger.info(f"LLM provider: {settings.LLM_PROVIDER}")
    logger.info(f"Embedding:   {settings.EMBEDDING_PROVIDER}")
    logger.info(f"Rerank:      {settings.RERANK_PROVIDER}")
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
