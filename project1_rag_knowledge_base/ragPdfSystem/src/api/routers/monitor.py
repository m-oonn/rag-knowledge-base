from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional, Any
from pydantic import BaseModel
from src.database.sql_session import get_db
from src.database.models import KnowledgeDocument, User, KnowledgeBase
from src.api.dependencies import get_current_user
from sqlalchemy import desc

router = APIRouter()

# Optional Celery imports
try:
    from celery.result import AsyncResult
    from src.worker.celery_app import celery_app
    _has_celery = True
except Exception:
    _has_celery = False


class QueueItem(BaseModel):
    task_id: Optional[str] = None
    filename: Optional[str] = None
    status: Optional[int] = None
    error_msg: Optional[str] = None

    class Config:
        from_attributes = True


@router.get("/queue", response_model=List[QueueItem])
def get_processing_queue(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Get documents currently being processed or recently queued."""
    docs = (
        db.query(KnowledgeDocument)
        .join(KnowledgeBase)
        .filter(KnowledgeBase.owner_id == current_user.id)
        .filter(KnowledgeDocument.status.in_([0, 1]))  # Uploading or Processing
        .order_by(desc(KnowledgeDocument.created_at))
        .limit(20)
        .all()
    )
    return [
        QueueItem(
            task_id=d.doc_uid,
            filename=d.filename,
            status=d.status,
            error_msg=d.error_msg,
        )
        for d in docs
    ]
