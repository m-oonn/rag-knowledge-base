from fastapi import APIRouter

router = APIRouter()

try:
    from celery.result import AsyncResult
    from src.worker.celery_app import celery_app
    _has_celery = True
except Exception:
    _has_celery = False


@router.get("/{task_id}")
def get_task_status(task_id: str):
    if not _has_celery:
        return {
            "task_id": task_id,
            "status": "unavailable",
            "result": "Celery not available (lightweight mode - documents are processed synchronously)",
        }
    task_result = AsyncResult(task_id, app=celery_app)
    return {
        "task_id": task_id,
        "status": task_result.status,
        "result": task_result.result,
    }
