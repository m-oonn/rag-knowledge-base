from fastapi import APIRouter

router = APIRouter()

# Optional Celery import
try:
    from src.worker.celery_app import celery_app  # noqa: F401
    _has_celery = True
except Exception:
    _has_celery = False


@router.get("")
async def health_check():
    """Basic health check (always available)."""
    return {"status": "ok", "version": "1.0.0", "celery": _has_celery}


@router.get("/queues")
async def get_queue_stats():
    """Queue stats (requires Celery)."""
    if not _has_celery:
        return {"status": "degraded", "message": "Celery not available (lightweight mode)"}

    try:
        from src.worker.celery_app import celery_app
        i = celery_app.control.inspect()
        if not i:
            return {"error": "Could not connect to Celery inspector"}

        active = i.active() or {}
        reserved = i.reserved() or {}
        scheduled = i.scheduled() or {}
        stats = i.stats() or {}

        return {
            "status": "ok",
            "summary": {
                "active_tasks": sum(len(tasks) for tasks in active.values()),
                "reserved_tasks": sum(len(tasks) for tasks in reserved.values()),
                "scheduled_tasks": sum(len(tasks) for tasks in scheduled.values()),
            },
            "details": {
                "active": active,
                "reserved": reserved,
                "scheduled": scheduled,
                "worker_stats": stats,
            },
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
