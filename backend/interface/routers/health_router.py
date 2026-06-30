"""Health check router."""
from fastapi import APIRouter, Request

from infrastructure.persistence.database import check_database_connection

router = APIRouter()


@router.get("/health")
async def health_check(request: Request):
    """GET /api/v1/health — reports database and model status."""
    db_ok = await check_database_connection()
    inferencer = getattr(request.app.state, "inferencer", None)
    model_loaded = inferencer is not None and inferencer.engine.is_loaded
    return {
        "status": "ok" if (db_ok and model_loaded) else "degraded",
        "database_connected": db_ok,
        "model_loaded": model_loaded,
    }
