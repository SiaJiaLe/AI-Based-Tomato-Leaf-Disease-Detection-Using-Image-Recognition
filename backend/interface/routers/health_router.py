"""Health check router."""
from fastapi import APIRouter

from infrastructure.persistence.database import check_database_connection

router = APIRouter()


@router.get("/health")
async def health_check():
    """GET /api/v1/health — service, database, and model status."""
    db_ok = await check_database_connection()
    return {
        "status": "ok" if db_ok else "degraded",
        "database_connected": db_ok,
        "model_loaded": False,
    }
