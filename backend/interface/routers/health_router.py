"""Health check router (scaffold)."""
from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health_check():
    """GET /api/v1/health — service and model status."""
    return {"status": "ok", "model_loaded": False}
