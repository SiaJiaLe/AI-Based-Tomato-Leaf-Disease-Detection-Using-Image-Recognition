"""Prediction API router (scaffold — returns 501 until implemented)."""
from fastapi import APIRouter, File, HTTPException, UploadFile

router = APIRouter()
NOT_IMPLEMENTED = HTTPException(status_code=501, detail="Not implemented")


@router.post("/predict")
async def predict_disease(image: UploadFile = File(...)):
    """POST /api/v1/predict — upload image and get disease prediction."""
    raise NOT_IMPLEMENTED


@router.get("/predictions")
async def list_predictions():
    """GET /api/v1/predictions — list prediction history."""
    raise NOT_IMPLEMENTED


@router.get("/predictions/{prediction_id}")
async def get_prediction_by_id(prediction_id: str):
    """GET /api/v1/predictions/{prediction_id} — fetch one prediction."""
    raise NOT_IMPLEMENTED
