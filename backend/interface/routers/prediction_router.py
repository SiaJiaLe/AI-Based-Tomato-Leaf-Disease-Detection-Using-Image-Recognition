"""Prediction API router."""
from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from application.dtos.prediction_response import PredictionResponse
from application.use_cases.get_prediction_history import GetPredictionHistoryUseCase
from application.use_cases.predict_disease import PredictDiseaseUseCase
from domain.services.confidence_policy import ConfidencePolicy
from domain.services.disease_service import DiseaseService
from infrastructure.ml.lesion_segmenter import LesionSegmenter
from infrastructure.ml.resnet34_inferencer import ResNet34Inferencer
from infrastructure.persistence.database import get_db_session
from infrastructure.persistence.postgres_prediction_repo import PostgresPredictionRepository
from infrastructure.persistence.postgres_treatment_repo import PostgresTreatmentRepository
from infrastructure.storage.image_store import ImageStore

router = APIRouter()

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/jpg"}


def _get_inferencer(request: Request) -> ResNet34Inferencer:
    return request.app.state.inferencer


async def _get_predict_use_case(
    request: Request,
    db: AsyncSession = Depends(get_db_session),
) -> PredictDiseaseUseCase:
    return PredictDiseaseUseCase(
        inferencer=_get_inferencer(request),
        segmenter=LesionSegmenter(),
        disease_service=DiseaseService(),
        confidence_policy=ConfidencePolicy(),
        prediction_repo=PostgresPredictionRepository(db),
        treatment_repo=PostgresTreatmentRepository(db),
        image_store=ImageStore(),
    )


async def _get_history_use_case(
    db: AsyncSession = Depends(get_db_session),
) -> GetPredictionHistoryUseCase:
    return GetPredictionHistoryUseCase(PostgresPredictionRepository(db))


@router.post("/predict", response_model=PredictionResponse)
async def predict_disease(
    image: UploadFile = File(...),
    use_case: PredictDiseaseUseCase = Depends(_get_predict_use_case),
):
    """POST /api/v1/predict — upload a tomato leaf image and get a disease prediction."""
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"File must be an image (jpeg/png/webp). Got: {image.content_type}",
        )
    image_bytes = await image.read()
    try:
        return await use_case.execute(image_bytes, filename=image.filename or "upload.jpg")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/predictions", response_model=list[PredictionResponse])
async def list_predictions(
    use_case: GetPredictionHistoryUseCase = Depends(_get_history_use_case),
):
    """GET /api/v1/predictions — list all past predictions, newest first."""
    return await use_case.list_all()


@router.get("/predictions/{prediction_id}", response_model=PredictionResponse)
async def get_prediction_by_id(
    prediction_id: str,
    use_case: GetPredictionHistoryUseCase = Depends(_get_history_use_case),
):
    """GET /api/v1/predictions/{id} — fetch one prediction by UUID."""
    result = await use_case.get_by_id(prediction_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return result
