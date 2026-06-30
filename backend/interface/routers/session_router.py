"""Diagnosis session router — multi-photo session endpoints."""
import io
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from application.dtos.prediction_response import PredictionResponse
from application.use_cases.predict_disease import PredictDiseaseUseCase
from domain.entities.diagnosis_session import DiagnosisSession
from domain.services.confidence_policy import ConfidencePolicy
from domain.services.disease_service import DiseaseService
from infrastructure.ml.lesion_segmenter import LesionSegmenter
from infrastructure.ml.resnet34_inferencer import ResNet34Inferencer
from infrastructure.persistence.database import get_db_session
from infrastructure.persistence.models import DiagnosisSessionRecord
from infrastructure.persistence.postgres_prediction_repo import PostgresPredictionRepository
from infrastructure.persistence.postgres_treatment_repo import PostgresTreatmentRepository
from infrastructure.storage.image_store import ImageStore

router = APIRouter()

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp", "image/jpg"}


class SessionCreateResponse(BaseModel):
    session_id: str
    prediction: PredictionResponse


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


@router.post("/diagnosis-sessions", response_model=SessionCreateResponse)
async def create_session(
    image: UploadFile = File(...),
    db: AsyncSession = Depends(get_db_session),
    use_case: PredictDiseaseUseCase = Depends(_get_predict_use_case),
):
    """POST /api/v1/diagnosis-sessions — start a new multi-photo session with the first image."""
    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="File must be an image.")

    session_id = str(uuid.uuid4())
    session_record = DiagnosisSessionRecord(id=session_id, photo_count=1)
    db.add(session_record)
    await db.commit()

    image_bytes = await image.read()
    prediction = await use_case.execute(
        image_bytes, filename=image.filename or "upload.jpg", session_id=session_id
    )

    # Update consolidated result
    session_record.consolidated_label = prediction.label
    session_record.consolidated_confidence = prediction.confidence
    await db.commit()

    return SessionCreateResponse(session_id=session_id, prediction=prediction)


@router.post("/diagnosis-sessions/{session_id}/photos", response_model=SessionCreateResponse)
async def add_photo_to_session(
    session_id: str,
    image: UploadFile = File(...),
    db: AsyncSession = Depends(get_db_session),
    use_case: PredictDiseaseUseCase = Depends(_get_predict_use_case),
):
    """POST /api/v1/diagnosis-sessions/{id}/photos — add another photo to an existing session."""
    from sqlalchemy import select, update

    result = await db.execute(
        select(DiagnosisSessionRecord).where(DiagnosisSessionRecord.id == session_id)
    )
    session_record = result.scalar_one_or_none()
    if session_record is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    if image.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="File must be an image.")

    image_bytes = await image.read()
    prediction = await use_case.execute(
        image_bytes, filename=image.filename or "upload.jpg", session_id=session_id
    )

    # Update photo count and consolidated result (latest prediction wins for simplicity)
    await db.execute(
        update(DiagnosisSessionRecord)
        .where(DiagnosisSessionRecord.id == session_id)
        .values(
            photo_count=DiagnosisSessionRecord.photo_count + 1,
            consolidated_label=prediction.label,
            consolidated_confidence=prediction.confidence,
        )
    )
    await db.commit()

    return SessionCreateResponse(session_id=session_id, prediction=prediction)


@router.get("/diagnosis-sessions/{session_id}")
async def get_session(
    session_id: str,
    db: AsyncSession = Depends(get_db_session),
):
    """GET /api/v1/diagnosis-sessions/{id} — get session summary."""
    from sqlalchemy import select

    result = await db.execute(
        select(DiagnosisSessionRecord).where(DiagnosisSessionRecord.id == session_id)
    )
    session = result.scalar_one_or_none()
    if session is None:
        raise HTTPException(status_code=404, detail="Session not found.")

    return {
        "session_id": str(session.id),
        "consolidated_label": session.consolidated_label,
        "consolidated_confidence": session.consolidated_confidence,
        "photo_count": session.photo_count,
        "created_at": session.created_at.isoformat() if session.created_at else None,
    }
