"""Treatment router — treatment options and outcome logging endpoints."""
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession

from domain.entities.treatment_log import TreatmentLog
from infrastructure.persistence.database import get_db_session
from infrastructure.persistence.postgres_treatment_repo import PostgresTreatmentRepository

router = APIRouter()


class TreatmentLogCreateRequest(BaseModel):
    prediction_id: str
    treatment_option_id: str


class OutcomeUpdateRequest(BaseModel):
    outcome: str


@router.get("/treatments")
async def list_treatments(
    disease: str,
    db: AsyncSession = Depends(get_db_session),
):
    """GET /api/v1/treatments?disease={label} — list treatment options for a disease."""
    repo = PostgresTreatmentRepository(db)
    options = await repo.get_for_disease(disease)
    return [
        {
            "id": t.id,
            "treatment_type": t.treatment_type,
            "product_name": t.product_name,
            "active_ingredient": t.active_ingredient,
            "application_method": t.application_method,
            "estimated_cost_myr": t.estimated_cost_myr,
        }
        for t in options
    ]


@router.post("/treatment-logs")
async def log_treatment(
    body: TreatmentLogCreateRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """POST /api/v1/treatment-logs — record that a treatment was applied."""
    repo = PostgresTreatmentRepository(db)
    log = TreatmentLog(
        id=str(uuid.uuid4()),
        prediction_id=body.prediction_id,
        treatment_option_id=body.treatment_option_id,
    )
    saved = await repo.save_log(log)
    return {"log_id": saved.id, "outcome": saved.outcome}


@router.patch("/treatment-logs/{log_id}/outcome")
async def update_treatment_outcome(
    log_id: str,
    body: OutcomeUpdateRequest,
    db: AsyncSession = Depends(get_db_session),
):
    """PATCH /api/v1/treatment-logs/{id}/outcome — record follow-up outcome."""
    allowed = {"improved", "worsened", "no_change", "pending"}
    if body.outcome not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"outcome must be one of: {', '.join(allowed)}",
        )
    repo = PostgresTreatmentRepository(db)
    await repo.update_log_outcome(log_id, body.outcome)
    return {"log_id": log_id, "outcome": body.outcome}
