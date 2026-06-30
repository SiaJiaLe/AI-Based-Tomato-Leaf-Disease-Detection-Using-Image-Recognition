"""PostgreSQL treatment repository — concrete implementation."""
import uuid
from datetime import datetime, timezone
from typing import List

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from domain.entities.treatment_log import TreatmentLog
from domain.entities.treatment_option import TreatmentOption
from domain.repositories.treatment_repository import TreatmentRepository
from infrastructure.persistence.models import TreatmentLogRecord, TreatmentOptionRecord


class PostgresTreatmentRepository(TreatmentRepository):
    def __init__(self, session: AsyncSession):
        self._session = session

    async def get_for_disease(self, disease_label: str) -> List[TreatmentOption]:
        result = await self._session.execute(
            select(TreatmentOptionRecord).where(
                TreatmentOptionRecord.disease_label == disease_label
            )
        )
        return [self._option_to_entity(r) for r in result.scalars().all()]

    async def save_log(self, log: TreatmentLog) -> TreatmentLog:
        record = TreatmentLogRecord(
            id=log.id,
            prediction_id=log.prediction_id,
            treatment_option_id=log.treatment_option_id,
            outcome=log.outcome,
        )
        self._session.add(record)
        await self._session.commit()
        await self._session.refresh(record)
        return self._log_to_entity(record)

    async def update_log_outcome(self, log_id: str, outcome: str) -> None:
        await self._session.execute(
            update(TreatmentLogRecord)
            .where(TreatmentLogRecord.id == log_id)
            .values(outcome=outcome, outcome_logged_at=datetime.now(timezone.utc))
        )
        await self._session.commit()

    @staticmethod
    def _option_to_entity(r: TreatmentOptionRecord) -> TreatmentOption:
        return TreatmentOption(
            id=str(r.id),
            disease_label=r.disease_label,
            treatment_type=r.treatment_type,
            product_name=r.product_name,
            active_ingredient=r.active_ingredient,
            application_method=r.application_method,
            estimated_cost_myr=float(r.estimated_cost_myr) if r.estimated_cost_myr else None,
            severity_min=r.severity_min,
            severity_max=r.severity_max,
        )

    @staticmethod
    def _log_to_entity(r: TreatmentLogRecord) -> TreatmentLog:
        return TreatmentLog(
            id=str(r.id),
            prediction_id=str(r.prediction_id),
            treatment_option_id=str(r.treatment_option_id),
            applied_at=r.applied_at,
            outcome=r.outcome,
            outcome_logged_at=r.outcome_logged_at,
        )
