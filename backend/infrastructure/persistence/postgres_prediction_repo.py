"""PostgreSQL prediction repository — concrete implementation."""
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from domain.entities.prediction import Prediction
from domain.repositories.prediction_repository import PredictionRepository
from infrastructure.persistence.models import PredictionRecord


class PostgresPredictionRepository(PredictionRepository):
    def __init__(self, session: AsyncSession):
        self._session = session

    async def save(self, prediction: Prediction) -> Prediction:
        record = PredictionRecord(
            id=prediction.prediction_id,
            label=prediction.label,
            confidence=prediction.confidence,
            advice=prediction.advice,
            image_path=prediction.image_path,
            severity_level=prediction.severity_level,
            affected_area_ratio=prediction.affected_area_ratio,
            is_low_confidence=prediction.is_low_confidence,
            certainty_label=prediction.certainty_label,
            session_id=prediction.session_id,
        )
        self._session.add(record)
        await self._session.commit()
        await self._session.refresh(record)
        return self._to_entity(record)

    async def list_all(self) -> List[Prediction]:
        result = await self._session.execute(
            select(PredictionRecord).order_by(PredictionRecord.created_at.desc())
        )
        return [self._to_entity(r) for r in result.scalars().all()]

    async def get_by_id(self, prediction_id: str) -> Optional[Prediction]:
        result = await self._session.execute(
            select(PredictionRecord).where(
                PredictionRecord.id == prediction_id
            )
        )
        record = result.scalar_one_or_none()
        return self._to_entity(record) if record else None

    @staticmethod
    def _to_entity(record: PredictionRecord) -> Prediction:
        return Prediction(
            prediction_id=str(record.id),
            label=record.label,
            confidence=record.confidence,
            advice=record.advice or "",
            image_path=record.image_path,
            severity_level=record.severity_level,
            affected_area_ratio=record.affected_area_ratio,
            is_low_confidence=bool(record.is_low_confidence),
            certainty_label=record.certainty_label or "high_certainty",
            session_id=str(record.session_id) if record.session_id else None,
            timestamp=record.created_at,
        )
