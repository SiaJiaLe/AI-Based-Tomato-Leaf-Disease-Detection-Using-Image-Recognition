"""GetPredictionHistoryUseCase — retrieves past predictions."""
from typing import List, Optional

from application.dtos.prediction_response import PredictionResponse
from domain.repositories.prediction_repository import PredictionRepository


class GetPredictionHistoryUseCase:
    def __init__(self, prediction_repo: PredictionRepository):
        self._prediction_repo = prediction_repo

    async def list_all(self) -> List[PredictionResponse]:
        predictions = await self._prediction_repo.list_all()
        return [self._to_dto(p) for p in predictions]

    async def get_by_id(self, prediction_id: str) -> Optional[PredictionResponse]:
        prediction = await self._prediction_repo.get_by_id(prediction_id)
        if prediction is None:
            return None
        return self._to_dto(prediction)

    @staticmethod
    def _to_dto(p) -> PredictionResponse:
        return PredictionResponse(
            prediction_id=p.prediction_id,
            label=p.label,
            confidence=p.confidence,
            advice=p.advice,
            severity_level=p.severity_level,
            affected_area_ratio=p.affected_area_ratio,
            is_low_confidence=p.is_low_confidence,
            certainty_label=p.certainty_label,
            alternative_labels=[],
            treatment_options=[],
            timestamp=p.timestamp.isoformat() if p.timestamp else "",
        )
