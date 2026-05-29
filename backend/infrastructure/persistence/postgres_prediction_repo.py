"""PostgreSQL prediction repository (scaffold)."""
from domain.repositories.prediction_repository import PredictionRepository


class PostgresPredictionRepository(PredictionRepository):
    """Concrete repository using PostgreSQL — implement in Phase 7."""

    async def save(self, prediction):
        raise NotImplementedError

    async def list_all(self):
        raise NotImplementedError

    async def get_by_id(self, prediction_id):
        raise NotImplementedError
