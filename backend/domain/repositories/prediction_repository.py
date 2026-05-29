"""Prediction repository port (scaffold)."""
from abc import ABC, abstractmethod
from typing import List, Optional

from domain.entities.prediction import Prediction


class PredictionRepository(ABC):
  @abstractmethod
  async def save(self, prediction: Prediction) -> None:
    pass

  @abstractmethod
  async def list_all(self) -> List[Prediction]:
    pass

  @abstractmethod
  async def get_by_id(self, prediction_id: str) -> Optional[Prediction]:
    pass
