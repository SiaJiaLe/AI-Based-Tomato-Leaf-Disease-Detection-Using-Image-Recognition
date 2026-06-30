"""Treatment repository port (abstract interface)."""
from abc import ABC, abstractmethod
from typing import List

from domain.entities.treatment_option import TreatmentOption
from domain.entities.treatment_log import TreatmentLog


class TreatmentRepository(ABC):
    @abstractmethod
    async def get_for_disease(self, disease_label: str) -> List[TreatmentOption]:
        pass

    @abstractmethod
    async def save_log(self, log: TreatmentLog) -> TreatmentLog:
        pass

    @abstractmethod
    async def update_log_outcome(self, log_id: str, outcome: str) -> None:
        pass
