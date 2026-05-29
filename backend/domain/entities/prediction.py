"""Prediction entity (scaffold)."""
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Prediction:
  prediction_id: str
  label: str
  confidence: float
  timestamp: datetime
