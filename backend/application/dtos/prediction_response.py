"""Prediction response DTO (scaffold)."""
from pydantic import BaseModel


class PredictionResponse(BaseModel):
  prediction_id: str
  label: str
  confidence: float
  advice: str
  timestamp: str
