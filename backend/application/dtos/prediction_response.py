"""Prediction response DTO — returned by all prediction endpoints."""
from typing import List, Optional
from pydantic import BaseModel


class TreatmentOptionDTO(BaseModel):
    id: str
    treatment_type: str
    product_name: str
    active_ingredient: Optional[str] = None
    application_method: Optional[str] = None
    estimated_cost_myr: Optional[float] = None


class AlternativeLabelDTO(BaseModel):
    label: str
    confidence: float


class PredictionResponse(BaseModel):
    prediction_id: str
    label: str
    confidence: float
    advice: str
    severity_level: Optional[str] = None
    affected_area_ratio: Optional[float] = None
    is_low_confidence: bool = False
    certainty_label: str = "high_certainty"
    alternative_labels: List[AlternativeLabelDTO] = []
    treatment_options: List[TreatmentOptionDTO] = []
    timestamp: str
