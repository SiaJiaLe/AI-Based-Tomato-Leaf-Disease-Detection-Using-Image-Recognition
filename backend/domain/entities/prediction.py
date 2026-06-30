"""Prediction entity."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Prediction:
    prediction_id: str
    label: str
    confidence: float
    advice: str
    severity_level: Optional[str] = None
    affected_area_ratio: Optional[float] = None
    is_low_confidence: bool = False
    certainty_label: str = "high_certainty"
    image_path: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
