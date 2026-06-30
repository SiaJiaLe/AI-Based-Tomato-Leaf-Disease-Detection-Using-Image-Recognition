"""Treatment log entity — records applied treatment and its outcome."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class TreatmentLog:
    id: str
    prediction_id: str
    treatment_option_id: str
    applied_at: datetime = field(default_factory=datetime.utcnow)
    outcome: str = "pending"
    outcome_logged_at: Optional[datetime] = None
