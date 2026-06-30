"""Diagnosis session entity — groups multiple photos into one consolidated result."""
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class DiagnosisSession:
    id: str
    consolidated_label: Optional[str] = None
    consolidated_confidence: Optional[float] = None
    photo_count: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
