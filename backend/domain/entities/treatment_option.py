"""Treatment option entity."""
from dataclasses import dataclass
from typing import Optional


@dataclass
class TreatmentOption:
    id: str
    disease_label: str
    treatment_type: str
    product_name: str
    active_ingredient: Optional[str]
    application_method: Optional[str]
    estimated_cost_myr: Optional[float]
    severity_min: Optional[str]
    severity_max: Optional[str]
