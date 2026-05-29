"""Disease value object (scaffold)."""
from dataclasses import dataclass


@dataclass
class Disease:
  name: str
  severity_hint: str
  treatment_tip: str
