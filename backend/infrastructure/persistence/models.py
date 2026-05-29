"""SQLAlchemy ORM models (scaffold) — define tables in Phase 7."""
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""

    pass

# Example (uncomment when implementing):
# class PredictionRecord(Base):
#     __tablename__ = "predictions"
#     prediction_id: Mapped[str] = mapped_column(String, primary_key=True)
#     ...
