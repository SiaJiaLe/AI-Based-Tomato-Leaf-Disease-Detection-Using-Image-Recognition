"""SQLAlchemy ORM models — maps domain entities to PostgreSQL tables."""
import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, Numeric, String, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class PredictionRecord(Base):
    __tablename__ = "predictions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    label: Mapped[str] = mapped_column(String(100), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    advice: Mapped[str | None] = mapped_column(Text)
    image_path: Mapped[str | None] = mapped_column(Text)
    severity_level: Mapped[str | None] = mapped_column(String(20))
    affected_area_ratio: Mapped[float | None] = mapped_column(Float)
    is_low_confidence: Mapped[bool] = mapped_column(Boolean, default=False)
    certainty_label: Mapped[str | None] = mapped_column(String(50))
    session_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True), ForeignKey("diagnosis_sessions.id"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class DiagnosisSessionRecord(Base):
    __tablename__ = "diagnosis_sessions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    consolidated_label: Mapped[str | None] = mapped_column(String(100))
    consolidated_confidence: Mapped[float | None] = mapped_column(Float)
    photo_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class TreatmentOptionRecord(Base):
    __tablename__ = "treatment_options"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    disease_label: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    treatment_type: Mapped[str] = mapped_column(String(20), nullable=False)
    product_name: Mapped[str] = mapped_column(String(200), nullable=False)
    active_ingredient: Mapped[str | None] = mapped_column(String(200))
    application_method: Mapped[str | None] = mapped_column(Text)
    estimated_cost_myr: Mapped[float | None] = mapped_column(Numeric(10, 2))
    severity_min: Mapped[str | None] = mapped_column(String(20))
    severity_max: Mapped[str | None] = mapped_column(String(20))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )


class TreatmentLogRecord(Base):
    __tablename__ = "treatment_logs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    prediction_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("predictions.id"), nullable=False
    )
    treatment_option_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("treatment_options.id"), nullable=False
    )
    applied_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    outcome: Mapped[str] = mapped_column(String(20), default="pending")
    outcome_logged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
