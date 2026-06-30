"""0002_tier1_diagnostic_intelligence

Revision ID: 0002
Revises: 0001
Create Date: 2026-06-30
"""
from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID
from alembic import op

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "diagnosis_sessions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("consolidated_label", sa.String(100)),
        sa.Column("consolidated_confidence", sa.Float),
        sa.Column("photo_count", sa.Integer, server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
    )

    op.create_foreign_key(
        "fk_predictions_session",
        "predictions",
        "diagnosis_sessions",
        ["session_id"],
        ["id"],
    )

    op.create_table(
        "treatment_options",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("disease_label", sa.String(100), nullable=False),
        sa.Column("treatment_type", sa.String(20), nullable=False),
        sa.Column("product_name", sa.String(200), nullable=False),
        sa.Column("active_ingredient", sa.String(200)),
        sa.Column("application_method", sa.Text),
        sa.Column("estimated_cost_myr", sa.Numeric(10, 2)),
        sa.Column("severity_min", sa.String(20)),
        sa.Column("severity_max", sa.String(20)),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
    )
    op.create_index(
        "idx_treatment_disease", "treatment_options", ["disease_label"]
    )

    op.create_table(
        "treatment_logs",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "prediction_id",
            UUID(as_uuid=True),
            sa.ForeignKey("predictions.id"),
            nullable=False,
        ),
        sa.Column(
            "treatment_option_id",
            UUID(as_uuid=True),
            sa.ForeignKey("treatment_options.id"),
            nullable=False,
        ),
        sa.Column(
            "applied_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
        sa.Column("outcome", sa.String(20), server_default="pending"),
        sa.Column("outcome_logged_at", sa.DateTime(timezone=True)),
    )


def downgrade() -> None:
    op.drop_table("treatment_logs")
    op.drop_index("idx_treatment_disease", table_name="treatment_options")
    op.drop_table("treatment_options")
    op.drop_constraint("fk_predictions_session", "predictions", type_="foreignkey")
    op.drop_table("diagnosis_sessions")
