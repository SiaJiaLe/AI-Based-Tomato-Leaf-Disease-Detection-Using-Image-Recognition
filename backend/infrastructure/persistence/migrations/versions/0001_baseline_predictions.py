"""0001_baseline_predictions

Revision ID: 0001
Revises:
Create Date: 2026-06-30
"""
from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID
from alembic import op

revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "predictions",
        sa.Column("id", UUID(as_uuid=True), primary_key=True),
        sa.Column("label", sa.String(100), nullable=False),
        sa.Column("confidence", sa.Float, nullable=False),
        sa.Column("advice", sa.Text),
        sa.Column("image_path", sa.Text),
        sa.Column("severity_level", sa.String(20)),
        sa.Column("affected_area_ratio", sa.Float),
        sa.Column("is_low_confidence", sa.Boolean, server_default="false"),
        sa.Column("certainty_label", sa.String(50)),
        sa.Column("session_id", UUID(as_uuid=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
        ),
    )


def downgrade() -> None:
    op.drop_table("predictions")
