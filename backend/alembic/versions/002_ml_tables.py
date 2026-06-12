"""ML tables: ml_model_registry, benchmark_results, delay_predictions, bottleneck_detections.

Revision ID: 002
Revises: 001
Create Date: 2026-06-12

"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "ml_model_registry",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("model_name", sa.String(100), nullable=False),
        sa.Column("binary_run_id", sa.String(64), nullable=False),
        sa.Column("regression_run_id", sa.String(64), nullable=True),
        sa.Column("root_cause_run_id", sa.String(64), nullable=True),
        sa.Column("feature_count", sa.Integer(), nullable=False),
        sa.Column("is_champion", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("promoted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "loaded_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )

    op.create_table(
        "benchmark_results",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("model_name", sa.String(100), nullable=False),
        sa.Column("run_id", sa.String(64), nullable=False),
        sa.Column("auc_roc", sa.Float(), nullable=True),
        sa.Column("average_precision", sa.Float(), nullable=True),
        sa.Column("precision_at_80_recall", sa.Float(), nullable=True),
        sa.Column("ece", sa.Float(), nullable=True),
        sa.Column(
            "evaluated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_benchmark_results_run_id", "benchmark_results", ["run_id"])

    op.create_table(
        "delay_predictions",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "production_order_id",
            sa.Uuid(),
            sa.ForeignKey("production_orders.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("model_version", sa.String(100), nullable=False),
        sa.Column("delay_probability", sa.Float(), nullable=False),
        sa.Column("delay_minutes_estimate", sa.Float(), nullable=True),
        sa.Column("root_cause", sa.String(100), nullable=True),
        sa.Column("confidence", sa.String(20), nullable=False),
        sa.Column("top_risk_factors", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "mitigating_factors", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("narrative", sa.Text(), nullable=True),
        sa.Column("shap_values", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "feature_snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_delay_predictions_production_order_id",
        "delay_predictions",
        ["production_order_id"],
    )

    op.create_table(
        "bottleneck_detections",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "machine_id",
            sa.Uuid(),
            sa.ForeignKey("machines.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("detected_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("severity", sa.String(20), nullable=False, server_default="'medium'"),
        sa.Column(
            "affected_order_count", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column("description", sa.Text(), nullable=True),
        sa.Column("resolved_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_bottleneck_detections_machine_id", "bottleneck_detections", ["machine_id"]
    )
    op.create_index(
        "ix_bottleneck_detections_detected_at", "bottleneck_detections", ["detected_at"]
    )


def downgrade() -> None:
    op.drop_table("bottleneck_detections")
    op.drop_table("delay_predictions")
    op.drop_table("benchmark_results")
    op.drop_table("ml_model_registry")
