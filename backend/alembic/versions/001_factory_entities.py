"""Factory entity tables: products, machines, operators, production_orders.

Revision ID: 001
Revises: None
Create Date: 2026-06-12

"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "products",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("sku", sa.String(50), nullable=False),
        sa.Column("product_family", sa.String(100), nullable=False),
        sa.Column("complexity_score", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column("operation_count", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("standard_hours", sa.Float(), nullable=False, server_default="8.0"),
        sa.Column("material_bom_complexity", sa.Integer(), nullable=False, server_default="1"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_products_sku", "products", ["sku"], unique=True)

    op.create_table(
        "machines",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("machine_code", sa.String(50), nullable=False),
        sa.Column("machine_type", sa.String(100), nullable=False),
        sa.Column("work_center", sa.String(100), nullable=False),
        sa.Column("oee_target", sa.Float(), nullable=False, server_default="0.85"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_machines_machine_code", "machines", ["machine_code"], unique=True)

    op.create_table(
        "machine_utilization_logs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "machine_id",
            sa.Uuid(),
            sa.ForeignKey("machines.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("snapshot_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("utilization_pct", sa.Float(), nullable=False),
        sa.Column("queue_depth", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "unplanned_downtime_hours", sa.Float(), nullable=False, server_default="0.0"
        ),
    )
    op.create_index(
        "ix_machine_utilization_logs_machine_id",
        "machine_utilization_logs",
        ["machine_id"],
    )
    op.create_index(
        "ix_machine_utilization_logs_snapshot_at",
        "machine_utilization_logs",
        ["snapshot_at"],
    )

    op.create_table(
        "operators",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("employee_id", sa.String(50), nullable=False),
        sa.Column("name", sa.String(150), nullable=False),
        sa.Column("skill_tier", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("experience_months", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("shift_type", sa.String(20), nullable=False, server_default="'day'"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_operators_employee_id", "operators", ["employee_id"], unique=True)

    op.create_table(
        "production_orders",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("order_number", sa.String(50), nullable=False),
        sa.Column(
            "product_id",
            sa.Uuid(),
            sa.ForeignKey("products.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "machine_id",
            sa.Uuid(),
            sa.ForeignKey("machines.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "operator_id",
            sa.Uuid(),
            sa.ForeignKey("operators.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("planned_start", sa.DateTime(timezone=True), nullable=False),
        sa.Column("planned_end", sa.DateTime(timezone=True), nullable=False),
        sa.Column("actual_start", sa.DateTime(timezone=True), nullable=True),
        sa.Column("actual_end", sa.DateTime(timezone=True), nullable=True),
        sa.Column("quantity", sa.Integer(), nullable=False),
        sa.Column("is_expedited", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("priority", sa.String(20), nullable=False, server_default="'normal'"),
        sa.Column("estimated_total_hours", sa.Float(), nullable=False),
        sa.Column("planned_lead_time_hours", sa.Float(), nullable=False),
        sa.Column("release_lag_hours", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column(
            "schedule_revision_count", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column(
            "material_availability_at_release",
            sa.Boolean(),
            nullable=False,
            server_default="true",
        ),
        sa.Column(
            "component_shortage_count", sa.Integer(), nullable=False, server_default="0"
        ),
        sa.Column(
            "changeover_required", sa.Boolean(), nullable=False, server_default="false"
        ),
        sa.Column(
            "changeover_complexity_score", sa.Float(), nullable=False, server_default="0.0"
        ),
        sa.Column("status", sa.String(20), nullable=False, server_default="'pending'"),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_production_orders_order_number", "production_orders", ["order_number"], unique=True
    )
    op.create_index(
        "ix_production_orders_planned_start", "production_orders", ["planned_start"]
    )
    op.create_index("ix_production_orders_status", "production_orders", ["status"])
    op.create_index(
        "ix_production_orders_product_id", "production_orders", ["product_id"]
    )
    op.create_index(
        "ix_production_orders_machine_id", "production_orders", ["machine_id"]
    )
    op.create_index(
        "ix_production_orders_operator_id", "production_orders", ["operator_id"]
    )


def downgrade() -> None:
    op.drop_table("production_orders")
    op.drop_table("operators")
    op.drop_table("machine_utilization_logs")
    op.drop_table("machines")
    op.drop_table("products")
