"""LLM and agent tables: chat_sessions, chat_messages, recommendations, operational_reports,
workflow_executions, audit_logs.

Revision ID: 003
Revises: 002
Create Date: 2026-06-12

"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "003"
down_revision: Union[str, None] = "002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "chat_sessions",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("session_token", sa.String(100), nullable=False),
        sa.Column(
            "context_snapshot", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("token_budget", sa.Integer(), nullable=False, server_default="4096"),
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
    op.create_index("ix_chat_sessions_session_token", "chat_sessions", ["session_token"], unique=True)

    op.create_table(
        "chat_messages",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column(
            "session_id",
            sa.Uuid(),
            sa.ForeignKey("chat_sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("role", sa.String(20), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("tool_calls", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("tool_results", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("model_used", sa.String(100), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_chat_messages_session_id", "chat_messages", ["session_id"])
    op.create_index("ix_chat_messages_created_at", "chat_messages", ["created_at"])

    op.create_table(
        "recommendations",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("title", sa.String(200), nullable=False),
        sa.Column("description", sa.Text(), nullable=False),
        sa.Column("category", sa.String(50), nullable=False),
        sa.Column("urgency", sa.String(20), nullable=False, server_default="'medium'"),
        sa.Column(
            "order_id",
            sa.Uuid(),
            sa.ForeignKey("production_orders.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "bottleneck_id",
            sa.Uuid(),
            sa.ForeignKey("bottleneck_detections.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("status", sa.String(20), nullable=False, server_default="'open'"),
        sa.Column("actioned_by", sa.String(150), nullable=True),
        sa.Column("actioned_at", sa.DateTime(timezone=True), nullable=True),
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
    op.create_index("ix_recommendations_order_id", "recommendations", ["order_id"])
    op.create_index("ix_recommendations_status", "recommendations", ["status"])

    op.create_table(
        "operational_reports",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("report_type", sa.String(50), nullable=False),
        sa.Column("report_date", sa.Date(), nullable=False),
        sa.Column("shift", sa.String(20), nullable=True),
        sa.Column("data_json", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("html_content", sa.Text(), nullable=True),
        sa.Column(
            "generated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_operational_reports_report_type", "operational_reports", ["report_type"]
    )
    op.create_index(
        "ix_operational_reports_report_date", "operational_reports", ["report_date"]
    )

    op.create_table(
        "workflow_executions",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("workflow_name", sa.String(100), nullable=False),
        sa.Column("trigger_type", sa.String(30), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="'running'"),
        sa.Column("input_data", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("output_data", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("error_message", sa.Text(), nullable=True),
        sa.Column(
            "started_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        "ix_workflow_executions_workflow_name", "workflow_executions", ["workflow_name"]
    )
    op.create_index("ix_workflow_executions_status", "workflow_executions", ["status"])

    op.create_table(
        "audit_logs",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("entity_type", sa.String(50), nullable=False),
        sa.Column("entity_id", sa.String(100), nullable=False),
        sa.Column("operation", sa.String(30), nullable=False),
        sa.Column("actor", sa.String(150), nullable=False, server_default="'system'"),
        sa.Column("data_before", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("data_after", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
    )
    op.create_index("ix_audit_logs_entity_type", "audit_logs", ["entity_type"])
    op.create_index("ix_audit_logs_entity_id", "audit_logs", ["entity_id"])
    op.create_index("ix_audit_logs_created_at", "audit_logs", ["created_at"])


def downgrade() -> None:
    op.drop_table("audit_logs")
    op.drop_table("workflow_executions")
    op.drop_table("operational_reports")
    op.drop_table("recommendations")
    op.drop_table("chat_messages")
    op.drop_table("chat_sessions")
