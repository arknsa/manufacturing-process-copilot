"""
backend/app/services/agent/tools/analytics.py
===============================================
Agent tools for machine performance, bottleneck detection, and shift-level KPIs.

Tools
-----
get_machine_history  — recent utilisation snapshots for a machine
get_bottlenecks      — list active (unresolved) bottleneck detections
get_shift_summary    — today's order completion/delay counts per shift
get_kpi_dashboard    — facility-wide KPIs: on-time rate, avg delay, throughput
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any

from sqlalchemy import and_, case, desc, func, select
from sqlalchemy.ext.asyncio import AsyncSession

from backend.app.db.models.bottleneck import BottleneckDetection
from backend.app.db.models.machine import Machine, MachineUtilizationLog
from backend.app.db.models.order import ProductionOrder
from backend.app.db.models.prediction import DelayPrediction

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

GET_MACHINE_HISTORY_SCHEMA: dict[str, Any] = {
    "description": (
        "Return recent utilisation log snapshots for a machine identified by its "
        "machine_code. Useful for diagnosing capacity pressure or downtime patterns."
    ),
    "parameters": {
        "machine_code": "str — machine identifier, e.g. 'MC-001'",
        "days": "int (optional, default 7) — how many days of history to return",
    },
}

GET_BOTTLENECKS_SCHEMA: dict[str, Any] = {
    "description": (
        "Return detected machine bottlenecks. By default only unresolved bottlenecks "
        "are returned, ordered by severity and detection time."
    ),
    "parameters": {
        "active_only": (
            "bool (optional, default true) — if true, exclude bottlenecks with a "
            "resolved_at timestamp"
        ),
    },
}

GET_SHIFT_SUMMARY_SCHEMA: dict[str, Any] = {
    "description": (
        "Return a summary of completed and delayed orders for a specific date. "
        "Defaults to today."
    ),
    "parameters": {
        "shift_date": (
            "str (optional) — ISO date string YYYY-MM-DD; defaults to today's date"
        ),
    },
}

GET_KPI_DASHBOARD_SCHEMA: dict[str, Any] = {
    "description": (
        "Return facility-wide KPIs for the last 30 days: on-time delivery rate, "
        "average delay minutes, total orders completed, and throughput by day."
    ),
    "parameters": {},
}


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

async def get_machine_history(
    db: AsyncSession,
    machine_code: str,
    days: int = 7,
) -> dict[str, Any]:
    """Utilisation snapshots for the last ``days`` days."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)

    machine_stmt = select(Machine).where(Machine.machine_code == machine_code)
    machine_result = await db.execute(machine_stmt)
    machine = machine_result.scalar_one_or_none()
    if machine is None:
        return {"error": f"Machine '{machine_code}' not found."}

    log_stmt = (
        select(MachineUtilizationLog)
        .where(
            and_(
                MachineUtilizationLog.machine_id == machine.id,
                MachineUtilizationLog.snapshot_at >= cutoff,
            )
        )
        .order_by(desc(MachineUtilizationLog.snapshot_at))
        .limit(100)
    )
    log_result = await db.execute(log_stmt)
    logs = log_result.scalars().all()

    snapshots = [
        {
            "snapshot_at": l.snapshot_at.isoformat(),
            "utilization_pct": round(l.utilization_pct, 3),
            "queue_depth": l.queue_depth,
            "unplanned_downtime_hours": l.unplanned_downtime_hours,
        }
        for l in logs
    ]

    avg_util = (
        sum(s["utilization_pct"] for s in snapshots) / len(snapshots) if snapshots else 0.0
    )
    total_downtime = sum(s["unplanned_downtime_hours"] for s in snapshots)

    return {
        "machine_code": machine_code,
        "machine_type": machine.machine_type,
        "work_center": machine.work_center,
        "oee_target": machine.oee_target,
        "period_days": days,
        "snapshots": snapshots[:20],  # Return most recent 20 for brevity
        "avg_utilization_pct": round(avg_util, 3),
        "total_unplanned_downtime_hours": round(total_downtime, 2),
        "snapshot_count": len(snapshots),
    }


async def get_bottlenecks(
    db: AsyncSession,
    active_only: bool = True,
) -> dict[str, Any]:
    """Active (or all) bottleneck detections with machine info."""
    stmt = (
        select(BottleneckDetection, Machine)
        .join(Machine, Machine.id == BottleneckDetection.machine_id, isouter=True)
        .order_by(
            desc(
                case(
                    (BottleneckDetection.severity == "critical", 4),
                    (BottleneckDetection.severity == "high", 3),
                    (BottleneckDetection.severity == "medium", 2),
                    else_=1,
                )
            ),
            desc(BottleneckDetection.detected_at),
        )
    )
    if active_only:
        stmt = stmt.where(BottleneckDetection.resolved_at.is_(None))

    result = await db.execute(stmt)
    rows = result.all()

    bottlenecks = [
        {
            "id": str(bn.id),
            "machine_code": m.machine_code if m else "unknown",
            "work_center": m.work_center if m else "unknown",
            "detected_at": bn.detected_at.isoformat(),
            "severity": bn.severity,
            "affected_order_count": bn.affected_order_count,
            "description": bn.description,
            "resolved_at": bn.resolved_at.isoformat() if bn.resolved_at else None,
        }
        for bn, m in rows
    ]

    return {
        "bottlenecks": bottlenecks,
        "count": len(bottlenecks),
        "active_only": active_only,
    }


async def get_shift_summary(
    db: AsyncSession,
    shift_date: str | None = None,
) -> dict[str, Any]:
    """Completed/delayed order counts for a date, grouped by order status."""
    try:
        target_date = (
            date.fromisoformat(shift_date) if shift_date else date.today()
        )
    except ValueError:
        return {"error": f"Invalid date format: '{shift_date}'. Use YYYY-MM-DD."}

    # Filter orders whose planned_start falls on target_date
    start_ts = datetime(
        target_date.year, target_date.month, target_date.day, 0, 0, 0,
        tzinfo=timezone.utc,
    )
    end_ts = start_ts + timedelta(days=1)

    stmt = (
        select(
            ProductionOrder.status,
            func.count().label("count"),
        )
        .where(
            and_(
                ProductionOrder.planned_start >= start_ts,
                ProductionOrder.planned_start < end_ts,
            )
        )
        .group_by(ProductionOrder.status)
    )
    result = await db.execute(stmt)
    rows = result.all()

    counts: dict[str, int] = {row.status: row.count for row in rows}
    total = sum(counts.values())
    completed = counts.get("completed", 0)
    delayed = counts.get("delayed", 0)
    on_time_rate = (completed / total) if total > 0 else 0.0

    # Average delay minutes for completed/delayed orders
    delay_stmt = (
        select(func.avg(DelayPrediction.delay_minutes_estimate))
        .join(
            ProductionOrder,
            and_(
                ProductionOrder.id == DelayPrediction.production_order_id,
                ProductionOrder.planned_start >= start_ts,
                ProductionOrder.planned_start < end_ts,
                ProductionOrder.status == "delayed",
            ),
        )
    )
    delay_result = await db.execute(delay_stmt)
    avg_delay_val = delay_result.scalar()

    return {
        "date": target_date.isoformat(),
        "total_orders": total,
        "completed": completed,
        "delayed": delayed,
        "pending": counts.get("pending", 0),
        "in_progress": counts.get("in_progress", 0),
        "on_time_rate": round(on_time_rate, 3),
        "avg_predicted_delay_minutes": (
            round(float(avg_delay_val), 1) if avg_delay_val else None
        ),
    }


async def get_kpi_dashboard(db: AsyncSession) -> dict[str, Any]:
    """Facility-wide KPIs for the last 30 days."""
    cutoff = datetime.now(timezone.utc) - timedelta(days=30)

    # Order-level aggregates
    order_stmt = (
        select(
            func.count().label("total"),
            func.sum(
                case((ProductionOrder.status == "completed", 1), else_=0)
            ).label("completed"),
            func.sum(
                case((ProductionOrder.status == "delayed", 1), else_=0)
            ).label("delayed"),
        )
        .where(ProductionOrder.planned_start >= cutoff)
    )
    order_result = await db.execute(order_stmt)
    o = order_result.one()

    total = int(o.total or 0)
    completed = int(o.completed or 0)
    delayed = int(o.delayed or 0)
    on_time_rate = completed / total if total > 0 else 0.0

    # Average predicted delay for delayed orders
    delay_stmt = (
        select(func.avg(DelayPrediction.delay_minutes_estimate))
        .join(
            ProductionOrder,
            and_(
                ProductionOrder.id == DelayPrediction.production_order_id,
                ProductionOrder.planned_start >= cutoff,
                ProductionOrder.status == "delayed",
            ),
        )
    )
    avg_delay_val = (await db.execute(delay_stmt)).scalar()

    # Active high-risk orders right now
    risk_stmt = (
        select(func.count())
        .select_from(DelayPrediction)
        .join(ProductionOrder, ProductionOrder.id == DelayPrediction.production_order_id)
        .where(
            and_(
                ProductionOrder.status.in_(["pending", "in_progress"]),
                DelayPrediction.delay_probability >= 0.65,
            )
        )
    )
    high_risk_count = (await db.execute(risk_stmt)).scalar() or 0

    return {
        "period_days": 30,
        "total_orders": total,
        "completed": completed,
        "delayed": delayed,
        "on_time_rate": round(on_time_rate, 3),
        "on_time_pct": f"{on_time_rate:.0%}",
        "avg_predicted_delay_minutes": (
            round(float(avg_delay_val), 1) if avg_delay_val else None
        ),
        "current_high_risk_active": int(high_risk_count),
    }
