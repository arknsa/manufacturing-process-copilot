#!/usr/bin/env python3
"""
backend/scripts/seed_db.py
===========================
Idempotent database seeder for the Manufacturing Process Copilot.

Reads ml/data/raw/synthetic_factory_data.csv and populates:
  1. products        — 15 rows (one per unique product_id in CSV)
  2. machines        —  8 rows (one per unique machine_id in CSV)
  3. operators       — 24 rows (one per unique operator_id in CSV)
  4. production_orders — all historical rows + a today-dated slice so
                        GET /api/v1/orders/today returns results

Idempotency:
  Every INSERT uses ON CONFLICT DO NOTHING on the natural-key column
  (sku, machine_code, employee_id, order_number).  Safe to run repeatedly.

Usage (from project root):
  python -m backend.scripts.seed_db
  python -m backend.scripts.seed_db --today-only     # skip historical, just today's slice
  python -m backend.scripts.seed_db --csv path/to/other.csv
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import uuid
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# ── path bootstrap so the script works when run with -m from project root ──
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.app.core.config import get_settings  # noqa: E402
from backend.app.db.models.machine import Machine  # noqa: E402
from backend.app.db.models.operator import Operator  # noqa: E402
from backend.app.db.models.order import ProductionOrder  # noqa: E402
from backend.app.db.models.product import Product  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Constants mirrored from generate_synthetic_data.py ──────────────────────

_PRIORITY_BUFFER: Dict[str, float] = {
    "critical": 1.15,
    "high":     1.50,
    "normal":   1.80,
    "low":      2.50,
}

# machine_type derived from _MACHINE_SPECS order in the generator script
_MACHINE_TYPE_BY_CODE: Dict[str, str] = {
    "MACH-001": "CNC_MILL",
    "MACH-002": "DRILL_PRESS",
    "MACH-003": "LATHE",
    "MACH-004": "LATHE",
    "MACH-005": "ASSEMBLY",
    "MACH-006": "INSPECTION",
    "MACH-007": "WELDING",
    "MACH-008": "PRESS",
}

_COMPLEXITY_FAMILY: Dict[float, str] = {
    0.25: "Precision Parts",
    0.55: "Assembly Units",
    0.85: "Complex Fabrication",
}

# How many of the most-recent simulation orders to rebase to today.
# 10 orders gives a realistic single-shift view on the dashboard.
_TODAY_SLICE_SIZE = 10


# ── CSV loading ──────────────────────────────────────────────────────────────

def load_csv(csv_path: Path) -> pd.DataFrame:
    log.info("Loading CSV: %s", csv_path)
    df = pd.read_csv(csv_path)
    log.info("  %d rows, %d columns", len(df), len(df.columns))
    return df


# ── Entity extraction ────────────────────────────────────────────────────────

def extract_products(df: pd.DataFrame) -> List[dict]:
    """One row per unique product_id.  Aggregates per-product attributes."""
    grp = (
        df.groupby("product_id")
        .agg(
            complexity_score=("product_complexity_score", "first"),
            operation_count=("operation_count", "first"),
            material_bom_complexity=("material_bom_complexity", "first"),
            standard_hours=("estimated_total_hours", "median"),
        )
        .reset_index()
    )
    rows = []
    for _, r in grp.iterrows():
        score = float(r["complexity_score"])
        rows.append({
            "id":                    uuid.uuid4(),
            "sku":                   str(r["product_id"]),
            "product_family":        _COMPLEXITY_FAMILY.get(round(score, 2), "General"),
            "complexity_score":      score,
            "operation_count":       int(r["operation_count"]),
            "standard_hours":        round(float(r["standard_hours"]), 4),
            "material_bom_complexity": int(r["material_bom_complexity"]),
        })
    log.info("  Extracted %d products", len(rows))
    return rows


def extract_machines(df: pd.DataFrame) -> List[dict]:
    """One row per unique machine_id.  OEE target = mean of machine_oee_30d."""
    grp = (
        df.groupby("machine_id")
        .agg(oee_target=("machine_oee_30d", "mean"))
        .reset_index()
    )
    rows = []
    for _, r in grp.iterrows():
        code = str(r["machine_id"])
        mtype = _MACHINE_TYPE_BY_CODE.get(code, "GENERAL")
        rows.append({
            "id":           uuid.uuid4(),
            "machine_code": code,
            "machine_type": mtype,
            "work_center":  mtype,          # work_center = machine_type at this scale
            "oee_target":   round(float(r["oee_target"]), 4),
        })
    log.info("  Extracted %d machines", len(rows))
    return rows


def extract_operators(df: pd.DataFrame) -> List[dict]:
    """One row per unique operator_id (first occurrence wins)."""
    first = df.drop_duplicates("operator_id").copy()
    rows = []
    for _, r in first.iterrows():
        emp_id = str(r["operator_id"])
        # operator_skill_tier_encoded: 0.0 → junior(0), 1.0 → mid(1), 2.0 → senior(2)
        tier = int(round(float(r["operator_skill_tier_encoded"])))
        rows.append({
            "id":                uuid.uuid4(),
            "employee_id":       emp_id,
            "name":              f"Operator {emp_id.split('-')[1]}",
            "skill_tier":        tier,
            "experience_months": int(r["operator_experience_months"]),
            "shift_type":        str(r["shift_type"]),
        })
    log.info("  Extracted %d operators", len(rows))
    return rows


# ── Order building ────────────────────────────────────────────────────────────

def _rebase_dt(iso_str: str, target_date: date, after: datetime | None = None) -> datetime:
    """Keep HH:MM:SS from iso_str but swap the calendar date to target_date (UTC).

    If `after` is given and the rebased result is earlier than `after`, add one
    day — this preserves validity for orders that originally crossed midnight.
    """
    dt = datetime.fromisoformat(str(iso_str))
    rebased = dt.replace(
        year=target_date.year,
        month=target_date.month,
        day=target_date.day,
        tzinfo=timezone.utc,
    )
    if after is not None and rebased <= after:
        rebased += timedelta(days=1)
    return rebased


def _derive_planned_end(planned_start: datetime, estimated_total_hours: float, priority: str) -> datetime:
    """Reconstruct planned_end (not stored in CSV) from the same formula the simulator used."""
    buf = _PRIORITY_BUFFER.get(priority, 1.80)
    return planned_start + timedelta(hours=estimated_total_hours * buf)


def _derive_status(actual_end: datetime | None, planned_end: datetime) -> str:
    if actual_end is None:
        return "pending"
    return "completed" if actual_end <= planned_end else "delayed"


def build_order_rows(
    df: pd.DataFrame,
    product_uuid_map: Dict[str, uuid.UUID],
    machine_uuid_map: Dict[str, uuid.UUID],
    operator_uuid_map: Dict[str, uuid.UUID],
    today: date,
    today_only: bool,
) -> List[dict]:
    """
    Returns two groups of order dicts:
      - historical orders with their original dates (unless today_only=True)
      - the last _TODAY_SLICE_SIZE orders rebased to today
    """
    df_sorted = df.sort_values("planned_start").reset_index(drop=True)

    today_slice = df_sorted.tail(_TODAY_SLICE_SIZE).copy()
    historical  = df_sorted.iloc[: -_TODAY_SLICE_SIZE].copy() if not today_only else pd.DataFrame()

    rows: List[dict] = []

    def _build_row(r: "pd.Series", target_date: date | None) -> dict:
        pid  = product_uuid_map.get(str(r["product_id"]))
        mid  = machine_uuid_map.get(str(r["machine_id"]))
        oid  = operator_uuid_map.get(str(r["operator_id"]))
        priority = str(r["priority"])
        est_h    = float(r["estimated_total_hours"])

        if target_date is not None:
            planned_start = _rebase_dt(r["planned_start"], target_date)
            actual_end    = _rebase_dt(r["actual_end"],    target_date, after=planned_start)
        else:
            planned_start = datetime.fromisoformat(str(r["planned_start"])).replace(tzinfo=timezone.utc)
            actual_end    = datetime.fromisoformat(str(r["actual_end"])).replace(tzinfo=timezone.utc)

        planned_end = _derive_planned_end(planned_start, est_h, priority)
        status      = _derive_status(actual_end, planned_end)
        now         = datetime.now(timezone.utc)

        return {
            "id":                               uuid.uuid4(),
            "order_number":                     str(r["order_id"]),
            "product_id":                       pid,
            "machine_id":                       mid,
            "operator_id":                      oid,
            "planned_start":                    planned_start,
            "planned_end":                      planned_end,
            "actual_start":                     planned_start if status in ("completed", "delayed") else None,
            "actual_end":                       actual_end,
            "quantity":                         int(r["quantity"]),
            "is_expedited":                     bool(int(r["is_expedited"])),
            "priority":                         priority,
            "estimated_total_hours":            est_h,
            "planned_lead_time_hours":          float(r["planned_lead_time_hours"]),
            "release_lag_hours":                float(r["release_lag_hours"]),
            "schedule_revision_count":          int(r["schedule_revision_count"]),
            "material_availability_at_release": bool(int(r["material_availability_at_release"])),
            "component_shortage_count":         int(r["component_shortage_count"]),
            "changeover_required":              bool(int(r["changeover_required"])),
            "changeover_complexity_score":      float(r["changeover_complexity_score"]),
            "status":                           status,
            "notes":                            None,
            "created_at":                       now,
            "updated_at":                       now,
        }

    for _, r in historical.iterrows():
        rows.append(_build_row(r, None))

    # Rebase today's slice — use distinct suffix to avoid order_number collisions
    # when run alongside historical rows (e.g. ORD-005400 vs ORD-005400-TODAY).
    for _, r in today_slice.iterrows():
        row = _build_row(r, today)
        row["order_number"] = row["order_number"] + "-TODAY"
        rows.append(row)

    log.info(
        "  Built %d historical + %d today's orders (%d total)",
        len(historical),
        len(today_slice),
        len(rows),
    )
    return rows


# ── DB helpers ────────────────────────────────────────────────────────────────

async def _upsert_products(session: AsyncSession, rows: List[dict]) -> Dict[str, uuid.UUID]:
    """Insert products, skip existing by sku.  Returns sku → uuid map."""
    stmt = (
        pg_insert(Product)
        .values(rows)
        .on_conflict_do_nothing(index_elements=["sku"])
    )
    await session.execute(stmt)

    result = await session.execute(
        text("SELECT sku, id FROM products")
    )
    return {row.sku: row.id for row in result}


async def _upsert_machines(session: AsyncSession, rows: List[dict]) -> Dict[str, uuid.UUID]:
    stmt = (
        pg_insert(Machine)
        .values(rows)
        .on_conflict_do_nothing(index_elements=["machine_code"])
    )
    await session.execute(stmt)

    result = await session.execute(
        text("SELECT machine_code, id FROM machines")
    )
    return {row.machine_code: row.id for row in result}


async def _upsert_operators(session: AsyncSession, rows: List[dict]) -> Dict[str, uuid.UUID]:
    stmt = (
        pg_insert(Operator)
        .values(rows)
        .on_conflict_do_nothing(index_elements=["employee_id"])
    )
    await session.execute(stmt)

    result = await session.execute(
        text("SELECT employee_id, id FROM operators")
    )
    return {row.employee_id: row.id for row in result}


async def _delete_today_orders(session: AsyncSession) -> int:
    """Delete all -TODAY orders so today's slice is always fresh on each run."""
    result = await session.execute(
        text("DELETE FROM production_orders WHERE order_number LIKE '%-TODAY'")
    )
    return result.rowcount


async def _upsert_orders(session: AsyncSession, rows: List[dict]) -> int:
    if not rows:
        return 0
    # Chunk to avoid hitting Postgres parameter limit (65535 / ~20 cols ≈ 3000 rows/chunk)
    chunk_size = 2000
    inserted   = 0
    for i in range(0, len(rows), chunk_size):
        chunk = rows[i : i + chunk_size]
        stmt  = (
            pg_insert(ProductionOrder)
            .values(chunk)
            .on_conflict_do_nothing(index_elements=["order_number"])
        )
        result = await session.execute(stmt)
        inserted += result.rowcount if result.rowcount != -1 else len(chunk)
    return inserted


# ── Main seeder ───────────────────────────────────────────────────────────────

async def seed(csv_path: Path, today_only: bool) -> None:
    settings = get_settings()
    engine = create_async_engine(
        settings.DATABASE_URL,
        echo=False,
        pool_pre_ping=True,
    )
    SessionLocal = async_sessionmaker(bind=engine, expire_on_commit=False, class_=AsyncSession)

    df = load_csv(csv_path)
    today = date.today()

    log.info("Step 1 — extracting entities from CSV …")
    product_rows  = extract_products(df)
    machine_rows  = extract_machines(df)
    operator_rows = extract_operators(df)

    async with SessionLocal() as session:
        async with session.begin():
            log.info("Step 2 — upserting products …")
            product_map = await _upsert_products(session, product_rows)
            log.info("  products in DB: %d", len(product_map))

            log.info("Step 3 — upserting machines …")
            machine_map = await _upsert_machines(session, machine_rows)
            log.info("  machines in DB: %d", len(machine_map))

            log.info("Step 4 — upserting operators …")
            operator_map = await _upsert_operators(session, operator_rows)
            log.info("  operators in DB: %d", len(operator_map))

            log.info("Step 5 — building order rows …")
            order_rows = build_order_rows(
                df,
                product_map,
                machine_map,
                operator_map,
                today,
                today_only,
            )

            log.info("Step 6a — deleting stale today-slice orders …")
            deleted = await _delete_today_orders(session)
            log.info("  deleted %d stale today-slice rows", deleted)

            log.info("Step 6b — upserting %d production orders …", len(order_rows))
            inserted = await _upsert_orders(session, order_rows)
            log.info("  rows inserted (new): %d  skipped (existing): %d",
                     inserted, len(order_rows) - inserted)

    await engine.dispose()

    log.info("=" * 60)
    log.info("Seeding complete.")
    log.info(
        "Run verification:\n"
        "  SELECT COUNT(*) FROM products;\n"
        "  SELECT COUNT(*) FROM machines;\n"
        "  SELECT COUNT(*) FROM operators;\n"
        "  SELECT COUNT(*) FROM production_orders;\n"
        "  SELECT COUNT(*) FROM production_orders\n"
        "    WHERE planned_start::date = CURRENT_DATE;"
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Seed PostgreSQL from synthetic_factory_data.csv")
    p.add_argument(
        "--csv",
        default=str(_PROJECT_ROOT / "ml" / "data" / "raw" / "synthetic_factory_data.csv"),
        help="Path to synthetic_factory_data.csv",
    )
    p.add_argument(
        "--today-only",
        action="store_true",
        help="Skip historical orders; only insert the today-rebased slice",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    csv_path = Path(args.csv)
    if not csv_path.exists():
        log.error("CSV not found: %s", csv_path)
        sys.exit(1)
    asyncio.run(seed(csv_path, args.today_only))
