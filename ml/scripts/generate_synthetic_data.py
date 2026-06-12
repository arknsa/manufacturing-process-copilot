#!/usr/bin/env python3
"""
ml/scripts/generate_synthetic_data.py

Generates synthetic manufacturing training data for the Manufacturing Process Copilot.
Produces a 50-column dataset (37 features + 4 targets + 9 metadata) aligned with the
schema contract defined in docs/02_dataset_schema.md.

Architecture:
    Layer 0  Foundation  — SimConfig, entity dataclasses, SeedManager
    Layer 1  Generation  — DemandGenerator, BreakdownGenerator, SetupRunTimeGenerator,
                           QualityOutcomeGenerator, AbsenteeismGenerator
    Layer 2  Engine      — Scheduler, StateManager
    Layer 3  Orchestration — FactorySimulation._simulate_day(), ._create_order()
    Layer 4  Collection  — FeatureCollector.snapshot(), OutcomeRecorder.record()
    Layer 5  Output      — DatasetBuilder, CalibrationChecker, ReportGenerator

Usage:
    python ml/scripts/generate_synthetic_data.py [--days 540] [--orders-per-day 10]
        [--machines 8] [--seed 42] [--output-dir ml/data]

Calibration validation run (120 days):
    python ml/scripts/generate_synthetic_data.py --days 120 --seed 42 --validate-only
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ─── Layer 0: Foundation ──────────────────────────────────────────────────────

SIM_BASE_DATE = datetime(2023, 1, 2, 8, 0, 0)  # Monday 08:00

FEATURE_COLS: Tuple[str, ...] = (
    "planned_lead_time_hours", "release_lag_hours", "schedule_revision_count",
    "is_expedited", "priority_encoded", "quantity", "operation_count",
    "estimated_total_hours", "schedule_tightness_ratio",
    "product_complexity_score", "material_bom_complexity",
    "is_month_end", "is_quarter_end",
    "machine_utilization_at_release", "work_center_queue_depth_at_release",
    "machine_oee_30d", "machine_unplanned_downtime_hours_30d",
    "days_since_last_planned_maintenance", "maintenance_due_within_order_window",
    "changeover_required", "changeover_complexity_score",
    "operator_experience_months", "operator_skill_tier_encoded",
    "operator_concurrent_order_count", "hours_into_shift_at_start",
    "shift_type_encoded",
    "material_availability_at_release", "component_shortage_count",
    "product_delay_rate_90d", "machine_delay_rate_90d", "operator_delay_rate_90d",
    "product_x_machine_delay_rate_90d", "product_first_pass_yield_90d",
    "machine_setup_overrun_rate_90d", "shift_delay_rate_30d",
    "planned_start_day_of_week", "planned_start_hour",
)
TARGET_COLS: Tuple[str, ...] = (
    "is_delayed", "delay_minutes", "delay_category", "delay_root_cause",
)
METADATA_COLS: Tuple[str, ...] = (
    "order_id", "po_number", "product_id", "machine_id", "operator_id",
    "priority", "shift_type", "planned_start", "actual_end",
)

COLD_START_DEFAULTS: Dict[str, float] = {
    "product_delay_rate_90d":           0.343,
    "machine_delay_rate_90d":           0.347,
    "operator_delay_rate_90d":          0.354,
    "product_x_machine_delay_rate_90d": 0.342,
    "product_first_pass_yield_90d":     0.916,
    "machine_setup_overrun_rate_90d":   0.521,
    "shift_delay_rate_30d":             0.358,
}

PRIORITY_BUFFER: Dict[str, float] = {
    "critical": 1.15, "high": 1.50, "normal": 1.80, "low": 2.50,
}
PRIORITY_ENCODE: Dict[str, int] = {
    "low": 0, "normal": 1, "high": 2, "critical": 3,
}
SHIFT_ENCODE: Dict[str, int] = {"morning": 0, "afternoon": 1, "night": 2}
SKILL_ENCODE: Dict[str, float] = {"junior": 0.0, "mid": 1.0, "senior": 2.0}
COMPLEXITY_SCORE: Dict[str, float] = {"LOW": 0.25, "MEDIUM": 0.55, "HIGH": 0.85}
BASE_FPY: Dict[str, float] = {"LOW": 0.970, "MEDIUM": 0.930, "HIGH": 0.880}

_MACHINE_SPECS: List[Dict] = [
    {"type": "CNC_MILL",    "mtbf_h": 720,  "mttr_h": 4.0, "pm_d": 90, "overrun": 1.06},
    {"type": "DRILL_PRESS", "mtbf_h": 1440, "mttr_h": 2.0, "pm_d": 60, "overrun": 1.03},
    {"type": "LATHE",       "mtbf_h": 960,  "mttr_h": 3.0, "pm_d": 60, "overrun": 1.05},
    {"type": "LATHE",       "mtbf_h": 960,  "mttr_h": 3.0, "pm_d": 60, "overrun": 1.05},
    {"type": "ASSEMBLY",    "mtbf_h": 2880, "mttr_h": 1.5, "pm_d": 30, "overrun": 1.08},
    {"type": "INSPECTION",  "mtbf_h": 4320, "mttr_h": 1.0, "pm_d": 30, "overrun": 1.01},
    {"type": "WELDING",     "mtbf_h": 480,  "mttr_h": 5.0, "pm_d": 45, "overrun": 1.07},
    {"type": "PRESS",       "mtbf_h": 600,  "mttr_h": 6.0, "pm_d": 60, "overrun": 1.05},
]


@dataclass
class SimConfig:
    simulation_days:          int   = 540
    target_orders_per_day:    int   = 10
    num_machines:             int   = 8
    num_products:             int   = 15
    num_operators_per_shift:  int   = 8
    seed:                     int   = 42
    machine_oee_mean:         float = 0.75
    machine_oee_std:          float = 0.08
    rework_fraction:          float = 0.80


@dataclass
class ProductEntity:
    product_id:            str
    complexity:            str   # LOW / MEDIUM / HIGH
    complexity_score:      float
    std_setup_min:         float  # standard setup time in minutes
    std_run_min_per_unit:  float  # standard run time per unit in minutes
    base_fpy:              float
    material_bom_complexity: int
    operation_count:       int


@dataclass
class MachineEntity:
    machine_id:            str
    machine_type:          str
    mtbf_hours:            float
    mttr_hours_mean:       float
    pm_interval_days:      int
    setup_overrun_tendency: float
    current_oee:           float
    age_years:             float
    last_maintenance_date: datetime
    free_at:               datetime  = field(default_factory=lambda: SIM_BASE_DATE)
    last_product_id:       Optional[str] = None
    downtime_log:          List[Tuple[datetime, float]] = field(default_factory=list)
    busy_intervals:        List[Tuple[datetime, datetime]] = field(default_factory=list)


@dataclass
class OperatorEntity:
    operator_id:              str
    skill_tier:               str   # junior / mid / senior
    shift_assignment:         str   # morning / afternoon / night
    setup_speed_multiplier:   float
    absenteeism_base_rate:    float
    experience_months:        int
    is_absent_today:          bool = False


@dataclass
class SupplierEntity:
    supplier_id: str
    reliability: float


@dataclass
class CompletedOrder:
    """Lightweight record stored in history for rolling-feature queries."""
    order_id:         str
    product_id:       str
    machine_id:       str
    operator_id:      str
    shift_type:       str
    planned_start:    datetime
    release_time:     datetime
    is_delayed:       int
    setup_overrun:    bool   # actual_setup > 1.5× std_setup (soft threshold)
    first_pass_pass:  bool   # True if quality passed


class SeedManager:
    def __init__(self, master_seed: int) -> None:
        rng = np.random.RandomState(master_seed)
        sub_seeds = rng.randint(0, 2**31 - 1, size=8)
        self.demand      = np.random.RandomState(int(sub_seeds[0]))
        self.breakdown   = np.random.RandomState(int(sub_seeds[1]))
        self.absenteeism = np.random.RandomState(int(sub_seeds[2]))
        self.setup_run   = np.random.RandomState(int(sub_seeds[3]))
        self.quality     = np.random.RandomState(int(sub_seeds[4]))
        self.material    = np.random.RandomState(int(sub_seeds[5]))
        self.scheduler   = np.random.RandomState(int(sub_seeds[6]))
        self.oee_init    = np.random.RandomState(int(sub_seeds[7]))


# ─── Calendar helpers ─────────────────────────────────────────────────────────

def nth_working_day(start: datetime, n: int) -> datetime:
    """Return the datetime n working days after start (Mon-Fri only)."""
    d = start
    count = 0
    while count < n:
        d = d + timedelta(days=1)
        if d.weekday() < 5:
            count += 1
    return d


def add_working_hours(dt: datetime, hours: float) -> datetime:
    """Add hours to dt, skipping full weekend days (Sat/Sun)."""
    result = dt + timedelta(hours=hours)
    days_added = 0
    while True:
        wd = result.weekday()
        if wd < 5:
            break
        if wd == 5:  # Saturday → skip to Monday
            result += timedelta(days=2)
        else:        # Sunday → skip to Monday
            result += timedelta(days=1)
        result = result.replace(hour=6, minute=0, second=0, microsecond=0)
        days_added += 1
        if days_added > 10:
            break
    return result


def shift_for_hour(hour: int) -> str:
    if 6 <= hour < 14:
        return "morning"
    elif 14 <= hour < 22:
        return "afternoon"
    return "night"


def is_month_end_day(dt: datetime) -> bool:
    """True if dt is in the last 4 working days of its calendar month."""
    import calendar
    last_day = calendar.monthrange(dt.year, dt.month)[1]
    end = datetime(dt.year, dt.month, last_day)
    working_days_from_end = 0
    cur = end
    while cur.date() >= dt.date():
        if cur.weekday() < 5:
            working_days_from_end += 1
        if cur.date() == dt.date():
            return working_days_from_end <= 4
        cur -= timedelta(days=1)
    return False


def is_quarter_end_day(dt: datetime) -> bool:
    """True if dt is in the last 4 working days of its calendar quarter."""
    quarter_end_months = {3, 6, 9, 12}
    if dt.month not in quarter_end_months:
        return False
    import calendar
    last_day = calendar.monthrange(dt.year, dt.month)[1]
    end = datetime(dt.year, dt.month, last_day)
    working_days_from_end = 0
    cur = end
    while cur.date() >= dt.date():
        if cur.weekday() < 5:
            working_days_from_end += 1
        if cur.date() == dt.date():
            return working_days_from_end <= 4
        cur -= timedelta(days=1)
    return False


def categorise_delay(minutes: int) -> str:
    if minutes <= 0:
        return "on_time"
    elif minutes <= 60:
        return "minor_delay"
    elif minutes <= 480:
        return "moderate_delay"
    elif minutes <= 1440:
        return "major_delay"
    return "critical_delay"


# ─── Layer 1: Generation ──────────────────────────────────────────────────────

class DemandGenerator:
    def __init__(self, seeds: SeedManager) -> None:
        self._rng = seeds.demand

    def daily_order_count(self, base_rate: float, calendar_dt: datetime) -> int:
        dow_factors = [1.10, 1.05, 1.00, 0.95, 0.90]  # Mon-Fri ±20%
        factor = dow_factors[calendar_dt.weekday()]
        if is_quarter_end_day(calendar_dt):
            factor *= 1.70
        elif is_month_end_day(calendar_dt):
            factor *= 1.35
        return max(1, int(self._rng.poisson(base_rate * factor)))

    def sample_lead_type(self) -> str:
        r = self._rng.uniform()
        if r < 0.30:
            return "rush"
        elif r < 0.80:
            return "normal"
        return "planned"

    def sample_priority(self, lead_type: str) -> str:
        if lead_type == "rush":
            return self._rng.choice(["high", "critical"], p=[0.60, 0.40])
        elif lead_type == "normal":
            return self._rng.choice(["normal", "high"], p=[0.65, 0.35])
        return "normal"

    def sample_quantity(self) -> int:
        raw = self._rng.lognormal(3.5, 0.8)
        return max(3, min(617, int(round(raw))))

    def sample_release_lag(self) -> float:
        raw = self._rng.lognormal(math.log(7.0), 1.04)
        return max(0.8, min(140.0, raw))

    def is_expedited(self, priority: str) -> bool:
        if priority == "critical":
            return bool(self._rng.uniform() < 0.35)
        return False


class SetupRunTimeGenerator:
    def __init__(self, seeds: SeedManager) -> None:
        self._rng = seeds.setup_run

    def sample_setup(
        self,
        std_setup_min: float,
        machine_overrun_tendency: float,
        operator_speed_mult: float,
        changeover_complexity: float,
    ) -> float:
        effective_mean = (
            std_setup_min * machine_overrun_tendency
            * operator_speed_mult * changeover_complexity
        )
        mu = math.log(effective_mean) - 0.5 * 0.28**2
        return float(self._rng.lognormal(mu, 0.28))

    def sample_run(
        self,
        std_run_min: float,
        quantity: int,
        hours_into_shift: float,
    ) -> float:
        base = std_run_min * quantity
        fatigue = 1.0 + max(0.0, (hours_into_shift - 4.0) * 0.025)
        mean = base * fatigue
        return float(max(1.0, self._rng.normal(mean, 0.10 * mean)))


class QualityOutcomeGenerator:
    def __init__(self, seeds: SeedManager) -> None:
        self._rng = seeds.quality

    def passes_inspection(
        self,
        base_fpy: float,
        skill_tier: str,
        current_oee: float,
    ) -> bool:
        skill_adj = {"junior": -0.03, "mid": 0.0, "senior": +0.02}.get(skill_tier, 0.0)
        oee_adj = (current_oee - 0.75) * 0.20  # ±5% over OEE range [0.35, 0.92]
        fpy = max(0.50, min(0.999, base_fpy + skill_adj + oee_adj))
        return bool(self._rng.uniform() < fpy)


class BreakdownGenerator:
    def __init__(self, seeds: SeedManager) -> None:
        self._rng = seeds.breakdown

    def order_breaks_down(
        self,
        machine: MachineEntity,
        days_since_pm: float,
    ) -> bool:
        base = 40.0 / machine.mtbf_hours
        pm_frac = min(1.5, days_since_pm / machine.pm_interval_days)
        pm_factor = 1.0 + pm_frac * 2.0
        oee_factor = 1.0 + max(0.0, (0.75 - machine.current_oee) * 5.0)
        p = min(0.25, base * pm_factor * oee_factor)
        return bool(self._rng.uniform() < p)

    def sample_repair_time(self, mttr_mean: float) -> float:
        return float(max(0.5, self._rng.exponential(mttr_mean)))


class AbsenteeismGenerator:
    def __init__(self, seeds: SeedManager) -> None:
        self._rng = seeds.absenteeism

    def is_absent(self, base_rate: float) -> bool:
        return bool(self._rng.uniform() < base_rate)


class MaterialChecker:
    def __init__(self, seeds: SeedManager) -> None:
        self._rng = seeds.material

    def check_availability(
        self, bom_complexity: int
    ) -> Tuple[bool, int]:
        if self._rng.uniform() > 0.118:
            return True, 0
        shortage_count = int(self._rng.choice([1, 2], p=[0.64, 0.36]))
        return False, shortage_count

    def sample_hold_hours(self, shortage_count: int) -> float:
        base = self._rng.lognormal(1.7, 0.75)
        return float(max(2.0, min(48.0, base * shortage_count)))


# ─── Layer 0: Entity initialisation ──────────────────────────────────────────

def _build_products(config: SimConfig, rng: np.random.RandomState) -> List[ProductEntity]:
    products = []
    # 5 LOW, 5 MEDIUM, 5 HIGH
    complexity_spec = (
        [("LOW",    20.0, 5.0,  2, 3)] * 5 +
        [("MEDIUM", 35.0, 9.0,  5, 4)] * 5 +
        [("HIGH",   60.0, 18.0, 9, 6)] * 5
    )
    for i, (ctype, setup_mean, run_mean, bom_max, ops_mid) in enumerate(complexity_spec, start=1):
        # Vary setup and run times slightly across products
        std_setup = max(5.0, rng.normal(setup_mean, setup_mean * 0.15))
        std_run = max(1.0, rng.lognormal(math.log(run_mean), 0.20))
        bom = int(rng.randint(max(2, bom_max - 3), bom_max + 1))
        # operation_count per routing length ranges
        ops_ranges = {"LOW": (1, 3), "MEDIUM": (2, 5), "HIGH": (4, 7)}
        lo, hi = ops_ranges[ctype]
        ops = int(rng.randint(lo, hi + 1))
        products.append(ProductEntity(
            product_id=f"PROD-{i:03d}",
            complexity=ctype,
            complexity_score=COMPLEXITY_SCORE[ctype],
            std_setup_min=std_setup,
            std_run_min_per_unit=std_run,
            base_fpy=BASE_FPY[ctype],
            material_bom_complexity=bom,
            operation_count=ops,
        ))
    return products


def _build_machines(
    config: SimConfig,
    rng: np.random.RandomState,
    base_date: datetime,
) -> List[MachineEntity]:
    specs = _MACHINE_SPECS[: config.num_machines]
    machines = []
    for i, spec in enumerate(specs, start=1):
        oee = float(np.clip(rng.normal(config.machine_oee_mean, config.machine_oee_std), 0.35, 0.92))
        age = float(rng.uniform(0.5, 8.0))
        # Last maintenance: random within PM interval
        days_since = int(rng.randint(0, spec["pm_d"]))
        last_maint = base_date - timedelta(days=days_since)
        machines.append(MachineEntity(
            machine_id=f"MACH-{i:03d}",
            machine_type=spec["type"],
            mtbf_hours=spec["mtbf_h"],
            mttr_hours_mean=spec["mttr_h"],
            pm_interval_days=spec["pm_d"],
            setup_overrun_tendency=spec["overrun"],
            current_oee=oee,
            age_years=age,
            last_maintenance_date=last_maint,
            free_at=base_date,
        ))
    return machines


def _build_operators(
    config: SimConfig,
    rng: np.random.RandomState,
) -> List[OperatorEntity]:
    operators = []
    idx = 1
    shifts = ["morning", "afternoon", "night"]
    tiers = [("junior", 1, 12, 1.12, 0.065),
             ("mid",    13, 48, 1.05, 0.040),
             ("senior", 49, 180, 0.95, 0.025)]
    for shift in shifts:
        # 2 junior, 4 mid, 2 senior per shift
        tier_counts = [("junior", 2), ("mid", 4), ("senior", 2)]
        for tier_name, count in tier_counts:
            t = next(t for t in tiers if t[0] == tier_name)
            for _ in range(count):
                exp = int(rng.randint(t[1], t[2] + 1))
                operators.append(OperatorEntity(
                    operator_id=f"OPR-{idx:03d}",
                    skill_tier=tier_name,
                    shift_assignment=shift,
                    setup_speed_multiplier=t[3],
                    absenteeism_base_rate=t[4],
                    experience_months=exp,
                ))
                idx += 1
    return operators


def _build_suppliers(rng: np.random.RandomState) -> List[SupplierEntity]:
    return [
        SupplierEntity(f"SUP-{i}", float(rng.uniform(0.80, 0.98)))
        for i in range(1, 4)
    ]


# ─── Layer 2: Engine (Scheduler / StateManager) ───────────────────────────────

class Scheduler:
    def __init__(self, machines: List[MachineEntity], rng: np.random.RandomState) -> None:
        self._machines = machines
        self._rng = rng

    def assign_machine(self, planned_start: datetime) -> MachineEntity:
        """Least-busy machine: prefer machines free at planned_start."""
        free = [m for m in self._machines if m.free_at <= planned_start]
        if free:
            return min(free, key=lambda m: m.free_at)
        return min(self._machines, key=lambda m: m.free_at)

    def assign_operator(
        self,
        operators: List[OperatorEntity],
        shift: str,
        planned_start: datetime,
    ) -> OperatorEntity:
        candidates = [
            op for op in operators
            if op.shift_assignment == shift and not op.is_absent_today
        ]
        if not candidates:
            candidates = [op for op in operators if not op.is_absent_today]
        if not candidates:
            candidates = operators
        return candidates[self._rng.randint(0, len(candidates))]


class StateManager:
    def __init__(self) -> None:
        self.completed_orders: List[CompletedOrder] = []

    def add(self, order: CompletedOrder) -> None:
        self.completed_orders.append(order)

    def query(
        self,
        before: datetime,
        window_days: int,
        product_id: Optional[str] = None,
        machine_id: Optional[str] = None,
        operator_id: Optional[str] = None,
        shift_type: Optional[str] = None,
    ) -> List[CompletedOrder]:
        cutoff = before - timedelta(days=window_days)
        results = []
        for o in self.completed_orders:
            if not (cutoff <= o.planned_start < before):
                continue
            if product_id and o.product_id != product_id:
                continue
            if machine_id and o.machine_id != machine_id:
                continue
            if operator_id and o.operator_id != operator_id:
                continue
            if shift_type and o.shift_type != shift_type:
                continue
            results.append(o)
        return results


# ─── Layer 4: Feature Collection ──────────────────────────────────────────────

class FeatureCollector:
    """Captures all 37 ML features at the prediction point (after changeover,
    before execution events). Leakage-free by construction: no execution outcomes
    are known at snapshot time."""

    def __init__(self, state: StateManager, revision_rng: np.random.RandomState) -> None:
        self._state = state
        self._revision_rng = revision_rng

    def _rate(self, orders: List[CompletedOrder], min_count: int, default: float) -> float:
        if len(orders) < min_count:
            return default
        return sum(o.is_delayed for o in orders) / len(orders)

    def _fpy(self, orders: List[CompletedOrder], min_count: int, default: float) -> float:
        if len(orders) < min_count:
            return default
        return sum(o.first_pass_pass for o in orders) / len(orders)

    def _overrun_rate(self, orders: List[CompletedOrder], min_count: int, default: float) -> float:
        if len(orders) < min_count:
            return default
        return sum(o.setup_overrun for o in orders) / len(orders)

    def snapshot(
        self,
        snapshot_time: datetime,
        creation_dt: datetime,
        planned_start: datetime,
        planned_end: datetime,
        planned_window_h: float,
        release_lag_h: float,
        is_expedited_flag: int,
        priority: str,
        quantity: int,
        product: ProductEntity,
        machine: MachineEntity,
        operator: OperatorEntity,
        changeover_required: int,
        changeover_complexity: float,
        material_available: int,
        shortage_count: int,
    ) -> Dict:

        # ── Temporal ─────────────────────────────────────────────────────────
        is_me = int(is_month_end_day(planned_start))
        is_qe = int(is_quarter_end_day(planned_start))
        dow = float(planned_start.weekday())  # 0=Mon … 4=Fri
        hour = int(planned_start.hour)

        # ── Order planning ────────────────────────────────────────────────────
        planned_lead_time_h = (planned_end - creation_dt).total_seconds() / 3600.0
        estimated_total_h = (product.std_setup_min + product.std_run_min_per_unit * quantity) / 60.0
        tightness = min(1.02, max(0.18, estimated_total_h / planned_window_h))
        _rev_p = {"normal": 0.02, "high": 0.05, "critical": 0.10}.get(priority, 0.02)
        rev_count = float(int(self._revision_rng.uniform() < _rev_p))

        # ── Machine state ──────────────────────────────────────────────────────
        days_since_pm = (snapshot_time - machine.last_maintenance_date).total_seconds() / 86400.0
        days_since_pm = max(0.0, days_since_pm)

        # OEE 30d (approximate from current OEE with small noise)
        oee_30d = float(np.clip(machine.current_oee + np.random.normal(0, 0.01), 0.35, 0.92))

        # Unplanned downtime 30d
        cutoff_30 = snapshot_time - timedelta(days=30)
        down_30d = sum(
            h for (t, h) in machine.downtime_log if t >= cutoff_30
        )

        # Machine utilization in past 24h
        cutoff_24 = snapshot_time - timedelta(hours=24)
        busy_h = sum(
            max(0, (min(e, snapshot_time) - max(s, cutoff_24)).total_seconds() / 3600.0)
            for (s, e) in machine.busy_intervals
            if s < snapshot_time and e > cutoff_24
        )
        utilization = min(1.0, busy_h / 24.0)

        # Queue depth at release (1 if machine is occupied at planned_start)
        queue_depth = float(int(machine.free_at > planned_start))

        # Maintenance due within order window
        pm_due = machine.last_maintenance_date + timedelta(days=machine.pm_interval_days)
        maint_due = int(
            planned_start <= pm_due <= planned_end
        )

        # ── Historical rolling features ────────────────────────────────────────
        prod_90 = self._state.query(snapshot_time, 90, product_id=product.product_id)
        mach_90 = self._state.query(snapshot_time, 90, machine_id=machine.machine_id)
        oper_90 = self._state.query(snapshot_time, 90, operator_id=operator.operator_id)
        prod_mach_90 = self._state.query(snapshot_time, 90,
                                         product_id=product.product_id,
                                         machine_id=machine.machine_id)
        shift_30 = self._state.query(snapshot_time, 30, shift_type=operator.shift_assignment)

        prod_delay_90 = self._rate(prod_90, 3, COLD_START_DEFAULTS["product_delay_rate_90d"])
        mach_delay_90 = self._rate(mach_90, 3, COLD_START_DEFAULTS["machine_delay_rate_90d"])
        oper_delay_90 = self._rate(oper_90, 3, COLD_START_DEFAULTS["operator_delay_rate_90d"])
        pm_delay_90   = self._rate(prod_mach_90, 3, COLD_START_DEFAULTS["product_x_machine_delay_rate_90d"])
        fpy_90        = self._fpy(prod_90, 3, COLD_START_DEFAULTS["product_first_pass_yield_90d"])
        overrun_90    = self._overrun_rate(mach_90, 3, COLD_START_DEFAULTS["machine_setup_overrun_rate_90d"])
        shift_delay_30 = self._rate(shift_30, 3, COLD_START_DEFAULTS["shift_delay_rate_30d"])

        # ── Hours into shift at start ─────────────────────────────────────────
        shift_start_h = {"morning": 6, "afternoon": 14, "night": 22}.get(operator.shift_assignment, 6)
        hrs_into_shift = float((planned_start.hour - shift_start_h) % 24)
        hrs_into_shift = min(7.5, max(0.0, hrs_into_shift))

        return {
            "planned_lead_time_hours":            planned_lead_time_h,
            "release_lag_hours":                  release_lag_h,
            "schedule_revision_count":            rev_count,
            "is_expedited":                       int(is_expedited_flag),
            "priority_encoded":                   PRIORITY_ENCODE[priority],
            "quantity":                           int(quantity),
            "operation_count":                    int(product.operation_count),
            "estimated_total_hours":              float(estimated_total_h),
            "schedule_tightness_ratio":           float(tightness),
            "product_complexity_score":           float(product.complexity_score),
            "material_bom_complexity":            int(product.material_bom_complexity),
            "is_month_end":                       is_me,
            "is_quarter_end":                     is_qe,
            "machine_utilization_at_release":     float(utilization),
            "work_center_queue_depth_at_release": queue_depth,
            "machine_oee_30d":                    float(oee_30d),
            "machine_unplanned_downtime_hours_30d": float(down_30d),
            "days_since_last_planned_maintenance": float(days_since_pm),
            "maintenance_due_within_order_window": maint_due,
            "changeover_required":                int(changeover_required),
            "changeover_complexity_score":        float(changeover_complexity),
            "operator_experience_months":         int(operator.experience_months),
            "operator_skill_tier_encoded":        float(SKILL_ENCODE[operator.skill_tier]),
            "operator_concurrent_order_count":    0.0,
            "hours_into_shift_at_start":          hrs_into_shift,
            "shift_type_encoded":                 int(SHIFT_ENCODE[operator.shift_assignment]),
            "material_availability_at_release":   int(material_available),
            "component_shortage_count":           float(shortage_count),
            "product_delay_rate_90d":             float(prod_delay_90),
            "machine_delay_rate_90d":             float(mach_delay_90),
            "operator_delay_rate_90d":            float(oper_delay_90),
            "product_x_machine_delay_rate_90d":   float(pm_delay_90),
            "product_first_pass_yield_90d":       float(fpy_90),
            "machine_setup_overrun_rate_90d":     float(overrun_90),
            "shift_delay_rate_30d":               float(shift_delay_30),
            "planned_start_day_of_week":          dow,
            "planned_start_hour":                 hour,
        }


# ─── Layer 3: Orchestration ───────────────────────────────────────────────────

class FactorySimulation:

    def __init__(self, config: SimConfig) -> None:
        self.config = config
        self._seeds = SeedManager(config.seed)
        product_rng = np.random.RandomState(config.seed + 1000)
        machine_rng = self._seeds.oee_init
        op_rng      = np.random.RandomState(config.seed + 2000)
        sup_rng     = np.random.RandomState(config.seed + 3000)

        self._products  = _build_products(config, product_rng)
        self._machines  = _build_machines(config, machine_rng, SIM_BASE_DATE)[:config.num_machines]
        self._operators = _build_operators(config, op_rng)
        self._suppliers = _build_suppliers(sup_rng)

        self._demand    = DemandGenerator(self._seeds)
        self._setup_run = SetupRunTimeGenerator(self._seeds)
        self._quality   = QualityOutcomeGenerator(self._seeds)
        self._breakdown = BreakdownGenerator(self._seeds)
        self._absent    = AbsenteeismGenerator(self._seeds)
        self._material  = MaterialChecker(self._seeds)
        self._state     = StateManager()
        self._scheduler = Scheduler(self._machines, self._seeds.scheduler)
        self._collector = FeatureCollector(self._state, np.random.RandomState(config.seed + 4000))

        self._order_counter = 0
        self._records: List[Dict] = []

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self) -> List[Dict]:
        log.info("Starting simulation: %d days, seed=%d", self.config.simulation_days, self.config.seed)
        for day_idx in range(self.config.simulation_days):
            calendar_date = nth_working_day(SIM_BASE_DATE - timedelta(days=1), day_idx + 1)
            self._simulate_day(day_idx, calendar_date)
            if (day_idx + 1) % 60 == 0:
                log.info("  Day %d/%d — %d orders so far",
                         day_idx + 1, self.config.simulation_days, len(self._records))
        log.info("Simulation complete: %d orders generated", len(self._records))
        return self._records

    # ── Per-day simulation ────────────────────────────────────────────────────

    def _simulate_day(self, day_idx: int, calendar_date: datetime) -> None:
        self._update_machines(calendar_date)
        self._process_absenteeism()
        n_orders = self._demand.daily_order_count(
            self.config.target_orders_per_day, calendar_date
        )
        creation_dt = calendar_date.replace(hour=8, minute=0, second=0, microsecond=0)
        for _ in range(n_orders):
            record = self._create_order(creation_dt, calendar_date)
            if record:
                self._records.append(record)

    def _update_machines(self, calendar_date: datetime) -> None:
        for machine in self._machines:
            # Degrade OEE slightly over time; recover after PM
            days_since_pm = (calendar_date - machine.last_maintenance_date).days
            if days_since_pm >= machine.pm_interval_days:
                # Trigger PM, reset
                machine.current_oee = float(np.clip(
                    machine.current_oee + 0.05,
                    machine.current_oee,
                    0.92,
                ))
                machine.last_maintenance_date = calendar_date
            else:
                decay = (days_since_pm / machine.pm_interval_days) * 0.002
                machine.current_oee = max(0.35, machine.current_oee - decay)

    def _process_absenteeism(self) -> None:
        for op in self._operators:
            op.is_absent_today = self._absent.is_absent(op.absenteeism_base_rate)

    # ── Single order simulation (causal core) ─────────────────────────────────

    def _create_order(
        self,
        creation_dt: datetime,
        calendar_date: datetime,
    ) -> Optional[Dict]:
        # ── Assign product, priority, quantity ────────────────────────────────
        complexity_weights = [0.3, 0.4, 0.3]
        complexity = self._seeds.demand.choice(["LOW", "MEDIUM", "HIGH"], p=complexity_weights)
        candidates = [p for p in self._products if p.complexity == complexity]
        product = candidates[self._seeds.scheduler.randint(0, len(candidates))]

        lead_type  = self._demand.sample_lead_type()
        priority   = self._demand.sample_priority(lead_type)
        quantity   = self._demand.sample_quantity()
        expedited  = self._demand.is_expedited(priority)

        # ── Compute estimated_total_hours ──────────────────────────────────────
        estimated_h = (product.std_setup_min + product.std_run_min_per_unit * quantity) / 60.0

        # ── Planned window and planned_end ─────────────────────────────────────
        buf_noise = float(self._seeds.setup_run.lognormal(0.0, 0.10))
        buf_noise = max(0.80, min(1.20, buf_noise))
        priority_buf = PRIORITY_BUFFER[priority] * buf_noise
        planned_window_h = estimated_h * priority_buf

        # ── Release time = planned_start ──────────────────────────────────────
        release_lag_h = self._demand.sample_release_lag()
        planned_start = add_working_hours(creation_dt, release_lag_h)
        planned_end   = planned_start + timedelta(hours=planned_window_h)

        # ── Assign machine and operator ────────────────────────────────────────
        machine  = self._scheduler.assign_machine(planned_start)
        shift    = shift_for_hour(planned_start.hour)
        operator = self._scheduler.assign_operator(self._operators, shift, planned_start)

        # ── Changeover ────────────────────────────────────────────────────────
        changeover_req = int(machine.last_product_id != product.product_id
                             and machine.last_product_id is not None)
        if changeover_req:
            chg_complexity = float(self._seeds.scheduler.uniform(1.5, 3.0))
        else:
            chg_complexity = 1.0

        # ── Material check ────────────────────────────────────────────────────
        mat_available, shortage_count = self._material.check_availability(
            product.material_bom_complexity
        )

        # ── FEATURE SNAPSHOT (prediction point) ───────────────────────────────
        snapshot_time = planned_start
        features = self._collector.snapshot(
            snapshot_time=snapshot_time,
            creation_dt=creation_dt,
            planned_start=planned_start,
            planned_end=planned_end,
            planned_window_h=planned_window_h,
            release_lag_h=release_lag_h,
            is_expedited_flag=int(expedited),
            priority=priority,
            quantity=quantity,
            product=product,
            machine=machine,
            operator=operator,
            changeover_required=changeover_req,
            changeover_complexity=chg_complexity,
            material_available=int(mat_available),
            shortage_count=shortage_count,
        )

        # ── Execution: extra_hours accumulator ────────────────────────────────
        extra_hours = 0.0
        delay_causes: List[str] = []
        days_since_pm = features["days_since_last_planned_maintenance"]

        # Material hold
        if not mat_available:
            hold_h = self._material.sample_hold_hours(shortage_count)
            extra_hours += hold_h
            delay_causes.append("material_unavailability")

        # Machine breakdown
        if self._breakdown.order_breaks_down(machine, days_since_pm):
            repair_h = self._breakdown.sample_repair_time(machine.mttr_hours_mean)
            extra_hours += repair_h
            machine.downtime_log.append((planned_start, repair_h))
            delay_causes.append("machine_breakdown")

        # Queue wait — only fires when machine utilisation > 0.70 (doc's explicit condition).
        # Below that threshold the scheduler finds available capacity without queueing.
        _util = features["machine_utilization_at_release"]
        if _util > 0.70 and machine.free_at > planned_start:
            wait_h = min(24.0, (machine.free_at - planned_start).total_seconds() / 3600.0)
            extra_hours += wait_h
            delay_causes.append("planning_schedule_conflict")

        # Setup time
        actual_setup_min = self._setup_run.sample_setup(
            std_setup_min=product.std_setup_min,
            machine_overrun_tendency=machine.setup_overrun_tendency,
            operator_speed_mult=operator.setup_speed_multiplier,
            changeover_complexity=chg_complexity,
        )
        if actual_setup_min > 1.70 * product.std_setup_min:
            delay_causes.append("setup_overrun")
        setup_overrun_soft = actual_setup_min > 1.50 * product.std_setup_min

        # Run time
        actual_run_min = self._setup_run.sample_run(
            std_run_min=product.std_run_min_per_unit,
            quantity=quantity,
            hours_into_shift=features["hours_into_shift_at_start"],
        )

        # Operator delay (overloaded / fatigue — not separately tracked as root cause)
        # Quality inspection
        passes_qc = self._quality.passes_inspection(
            product.base_fpy, operator.skill_tier, machine.current_oee
        )
        if not passes_qc:
            rework_h = self.config.rework_fraction * features["estimated_total_hours"]
            extra_hours += rework_h
            delay_causes.append("quality_failure_rework")

        # ── Compute actual_end (doc formula)
        # actual_end = planned_start + processing_h + extra_hours
        # extra_hours contains queue_wait (when util>0.70), material hold, repair, rework.
        processing_h = min(200.0, (actual_setup_min + actual_run_min) / 60.0)
        extra_hours  = min(120.0, extra_hours)
        actual_end   = planned_start + timedelta(hours=processing_h + extra_hours)

        # ── Update machine state (physical scheduling) ─────────────────────────
        # Physical machine occupancy: starts when machine is free, takes processing_h.
        phys_start = max(machine.free_at, planned_start)
        phys_end   = phys_start + timedelta(hours=processing_h)
        machine.busy_intervals.append((phys_start, phys_end))
        if len(machine.busy_intervals) > 300:
            machine.busy_intervals = machine.busy_intervals[-150:]
        machine.free_at = phys_end
        machine.last_product_id = product.product_id

        # ── Outcomes ──────────────────────────────────────────────────────────
        is_delayed = int(actual_end > planned_end)
        delay_min  = max(0, int((actual_end - planned_end).total_seconds() / 60.0))

        if not is_delayed:
            root_cause = "none"
        elif len(delay_causes) == 0:
            root_cause = "none"
        elif len(delay_causes) == 1:
            root_cause = delay_causes[0]
        else:
            root_cause = "multiple_causes"

        delay_cat = categorise_delay(delay_min)

        # ── Update order history ───────────────────────────────────────────────
        self._order_counter += 1
        order_id = f"ORD-{self._order_counter:06d}"
        po_id    = f"PO-{self._order_counter:05d}"

        self._state.add(CompletedOrder(
            order_id=order_id,
            product_id=product.product_id,
            machine_id=machine.machine_id,
            operator_id=operator.operator_id,
            shift_type=shift,
            planned_start=planned_start,
            release_time=planned_start,
            is_delayed=is_delayed,
            setup_overrun=setup_overrun_soft,
            first_pass_pass=passes_qc,
        ))

        # ── Assemble full record ───────────────────────────────────────────────
        record = {**features}
        record["is_delayed"]       = is_delayed
        record["delay_minutes"]    = delay_min
        record["delay_category"]   = delay_cat
        record["delay_root_cause"] = root_cause
        record["order_id"]         = order_id
        record["po_number"]        = po_id
        record["product_id"]       = product.product_id
        record["machine_id"]       = machine.machine_id
        record["operator_id"]      = operator.operator_id
        record["priority"]         = priority
        record["shift_type"]       = shift
        record["planned_start"]    = planned_start.isoformat()
        record["actual_end"]       = actual_end.isoformat()
        return record


# ─── Layer 5: Output ─────────────────────────────────────────────────────────

class DatasetBuilder:
    """Converts raw records to a schema-validated 50-column DataFrame."""

    COLUMN_ORDER: Tuple[str, ...] = FEATURE_COLS + TARGET_COLS + METADATA_COLS

    DTYPE_MAP: Dict[str, str] = {
        "planned_lead_time_hours": "float64",
        "release_lag_hours": "float64",
        "schedule_revision_count": "float64",
        "is_expedited": "int64",
        "priority_encoded": "int64",
        "quantity": "int64",
        "operation_count": "int64",
        "estimated_total_hours": "float64",
        "schedule_tightness_ratio": "float64",
        "product_complexity_score": "float64",
        "material_bom_complexity": "int64",
        "is_month_end": "int64",
        "is_quarter_end": "int64",
        "machine_utilization_at_release": "float64",
        "work_center_queue_depth_at_release": "float64",
        "machine_oee_30d": "float64",
        "machine_unplanned_downtime_hours_30d": "float64",
        "days_since_last_planned_maintenance": "float64",
        "maintenance_due_within_order_window": "int64",
        "changeover_required": "int64",
        "changeover_complexity_score": "float64",
        "operator_experience_months": "int64",
        "operator_skill_tier_encoded": "float64",
        "operator_concurrent_order_count": "float64",
        "hours_into_shift_at_start": "float64",
        "shift_type_encoded": "int64",
        "material_availability_at_release": "int64",
        "component_shortage_count": "float64",
        "product_delay_rate_90d": "float64",
        "machine_delay_rate_90d": "float64",
        "operator_delay_rate_90d": "float64",
        "product_x_machine_delay_rate_90d": "float64",
        "product_first_pass_yield_90d": "float64",
        "machine_setup_overrun_rate_90d": "float64",
        "shift_delay_rate_30d": "float64",
        "planned_start_day_of_week": "float64",
        "planned_start_hour": "int64",
        "is_delayed": "int64",
        "delay_minutes": "int64",
        "delay_category": "object",
        "delay_root_cause": "object",
        "order_id": "object",
        "po_number": "object",
        "product_id": "object",
        "machine_id": "object",
        "operator_id": "object",
        "priority": "object",
        "shift_type": "object",
        "planned_start": "object",
        "actual_end": "object",
    }

    def build(self, records: List[Dict]) -> pd.DataFrame:
        df = pd.DataFrame(records, columns=list(self.COLUMN_ORDER))
        for col, dtype in self.DTYPE_MAP.items():
            if col in df.columns:
                try:
                    df[col] = df[col].astype(dtype)
                except (ValueError, TypeError):
                    pass
        self._validate(df)
        return df

    def _validate(self, df: pd.DataFrame) -> None:
        assert len(df.columns) == 50, f"Expected 50 columns, got {len(df.columns)}"
        assert df.isnull().sum().sum() == 0, "Null values detected — leakage risk"
        for col in FEATURE_COLS:
            assert col in df.columns, f"Missing feature: {col}"
        for col in TARGET_COLS:
            assert col in df.columns, f"Missing target: {col}"
        for col in METADATA_COLS:
            assert col in df.columns, f"Missing metadata: {col}"

    @staticmethod
    def split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        df_sorted = df.sort_values("planned_start").reset_index(drop=True)
        n = len(df_sorted)
        n_train = int(round(n * 0.71))
        n_val   = int(round(n * 0.18))
        train = df_sorted.iloc[:n_train].copy()
        val   = df_sorted.iloc[n_train:n_train + n_val].copy()
        test  = df_sorted.iloc[n_train + n_val:].copy()
        return train, val, test


class CalibrationChecker:
    """Validates five assertions from the architecture specification."""

    def check(self, df: pd.DataFrame) -> Dict:
        results: Dict = {}
        delayed = df[df["is_delayed"] == 1]
        n_total = len(df)
        n_delayed = len(delayed)
        delay_rate = n_delayed / n_total if n_total > 0 else 0.0

        # 1. Delay rate 22–40%
        results["delay_rate"] = {
            "value": round(delay_rate * 100, 2),
            "pass": 22.0 <= delay_rate * 100 <= 40.0,
            "target": "22–40%",
        }

        # 2. Utilisation causal
        high_util = df[df["machine_utilization_at_release"] > 0.70]["is_delayed"].mean()
        low_util  = df[df["machine_utilization_at_release"] <= 0.70]["is_delayed"].mean()
        results["utilisation_causal"] = {
            "high_util_delay_rate": round(float(high_util or 0) * 100, 2),
            "low_util_delay_rate":  round(float(low_util or 0) * 100, 2),
            "pass": bool(high_util > low_util),
            "target": "high_util_delay > low_util_delay",
        }

        # 3. Material causal
        no_mat  = df[df["material_availability_at_release"] == 0]["is_delayed"].mean()
        has_mat = df[df["material_availability_at_release"] == 1]["is_delayed"].mean()
        results["material_causal"] = {
            "no_material_delay_rate": round(float(no_mat or 0) * 100, 2),
            "has_material_delay_rate": round(float(has_mat or 0) * 100, 2),
            "pass": bool(no_mat > has_mat),
            "target": "no_material_delay > has_material_delay (69.3% vs 32.1%)",
        }

        # 4. Operator skill causal
        senior = df[df["operator_skill_tier_encoded"] == 2.0]["is_delayed"].mean()
        junior = df[df["operator_skill_tier_encoded"] == 0.0]["is_delayed"].mean()
        results["skill_causal"] = {
            "senior_delay_rate": round(float(senior or 0) * 100, 2),
            "junior_delay_rate": round(float(junior or 0) * 100, 2),
            "pass": bool(senior <= junior),
            "target": "senior_delay ≤ junior_delay",
        }

        # 5. Feature completeness
        missing = int(df[list(FEATURE_COLS)].isnull().sum().sum())
        results["feature_completeness"] = {
            "missing_values": missing,
            "pass": missing == 0,
            "target": "0 missing values",
        }

        # Root cause distribution
        if n_delayed > 0:
            rc_dist = (
                delayed["delay_root_cause"]
                .value_counts(normalize=True)
                .mul(100)
                .round(2)
                .to_dict()
            )
        else:
            rc_dist = {}
        results["root_cause_distribution"] = rc_dist

        all_pass = all(v["pass"] for v in results.values() if isinstance(v, dict) and "pass" in v)
        results["all_pass"] = all_pass
        return results


class ReportGenerator:
    def generate(
        self,
        df: pd.DataFrame,
        config: SimConfig,
        calibration: Dict,
        train: pd.DataFrame,
        val: pd.DataFrame,
        test: pd.DataFrame,
    ) -> Dict:
        n = len(df)
        delayed = df[df["is_delayed"] == 1]
        return {
            "run_parameters": {
                "simulation_days": config.simulation_days,
                "target_orders_per_day": config.target_orders_per_day,
                "num_machines": config.num_machines,
                "seed": config.seed,
                "rework_fraction": config.rework_fraction,
            },
            "dataset_summary": {
                "total_rows": n,
                "total_columns": len(df.columns),
                "feature_columns": len(FEATURE_COLS),
                "target_columns": len(TARGET_COLS),
                "metadata_columns": len(METADATA_COLS),
                "delay_rate_pct": round(len(delayed) / n * 100, 2) if n > 0 else 0,
                "mean_delay_minutes": round(float(delayed["delay_minutes"].mean()), 1) if len(delayed) > 0 else 0,
                "median_delay_minutes": round(float(delayed["delay_minutes"].median()), 1) if len(delayed) > 0 else 0,
                "p95_delay_minutes": round(float(delayed["delay_minutes"].quantile(0.95)), 1) if len(delayed) > 0 else 0,
            },
            "root_cause_distribution": calibration.get("root_cause_distribution", {}),
            "calibration_checks": {
                k: v for k, v in calibration.items()
                if k not in ("root_cause_distribution", "all_pass")
            },
            "calibration_all_pass": calibration.get("all_pass", False),
            "train_val_test_split": {
                "train_rows": len(train),
                "val_rows": len(val),
                "test_rows": len(test),
                "train_pct": round(len(train) / n * 100, 1) if n > 0 else 0,
                "val_pct":   round(len(val) / n * 100, 1) if n > 0 else 0,
                "test_pct":  round(len(test) / n * 100, 1) if n > 0 else 0,
            },
            "schema_contract": {
                "column_count_valid": len(df.columns) == 50,
                "missing_values": int(df.isnull().sum().sum()),
            },
        }


# ─── CLI entry point ──────────────────────────────────────────────────────────

def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate synthetic manufacturing data for MPC ML pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--days",            type=int, default=540,         help="Simulation days")
    parser.add_argument("--orders-per-day",  type=int, default=10,          help="Mean orders per day")
    parser.add_argument("--machines",        type=int, default=8,           help="Number of machines")
    parser.add_argument("--seed",            type=int, default=42,          help="Random seed")
    parser.add_argument("--output-dir",      type=str, default="ml/data",   help="Output directory")
    parser.add_argument("--rework-fraction", type=float, default=0.80,      help="Rework fraction")
    parser.add_argument("--validate-only",   action="store_true",           help="Run 120-day validation, no 540d output")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)

    days = 120 if args.validate_only else args.days
    config = SimConfig(
        simulation_days=days,
        target_orders_per_day=args.orders_per_day,
        num_machines=args.machines,
        seed=args.seed,
        rework_fraction=args.rework_fraction,
    )

    # ── Run simulation ────────────────────────────────────────────────────────
    sim = FactorySimulation(config)
    records = sim.run()

    if not records:
        log.error("No orders generated — check simulation parameters.")
        return 1

    # ── Build dataset ─────────────────────────────────────────────────────────
    builder = DatasetBuilder()
    try:
        df = builder.build(records)
    except AssertionError as exc:
        log.error("Schema validation failed: %s", exc)
        return 1

    log.info("Dataset: %d rows × %d columns", len(df), len(df.columns))

    # ── Calibration check ──────────────────────────────────────────────────────
    checker   = CalibrationChecker()
    calib     = checker.check(df)
    status    = "PASS" if calib["all_pass"] else "WARN"
    log.info("Calibration: %s", status)
    for key, val in calib.items():
        if isinstance(val, dict) and "pass" in val:
            icon = "✓" if val["pass"] else "✗"
            log.info("  %s %s: %s", icon, key, val)

    # ── Train/val/test split ───────────────────────────────────────────────────
    train, val, test = DatasetBuilder.split(df)
    log.info("Split: train=%d  val=%d  test=%d", len(train), len(val), len(test))

    # ── Write outputs ──────────────────────────────────────────────────────────
    out_root = Path(args.output_dir)
    raw_dir  = out_root / "raw"
    proc_dir = out_root / "processed"
    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    if not args.validate_only:
        df.to_csv(raw_dir / "synthetic_factory_data.csv", index=False)
        log.info("Wrote %s", raw_dir / "synthetic_factory_data.csv")

        train.to_csv(proc_dir / "train.csv", index=False)
        val.to_csv(proc_dir / "val.csv",   index=False)
        test.to_csv(proc_dir / "test.csv",  index=False)
        log.info("Wrote train/val/test splits")
    else:
        log.info("[--validate-only] Skipping file writes for 120-day run")

    report = ReportGenerator().generate(df, config, calib, train, val, test)
    report_path = raw_dir / "simulation_report.json"
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)
    log.info("Wrote %s", report_path)

    # ── Summary ────────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Orders: %d  |  Delayed: %.1f%%  |  Calibration: %s",
             len(df),
             calib["delay_rate"]["value"],
             status)
    rc = calib.get("root_cause_distribution", {})
    for cause in ("setup_overrun", "material_unavailability", "machine_breakdown",
                  "quality_failure_rework", "none", "multiple_causes",
                  "planning_schedule_conflict"):
        log.info("  %-32s %5.1f%%", cause, rc.get(cause, 0.0))
    log.info("=" * 60)

    return 0 if calib["all_pass"] else 2


if __name__ == "__main__":
    sys.exit(main())
