"""
backend/app/schemas/predictions.py
=====================================
Pydantic request / response schemas for the ML prediction endpoints.

OrderFeatures   — 38 raw input features (matches mpc_ml FEATURE_COLS).
                  The 8 cold-start rolling features are Optional[float]
                  so callers can omit them for new products / machines.
DelayPrediction — full prediction response; mirrors ExplanationResult.
Batch*          — wrappers for the batch endpoint.
ModelInfo       — metadata returned by GET /models/current.
FeatureImportanceItem — one row from GET /models/feature-importance.
"""

from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class OrderFeatures(BaseModel):
    # ── Order planning ───────────────────────────────────────────────────
    planned_lead_time_hours: float
    release_lag_hours: float
    schedule_revision_count: float
    is_expedited: int
    priority_encoded: int
    quantity: int
    operation_count: int
    estimated_total_hours: float
    schedule_tightness_ratio: float
    # ── Product characteristics ──────────────────────────────────────────
    product_complexity_score: float
    material_bom_complexity: int
    # ── Temporal flags ───────────────────────────────────────────────────
    is_month_end: int
    is_quarter_end: int
    # ── Machine state ────────────────────────────────────────────────────
    machine_utilization_at_release: float
    work_center_queue_depth_at_release: float
    machine_oee_30d: float
    machine_unplanned_downtime_hours_30d: float
    days_since_last_planned_maintenance: float
    maintenance_due_within_order_window: int
    changeover_required: int
    changeover_complexity_score: float
    # ── Operator state ───────────────────────────────────────────────────
    operator_experience_months: int
    operator_skill_tier_encoded: float
    operator_concurrent_order_count: float = 0.0
    hours_into_shift_at_start: float
    shift_type_encoded: int
    # ── Material state ───────────────────────────────────────────────────
    material_availability_at_release: int
    component_shortage_count: float
    # ── Historical rolling (optional — cold-start NaN allowed) ───────────
    product_delay_rate_90d: Optional[float] = None
    machine_delay_rate_90d: Optional[float] = None
    operator_delay_rate_90d: Optional[float] = None
    product_x_machine_delay_rate_90d: Optional[float] = None
    product_first_pass_yield_90d: Optional[float] = None
    machine_setup_overrun_rate_90d: Optional[float] = None
    shift_delay_rate_30d: Optional[float] = None
    machine_avg_delay_minutes_90d: Optional[float] = None
    # ── Additional temporal ──────────────────────────────────────────────
    planned_start_day_of_week: float
    planned_start_hour: int

    def to_feature_dict(self) -> dict:
        """Return a flat dict with None values replaced by float('nan')."""
        return {
            k: (float("nan") if v is None else v)
            for k, v in self.model_dump().items()
        }


class BatchPredictionRequest(BaseModel):
    orders: List[OrderFeatures] = Field(..., max_length=100)


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class FactorExplanationResponse(BaseModel):
    feature_name: str
    human_label: str
    value: Any
    shap_contribution: float
    direction: str   # 'increases_risk' | 'reduces_risk'
    magnitude: str   # 'high' | 'medium' | 'low'


class DelayPrediction(BaseModel):
    delay_probability: float = Field(..., ge=0.0, le=1.0)
    delay_minutes_estimate: Optional[float] = None
    root_cause: str
    confidence: str  # 'high' | 'medium' | 'low'
    top_risk_factors: List[FactorExplanationResponse]
    mitigating_factors: List[FactorExplanationResponse]
    narrative: str


class BatchPredictionResponse(BaseModel):
    predictions: List[DelayPrediction]
    count: int


class ModelInfo(BaseModel):
    binary_run_id: str
    regression_run_id: str
    root_cause_run_id: str
    feature_count: int
    loaded_at: str


class FeatureImportanceItem(BaseModel):
    rank: int
    feature_name: str
    human_label: str
    mean_abs_shap: float
