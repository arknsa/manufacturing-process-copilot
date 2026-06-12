"""
frontend/utils/formatting.py
==============================
Pure formatting helpers — no Streamlit, no HTTP.
These are safe to import from anywhere including tests.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Risk / probability
# ---------------------------------------------------------------------------

def prob_to_risk_label(p: float) -> str:
    """Return a human-readable risk tier for a delay probability."""
    if p >= 0.65:
        return "High Risk"
    if p >= 0.40:
        return "Medium Risk"
    return "Low Risk"


def risk_colour(p: float) -> str:
    """Return a hex colour for a delay probability (red / amber / green)."""
    if p >= 0.65:
        return "#e74c3c"
    if p >= 0.40:
        return "#f39c12"
    return "#2ecc71"


def risk_emoji(p: float) -> str:
    if p >= 0.65:
        return "🔴"
    if p >= 0.40:
        return "🟡"
    return "🟢"


def format_probability(p: float) -> str:
    """Format as a percentage string: 0.723 → '72.3%'."""
    return f"{p * 100:.1f}%"


# ---------------------------------------------------------------------------
# Time / duration
# ---------------------------------------------------------------------------

def minutes_to_display(minutes: int | float) -> str:
    """Convert an integer minute count to a human-readable string.

    Examples: 45 → '45m',  90 → '1h 30m',  120 → '2h'
    """
    minutes = int(minutes)
    if minutes < 60:
        return f"{minutes}m"
    hours, rem = divmod(minutes, 60)
    return f"{hours}h" if rem == 0 else f"{hours}h {rem}m"


def format_duration_hours(hours: float) -> str:
    """Convert fractional hours to a display string: 1.5 → '1h 30m'."""
    return minutes_to_display(int(hours * 60))


# ---------------------------------------------------------------------------
# Domain label mappings
# ---------------------------------------------------------------------------

_ROOT_CAUSE_LABELS: dict[str, str] = {
    "material_unavailability": "Material Unavailability",
    "machine_breakdown": "Machine Breakdown",
    "operator_shortage": "Operator Shortage",
    "schedule_overrun": "Schedule Overrun",
    "quality_rework": "Quality Rework",
    "setup_overrun": "Setup Overrun",
    "changeover_delay": "Changeover Delay",
    "capacity_constraint": "Capacity Constraint",
}


def root_cause_to_display(rc: str) -> str:
    return _ROOT_CAUSE_LABELS.get(rc, rc.replace("_", " ").title())


_CONFIDENCE_COLOURS: dict[str, str] = {
    "high": "#1abc9c",
    "medium": "#3498db",
    "low": "#95a5a6",
}


def confidence_badge_colour(confidence: str) -> str:
    return _CONFIDENCE_COLOURS.get(confidence.lower(), "#95a5a6")


_STATUS_COLOURS: dict[str, str] = {
    "pending": "#3498db",
    "in_progress": "#f39c12",
    "completed": "#2ecc71",
    "delayed": "#e74c3c",
    "cancelled": "#95a5a6",
}


def status_colour(status: str) -> str:
    return _STATUS_COLOURS.get(status.lower(), "#95a5a6")


def status_to_display(status: str) -> str:
    return status.replace("_", " ").title()
