"""
ml/tests/conftest.py

Pytest configuration for the mpc_ml test suite.

Adds ``ml/src`` to ``sys.path`` so that ``mpc_ml`` is importable without a
package install step.  This mirrors how the package will be consumed once a
``pyproject.toml`` is added to ``ml/``.
"""
from __future__ import annotations

import sys
from pathlib import Path

# ml/src must be on sys.path before any test module imports mpc_ml
_ML_SRC = str(Path(__file__).parent.parent / "src")
if _ML_SRC not in sys.path:
    sys.path.insert(0, _ML_SRC)
