"""
resp_metrics — Cycle-by-cycle ventilatory metric computation.

This package provides tools to compute standard ventilatory variables from flow
(and optionally volume) signals, using inspiratory ("INSPI") and expiratory
("EXPI") comment annotations exported from LabChart.

Key functions:
- cycles_from_comments: Identify respiratory cycles based on comment timestamps.
- ventilatory_from_cycles: Spontaneous-breathing ventilatory variables.
- mechanical_from_cycles: Mechanical-ventilation per-cycle variables (ventilatory + mechanical).
- compute_from_labchart: High-level one-call API.
"""

from .cycles import cycles_from_comments
from .ventilatory import ventilatory_from_cycles
from .mechanical_vent import mechanical_from_cycles
from .api import compute_from_labchart

__all__ = [
    "cycles_from_comments",
    "ventilatory_from_cycles",
    "mechanical_from_cycles",
    "compute_from_labchart",
]

__version__ = "0.1.0"