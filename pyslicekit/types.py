"""
pyslicekit.types
~~~~~~~~~~~~~~~~
Core data structures used throughout the library.

Usage:
    from pyslicekit.types import SliceResult, MetricDirection, METRIC_REGISTRY
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class MetricDirection(Enum):
    """
    Describes whether a higher or lower metric value is better.

    Used by the renderer to flip the colour scale so that
    'red always means bad' regardless of metric type.

    - LOWER_IS_BETTER: MAE, RMSE  → positive gap is bad (segment is worse)
    - HIGHER_IS_BETTER: accuracy, F1, R² → negative gap is bad (segment is worse)
    """

    LOWER_IS_BETTER = "lower_is_better"
    HIGHER_IS_BETTER = "higher_is_better"


# ---------------------------------------------------------------------------
# Metric registry
# Maps each supported metric name → its direction.
# Renderer and stats modules consult this; never hardcode direction elsewhere.
# ---------------------------------------------------------------------------

METRIC_REGISTRY: Dict[str, MetricDirection] = {
    "accuracy": MetricDirection.HIGHER_IS_BETTER,
    "f1": MetricDirection.HIGHER_IS_BETTER,
    "f1_macro": MetricDirection.HIGHER_IS_BETTER,
    "f1_weighted": MetricDirection.HIGHER_IS_BETTER,
    "precision": MetricDirection.HIGHER_IS_BETTER,
    "recall": MetricDirection.HIGHER_IS_BETTER,
    "r2": MetricDirection.HIGHER_IS_BETTER,
    "mae": MetricDirection.LOWER_IS_BETTER,
    "rmse": MetricDirection.LOWER_IS_BETTER,
    "mse": MetricDirection.LOWER_IS_BETTER,
}

SUPPORTED_METRICS = list(METRIC_REGISTRY.keys())


@dataclass
class SliceResult:
    """
    Holds the evaluation result for a single data segment.

    A segment is defined by one or more (column, value) pairs.
    For example: [("gender", "female"), ("region", "north")]

    Attributes
    ----------
    slice_def : list of (column, value) tuples
        The column-value pairs that define this segment.
        Single-column slice: [("gender", "female")]
        Two-column slice:   [("gender", "female"), ("age_bin", "Q1")]

    n : int
        Number of rows in this segment.

    metric_name : str
        The metric computed (e.g. "accuracy", "mae").

    metric_value : float
        The metric value for this segment.

    overall_metric : float
        The metric value across the full test set (baseline).

    gap : float
        metric_value - overall_metric.
        Sign interpretation depends on MetricDirection:
        
        - HIGHER_IS_BETTER → negative gap = segment underperforms
        - LOWER_IS_BETTER  → positive gap = segment underperforms

    is_significant : bool
        True if the gap is statistically significant (p < 0.05).
        Set to False when n < 30 (test unreliable at small n).

    low_n : bool
        True when n < min_samples. Result is included but flagged.
        Renderer displays a warning overlay on these cells.

    p_value : float or None
        The p-value from the significance test.
        None when the test could not be run (e.g. n=0, all same label).

    test_used : str or None
        Name of the statistical test applied:
        "proportion_z", "fisher_exact", "bootstrap_ci", or None.

    extra : dict
        Reserved for future use (confidence intervals, etc.).
    """

    slice_def: List[Tuple[str, Any]]
    n: int
    metric_name: str
    metric_value: float
    overall_metric: float
    gap: float
    is_significant: bool = False
    low_n: bool = False
    p_value: Optional[float] = None
    test_used: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    @property
    def label(self) -> str:
        """
        Human-readable segment label, e.g. 'gender=female & age_bin=Q1'.
        Used by the renderer for axis labels and CSV column headers.
        """
        parts = [f"{col}={val}" for col, val in self.slice_def]
        return " & ".join(parts)

    @property
    def direction(self) -> MetricDirection:
        """Looks up the metric direction from the registry."""
        return METRIC_REGISTRY[self.metric_name]

    @property
    def is_underperforming(self) -> bool:
        """
        True when the segment genuinely performs worse than baseline,
        taking metric direction into account.
        """
        if self.direction == MetricDirection.HIGHER_IS_BETTER:
            return self.gap < 0
        return self.gap > 0

    @property
    def abs_gap(self) -> float:
        """Absolute gap — used for sort ordering."""
        return abs(self.gap)

    def __repr__(self) -> str:
        sig_marker = "*" if self.is_significant else ""
        low_marker = " [low-n]" if self.low_n else ""
        return (
            f"SliceResult({self.label!r}, n={self.n}, "
            f"{self.metric_name}={self.metric_value:.4f}, "
            f"gap={self.gap:+.4f}{sig_marker}{low_marker})"
        )
