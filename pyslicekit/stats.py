"""
pyslicekit.stats
~~~~~~~~~~~~~~~~
Metric computation, gap scoring, and statistical significance testing.

Responsibilities:
  1. Metric dispatcher — computes the correct sklearn metric per segment
  2. Gap scorer — segment_metric minus overall_metric (signed)
  3. Significance flagger — picks the right test automatically:
       classifiers: proportion z-test (n>=30) or Fisher's exact (n<30)
       regression:  bootstrap confidence interval (1000 resamples)
  4. Sample-size annotator — attaches low_n flag and warns in message

Usage:
    from pyslicekit.stats import compute_overall_metric, evaluate_segment
"""

from __future__ import annotations

import math
import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats as scipy_stats

from pyslicekit.exceptions import PySliceKitMetricError
from pyslicekit.types import METRIC_REGISTRY, MetricDirection, SliceResult

# ---------------------------------------------------------------------------
# Metric computation helpers
# ---------------------------------------------------------------------------

# Lazy import of sklearn metrics to avoid import-time cost
def _get_metric_fn(metric: str):
    """Return the sklearn metric function for the given metric name."""
    from sklearn import metrics as skm

    _dispatch = {
        "accuracy":    skm.accuracy_score,
        "f1":          lambda yt, yp: float("nan") if (np.sum(yt) == 0 and np.sum(yp) == 0) else skm.f1_score(yt, yp, average="binary", zero_division=0),
        "f1_macro":    lambda yt, yp: skm.f1_score(yt, yp, average="macro", zero_division=0),
        "f1_weighted": lambda yt, yp: skm.f1_score(yt, yp, average="weighted", zero_division=0),
        "precision":   lambda yt, yp: skm.precision_score(yt, yp, average="binary", zero_division=0),
        "recall":      lambda yt, yp: skm.recall_score(yt, yp, average="binary", zero_division=0),
        "r2":          skm.r2_score,
        "mae":         skm.mean_absolute_error,
        "rmse":        lambda yt, yp: math.sqrt(skm.mean_squared_error(yt, yp)),
        "mse":         skm.mean_squared_error,
    }
    return _dispatch[metric]


# ---------------------------------------------------------------------------
# Public: compute overall baseline metric
# ---------------------------------------------------------------------------


def compute_overall_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str,
) -> float:
    """
    Compute the metric across the entire test set.

    This is the baseline every segment gap is measured against.

    Parameters
    ----------
    y_true : ground truth
    y_pred : predictions
    metric : metric name from SUPPORTED_METRICS

    Returns
    -------
    float
    """
    fn = _get_metric_fn(metric)
    try:
        return float(fn(y_true, y_pred))
    except Exception as exc:
        raise PySliceKitMetricError(
            f"Failed to compute overall {metric!r}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Public: evaluate one segment → SliceResult
# ---------------------------------------------------------------------------


def evaluate_segment(
    segment: Dict[str, Any],
    metric: str,
    overall_metric: float,
    is_regression: bool,
    min_samples: int,
) -> SliceResult:
    """
    Compute metric, gap, and significance for one segment dict.

    Parameters
    ----------
    segment : dict produced by slicer.build_segments()
    metric : metric name string
    overall_metric : full-dataset baseline metric value
    is_regression : True for regression tasks, False for classification
    min_samples : floor below which low_n=True is set

    Returns
    -------
    SliceResult
    """
    y_true = segment["y_true"]
    y_pred = segment["y_pred"]
    n = segment["n"]
    low_n = segment["low_n"]

    # --- metric value ---
    metric_value = _safe_metric(y_true, y_pred, metric, segment["slice_def"])

    # --- gap ---
    if metric_value is None or math.isnan(metric_value):
        gap = float("nan")
        metric_value = float("nan")
    else:
        gap = metric_value - overall_metric

    # --- significance ---
    is_significant, p_value, test_used = _run_significance_test(
        y_true=y_true,
        y_pred=y_pred,
        n=n,
        metric=metric,
        overall_metric=overall_metric,
        metric_value=metric_value,
        is_regression=is_regression,
        low_n=low_n,
    )

    return SliceResult(
        slice_def=segment["slice_def"],
        n=n,
        metric_name=metric,
        metric_value=metric_value,
        overall_metric=overall_metric,
        gap=gap,
        is_significant=is_significant,
        low_n=low_n,
        p_value=p_value,
        test_used=test_used,
    )


# ---------------------------------------------------------------------------
# Metric computation (safe — returns NaN on structural failure)
# ---------------------------------------------------------------------------


def _safe_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str,
    slice_def: List[Tuple[str, Any]],
) -> Optional[float]:
    """
    Compute the metric, returning NaN on structural failures.

    Structural failures (e.g. all same label in y_true making F1 undefined)
    are surfaced as NaN in SliceResult, not as exceptions.
    Non-structural failures (metric function itself is broken) re-raise.
    """
    fn = _get_metric_fn(metric)
    try:
        val = fn(y_true, y_pred)
        return float(val)
    except ZeroDivisionError:
        return float("nan")
    except ValueError as exc:
        # Structural: single class in segment for binary metric etc.
        warnings.warn(
            f"Could not compute {metric!r} for segment "
            f"{[f'{c}={v}' for c, v in slice_def]}: {exc}. "
            "metric_value will be NaN for this segment.",
            UserWarning,
            stacklevel=4,
        )
        return float("nan")
    except Exception as exc:
        raise PySliceKitMetricError(
            f"Unexpected error computing {metric!r} for segment "
            f"{slice_def!r}: {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Significance testing — three code paths
# ---------------------------------------------------------------------------


def _run_significance_test(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n: int,
    metric: str,
    overall_metric: float,
    metric_value: float,
    is_regression: bool,
    low_n: bool,
) -> Tuple[bool, Optional[float], Optional[str]]:
    """
    Choose and run the appropriate significance test.

    Returns (is_significant, p_value, test_name)

    Test selection logic:
      - NaN metric_value → cannot test → (False, None, None)
      - low_n (n < min_samples) → unreliable → (False, None, "skipped_low_n")
      - regression → bootstrap CI
      - classification, n >= 30 → proportion z-test
      - classification, n < 30  → Fisher's exact test
    """
    # Cannot test on NaN
    if math.isnan(metric_value) or math.isnan(overall_metric):
        return False, None, None

    # Low-n: compute but mark unreliable
    if low_n:
        return False, None, "skipped_low_n"

    if is_regression:
        return _bootstrap_ci_test(y_true, y_pred, metric, overall_metric)

    # Classification path
    if n >= 30:
        return _proportion_z_test(y_true, y_pred, overall_metric)
    else:
        return _fisher_exact_test(y_true, y_pred, overall_metric)


def _proportion_z_test(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    overall_accuracy: float,
) -> Tuple[bool, float, str]:
    """
    Two-proportion z-test comparing segment accuracy to overall accuracy.

    Uses the overall accuracy as the null proportion.
    p < 0.05 → significant.

    Only valid for binary-like accuracy comparison.
    For multi-class F1 metrics this is an approximation.
    """
    n = len(y_true)
    # Number of correct predictions in this segment
    n_correct = int(np.sum(y_true == y_pred))
    p_segment = n_correct / n
    p_overall = overall_accuracy

    # Standard error under the null
    se = math.sqrt(p_overall * (1 - p_overall) / n)
    if se == 0:
        return False, 1.0, "proportion_z"

    z = (p_segment - p_overall) / se
    # Two-tailed p-value
    p_value = float(2 * (1 - scipy_stats.norm.cdf(abs(z))))
    return p_value < 0.05, p_value, "proportion_z"


def _fisher_exact_test(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    overall_accuracy: float,
) -> Tuple[bool, float, str]:
    """
    Fisher's exact test for small samples (n < 30).

    Constructs a 2×2 contingency table:
        [correct_segment, wrong_segment]
        [correct_expected, wrong_expected]

    where expected counts are derived from overall_accuracy × n.
    """
    n = len(y_true)
    correct_seg = int(np.sum(y_true == y_pred))
    wrong_seg = n - correct_seg

    expected_correct = round(overall_accuracy * n)
    expected_wrong = n - expected_correct

    # Ensure no zero rows/cols to avoid degenerate table
    if expected_correct == 0 or expected_wrong == 0:
        return False, 1.0, "fisher_exact"

    table = [
        [correct_seg, wrong_seg],
        [expected_correct, expected_wrong],
    ]
    _, p_value = scipy_stats.fisher_exact(table)
    return float(p_value) < 0.05, float(p_value), "fisher_exact"


def _bootstrap_ci_test(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric: str,
    overall_metric: float,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_state: int = 42,
) -> Tuple[bool, Optional[float], str]:
    """
    Bootstrap confidence interval test for regression metrics.

    Resamples the segment 1000 times with replacement, computes the
    metric each time, and checks whether the overall_metric falls
    outside the (alpha/2, 1 - alpha/2) percentile interval.

    If overall_metric is outside the CI → significant difference.

    Returns a pseudo p-value = proportion of bootstrap samples
    where the metric is at least as extreme as overall_metric.
    """
    rng = np.random.default_rng(random_state)
    fn = _get_metric_fn(metric)
    n = len(y_true)

    bootstrap_metrics = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            val = float(fn(y_true[idx], y_pred[idx]))
            if not math.isnan(val):
                bootstrap_metrics.append(val)
        except Exception:
            continue

    if len(bootstrap_metrics) < 10:
        # Too few valid bootstrap samples
        return False, None, "bootstrap_ci"

    arr = np.array(bootstrap_metrics)
    lower = float(np.percentile(arr, 100 * alpha / 2))
    upper = float(np.percentile(arr, 100 * (1 - alpha / 2)))

    is_significant = overall_metric < lower or overall_metric > upper

    # Pseudo p-value: fraction of bootstrap samples more extreme than overall
    direction = METRIC_REGISTRY[metric]
    if direction == MetricDirection.HIGHER_IS_BETTER:
        pseudo_p = float(np.mean(arr >= overall_metric))
    else:
        pseudo_p = float(np.mean(arr <= overall_metric))

    return is_significant, pseudo_p, "bootstrap_ci"


# ---------------------------------------------------------------------------
# Task type detection
# ---------------------------------------------------------------------------


def detect_task_type(y_true: np.ndarray, metric: str) -> bool:
    """
    Return True if the task is regression, False if classification.

    Decision is based on the metric name — the user already specified this.
    Regression metrics: mae, rmse, mse, r2
    Classification metrics: accuracy, f1*, precision, recall
    """
    regression_metrics = {"mae", "rmse", "mse", "r2"}
    return metric in regression_metrics
