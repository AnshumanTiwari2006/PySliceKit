"""
pyslicekit.validators
~~~~~~~~~~~~~~~~~~~~~
Input validation guards for evaluate().

All validation runs before any slicing or metric computation.
A failed guard raises PySliceKitValidationError with a message
that names the exact problem — never a generic error.

Usage:
    from pyslicekit.validators import validate_inputs
"""

from __future__ import annotations

from typing import Any, List, Optional

import numpy as np
import pandas as pd

from pyslicekit.exceptions import PySliceKitValidationError
from pyslicekit.types import SUPPORTED_METRICS


def validate_inputs(
    model: Any,
    df: pd.DataFrame,
    y_true: Any,
    y_pred: Any,
    slice_cols: List[str],
    metric: str,
    min_samples: int,
    depth: int,
) -> None:
    """
    Run all input guards in sequence.

    Raises PySliceKitValidationError on the first failure found.
    Does not raise on warnings (low_n, etc.) — those are handled
    downstream as flags on SliceResult.

    Parameters
    ----------
    model : any sklearn-compatible model
    df : the test DataFrame (features only, no target)
    y_true : ground truth labels or values
    y_pred : model predictions
    slice_cols : list of column names to slice on
    metric : metric name string
    min_samples : minimum segment size (int >= 1)
    depth : cross-product depth (1 or 2 only in V1)
    """
    _validate_model(model)
    _validate_dataframe(df)
    _validate_labels(y_true, y_pred, df)
    _validate_slice_cols(slice_cols, df)
    _validate_metric(metric)
    _validate_min_samples(min_samples)
    _validate_depth(depth)


# ---------------------------------------------------------------------------
# Individual guards — each raises with a specific message
# ---------------------------------------------------------------------------


def _validate_model(model: Any) -> None:
    """
    Model must have a predict() method.
    The library calls predict() internally only for task-type detection.
    It never calls fit().
    """
    if not hasattr(model, "predict"):
        raise PySliceKitValidationError(
            f"model of type {type(model).__name__!r} has no predict() method. "
            "pyslicekit requires a fitted sklearn-compatible model."
        )


def _validate_dataframe(df: pd.DataFrame) -> None:
    """DataFrame must be non-empty and a proper pandas DataFrame."""
    if not isinstance(df, pd.DataFrame):
        raise PySliceKitValidationError(
            f"df must be a pandas DataFrame, got {type(df).__name__!r}."
        )
    if df.empty:
        raise PySliceKitValidationError(
            "df is empty (0 rows). Nothing to evaluate."
        )


def _validate_labels(
    y_true: Any,
    y_pred: Any,
    df: pd.DataFrame,
) -> None:
    """
    y_true and y_pred must:
    - be array-like (list, np.ndarray, pd.Series)
    - have the same length as df
    - have the same length as each other
    - not be all-NaN
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    if y_true.ndim == 0 or y_pred.ndim == 0:
        raise PySliceKitValidationError(
            "y_true and y_pred must be 1-D arrays, not scalars."
        )

    if len(y_true) != len(y_pred):
        raise PySliceKitValidationError(
            f"y_true has {len(y_true)} rows but y_pred has {len(y_pred)} rows. "
            "They must have the same length."
        )

    if len(y_true) != len(df):
        raise PySliceKitValidationError(
            f"y_true has {len(y_true)} rows but df has {len(df)} rows. "
            "y_true, y_pred, and df must all represent the same test set."
        )

    try:
        if np.all(np.isnan(y_true.astype(float))):
            raise PySliceKitValidationError(
                "y_true is all NaN. Cannot compute any metric."
            )
    except (ValueError, TypeError):
        pass


def _validate_slice_cols(slice_cols: List[str], df: pd.DataFrame) -> None:
    """
    slice_cols must:
    - be a non-empty list
    - contain only strings
    - reference columns that exist in df
    """
    if not isinstance(slice_cols, list) or len(slice_cols) == 0:
        raise PySliceKitValidationError(
            "slice_cols must be a non-empty list of column name strings. "
            "Example: slice_cols=['gender', 'region']"
        )

    non_strings = [c for c in slice_cols if not isinstance(c, str)]
    if non_strings:
        raise PySliceKitValidationError(
            f"slice_cols contains non-string values: {non_strings!r}. "
            "All entries must be column name strings."
        )

    missing = [c for c in slice_cols if c not in df.columns]
    if missing:
        available = list(df.columns)
        raise PySliceKitValidationError(
            f"slice_cols references columns not found in df: {missing!r}. "
            f"Available columns: {available!r}"
        )


def _validate_metric(metric: str) -> None:
    """metric must be one of the supported metric names."""
    if metric not in SUPPORTED_METRICS:
        raise PySliceKitValidationError(
            f"metric={metric!r} is not supported. "
            f"Choose from: {SUPPORTED_METRICS}"
        )


def _validate_min_samples(min_samples: int) -> None:
    """min_samples must be a positive integer."""
    if not isinstance(min_samples, int) or min_samples < 1:
        raise PySliceKitValidationError(
            f"min_samples must be a positive integer, got {min_samples!r}."
        )


def _validate_depth(depth: int) -> None:
    """
    depth must be 1 or 2.
    depth=3 is not supported in V1 — raises rather than silently ignoring.
    """
    if depth not in (1, 2):
        raise PySliceKitValidationError(
            f"depth={depth!r} is not supported. "
            "Use depth=1 (single-column slices) or depth=2 (two-column intersections). "
            "depth=3+ causes combinatorial explosion and is not available in V1."
        )
