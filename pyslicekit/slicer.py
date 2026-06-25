"""
pyslicekit.slicer
~~~~~~~~~~~~~~~~~
Segment engine: turns slice_cols into a list of usable (mask, slice_def) pairs.

Responsibilities:
  1. Detect numeric columns and bin them into labelled quartiles
  2. Detect high-cardinality columns and warn (but still process)
  3. Build all column combinations up to the requested depth
  4. For each combination, enumerate all value tuples
  5. Apply the min_samples floor — keep segment, attach low_n flag

Usage:
    from pyslicekit.slicer import build_segments
"""

from __future__ import annotations

import warnings
from itertools import combinations, product
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np
import pandas as pd

# Max unique values a column may have before it is considered high-cardinality.
# High-cardinality columns are binned if numeric, or warned-about if categorical.
_CARDINALITY_WARN_THRESHOLD = 20

# Number of quantile bins for continuous columns.
_N_QUANTILE_BINS = 4

# Bin label template used in SliceResult.label — kept human-readable.
_BIN_LABEL_TEMPLATE = "Q{i}({lo:.3g}–{hi:.3g})"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def build_segments(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    slice_cols: List[str],
    depth: int,
    min_samples: int,
) -> List[Dict[str, Any]]:
    """
    Build all usable segments from slice_cols.

    Returns a list of segment dicts, each containing:
        {
            "slice_def": [("col", "value"), ...],
            "mask":      boolean np.ndarray aligned with df index,
            "n":         int,
            "low_n":     bool,
            "y_true":    np.ndarray subset,
            "y_pred":    np.ndarray subset,
        }

    Segments with n == 0 are dropped entirely.
    Segments with 0 < n < min_samples are included with low_n=True.
    """
    # Step 1 — pre-process columns: bin numerics, warn on high-cardinality cats
    processed = _preprocess_columns(df, slice_cols)

    # Step 2 — build combinations up to requested depth
    col_combos = _column_combinations(list(processed.keys()), depth)

    # Step 3 — enumerate all value tuples for each combination
    segments = []
    for combo in col_combos:
        for seg in _enumerate_segments(df, processed, combo, y_true, y_pred, min_samples):
            segments.append(seg)

    return segments


# ---------------------------------------------------------------------------
# Step 1 — column pre-processor
# ---------------------------------------------------------------------------


def _preprocess_columns(
    df: pd.DataFrame,
    slice_cols: List[str],
) -> Dict[str, pd.Series]:
    """
    Returns a dict of {col_name: processed_series} where:
    - numeric columns are replaced by a string-labelled bin series
    - categorical / object columns are used as-is (cast to string)
    - high-cardinality categoricals trigger a warning (but are kept)

    The processed series always contains string values so downstream
    code can treat all columns uniformly.
    """
    processed: Dict[str, pd.Series] = {}

    for col in slice_cols:
        series = df[col]

        if _is_numeric(series):
            processed[col] = _bin_numeric(series, col)
        else:
            str_series = series.astype(str)
            n_unique = str_series.nunique()
            if n_unique > _CARDINALITY_WARN_THRESHOLD:
                warnings.warn(
                    f"Column {col!r} has {n_unique} unique values. "
                    f"This will produce {n_unique} segments for this column alone. "
                    "Consider grouping values or choosing a lower-cardinality column.",
                    UserWarning,
                    stacklevel=4,
                )
            processed[col] = str_series

    return processed


def _is_numeric(series: pd.Series) -> bool:
    """True for integer and float dtypes."""
    return pd.api.types.is_numeric_dtype(series)


def _bin_numeric(series: pd.Series, col_name: str) -> pd.Series:
    """
    Convert a numeric series to a labelled quartile bin series.

    Uses pd.qcut with 4 bins. Duplicate edges are handled with
    duplicates='drop', so some columns may produce fewer than 4 bins.

    Labels are human-readable: "Q1(18.0–34.0)" etc.
    NaN values in the series become the string "nan" (treated as a category).
    """
    try:
        binned, bin_edges = pd.qcut(
            series,
            q=_N_QUANTILE_BINS,
            retbins=True,
            duplicates="drop",
        )
        # Build readable label for each interval
        label_map: Dict[Any, str] = {}
        for i, interval in enumerate(binned.cat.categories):
            label = _BIN_LABEL_TEMPLATE.format(
                i=i + 1,
                lo=interval.left,
                hi=interval.right,
            )
            label_map[interval] = label

        result = binned.map(label_map).astype(str)
        return result

    except ValueError:
        # Fallback: all values identical — treat as single-value categorical
        warnings.warn(
            f"Column {col_name!r} could not be binned into quartiles "
            "(all values may be identical). Treating as a single-value categorical.",
            UserWarning,
            stacklevel=5,
        )
        return series.astype(str)


# ---------------------------------------------------------------------------
# Step 2 — column combinations
# ---------------------------------------------------------------------------


def _column_combinations(
    cols: List[str],
    depth: int,
) -> List[Tuple[str, ...]]:
    """
    Return all column combinations of size 1 through depth.

    depth=1 → [(col_a,), (col_b,), ...]
    depth=2 → [(col_a,), (col_b,), (col_a, col_b), ...]
    """
    result = []
    for d in range(1, depth + 1):
        result.extend(combinations(cols, d))
    return result


# ---------------------------------------------------------------------------
# Step 3 — segment enumeration
# ---------------------------------------------------------------------------


def _enumerate_segments(
    df: pd.DataFrame,
    processed: Dict[str, pd.Series],
    combo: Tuple[str, ...],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    min_samples: int,
) -> Generator[Dict[str, Any], None, None]:
    """
    Yield one segment dict per unique value combination in combo.

    For a single column ("gender",) with values ["male", "female"],
    yields two segments.

    For two columns ("gender", "age_bin") with 2 and 4 values respectively,
    yields up to 8 segments (fewer if some combos have 0 rows).
    """
    # Collect unique values per column in this combo
    value_lists = [processed[col].unique().tolist() for col in combo]

    for value_tuple in product(*value_lists):
        # Build boolean mask: row must match ALL (col, value) pairs
        mask = np.ones(len(df), dtype=bool)
        for col, val in zip(combo, value_tuple):
            mask &= (processed[col].values == val)

        n = int(mask.sum())

        # Drop empty segments entirely — nothing to compute
        if n == 0:
            continue

        slice_def = list(zip(combo, value_tuple))
        low_n = n < min_samples

        yield {
            "slice_def": slice_def,
            "mask": mask,
            "n": n,
            "low_n": low_n,
            "y_true": y_true[mask],
            "y_pred": y_pred[mask],
        }
