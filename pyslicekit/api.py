"""
pyslicekit.api
~~~~~~~~~~~~~~
Public entry point for the library.

Usage:
    from pyslicekit import evaluate
    results = evaluate(model, df, y_true, y_pred, slice_cols=["region", "age"])
"""

from __future__ import annotations

from typing import Any, List, Optional

import pandas as pd

from pyslicekit.exceptions import PySliceKitNoSegmentsError
from pyslicekit.renderer import render
from pyslicekit.slicer import build_segments
from pyslicekit.stats import compute_overall_metric, detect_task_type, evaluate_segment
from pyslicekit.types import SliceResult
from pyslicekit.validators import validate_inputs


def evaluate(
    model: Any,
    df: pd.DataFrame,
    y_true: Any,
    y_pred: Any,
    slice_cols: List[str],
    metric: str = "accuracy",
    min_samples: int = 30,
    depth: int = 2,
    render_visuals: bool = True,
    **render_kwargs: Any,
) -> List[SliceResult]:
    """
    Evaluate a machine learning model across different slices (subgroups) of your data to discover hidden areas of poor performance.

    This function is the main engine of PySliceKit. It takes your data, automatically chunks it up into subgroups based on the columns you provide, tests your model on those specific groups, and highlights the ones where your model is secretly failing.

    .. code-block:: python

        import pyslicekit

        # Find the exact segments where your model underperforms!
        results = pyslicekit.evaluate(
            model=my_model,
            df=my_dataframe,
            y_true=y_actuals,
            y_pred=y_predictions,
            slice_cols=["Age", "Geography"],
            metric="accuracy",
            depth=2,
            render_visuals=True,
            top_n=15
        )

    **Parameters:**

    * ``model`` (Any) – Your trained machine learning model. It just needs a standard `.predict()` method. We never train your model, we only test it!
    * ``df`` (pd.DataFrame) – Your feature dataset. This is the data that contains the columns you want to slice (like Age, Income, City, etc).
    * ``y_true`` (array-like) – The actual, correct answers (the ground truth).
    * ``y_pred`` (array-like) – The answers your model predicted.
    * ``slice_cols`` (List[str]) – A list of column names from your `df` that you want to investigate. E.g., `["Age", "Geography"]`.
    * ``metric`` (str, optional) – The mathematical way you want to measure success. Examples: "accuracy", "f1", "mae", "rmse". ``Default is "accuracy"``.
    * ``min_samples`` (int, optional) – The minimum number of data points needed in a group for us to trust the math. If a group has fewer people than this, we still show it but flag it with a low-sample warning. ``Default is 30``.
    * ``depth`` (int, optional) – How deep should we combine columns? `1` means we check Age, then we check Geography. `2` means we cross them and check "Age AND Geography" together. ``Default is 2``.
    * ``render_visuals`` (bool, optional) – Do you want us to automatically draw the beautiful Heatmap and Bar charts for you? ``Default is True``.
    * ``**render_kwargs`` (Any) – Extra commands for the chart drawing. For example: `top_n=15` to only show the top 15 worst segments in the bar chart (``Default `top_n` is 15``), or `figsize_heatmap=(12, 6)` to change the size of the heatmap figure.

    Returns
    -------
    List[SliceResult]
        A list of result objects, one for each segment tested, sorted so the absolute worst performing segments are exactly at the top!
    """
    # 1. Validate inputs
    validate_inputs(
        model=model,
        df=df,
        y_true=y_true,
        y_pred=y_pred,
        slice_cols=slice_cols,
        metric=metric,
        min_samples=min_samples,
        depth=depth,
    )

    # 2. Detect task type
    is_regression = detect_task_type(y_true, metric)

    # 3. Compute overall baseline
    overall_metric = compute_overall_metric(y_true, y_pred, metric)

    # 4. Build segments
    segment_dicts = build_segments(
        df=df,
        y_true=y_true,
        y_pred=y_pred,
        slice_cols=slice_cols,
        depth=depth,
        min_samples=min_samples,
    )

    # 5. Evaluate segments
    results = []
    for seg in segment_dicts:
        result = evaluate_segment(
            segment=seg,
            metric=metric,
            overall_metric=overall_metric,
            is_regression=is_regression,
            min_samples=min_samples,
        )
        results.append(result)

    # Filter out empty segments (where n=0, though slicer drops them)
    results = [r for r in results if r.n > 0]

    if not results or all(r.low_n for r in results):
        raise PySliceKitNoSegmentsError(
            "All candidate segments were dropped or had n < min_samples. "
            "Try choosing different slice columns or lowering min_samples."
        )

    # 6. Sort results by absolute gap descending (worst first)
    results.sort(key=lambda r: r.abs_gap, reverse=True)

    # 7. Render visuals
    if render_visuals:
        render(results, **render_kwargs)

    return results
