"""
pyslicekit.exceptions
~~~~~~~~~~~~~~~~~~~~~
Typed error classes for pyslicekit.

Every error raised by the library is a subclass of PySliceKitError,
so users can catch all library errors with a single except clause
while still being able to distinguish specific failure modes.

Usage:
    from pyslicekit.exceptions import (
        PySliceKitError,
        PySliceKitValidationError,
        PySliceKitNoSegmentsError,
        PySliceKitMetricError,
        PySliceKitRenderError,
    )
"""


class PySliceKitError(Exception):
    """
    Base class for all pyslicekit errors.

    Catch this to handle any library error generically:

    .. code-block:: python

        try:
            results = pyslicekit.evaluate(...)
        except PySliceKitError as e:
            print(f"pyslicekit failed: {e}")
    """


class PySliceKitValidationError(PySliceKitError):
    """
    Raised when the inputs to evaluate() fail validation.

    Common causes:
    - y_true and y_pred have different lengths
    - slice_cols contains column names not present in df
    - metric name is not in SUPPORTED_METRICS
    - model has no predict() method
    - df is empty

    The error message always names the specific problem.
    
    **What triggers this (Example):**
    
    .. code-block:: python

        # ❌ WRONG: Passing a metric that doesn't exist
        pyslicekit.evaluate(..., metric="made_up_metric")
        # Raises: PySliceKitValidationError("Metric 'made_up_metric' is not supported.")

        # ❌ WRONG: y_true and y_pred lengths don't match
        pyslicekit.evaluate(..., y_true=[1, 0, 1], y_pred=[1, 0])
        # Raises: PySliceKitValidationError("Length mismatch: y_true has 3, y_pred has 2")
    """


class PySliceKitNoSegmentsError(PySliceKitError):
    """
    Raised when slicing produces zero usable segments.

    This happens when every candidate segment has n < min_samples
    and there is nothing left to evaluate.

    Includes a suggestion to lower min_samples or change slice_cols.

    **What triggers this (Example):**

    .. code-block:: python

        # ❌ WRONG: Setting min_samples too high for a small dataset
        # If your df only has 100 rows, and you ask for min_samples=200, 
        # all segments will be dropped!
        pyslicekit.evaluate(..., df=small_df, min_samples=200)
        # Raises: PySliceKitNoSegmentsError("All candidate segments were dropped...")
    """


class PySliceKitMetricError(PySliceKitError):
    """
    Raised when metric computation fails for a structural reason.

    This is distinct from a NaN result (which is handled gracefully
    inside SliceResult). This error fires when the metric function
    itself raises — e.g. when y_pred contains values outside [0, 1]
    for a metric that requires probabilities.

    Includes the segment label and original exception in the message.
    """


class PySliceKitRenderError(PySliceKitError):
    """
    Raised when the renderer fails to produce a figure.

    Common cause: matplotlib backend not available in the environment
    (e.g. headless server with no display). 
    
    **What triggers this (Example):**

    .. code-block:: python

        # ❌ WRONG: Running render_visuals=True on a headless Linux server
        pyslicekit.evaluate(..., render_visuals=True)
        # Raises: PySliceKitRenderError("Renderer failed: No display found...")
        
        # ✅ FIX: Set render_visuals=False when in a CI pipeline or headless server
        pyslicekit.evaluate(..., render_visuals=False)
    """
