import math
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.datasets import load_iris, fetch_california_housing
import os

from pyslicekit import evaluate, to_csv
from pyslicekit.exceptions import PySliceKitValidationError, PySliceKitNoSegmentsError
from pyslicekit.slicer import build_segments
from pyslicekit.stats import _run_significance_test

# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def test_slicer():
    # 5-col DataFrame with known categoricals
    df = pd.DataFrame({
        "col1": ["A", "B", "A", "B", "C"],
        "col2": ["X", "X", "Y", "Y", "Z"],
        "col3": [1, 2, 3, 4, 5],
        "col4": ["M", "M", "M", "N", "N"],
        "col5": range(25, 30)
    })
    y_true = np.ones(5)
    y_pred = np.ones(5)
    
    # Test depth=1
    segments_d1 = build_segments(df, y_true, y_pred, slice_cols=["col1", "col2"], depth=1, min_samples=1)
    # col1 has 3 unique (A, B, C) -> 3 segments
    # col2 has 3 unique (X, Y, Z) -> 3 segments
    assert len(segments_d1) == 6
    
    # Test depth=2
    segments_d2 = build_segments(df, y_true, y_pred, slice_cols=["col1", "col2"], depth=2, min_samples=1)
    # depth 1: 6 segments. depth 2 combos of col1 x col2: (A,X), (B,X), (A,Y), (B,Y), (C,Z) -> 5 valid segments. Total 11
    assert len(segments_d2) == 11

def test_stats():
    from pyslicekit.stats import evaluate_segment
    segment = {
        "slice_def": [("col", "val")],
        "mask": np.array([True, True, True, True]),
        "n": 4,
        "low_n": False,
        "y_true": np.array([1, 1, 0, 0]),
        "y_pred": np.array([1, 0, 0, 1]), # accuracy = 0.5
    }
    
    res = evaluate_segment(segment, metric="accuracy", overall_metric=0.8, is_regression=False, min_samples=2)
    assert math.isclose(res.metric_value, 0.5, abs_tol=1e-4)
    # accuracy is HIGHER_IS_BETTER. gap = segment_metric - overall_metric = 0.5 - 0.8 = -0.3
    assert math.isclose(res.gap, -0.3, abs_tol=1e-4)
    assert res.is_underperforming == True

def test_significance_flagger():
    # 1000-row segment that performs identically to baseline
    y_true = np.ones(1000)
    y_pred = np.ones(1000)
    y_pred[:200] = 0 # 80% accuracy
    
    is_sig, p, test = _run_significance_test(y_true, y_pred, 1000, "accuracy", 0.8, 0.8, False, False)
    assert not is_sig
    
    # 500-row segment with a 15-point accuracy drop
    y_true_drop = np.ones(500)
    y_pred_drop = np.ones(500)
    y_pred_drop[:175] = 0 # 325/500 = 65% accuracy (15 points drop from 80%)
    
    is_sig, p, test = _run_significance_test(y_true_drop, y_pred_drop, 500, "accuracy", 0.8, 0.65, False, False)
    assert is_sig
    assert p < 0.05

# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------

def test_end_to_end_happy_path():
    iris = load_iris()
    X, y = iris.data, iris.target
    df = pd.DataFrame(X, columns=iris.feature_names)
    df["dummy_cat"] = np.where(y == 0, "A", np.where(y == 1, "B", "C"))
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X, y)
    y_pred = model.predict(X)
    
    # Inject bad segment: deliberate mislabeling in "B"
    y_pred[df["dummy_cat"] == "B"] = 0
    
    results = evaluate(model, df, y, y_pred, slice_cols=["dummy_cat"], min_samples=10, render_visuals=False)
    
    # Assert segment appears in top 3 with negative gap and significant=True
    top_labels = [r.label for r in results[:3]]
    assert "dummy_cat=B" in top_labels
    bad_res = next(r for r in results if r.label == "dummy_cat=B")
    assert bad_res.gap < 0
    assert bad_res.is_significant

def test_regression_path():
    cali = fetch_california_housing()
    X = cali.data[:1000] # smaller for speed
    y = cali.target[:1000]
    df = pd.DataFrame(X, columns=cali.feature_names)
    
    # Mock ocean proximity since it's not in numeric version
    df["ocean_proximity"] = np.random.choice(["INLAND", "NEAR BAY", "<1H OCEAN"], size=1000)
    
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    
    results = evaluate(model, df, y, y_pred, slice_cols=["ocean_proximity"], metric="mae", render_visuals=False)
    
    # MAE per segment differs meaningfully
    gaps = [r.gap for r in results]
    assert len(set(gaps)) > 1
    
    # Sorted correctly by absolute gap
    abs_gaps = [r.abs_gap for r in results]
    assert abs_gaps == sorted(abs_gaps, reverse=True)

def test_export_round_trip(tmpdir):
    df = pd.DataFrame({"col": ["A", "B", "A", "B"] * 10})
    y_true = np.ones(40)
    y_pred = np.ones(40)
    y_pred[:10] = 0
    
    class DummyModel:
        def predict(self, x): return x
    
    results = evaluate(DummyModel(), df, y_true, y_pred, slice_cols=["col"], min_samples=5, render_visuals=False)
    filepath = os.path.join(tmpdir, "audit.csv")
    to_csv(results, filepath)
    
    df_in = pd.read_csv(filepath)
    assert len(df_in) == len(results)
    
    for r, row_gap in zip(results, df_in["gap"]):
        assert math.isclose(r.gap, row_gap, abs_tol=1e-4)

# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_all_empty_segments():
    df = pd.DataFrame({"col": ["A", "B"]})
    y = np.array([1, 1])
    class DummyModel:
        def predict(self, x): return x
    
    with pytest.raises(PySliceKitNoSegmentsError):
        evaluate(DummyModel(), df, y, y, slice_cols=["col"], min_samples=5, render_visuals=False)

def test_single_class_segment():
    # Segment where all y_true = 0 and y_pred = 0. F1 is undefined (0/0).
    df = pd.DataFrame({"col": ["A"] * 50})
    y_true = np.zeros(50)
    y_pred = np.zeros(50)
    class DummyModel:
        def predict(self, x): return x
    
    results = evaluate(DummyModel(), df, y_true, y_pred, slice_cols=["col"], metric="f1", min_samples=2, render_visuals=False)
    # Handled gracefully with NaN
    assert math.isnan(results[0].metric_value)

def test_mismatched_lengths():
    df = pd.DataFrame({"col": ["A"] * 100})
    y_true = np.ones(100)
    y_pred = np.ones(99)
    class DummyModel:
        def predict(self, x): return x
    
    with pytest.raises(PySliceKitValidationError, match="y_true has 100 rows but y_pred has 99 rows"):
        evaluate(DummyModel(), df, y_true, y_pred, slice_cols=["col"], render_visuals=False)

def test_numeric_only_dataframe():
    df = pd.DataFrame({"num": range(100)})
    y_true = np.ones(100)
    y_pred = np.ones(100)
    class DummyModel:
        def predict(self, x): return x
    
    results = evaluate(DummyModel(), df, y_true, y_pred, slice_cols=["num"], min_samples=10, render_visuals=False)
    assert len(results) > 0 # Auto-binning fired
