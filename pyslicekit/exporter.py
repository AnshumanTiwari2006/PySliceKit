"""
pyslicekit.exporter
~~~~~~~~~~~~~~~~~~~
Export SliceResult lists to CSV or JSON.

Usage:
    from pyslicekit.exporter import to_csv, to_json
"""

import csv
import json
from typing import List

from pyslicekit.types import SliceResult


def to_csv(results: List[SliceResult], filepath: str) -> None:
    """
    Export your entire slice evaluation into a clean, easy-to-read CSV file.

    .. code-block:: python

        import pyslicekit
        from pyslicekit.exporter import to_csv, to_json

        # Save your findings to show your manager or colleagues
        to_csv(results, "audit_results.csv")

    **Parameters:**
    
    * ``results`` (List[SliceResult]) – The exact list of results that the `evaluate()` function gave you.
    * ``filepath`` (str) – Where do you want to save the file? (e.g. "my_results.csv")
    """
    if not results:
        return

    # Extract field names
    fieldnames = [
        "segment", "n", "metric", "metric_value", "overall_metric",
        "gap", "is_significant", "low_n", "p_value", "test_used"
    ]

    with open(filepath, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({
                "segment": r.label,
                "n": r.n,
                "metric": r.metric_name,
                "metric_value": r.metric_value,
                "overall_metric": r.overall_metric,
                "gap": r.gap,
                "is_significant": r.is_significant,
                "low_n": r.low_n,
                "p_value": r.p_value,
                "test_used": r.test_used or "",
            })


def to_json(results: List[SliceResult], filepath: str) -> None:
    """
    Export your slice evaluation into a structured JSON file.
    
    This is perfect if you want to take the results and feed them into a web dashboard or another automated system.

    .. code-block:: python

        import pyslicekit
        from pyslicekit.exporter import to_csv, to_json

        # Save as JSON for your web app
        to_json(results, "audit_results.json")

    **Parameters:**
    
    * ``results`` (List[SliceResult]) – The exact list of results that the `evaluate()` function gave you.
    * ``filepath`` (str) – Where do you want to save the file? (e.g. "my_results.json")
    """
    if not results:
        with open(filepath, mode="w", encoding="utf-8") as f:
            json.dump([], f)
        return

    data = []
    for r in results:
        data.append({
            "segment": r.label,
            "slice_def": r.slice_def,
            "n": r.n,
            "metric": r.metric_name,
            "metric_value": r.metric_value,
            "overall_metric": r.overall_metric,
            "gap": r.gap,
            "is_significant": r.is_significant,
            "low_n": r.low_n,
            "p_value": r.p_value,
            "test_used": r.test_used,
        })

    with open(filepath, mode="w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
