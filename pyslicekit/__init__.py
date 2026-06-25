"""
pyslicekit
~~~~~~~~~~
A library for discovering underperforming segments in machine learning models.
"""

from pyslicekit.api import evaluate
from pyslicekit.exporter import to_csv, to_json
from pyslicekit.types import SliceResult
from pyslicekit.exceptions import PySliceKitError

__all__ = ["evaluate", "to_csv", "to_json", "SliceResult", "PySliceKitError"]
__version__ = "0.1.0"
