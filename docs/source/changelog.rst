Changelog
=========

All notable changes to this project will be documented in this file.

[v0.1.0] - Initial Release
--------------------------

Features
^^^^^^^^
* **Core evaluate() Engine**: Automatically slice datasets up to depth $N$ and test ML model performance across subgroups.
* **Statistical Rigor**: Auto-switching statistical backends (Z-Test, Fisher's Exact, Bootstrapping) to guarantee robust comparisons against global baseline metrics.
* **Visual Output System**: Integrated ``matplotlib`` renderers generating automated, color-coded heatmaps and "Worst Segments" bar charts.
* **Universal Metric Support**: Built-in support for classification metrics (``accuracy``, ``f1``, ``precision``, ``recall``) and regression metrics (``mae``, ``rmse``, ``r2``, ``mse``).
* **Exporters**: Added ``pyslicekit.to_csv()`` and ``pyslicekit.to_json()`` for seamless external auditing and dashboard integration.
* **Documentation Suite**: Sphinx-based documentation containing Getting Started guide, User Guide, FAQ, and complete API reference.

Limitations
^^^^^^^^^^^
* Visualizations currently rely entirely on ``matplotlib`` (interactive charts like Plotly are not yet supported).
* Very deep slicing (``depth > 3``) on wide datasets may cause significant performance degradation and memory usage due to combinatorial explosion.

Known Issues
^^^^^^^^^^^^
* Overlapping text on the heatmap Y-axis labels when column names or categorical string values are exceptionally long.
* High variance in Bootstrap Confidence Intervals when dealing with extremely small, skewed regression segments near the ``min_samples`` boundary.

What's next
^^^^^^^^^^^
* **Performance Enhancements**: Integration of multiprocessing/parallelization to handle massive datasets and deeper slice combinations faster.
* **Interactive Dashboard**: Export capabilities that automatically generate an interactive HTML dashboard.
* **Custom Metrics**: Allowing users to pass their own callable metric functions instead of just pre-defined string names.
