# PySliceKit

[![PyPI version](https://badge.fury.io/py/pyslicekit.svg)](https://badge.fury.io/py/pyslicekit)
[![Tests](https://img.shields.io/badge/Tests-passing-brightgreen.svg)](https://github.com/AnshumanTiwari2006/PySliceKit/actions/workflows/test.yml)
[![Docs](https://img.shields.io/badge/docs-passing-brightgreen.svg)](https://AnshumanTiwari2006.github.io/PySliceKit/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![GitHub stars](https://img.shields.io/github/stars/AnshumanTiwari2006/PySliceKit.svg?style=social&label=Star)](https://github.com/AnshumanTiwari2006/PySliceKit)

**PySliceKit** is a Python library that helps you automatically discover exactly where your machine learning models are secretly failing.

Global metrics like 95% Accuracy or a low RMSE can be dangerously misleading. Your model might perform perfectly on the majority of your data, while severely underperforming on specific subgroups (e.g., specific age groups, geographic regions, or combined minority segments). 

PySliceKit solves this by automatically "slicing" your feature dataset, evaluating your model's performance on every single subgroup, calculating statistical significance, and returning a beautiful heatmap and bar chart of your model's worst-performing blind spots.

---

## Quick Links
* **Documentation**: [AnshumanTiwari2006.github.io/PySliceKit](https://AnshumanTiwari2006.github.io/PySliceKit/)
* **Source Code**: [GitHub Repository](https://github.com/AnshumanTiwari2006/PySliceKit)
* **Changelog**: [CHANGELOG.md](https://github.com/AnshumanTiwari2006/PySliceKit/blob/main/CHANGELOG.md)
* **Discussions**: [Community Board](https://github.com/AnshumanTiwari2006/PySliceKit/discussions)
* **Issue Tracker**: [Bug Reports](https://github.com/AnshumanTiwari2006/PySliceKit/issues)

## Full Documentation
**[Read the full PySliceKit Documentation here](https://AnshumanTiwari2006.github.io/PySliceKit/)** for the Getting Started guide, complete User Guide, and comprehensive API Reference.

## Quick Start

### Installation
```bash
pip install pyslicekit
```

### Usage
```python
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pyslicekit

# 1. Load data and train your model
data = load_breast_cancer(as_frame=True)
df = data.frame
X = df.drop(columns=['target'])
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# 2. Let PySliceKit find the blind spots!
results = pyslicekit.evaluate(
    model=model,
    df=X_test,
    y_true=y_test,
    y_pred=model.predict(X_test),
    slice_cols=["mean radius", "mean texture"],
    metric="f1"
)
```

### What does this do?
When you call `pyslicekit.evaluate()`, the library automatically chunks your test dataset into subgroups (like specific age brackets and regions), evaluates your model's F1 score on each subgroup independently, and applies statistical tests (like a Z-Test or Fisher's Exact) to prove if the performance drop is mathematically significant.

### What does the output show?
You will automatically receive two powerful visualizations. The **Heatmap** shows the raw gaps across all subgroups—the darker the red, the worse your model performs compared to its global average.

![Heatmap Visualization](https://raw.githubusercontent.com/AnshumanTiwari2006/PySliceKit/main/docs/source/_static/cancer_heatmap.png)

The **Worst Segments Bar Chart** automatically extracts and sorts the worst offenders. Statistically significant failures are solid red, while non-significant drops are faded, giving you a perfectly prioritized list of areas to fix.

![Bar Chart Visualization](https://raw.githubusercontent.com/AnshumanTiwari2006/PySliceKit/main/docs/source/_static/cancer_bar.png)

## Features
* **Model Agnostic**: Works with any model that has a `.predict()` method (scikit-learn, XGBoost, PyTorch, etc.).
* **Automatic Statistics**: Chooses between Z-tests, Fisher's Exact Test, and Bootstrap CIs based on your task type and sample sizes.
* **Beautiful Visualizations**: Instantly generates heatmaps and bar charts of your model's weak spots.
* **Exporters**: Easily export findings to JSON or CSV for audits or dashboards.

## License
Licensed under the [MIT License](https://github.com/AnshumanTiwari2006/PySliceKit/blob/main/LICENSE) (See GitHub for details).
