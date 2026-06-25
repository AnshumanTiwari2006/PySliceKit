Getting Started
===============

Installation and Dependencies
-----------------------------
You can install PySliceKit using pip:

.. code-block:: bash

    pip install pyslicekit

PySliceKit relies on standard data science libraries:

* ``pandas`` (>= 1.0.0)
* ``numpy`` (>= 1.18.0)
* ``scikit-learn`` (>= 0.22.0)
* ``scipy`` (>= 1.4.0)
* ``matplotlib`` (>= 3.2.0)

Supported Metrics
-----------------
You must pass a valid string to the ``metric`` parameter. PySliceKit automatically understands whether higher or lower is better, and automatically selects the correct statistical test for the task type.

.. list-table::
   :header-rows: 1

   * - Metric string
     - Task
     - Direction
     - Test used
   * - ``accuracy``
     - Classification
     - higher is better
     - Z-test / Fisher
   * - ``f1``, ``f1_macro``, ``f1_weighted``
     - Classification
     - higher is better
     - Z-test / Fisher
   * - ``precision``, ``recall``
     - Classification
     - higher is better
     - Z-test / Fisher
   * - ``mae``, ``rmse``, ``mse``
     - Regression
     - lower is better
     - Bootstrap CI
   * - ``r2``
     - Regression
     - higher is better
     - Bootstrap CI

What it Returns
---------------
The ``pyslicekit.evaluate()`` function returns a list of ``SliceResult`` objects, sorted by absolute gap (worst performing segments first). 

You can loop through them or extract the exact properties you need:

.. code-block:: python

    for result in results[:5]:  # top 5 worst
        print(f"Segment: {result.label}")
        print(f"Gap: {result.gap:.3f}")
        print(f"Significant: {result.is_significant}")

Complete Minimal Example
------------------------

.. code-block:: python

    import pandas as pd
    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    import pyslicekit

    # 1. Load your data and train a model
    cancer = load_breast_cancer(as_frame=True)
    df = cancer.frame
    X = df.drop(columns=['target'])
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # 2. Evaluate!
    results = pyslicekit.evaluate(
        model=model,
        df=X_test,
        y_true=y_test,
        y_pred=y_pred,
        slice_cols=["mean radius", "mean texture"],
        metric="f1",
        render_visuals=True,
        top_n=15
    )
