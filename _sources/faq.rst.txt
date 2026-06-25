Frequently Asked Questions (FAQ)
================================

1. What is PySliceKit?
----------------------
PySliceKit is a Python library designed to automatically identify, quantify, and visualize the hidden subgroups in your dataset where your machine learning model is underperforming.

2. Why shouldn't I just use SHAP or LIME?
-----------------------------------------
SHAP and LIME provide **feature importance** (why a model made a specific prediction). PySliceKit provides **cohort performance profiling** (where the model is failing in aggregate). They are complementary tools: use PySliceKit to find the failing cohort, and SHAP to understand why it failed.

3. Do I need to re-train my model?
----------------------------------
No! PySliceKit is completely model-agnostic. It only requires your trained model's ``.predict()`` output and the raw features.

4. What metrics are supported?
------------------------------
For classification: ``accuracy``, ``f1``, ``precision``, ``recall``. For regression: ``mae``, ``rmse``, ``mse``, ``r2``.

5. How does PySliceKit know if a drop in performance is "real"?
---------------------------------------------------------------
It automatically applies statistical significance testing (Z-Tests for proportions, Bootstrap CI for continuous metrics) to differentiate true failures from random noise.

6. What does the `depth` parameter do?
--------------------------------------
Depth determines how many columns are combined to form a segment. Depth 1 checks single columns (``Age``, ``Gender``). Depth 2 checks pairwise intersections (``Age AND Gender``).

7. My dataset is massive, will depth=3 be too slow?
---------------------------------------------------
Yes, high depth on datasets with many categorical unique values creates a combinatorial explosion. Stick to depth 1 or 2 for wide datasets, or explicitly limit the ``slice_cols`` parameter.

8. Can I pass PyTorch or TensorFlow models?
-------------------------------------------
Yes, as long as you wrap the model so that ``model.predict(X)`` returns a flat numpy array of predictions.

9. Why are some bars in the bar chart faded?
--------------------------------------------
Faded bars represent segments where the model performs worse than average, but the difference is NOT statistically significant (p >= 0.05). Solid red bars are statistically significant failures.

10. What does the warning icon mean in the heatmap?
---------------------------------------------------
The warning icon (or ``[low-n]`` tag) indicates that a specific segment contains fewer samples than your ``min_samples`` threshold (default 30). The math is less reliable for tiny cohorts.

11. Why does the renderer enforce "Red means bad"?
--------------------------------------------------
To reduce cognitive load. A drop in accuracy is bad (negative gap), but a drop in RMSE is good (positive gap). The renderer automatically aligns the colors so red *always* indicates a negative outcome for the user.

12. Can I export the raw data instead of visualizations?
--------------------------------------------------------
Yes! The ``pyslicekit.evaluate()`` function returns a list of ``SliceResult`` objects. You can also use ``pyslicekit.to_csv()`` and ``pyslicekit.to_json()``.

13. Does PySliceKit automatically bin continuous columns?
---------------------------------------------------------
Yes. If you pass a continuous numeric column (like Income or Age), PySliceKit will automatically bin it into quartiles (Q1, Q2, Q3, Q4) to evaluate the cohorts.

14. How can I change the chart colors?
--------------------------------------
Currently, PySliceKit uses a fixed, color-blind friendly palette tailored for dark themes. Future releases will allow passing custom matplotlib ``cmap`` objects.

15. Does it support multi-class classification?
-----------------------------------------------
Currently, the built-in metrics are heavily optimized for binary classification and regression. For multi-class, you should binarize your labels (One-vs-Rest) before passing them in.

16. What happens if I pass highly correlated columns?
-----------------------------------------------------
PySliceKit treats them independently when evaluating slices. If ``Age`` and ``Tenure`` are highly correlated, you will likely see identical performance drops flagged for both.

17. Can I integrate PySliceKit into my CI/CD pipeline?
------------------------------------------------------
Absolutely. Since you can run it purely in code (by setting ``render_visuals=False``), you can write a script that asserts no cohort drops below a specific F1-score threshold before allowing a model deployment.

18. What is the difference between `metric_value` and `overall_metric`?
-----------------------------------------------------------------------
``overall_metric`` is the performance of your model on the *entire* dataset you provided. ``metric_value`` is the performance on the *specific* slice being evaluated. The ``gap`` is the difference between them.

19. How do I handle missing data (NaNs) in my slice columns?
------------------------------------------------------------
You should impute or drop NaNs in your ``df`` before passing it to ``evaluate()``. PySliceKit does not currently process or slice explicitly on ``NaN`` as a distinct category.

20. Is PySliceKit free for commercial use?
------------------------------------------
Yes! PySliceKit is distributed under the open-source MIT License.
