# Statistical Methods in PySliceKit

PySliceKit is designed to identify machine-learning performance differences across data subgroups.

This document explains the statistical methods used by PySliceKit, including subgroup performance gaps, classification tests, bootstrap analysis, and multiple-comparisons correction.

---

## 1. Global Performance and Subgroup Performance

Let the complete evaluation dataset be:

\[
D = \{(x_i, y_i, \hat{y}_i)\}_{i=1}^{N}
\]

where:

- \(x_i\) represents the input features.
- \(y_i\) represents the actual target.
- \(\hat{y}_i\) represents the model prediction.
- \(N\) represents the total number of observations.

PySliceKit calculates the selected evaluation metric on the complete dataset.

This is called the **global metric**:

\[
M_{\text{global}} = M(D)
\]

For a subgroup \(S\), PySliceKit calculates:

\[
M_{\text{segment}} = M(S)
\]

A subgroup may represent:

- A single feature value.
- A numerical feature range.
- A categorical group.
- A combination of multiple feature values.

---

## 2. Performance Gap

PySliceKit calculates the difference between subgroup performance and global performance using:

\[
\text{Gap} =
M_{\text{segment}} - M_{\text{global}}
\]

The interpretation of the gap depends on the selected metric.

### Metrics where higher values are better

Examples include:

- Accuracy
- F1 score
- Precision
- Recall
- \(R^2\)

For these metrics:

- A positive gap indicates better subgroup performance.
- A negative gap indicates worse subgroup performance.

### Metrics where lower values are better

Examples include:

- Mean Absolute Error
- Root Mean Squared Error
- Mean Squared Error

For these metrics:

- A positive gap indicates worse subgroup performance.
- A negative gap indicates better subgroup performance.

The metric direction should always be considered when interpreting subgroup results.

---

## 3. Subgroup Construction

PySliceKit can evaluate individual features and combinations of features.

For example, a dataset may be divided into subgroups such as:

- `Age = Q1`
- `Age = Q3`
- `Region = North`
- `Age = Q3 AND Region = North`

For a subgroup defined by multiple features, we can represent the subgroup as:

\[
S =
\{x : x_{j_1}=v_1,\ldots,x_{j_k}=v_k\}
\]

The slicing depth determines the maximum number of columns used to construct a subgroup.

Increasing the depth can reveal more specific performance gaps. However, it may also:

- Increase the number of subgroups.
- Reduce subgroup sample sizes.
- Increase computation time.
- Increase memory usage.

---

## 4. Numerical Feature Binning

Continuous numerical features may contain many unique values.

Creating a separate subgroup for every unique value is often impractical. PySliceKit can therefore group numerical values into ranges.

One possible approach is quartile-based binning.

A feature may be divided into:

- Q1 — lowest 25% of values.
- Q2 — 25th to 50th percentile.
- Q3 — 50th to 75th percentile.
- Q4 — highest 25% of values.

The exact grouping depends on the implementation and configuration used by the library.

Binning allows the analysis to focus on ranges of values rather than individual observations.

---

## 5. Classification Metrics

For classification tasks, PySliceKit supports metrics such as:

- Accuracy
- F1 score
- Precision
- Recall
- Macro F1
- Weighted F1

The selected metric is calculated globally and separately for each subgroup.

For example, classification accuracy is defined as:

\[
\text{Accuracy}
=
\frac{\text{Number of correct predictions}}
{\text{Total number of predictions}}
\]

For a subgroup \(S\), subgroup accuracy is:

\[
\text{Accuracy}_{S}
=
\frac{\text{Correct predictions in }S}
{|S|}
\]

where \(|S|\) is the number of observations in the subgroup.

---

## 6. Two-Proportion Z-Test

For classification analyses based on proportions, PySliceKit may use a two-proportion Z-Test when the sample-size conditions are suitable.

Let:

- \(p_s\) be the proportion of correct predictions in the subgroup.
- \(p_g\) be the corresponding global proportion.
- \(n_s\) be the subgroup sample size.

The test compares the subgroup proportion with the reference proportion.

A simplified test statistic can be written as:

\[
z =
\frac{p_s-p_g}
{\sqrt{\frac{p_g(1-p_g)}{n_s}}}
\]

The test statistic is used to calculate a p-value under the assumptions of the selected statistical test.

The result should be interpreted together with:

- The subgroup sample size.
- The size of the performance gap.
- The assumptions of the test.
- The number of hypotheses tested.

---

## 7. Fisher's Exact Test

For small classification subgroups, an exact test may be more appropriate than a large-sample approximation.

The observations can be represented using a contingency table:

| | Correct | Incorrect |
|---|---:|---:|
| Subgroup | \(a\) | \(b\) |
| Reference group | \(c\) | \(d\) |

Fisher's Exact Test calculates the probability of observing a table at least as extreme as the observed table under the null hypothesis.

The test does not depend on the same large-sample approximation as the Z-Test.

However, very small samples may still provide limited statistical power.

---

## 8. Bootstrap Analysis

For regression metrics such as MAE, RMSE, and MSE, the sampling distribution of the metric may be difficult to model analytically.

PySliceKit can use bootstrap resampling to estimate the uncertainty of a metric.

The bootstrap process is:

1. Select the observations belonging to a subgroup.
2. Draw a sample of the same size with replacement.
3. Calculate the selected metric on the resampled data.
4. Repeat the process a specified number of times.
5. Use the resulting distribution to estimate uncertainty.

If \(B\) bootstrap samples are generated, the resulting metric estimates can be represented as:

\[
M_1^*, M_2^*, \ldots, M_B^*
\]

These values form the empirical bootstrap distribution.

---

## 9. Bootstrap Confidence Intervals

A percentile-based confidence interval can be calculated from the bootstrap distribution.

For a 95% confidence interval:

\[
CI_{95\%}
=
\left[
Q_{0.025},
Q_{0.975}
\right]
\]

where:

- \(Q_{0.025}\) is the 2.5th percentile.
- \(Q_{0.975}\) is the 97.5th percentile.

The interval provides an estimate of the uncertainty surrounding the metric.

A confidence interval should not be interpreted as proof that a subgroup is practically important or causally different from the global population.

---

## 10. Two-Tailed Bootstrap Significance

A two-tailed test considers deviations in both directions.

The null hypothesis can be expressed as:

\[
H_0:
M_{\text{segment}} = M_{\text{global}}
\]

The alternative hypothesis is:

\[
H_1:
M_{\text{segment}} \neq M_{\text{global}}
\]

This approach considers both:

- Subgroups performing substantially worse than the global baseline.
- Subgroups performing substantially better than the global baseline.

The direction of the gap must then be interpreted according to the metric.

---

## 11. Bootstrap P-Value Resolution

Bootstrap p-values are estimated using a finite number of resamples.

If \(B\) bootstrap resamples are used, the resulting empirical probabilities have finite resolution.

Consequently:

- A small number of resamples may produce coarse p-value estimates.
- Increasing the number of resamples can improve probability resolution.
- Extremely small significance thresholds may be difficult to estimate reliably.
- Multiple-comparisons correction can produce thresholds smaller than the available bootstrap resolution.

The number of bootstrap resamples should therefore be considered when interpreting bootstrap-based statistical results.

---

## 12. The Multiple-Comparisons Problem

PySliceKit may evaluate many subgroups during a single analysis.

Each subgroup can represent a separate statistical hypothesis.

If every hypothesis is tested at a significance level of:

\[
\alpha = 0.05
\]

the probability of obtaining at least one false-positive result increases as the number of tests grows.

For \(m\) independent tests, the probability of obtaining at least one false positive can be expressed as:

\[
1-(1-\alpha)^m
\]

This is known as the **multiple-comparisons problem**.

Multiple-comparisons correction methods help reduce the risk of over-interpreting apparently significant results.

---

## 13. Bonferroni Correction

The Bonferroni method controls the family-wise error rate by adjusting the significance threshold.

If \(m\) hypotheses are tested and the desired significance level is \(\alpha\), the adjusted threshold is:

\[
\alpha_{\text{adjusted}}
=
\frac{\alpha}{m}
\]

A hypothesis is considered significant under the Bonferroni threshold when:

\[
p_i
\leq
\frac{\alpha}{m}
\]

where \(p_i\) is the original p-value for hypothesis \(i\).

### Characteristics

Bonferroni correction:

- Is simple to calculate.
- Controls the family-wise error rate under common assumptions.
- Can be conservative.
- May reduce statistical power when many hypotheses are tested.

---

## 14. Benjamini–Hochberg Correction

The Benjamini–Hochberg procedure controls the false discovery rate.

Suppose there are \(m\) p-values.

First, sort the p-values in ascending order:

\[
p_{(1)}
\leq
p_{(2)}
\leq
\cdots
\leq
p_{(m)}
\]

For a selected false-discovery-rate threshold \(q\), calculate:

\[
\frac{i}{m}q
\]

for each ranked p-value \(p_{(i)}\).

Find the largest index \(k\) satisfying:

\[
p_{(k)}
\leq
\frac{k}{m}q
\]

The hypotheses associated with the qualifying ranked p-values are considered significant under the procedure.

### Characteristics

Benjamini–Hochberg correction:

- Controls the false discovery rate under appropriate assumptions.
- Can be less conservative than Bonferroni correction.
- Is useful when many subgroup hypotheses are evaluated.
- Does not guarantee that every reported result is a true positive.

---

## 15. Corrected and Uncorrected Results

A subgroup may be statistically significant before correction but not remain significant after correction.

Users should compare:

- The original p-value.
- The corrected significance decision.
- The subgroup sample size.
- The size of the performance gap.
- The selected correction method.
- The practical importance of the finding.

Statistical significance and practical significance are not the same.

A very small performance difference may be statistically significant in a large dataset, while a large performance difference may not be statistically significant in a small subgroup.

---

## 16. Statistical Interpretation

A statistically significant result does not automatically prove that a model is unfair, defective, or causally biased.

It indicates that the observed result is inconsistent with the assumptions of the selected null hypothesis and statistical procedure at the chosen threshold.

Subgroup results should be interpreted alongside:

- Data quality.
- Segment size.
- Feature definitions.
- Sampling procedures.
- Multiple-comparisons correction.
- Model limitations.
- Distribution differences.
- Practical impact.
- Domain-specific considerations.

PySliceKit is intended to identify potential model blind spots that require further investigation.

It does not independently establish causation or explain why a performance difference exists.

---

## 17. Reproducibility

Statistical results may depend on:

- The dataset.
- The model.
- The selected metric.
- The selected slicing columns.
- The slicing depth.
- The minimum subgroup size.
- The statistical test.
- The number of bootstrap resamples.
- Random seeds and resampling procedures.
- The correction method.

For reproducible analyses, users should record the configuration and environment used to generate the results.

---

## 18. Further Reading

For practical usage instructions, see:

- [README](README.md)
- [Getting Started](https://anshumantiwari2006.github.io/PySliceKit/)
- [PySliceKit Documentation](https://anshumantiwari2006.github.io/PySliceKit/)
- [Statistical Correction Guide](https://anshumantiwari2006.github.io/PySliceKit/correction_guide.html)

For the source code and development history, visit the [PySliceKit repository](https://github.com/AnshumanTiwari2006/PySliceKit).

---

## Disclaimer

The mathematical explanations in this document describe the statistical concepts used in subgroup analysis.

The exact implementation, assumptions, and available parameters should be verified against the corresponding PySliceKit source code and API documentation.
Do fork and edit in, if you need something to add here.
