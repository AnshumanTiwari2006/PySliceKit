"""
pyslicekit.correction
~~~~~~~~~~~~~~~~~~~~~
Applies multiple-comparisons correction to evaluate() results.
"""

from typing import List
from pyslicekit.types import SliceResult
from statsmodels.stats.multitest import multipletests

def apply_correction(
    results: List[SliceResult],
    method: str = "fdr_bh",
    alpha: float = 0.05
) -> List[SliceResult]:
    """
    Applies multiple-comparisons correction to the evaluated segments.

    Multiple-comparisons correction addresses the problem of inflated false
    positive rates when conducting many statistical tests simultaneously.
    When evaluate() runs at depth=2, it can generate hundreds of simultaneous
    hypothesis tests (one per segment with test_used in {"proportion_z",
    "fisher_exact", "bootstrap_ci"}). This meaningfully inflates the chance
    of observing is_significant=True purely by noise.

    - "bonferroni" controls the family-wise error rate (FWER) and is very
      conservative. It guarantees the probability of even one false positive
      across all tests is <= alpha.
    - "fdr_bh" (Benjamini-Hochberg) controls the false discovery rate (FDR).
      It ensures that out of all segments declared significant, only alpha
      proportion are expected to be false positives. This is standard in
      most applied contexts because it is more powerful than Bonferroni.

    Parameters
    ----------
    results : List[SliceResult]
        The results from evaluate().
    method : str
        The correction method to use, e.g., "bonferroni" or "fdr_bh".
    alpha : float
        The alpha level for the correction (default 0.05).

    Returns
    -------
    List[SliceResult]
        The mutated results list, for chaining.
    """
    if not results:
        raise ValueError("results list is empty.")

    # Only include results that were actually tested (p_value is not None)
    eligible_indices = [i for i, r in enumerate(results) if r.p_value is not None]
    
    if not eligible_indices:
        raise ValueError("zero results have a non-None p_value (nothing testable).")

    p_values = [results[i].p_value for i in eligible_indices]

    reject, pvals_corrected, _, _ = multipletests(
        pvals=p_values,
        alpha=alpha,
        method=method
    )

    for idx, p_val_corr, is_sig_corr in zip(eligible_indices, pvals_corrected, reject):
        r = results[idx]
        r.extra["p_value_corrected"] = float(p_val_corr)
        r.extra["is_significant_corrected"] = bool(is_sig_corr)
        r.extra["correction_method"] = method

    return results
