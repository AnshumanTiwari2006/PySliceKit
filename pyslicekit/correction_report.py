"""
pyslicekit.correction_report
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Exports a plain-text report comparing significance before and after correction.
"""

from typing import List
from pyslicekit.types import SliceResult
from pyslicekit.correction import apply_correction

def export_correction_report(
    results: List[SliceResult],
    filepath: str,
    method: str = "fdr_bh",
    alpha: float = 0.05
) -> None:
    """
    Export a plain-text report comparing significance before and after
    multiple-comparisons correction.

    Parameters
    ----------
    results : List[SliceResult]
        Results returned by ``pyslicekit.evaluate()``.
    filepath : str
        Destination path for the text report.
    method : str, optional
        Multiple-comparisons correction method, such as ``"fdr_bh"`` or
        ``"bonferroni"``. Default is ``"fdr_bh"``.
    alpha : float, optional
        Significance level used for correction. Default is ``0.05``.

    Returns
    -------
    None
        The report is written to ``filepath``. If ``results`` is empty,
        the function returns without writing a report.
    """
    if not results:
        return
        
    # Check if we need to apply correction
    first_eligible = next((r for r in results if r.p_value is not None), None)
    if first_eligible and "p_value_corrected" not in first_eligible.extra:
        apply_correction(results, method=method, alpha=alpha)
        
    eligible_results = [r for r in results if r.p_value is not None]
    
    total_segments = len(results)
    total_eligible = len(eligible_results)
    
    before_count = sum(1 for r in eligible_results if r.is_significant)
    after_count = sum(1 for r in eligible_results if r.extra.get("is_significant_corrected"))

    lost_sig = [
        r for r in eligible_results
        if r.is_significant and not r.extra.get("is_significant_corrected")
    ]
    
    kept_sig = [
        r for r in eligible_results
        if r.is_significant and r.extra.get("is_significant_corrected")
    ]

    with open(filepath, "w", encoding="utf-8") as f:
        # 1. Header block
        f.write(f"Total segments evaluated: {total_segments}\n")
        f.write(f"Eligible for testing: {total_eligible}\n")
        f.write(f"Correction method: {method}\n")
        f.write(f"Alpha value: {alpha}\n\n")
        
        # 2. Summary line
        f.write(f"Significant before correction: {before_count}\n")
        f.write(f"Significant after correction: {after_count}\n\n")
        
        # 3. Lost significance
        f.write("SEGMENTS THAT LOST SIGNIFICANCE AFTER CORRECTION:\n")
        for r in lost_sig:
            f.write(f"{r.label} | n={r.n} | gap={r.gap:.4f} | p_value={r.p_value:.4f} | p_value_corrected={r.extra['p_value_corrected']:.4f}\n")
        f.write("\n")
        
        # 4. Kept significance
        f.write("SEGMENTS THAT REMAIN SIGNIFICANT AFTER CORRECTION:\n")
        for r in kept_sig:
            f.write(f"{r.label} | n={r.n} | gap={r.gap:.4f} | p_value={r.p_value:.4f} | p_value_corrected={r.extra['p_value_corrected']:.4f}\n")
