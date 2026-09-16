"""
pyslicekit.correction_render
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Renders a comparison chart for segments that were originally significant.
"""

from typing import List, Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from pyslicekit.types import SliceResult
from pyslicekit.correction import apply_correction
from pyslicekit.exceptions import PySliceKitRenderError

bad_color = "#E24B4A"
good_color = "#1D9E75"
neutral_color = "#D3D1C7"
hatch_pattern = "//"

def render_correction_comparison(
    results: List[SliceResult],
    method: str = "fdr_bh",
    alpha: float = 0.05,
    figsize: Tuple[int, int] = (14, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    
    if not results:
        raise PySliceKitRenderError("results list is empty. Nothing to render.")

    # Check if correction has been applied
    first_eligible = next((r for r in results if r.p_value is not None), None)
    if first_eligible and "p_value_corrected" not in first_eligible.extra:
        apply_correction(results, method=method, alpha=alpha)

    # Restrict to originally significant BUT lost significance after correction
    display = [r for r in results if r.is_significant and not r.extra.get("is_significant_corrected", False)]
    
    if not display:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No segments lost significance.", ha="center", va="center")
        ax.axis("off")
        return fig

    # Sort by abs(gap) descending
    display = sorted(display, key=lambda r: r.abs_gap, reverse=True)

    labels = []
    gaps = []
    colors = []
    hatches = []
    annotations = []
    flipped_count = 0

    for r in reversed(display): # Reversed so worst is at top
        labels.append(r.label)
        gaps.append(r.gap)
        
        # Solid fill
        colors.append(bad_color if r.is_underperforming else good_color)
        
        survived = r.extra.get("is_significant_corrected", False)
        if not survived:
            hatches.append(hatch_pattern)
            annotations.append("lost significance")
            flipped_count += 1
        else:
            hatches.append("")
            annotations.append("survived")

    try:
        fig, ax = plt.subplots(figsize=figsize)
        
        y_pos = np.arange(len(labels))
        bars = ax.barh(y_pos, gaps, color=colors, edgecolor="white", linewidth=0.8)
        
        for bar, hatch in zip(bars, hatches):
            bar.set_hatch(hatch)
            
        for i, (bar, ann, gap, color) in enumerate(zip(bars, annotations, gaps, colors)):
            x_pos = gap / 2
            
            # Simple text color selection
            text_color = "white" if color in {bad_color, good_color} else "#2C2C2A"
            
            ax.text(
                x_pos, i,
                ann,
                va="center", ha="center",
                fontsize=8, color=text_color, weight="bold"
            )
            
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=9)
        ax.axvline(x=0, color="#5F5E5A", linewidth=0.8, linestyle="--")
        
        ax.set_title(f"pyslicekit — significance before vs after {method} correction", fontsize=11, weight="bold")
        
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        
        footer_text = (
            f"Correction method: {method} (alpha={alpha}) | "
            f"Showing {len(display)} originally significant segments | "
            f"{flipped_count} lost significance"
        )
        
        fig.text(0.5, 0.02, footer_text, ha="center", fontsize=9, color="#5F5E5A")
        
        if save_path:
            fig.savefig(save_path, bbox_inches="tight")
            
        return fig
    except Exception as exc:
        raise PySliceKitRenderError(f"Renderer failed: {exc}") from exc
