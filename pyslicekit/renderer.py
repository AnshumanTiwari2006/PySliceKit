"""
pyslicekit.renderer
~~~~~~~~~~~~~~~~~~~
Visual output: heatmap + ranked bar chart with sample-size overlay.

Responsibilities:
  1. heatmap_figure  — columns × values grid, cell = metric, colour = severity
  2. bar_figure      — top-N worst segments ranked by |gap|
  3. _severity_color — metric-direction-aware colour scale (red always = bad)
  4. Theme manager   — consistent fonts, sizes, colour palette

The renderer reads gap values from SliceResult objects directly.
It does NOT respect list order for colouring — it always re-sorts by gap.

Usage:
    from pyslicekit.renderer import render
"""

from __future__ import annotations

import math
import warnings
from typing import List, Optional, Tuple

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from pyslicekit.exceptions import PySliceKitRenderError
from pyslicekit.types import MetricDirection, SliceResult

# ---------------------------------------------------------------------------
# Colour palette — metric-direction-aware
# ---------------------------------------------------------------------------

# Colours for underperforming segments (bad)
_BAD_COLOR = "#E24B4A"       # coral-red
_BAD_COLOR_MILD = "#F09595"  # light red

# Colours for overperforming segments (good)
_GOOD_COLOR = "#1D9E75"      # teal
_GOOD_COLOR_MILD = "#9FE1CB" # light teal

# Neutral (baseline / near-zero gap)
_NEUTRAL_COLOR = "#D3D1C7"

# No-data cell
_NODATA_COLOR = "#F1EFE8"
_NODATA_TEXT = "#888780"

# Low-n warning hatch
_LOWN_HATCH = "//"

# Significance marker suffix
_SIG_MARKER = " *"

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def render(
    results: List[SliceResult],
    top_n: int = 15,
    show: bool = True,
    figsize_heatmap: Tuple[int, int] = (12, 6),
    figsize_bar: Tuple[int, int] = (10, 6),
    title_prefix: str = "pyslicekit",
) -> Tuple[plt.Figure, plt.Figure]:
    """
    Produce both figures: heatmap and ranked bar chart.

    Parameters
    ----------
    results : list of SliceResult (from evaluate())
    top_n : maximum segments to show in the bar chart
    show : call plt.show() after building figures
    figsize_heatmap : (width, height) for the heatmap figure
    figsize_bar : (width, height) for the bar chart figure
    title_prefix : prefix for figure titles

    Returns
    -------
    (fig_heatmap, fig_bar) — both matplotlib Figure objects
    """
    if not results:
        raise PySliceKitRenderError(
            "results list is empty. Nothing to render. "
            "This usually means all segments were filtered out. "
            "Try lowering min_samples."
        )

    try:
        fig_heatmap = _build_heatmap(results, figsize_heatmap, title_prefix)
        fig_bar = _build_bar(results, top_n, figsize_bar, title_prefix)
    except Exception as exc:
        raise PySliceKitRenderError(
            f"Renderer failed: {exc}. "
        ) from exc

    if show:
        plt.show()

    return fig_heatmap, fig_bar


# ---------------------------------------------------------------------------
# Heatmap
# ---------------------------------------------------------------------------


def _build_heatmap(
    results: List[SliceResult],
    figsize: Tuple[int, int],
    title_prefix: str,
) -> plt.Figure:
    """
    Build a grid heatmap for single-column slices only.

    Each row = one slice column. Each column = one unique value in that column.
    Cell fill = severity colour based on gap. Cell text = metric value + n.

    Two-column (cross-product) slices are excluded from the heatmap
    because they cannot be laid out on a 2D grid cleanly.
    They appear in the bar chart only.
    """
    # Filter to depth=1 results only
    single_col = [r for r in results if len(r.slice_def) == 1]

    if not single_col:
        # All results are multi-column — return a placeholder figure
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(
            0.5, 0.5,
            "No single-column slices to display.\nSee bar chart for segment gaps.",
            ha="center", va="center", fontsize=12,
            transform=ax.transAxes,
        )
        ax.axis("off")
        fig.suptitle(f"{title_prefix} — segment heatmap", fontsize=14, weight="bold")
        return fig

    # Group by column name
    col_groups: dict = {}
    for r in single_col:
        col = r.slice_def[0][0]
        col_groups.setdefault(col, []).append(r)

    # Grid dimensions
    n_rows = len(col_groups)
    n_cols = max(len(v) for v in col_groups.values())

    fig, axes = plt.subplots(
        n_rows, 1,
        figsize=figsize,
        squeeze=False,
    )
    fig.suptitle(
        f"{title_prefix} — per-segment performance heatmap",
        fontsize=13, weight="bold", y=1.01,
    )

    metric_name = single_col[0].metric_name
    direction = single_col[0].direction

    for row_idx, (col_name, col_results) in enumerate(col_groups.items()):
        ax = axes[row_idx, 0]

        # Sort values alphabetically for consistent layout
        col_results_sorted = sorted(col_results, key=lambda r: str(r.slice_def[0][1]))
        n_values = len(col_results_sorted)

        ax.set_xlim(0, 1)
        ax.set_ylim(0, n_values)
        ax.axis("off")

        # Row label
        ax.text(
            -0.01, 0.5, col_name,
            ha="right", va="center",
            fontsize=10, weight="bold",
            transform=ax.transAxes,
        )

        for cell_idx, r in enumerate(col_results_sorted):
            # Draw from top to bottom
            y0 = n_values - cell_idx - 1

            fill_color = _severity_color(r, direction)
            rect = mpatches.FancyBboxPatch(
                (0.01, y0 + 0.05),
                0.98,
                0.9,
                boxstyle="round,pad=0.01",
                facecolor=fill_color,
                edgecolor="white",
                linewidth=1.5,
            )
            ax.add_patch(rect)

            # Low-n hatch
            if r.low_n:
                hatch_rect = mpatches.FancyBboxPatch(
                    (0.01, y0 + 0.05),
                    0.98,
                    0.9,
                    boxstyle="round,pad=0.01",
                    facecolor="none",
                    edgecolor="#888780",
                    linewidth=0,
                    hatch=_LOWN_HATCH,
                    alpha=0.4,
                )
                ax.add_patch(hatch_rect)

            # Value label
            value_str = (
                f"{r.metric_value:.3f}"
                if not math.isnan(r.metric_value)
                else "n/a"
            )
            sig_str = _SIG_MARKER if r.is_significant else ""
            text_color = _text_color_for_bg(fill_color)

            cx = 0.5
            ax.text(
                cx, y0 + 0.65, f"{r.slice_def[0][1]}{sig_str}",
                ha="center", va="center",
                fontsize=8.5, color=text_color, weight="bold",
            )
            ax.text(
                cx, y0 + 0.35, f"{value_str}   n={r.n}",
                ha="center", va="center",
                fontsize=8.5, color=text_color,
            )

    # Shared legend
    legend_elements = [
        mpatches.Patch(facecolor=_BAD_COLOR, label="Underperforms"),
        mpatches.Patch(facecolor=_GOOD_COLOR, label="Outperforms"),
        mpatches.Patch(facecolor=_NEUTRAL_COLOR, label="Near baseline"),
        mpatches.Patch(facecolor="white", hatch=_LOWN_HATCH,
                       edgecolor="#888780", label="Low sample count"),
    ]
    
    # Adjust layout FIRST to leave 12% empty space at the bottom of the figure
    plt.tight_layout(rect=[0, 0.12, 1, 0.96])
    
    fig.legend(
        handles=legend_elements,
        loc="upper center",
        ncol=4,
        fontsize=9,
        frameon=False,
        bbox_to_anchor=(0.5, 0.12),
    )
    overall_val = single_col[0].overall_metric
    if direction == MetricDirection.HIGHER_IS_BETTER:
        red_txt = f"Red = segment {metric_name} lower than overall (worse)"
        green_txt = f"Green = segment {metric_name} higher than overall (better)"
    else:
        red_txt = f"Red = segment {metric_name} higher than overall (worse)"
        green_txt = f"Green = segment {metric_name} lower than overall (better)"

    footer_text = (
        f"Overall {metric_name} = {overall_val:.3f}  |  "
        f"{red_txt}  |  {green_txt}\n"
        f"* = statistically significant gap (p < 0.05)"
    )

    fig.text(
        0.5, 0.02,
        footer_text,
        ha="center", fontsize=9, color="#5F5E5A",
    )

    return fig


# ---------------------------------------------------------------------------
# Ranked bar chart
# ---------------------------------------------------------------------------


def _build_bar(
    results: List[SliceResult],
    top_n: int,
    figsize: Tuple[int, int],
    title_prefix: str,
) -> plt.Figure:
    """
    Horizontal bar chart — top-N worst performing segments by |gap|.

    Always re-sorts by abs_gap descending regardless of input list order.
    This enforces the sort-order contract from the blueprint.
    """
    # Sort by absolute gap descending — worst first
    sorted_results = sorted(results, key=lambda r: r.abs_gap, reverse=True)
    display = sorted_results[:top_n]

    if not display:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No segments to display.", ha="center", va="center")
        ax.axis("off")
        return fig

    metric_name = display[0].metric_name
    direction = display[0].direction

    labels = []
    gaps = []
    colors = []
    hatches = []
    annotations = []

    for r in reversed(display):  # reversed so worst is at top of horizontal chart
        sig_mark = " *" if r.is_significant else ""
        labels.append(f"{r.label}{sig_mark}")
        gaps.append(r.gap)
        colors.append(_severity_color(r, direction))
        hatches.append(_LOWN_HATCH if r.low_n else "")
        n_label = f"n={r.n}"
        if r.low_n:
            n_label += " ⚠"
        annotations.append(n_label)

    fig, ax = plt.subplots(figsize=figsize)

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, gaps, color=colors, edgecolor="white", linewidth=0.8)

    # Apply hatch for low-n bars
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    # Annotations: metric gap value + n
    for i, (bar, ann, gap, color) in enumerate(zip(bars, annotations, gaps, colors)):
        x_pos = gap / 2
        ha = "center"
        gap_str = f"{gap:+.3f}" if not math.isnan(gap) else "n/a"
        
        # Color the text to contrast with the bar's background color
        text_color = _text_color_for_bg(color)
        
        ax.text(
            x_pos, i,
            f"{gap_str}  {ann}",
            va="center", ha=ha,
            fontsize=8, color=text_color, weight="bold",
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.axvline(x=0, color="#5F5E5A", linewidth=0.8, linestyle="--")
    ax.set_xlabel(f"Gap vs overall {metric_name} (segment − overall)", fontsize=10)
    ax.set_title(
        f"{title_prefix} — top {len(display)} segments by performance gap\n"
        f"* = statistically significant  ⚠ = low sample count",
        fontsize=11, weight="bold",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    
    # Leave 8% space at bottom for the explanation string
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    
    overall_val = display[0].overall_metric
    if direction == MetricDirection.HIGHER_IS_BETTER:
        red_txt = f"Red = segment {metric_name} lower than overall (worse)"
        green_txt = f"Green = segment {metric_name} higher than overall (better)"
    else:
        red_txt = f"Red = segment {metric_name} higher than overall (worse)"
        green_txt = f"Green = segment {metric_name} lower than overall (better)"
        
    footer_text = f"Overall {metric_name} = {overall_val:.3f}  |  {red_txt}  |  {green_txt}"

    fig.text(
        0.5, 0.02,
        footer_text,
        ha="center", fontsize=9, color="#5F5E5A",
    )

    return fig


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------


def _severity_color(result: SliceResult, direction: MetricDirection) -> str:
    """
    Return a fill colour for a segment cell based on how bad the gap is.

    Rule: red always = bad, green always = good, grey = near-baseline.
    The mapping from gap sign to "good/bad" depends on metric direction.
    """
    if math.isnan(result.gap):
        return _NODATA_COLOR

    threshold_mild = 0.02   # 2 percentage points — mild deviation
    threshold_strong = 0.05  # 5 percentage points — strong deviation

    is_bad = result.is_underperforming
    magnitude = result.abs_gap

    if magnitude < threshold_mild:
        return _NEUTRAL_COLOR

    if is_bad:
        return _BAD_COLOR if magnitude >= threshold_strong else _BAD_COLOR_MILD
    else:
        return _GOOD_COLOR if magnitude >= threshold_strong else _GOOD_COLOR_MILD


def _text_color_for_bg(bg_color: str) -> str:
    """
    Return white text for dark backgrounds, dark text for light backgrounds.
    Threshold: luminance < 0.5 → white text.
    """
    dark_backgrounds = {_BAD_COLOR, _GOOD_COLOR}
    return "white" if bg_color in dark_backgrounds else "#2C2C2A"


