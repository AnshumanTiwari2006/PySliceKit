# """
# generate_extra_figures.py

# Regenerates the four dataset audits (california_housing, breast_cancer,
# adult_income, diabetes) and produces five additional figures for the paper:

#   1. raw_vs_corrected_scatter.png   -- raw p vs BH-corrected p, all datasets pooled
#   2. effect_size_vs_pvalue.png      -- |gap| vs raw p, colored by survived/lost
#   4. bonferroni_vs_test_count.png   -- threshold curve + bootstrap resolution floor
#   6. pvalue_by_test_type_boxplot.png -- bootstrap vs z-test/fisher p-value spread
#   7. survival_flow.png              -- before -> after BH flow, per dataset

# Run with:  python generate_extra_figures.py
# Requires: pyslicekit (with the fixed stats.py) installed/importable,
# scikit-learn, pandas, numpy, matplotlib, statsmodels.

# Everything is saved to ./verification_output/extra_figures/
# """

# import os
# from pathlib import Path
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt

# from sklearn.datasets import (
#     fetch_california_housing, load_breast_cancer, fetch_openml, load_diabetes
# )
# from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
# from sklearn.linear_model import LogisticRegression, Ridge
# from sklearn.model_selection import train_test_split

# import pyslicekit
# from pyslicekit.correction import apply_correction

# # Use pathlib and an absolute path so Windows cannot choke on mixed / and \
# # separators, and create every parent directory that might be missing.
# OUT_DIR = Path("verification_output") / "extra_figures"
# OUT_DIR = OUT_DIR.resolve()
# OUT_DIR.mkdir(parents=True, exist_ok=True)


# def savepath(filename: str) -> str:
#     """Return an absolute, OS-correct path string for a file inside OUT_DIR."""
#     return str(OUT_DIR / filename)


# def safe_savefig(fig, filename: str, **kwargs):
#     """
#     Save a figure with retries, to survive transient Windows file locks
#     (antivirus real-time scanning or the search indexer briefly holding the
#     file) which can cause OSError [Errno 22] on an otherwise valid path.
#     """
#     import time
#     path = savepath(filename)
#     last_exc = None
#     for attempt in range(5):
#         try:
#             fig.savefig(path, **kwargs)
#             return path
#         except OSError as exc:
#             last_exc = exc
#             time.sleep(0.5 * (attempt + 1))
#     raise RuntimeError(
#         f"Failed to save {filename!r} after 5 attempts. "
#         f"Last error: {last_exc}. This is usually a transient Windows file "
#         f"lock (antivirus or search indexer) -- try running the script again, "
#         f"or temporarily disable real-time antivirus scanning for this folder."
#     ) from last_exc

# # Color palette -- consistent with the rest of the paper's figures
# COLOR_SURVIVED = "#1D9E75"   # teal
# COLOR_LOST     = "#E24B4A"   # red
# COLOR_NEUTRAL  = "#D3D1C7"   # grey
# COLOR_BOOTSTRAP = "#E24B4A"
# COLOR_CLASSIFICATION = "#1D9E75"


# # ---------------------------------------------------------------------------
# # Step 1 -- rebuild all four audits (same setup as the main pipeline)
# # ---------------------------------------------------------------------------

# def build_california_housing():
#     cali = fetch_california_housing(as_frame=True)
#     df = cali.frame
#     np.random.seed(42)
#     df["ocean_proximity"] = np.random.choice(
#         ["INLAND", "NEAR BAY", "NEAR OCEAN", "<1H OCEAN"], size=len(df)
#     )
#     X = df.drop(columns=["MedHouseVal"])
#     y = df["MedHouseVal"]
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#     model = RandomForestRegressor(n_estimators=100, random_state=42)
#     model.fit(X_train.drop(columns=["ocean_proximity"]), y_train)
#     y_pred = model.predict(X_test.drop(columns=["ocean_proximity"]))
#     results = pyslicekit.evaluate(
#         model=model, df=X_test, y_true=y_test.values, y_pred=y_pred,
#         slice_cols=["HouseAge", "ocean_proximity"], metric="mae",
#         min_samples=50, depth=2, render_visuals=False,
#     )
#     return results


# def build_breast_cancer():
#     cancer = load_breast_cancer(as_frame=True)
#     df_c = cancer.frame
#     X_c = df_c.drop(columns=["target"])
#     y_c = df_c["target"]
#     X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
#         X_c, y_c, test_size=0.25, random_state=42
#     )
#     clf = LogisticRegression(max_iter=5000)
#     clf.fit(X_train_c, y_train_c)
#     y_pred_c = clf.predict(X_test_c)
#     results = pyslicekit.evaluate(
#         model=clf, df=X_test_c, y_true=y_test_c.values, y_pred=y_pred_c,
#         slice_cols=["mean radius", "mean texture"], metric="f1",
#         min_samples=10, depth=2, render_visuals=False,
#     )
#     return results


# def build_adult_income():
#     adult = fetch_openml(data_id=1590, as_frame=True)
#     df_a = adult.frame.dropna()
#     X_a = df_a.drop(columns=["class"])
#     y_a = (df_a["class"] == ">50K").astype(int)
#     X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
#         X_a, y_a, test_size=0.2, random_state=42
#     )
#     numeric_cols = X_train_a.select_dtypes(include=[np.number]).columns
#     clf_a = RandomForestClassifier(n_estimators=50, random_state=42)
#     clf_a.fit(X_train_a[numeric_cols], y_train_a)
#     y_pred_a = clf_a.predict(X_test_a[numeric_cols])
#     results = pyslicekit.evaluate(
#         model=clf_a, df=X_test_a, y_true=y_test_a.values, y_pred=y_pred_a,
#         slice_cols=["sex", "race"], metric="accuracy",
#         min_samples=30, depth=2, render_visuals=False,
#     )
#     return results


# def build_diabetes():
#     diabetes = load_diabetes(as_frame=True)
#     df_d = diabetes.frame
#     X_d = df_d.drop(columns=["target"])
#     y_d = df_d["target"]
#     X_d["age_group"] = pd.qcut(X_d["age"], q=3, labels=["Young", "Middle", "Old"])
#     X_d["bmi_group"] = pd.qcut(X_d["bmi"], q=3, labels=["Low", "Med", "High"])
#     X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
#         X_d, y_d, test_size=0.2, random_state=42
#     )
#     model_d = Ridge()
#     model_d.fit(X_train_d.drop(columns=["age_group", "bmi_group"]), y_train_d)
#     y_pred_d = model_d.predict(X_test_d.drop(columns=["age_group", "bmi_group"]))
#     results = pyslicekit.evaluate(
#         model=model_d, df=X_test_d, y_true=y_test_d.values, y_pred=y_pred_d,
#         slice_cols=["age_group", "bmi_group"], metric="mae",
#         min_samples=15, depth=2, render_visuals=False,
#     )
#     return results


# print("Rebuilding all four audits...")
# datasets = {
#     "california_housing": build_california_housing(),
#     "breast_cancer": build_breast_cancer(),
#     "adult_income": build_adult_income(),
#     "diabetes": build_diabetes(),
# }

# # Apply BH correction to every dataset
# for name, results in datasets.items():
#     apply_correction(results, method="fdr_bh", alpha=0.05)
#     apply_correction(results, method="bonferroni", alpha=0.05)
#     print(f"  {name}: {len(results)} total segments")


# # ---------------------------------------------------------------------------
# # Build one flat table of every eligible segment across all datasets
# # ---------------------------------------------------------------------------

# rows = []
# for dataset_name, results in datasets.items():
#     for r in results:
#         if r.p_value is None:
#             continue
#         test_group = "bootstrap_ci" if r.test_used == "bootstrap_ci" else "classification"
#         # Normalized effect size: |gap| as a fraction of the overall metric.
#         # Raw |gap| is not comparable across datasets because metrics live on
#         # very different scales (e.g. MAE in the tens for diabetes vs
#         # accuracy/F1 bounded in [0, 1]). Dividing by the overall metric puts
#         # every segment's effect size on a roughly comparable relative scale.
#         overall = r.overall_metric
#         norm_gap = (r.abs_gap / abs(overall)) if overall not in (0, None) else float("nan")

#         rows.append({
#             "dataset": dataset_name,
#             "label": r.label,
#             "n": r.n,
#             "test_used": r.test_used,
#             "test_group": test_group,
#             "gap": r.gap,
#             "abs_gap": r.abs_gap,
#             "norm_gap": norm_gap,
#             "raw_p": r.p_value,
#             "bh_p": r.extra.get("p_value_corrected"),
#             "bh_sig": r.extra.get("is_significant_corrected"),
#             "sig_before": r.is_significant,
#         })

# table = pd.DataFrame(rows)
# table.to_csv(savepath("pooled_segment_table.csv"), index=False)
# print(f"\nPooled table: {len(table)} eligible segments across all datasets.")


# # ---------------------------------------------------------------------------
# # Figure 1 -- raw p-value vs BH-corrected p-value, all datasets pooled
# # ---------------------------------------------------------------------------

# fig, ax = plt.subplots(figsize=(8, 8))

# for group, color, label in [
#     ("bootstrap_ci", COLOR_BOOTSTRAP, "Bootstrap CI (regression)"),
#     ("classification", COLOR_CLASSIFICATION, "z-test / Fisher (classification)"),
# ]:
#     sub = table[table["test_group"] == group]
#     ax.scatter(
#         sub["raw_p"].clip(lower=1e-6),
#         sub["bh_p"].clip(lower=1e-6),
#         color=color, edgecolor="white", linewidth=0.6, s=70, alpha=0.85, label=label,
#     )

# lims = [1e-6, 1.5]
# ax.plot(lims, lims, linestyle="--", color="#5F5E5A", linewidth=1, label="y = x (no change)")
# ax.set_xscale("log")
# ax.set_yscale("log")
# ax.set_xlim(lims)
# ax.set_ylim(lims)
# ax.set_xlabel("Raw p-value (log scale)", fontsize=11)
# ax.set_ylabel("BH-corrected p-value (log scale)", fontsize=11)
# ax.set_title("Raw vs BH-corrected p-value, all datasets pooled", fontsize=12, weight="bold")
# ax.legend(fontsize=9, loc="upper left")
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)
# fig.tight_layout()
# safe_savefig(fig, "1_raw_vs_corrected_scatter.png", dpi=300)
# plt.close(fig)
# print("Saved: 1_raw_vs_corrected_scatter.png")
# print("  Caption: Raw p-values compared to their Benjamini-Hochberg corrected")
# print("  values for every eligible segment across all four datasets. Points")
# print("  further from the diagonal were penalized more heavily by correction.")


# # ---------------------------------------------------------------------------
# # Figure 2 -- effect size (|gap|) vs raw p-value, colored by survival
# # ---------------------------------------------------------------------------

# fig, ax = plt.subplots(figsize=(9, 6))

# survived = table[table["sig_before"] & table["bh_sig"]]
# lost = table[table["sig_before"] & ~table["bh_sig"]]
# never = table[~table["sig_before"]]

# # Use norm_gap (|gap| as a fraction of the overall metric) rather than raw
# # gap, since raw gap is not comparable across datasets with metrics on very
# # different scales (e.g. MAE in the tens for diabetes vs accuracy/F1 in [0,1]).
# ax.scatter(never["norm_gap"], never["raw_p"].clip(lower=1e-6),
#            color=COLOR_NEUTRAL, edgecolor="white", linewidth=0.5, s=45,
#            alpha=0.6, label="Never significant")
# ax.scatter(lost["norm_gap"], lost["raw_p"].clip(lower=1e-6),
#            color=COLOR_LOST, edgecolor="white", linewidth=0.6, s=80,
#            label="Lost significance under BH")
# ax.scatter(survived["norm_gap"], survived["raw_p"].clip(lower=1e-6),
#            color=COLOR_SURVIVED, edgecolor="white", linewidth=0.6, s=80,
#            label="Survived BH correction")

# ax.axhline(0.05, color="#5F5E5A", linestyle="--", linewidth=1)
# ax.text(ax.get_xlim()[1] * 0.98, 0.05, "alpha = 0.05", va="bottom", ha="right",
#         fontsize=8, color="#5F5E5A")
# ax.set_yscale("log")
# ax.set_xlabel("Normalized effect size, |gap| / overall metric", fontsize=11)
# ax.set_ylabel("Raw p-value (log scale)", fontsize=11)
# ax.set_title("Normalized effect size vs raw p-value, colored by correction outcome",
#              fontsize=12, weight="bold")
# ax.legend(fontsize=9)
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)
# fig.tight_layout()
# safe_savefig(fig, "2_effect_size_vs_pvalue.png", dpi=300)
# plt.close(fig)
# print("Saved: 2_effect_size_vs_pvalue.png")
# print("  Caption: Segment effect size (as a fraction of the overall metric,")
# print("  so datasets with different metric scales are comparable) against")
# print("  raw p-value, colored by correction outcome. Segments in the upper")
# print("  region combine a real-looking gap with weak statistical support.")


# # ---------------------------------------------------------------------------
# # Figure 4 -- Bonferroni threshold vs number of tests, with resolution floor
# # ---------------------------------------------------------------------------

# N_BOOTSTRAP = 1000
# RESOLUTION_FLOOR = 2 / N_BOOTSTRAP  # smallest possible two-tailed bootstrap p-value

# m_range = np.arange(1, 51)
# thresholds = 0.05 / m_range

# fig, ax = plt.subplots(figsize=(9, 6))
# ax.plot(m_range, thresholds, color="#2C2C2A", linewidth=2, label="Bonferroni threshold (0.05 / m)")
# ax.axhline(RESOLUTION_FLOOR, color=COLOR_LOST, linestyle="--", linewidth=1.5,
#            label=f"Bootstrap resolution floor ({RESOLUTION_FLOOR:.4f}, n_bootstrap=1000)")

# crossing_m = 0.05 / RESOLUTION_FLOOR
# ax.axvline(crossing_m, color=COLOR_LOST, linestyle=":", linewidth=1)
# ax.fill_betweenx([0, thresholds.max()], crossing_m, m_range.max(),
#                   color=COLOR_LOST, alpha=0.08,
#                   label=f"Beyond m={crossing_m:.0f}: floor exceeds threshold")

# # Mark actual dataset points
# for m_actual, name, offset in [(24, "california_housing (m=24)", 1), (7, "diabetes (m=7)", -1)]:
#     thr = 0.05 / m_actual
#     ax.scatter([m_actual], [thr], color="#2C2C2A", zorder=5, s=60)
#     ax.annotate(name, (m_actual, thr), textcoords="offset points",
#                 xytext=(6, 8 * offset), fontsize=8)

# ax.set_xlabel("Number of simultaneous tests (m)", fontsize=11)
# ax.set_ylabel("Threshold p-value", fontsize=11)
# ax.set_title("Bonferroni threshold vs test count, against the bootstrap resolution floor",
#              fontsize=12, weight="bold")
# ax.legend(fontsize=8, loc="upper right")
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)
# fig.tight_layout()
# safe_savefig(fig, "4_bonferroni_vs_test_count.png", dpi=300)
# plt.close(fig)
# print("Saved: 4_bonferroni_vs_test_count.png")
# print(f"  Caption: The Bonferroni threshold falls below the bootstrap test's")
# print(f"  minimum resolvable p-value once more than {crossing_m:.0f} tests are")
# print(f"  run at n_bootstrap=1000, making it mathematically impossible for any")
# print(f"  bootstrap-tested segment to survive correction beyond that point.")


# # ---------------------------------------------------------------------------
# # Figure 6 -- raw p-value distribution by test type (boxplot)
# # ---------------------------------------------------------------------------

# fig, ax = plt.subplots(figsize=(7, 6))

# groups_data = [
#     table[table["test_group"] == "bootstrap_ci"]["raw_p"].values,
#     table[table["test_group"] == "classification"]["raw_p"].values,
# ]
# bp = ax.boxplot(
#     groups_data, labels=["Bootstrap CI\n(regression)", "z-test / Fisher\n(classification)"],
#     patch_artist=True, widths=0.5,
# )
# for patch, color in zip(bp["boxes"], [COLOR_BOOTSTRAP, COLOR_CLASSIFICATION]):
#     patch.set_facecolor(color)
#     patch.set_alpha(0.75)
# for median in bp["medians"]:
#     median.set_color("#2C2C2A")
#     median.set_linewidth(1.5)

# ax.axhline(0.05, color="#5F5E5A", linestyle="--", linewidth=1)
# ax.text(2.35, 0.05, "alpha = 0.05", va="bottom", ha="right", fontsize=8, color="#5F5E5A")
# ax.set_ylabel("Raw p-value", fontsize=11)
# ax.set_title("Raw p-value spread by significance test type", fontsize=12, weight="bold")
# ax.spines["top"].set_visible(False)
# ax.spines["right"].set_visible(False)
# fig.tight_layout()
# safe_savefig(fig, "6_pvalue_by_test_type_boxplot.png", dpi=300)
# plt.close(fig)
# print("Saved: 6_pvalue_by_test_type_boxplot.png")
# print("  Caption: Distribution of raw p-values pooled across all four datasets,")
# print("  split by the underlying significance test. Classification results")
# print("  cluster toward the extremes; bootstrap results sit closer to the")
# print("  uncorrected threshold, making them more sensitive to correction.")


# # ---------------------------------------------------------------------------
# # Figure 7 -- before/after BH flow, one panel per dataset
# # ---------------------------------------------------------------------------

# fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)

# for ax, (name, results) in zip(axes, datasets.items()):
#     eligible = [r for r in results if r.p_value is not None]
#     before = sum(1 for r in eligible if r.is_significant)
#     survived = sum(1 for r in eligible if r.is_significant and r.extra.get("is_significant_corrected"))
#     lost = before - survived
#     total = len(eligible)
#     never = total - before

#     # Simple two-stage stacked flow using bars at x=0 (before) and x=1 (after)
#     ax.bar([0], [never], bottom=[before], color=COLOR_NEUTRAL, width=0.4, label="Never significant")
#     ax.bar([0], [survived], bottom=[0], color=COLOR_SURVIVED, width=0.4)
#     ax.bar([0], [lost], bottom=[survived], color=COLOR_LOST, width=0.4)

#     ax.bar([1], [never], bottom=[survived], color=COLOR_NEUTRAL, width=0.4)
#     ax.bar([1], [survived], bottom=[0], color=COLOR_SURVIVED, width=0.4)

#     # connecting lines for the "survived" and "lost" blocks
#     ax.plot([0.2, 0.8], [survived, survived], color=COLOR_SURVIVED, linewidth=1, alpha=0.5)
#     ax.plot([0.2, 0.8], [survived + lost, survived], color=COLOR_LOST, linewidth=1, alpha=0.5)

#     ax.set_xticks([0, 1])
#     ax.set_xticklabels(["Before", "After BH"])
#     ax.set_title(f"{name}\n(m={total})", fontsize=10)
#     ax.set_xlim(-0.4, 1.4)

# axes[0].set_ylabel("Segment count", fontsize=11)

# handles = [
#     plt.Rectangle((0, 0), 1, 1, color=COLOR_SURVIVED, label="Survived BH"),
#     plt.Rectangle((0, 0), 1, 1, color=COLOR_LOST, label="Lost to BH"),
#     plt.Rectangle((0, 0), 1, 1, color=COLOR_NEUTRAL, label="Never significant"),
# ]
# fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 1.05))
# fig.suptitle("Segment flow before and after BH correction, by dataset", fontsize=13, weight="bold", y=1.1)
# fig.tight_layout()
# safe_savefig(fig, "7_survival_flow.png", dpi=300, bbox_inches="tight")
# plt.close(fig)
# print("Saved: 7_survival_flow.png")
# print("  Caption: Segment counts before and after BH correction for each")
# print("  dataset, showing how many originally significant segments survived,")
# print("  were lost, versus segments that were never significant to begin with.")
# print("  Note: the diabetes panel shows zero significant segments both before")
# print("  and after correction -- this is a null-result baseline, not a case")
# print("  of correction removing findings, since nothing was flagged to begin")
# print("  with. Worth stating this explicitly in the figure caption in the paper.")

# print(f"\nAll figures saved to {OUT_DIR}/")



























"""
generate_extra_figures.py

Regenerates the four dataset audits (california_housing, breast_cancer,
adult_income, diabetes) and produces five additional figures for the paper:

  1. raw_vs_corrected_scatter.png   -- raw p vs BH-corrected p, all datasets pooled
  2. effect_size_vs_pvalue.png      -- |gap| vs raw p, colored by survived/lost
  4. bonferroni_vs_test_count.png   -- threshold curve + bootstrap resolution floor
  6. pvalue_by_test_type_boxplot.png -- bootstrap vs z-test/fisher p-value spread
  7. survival_flow.png              -- before -> after BH flow, per dataset

Run with:  python generate_extra_figures.py
Requires: pyslicekit (with the fixed stats.py) installed/importable,
scikit-learn, pandas, numpy, matplotlib, statsmodels.

Everything is saved to ./verification_output/extra_figures/
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import (
    fetch_california_housing, load_breast_cancer, fetch_openml, load_diabetes
)
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import train_test_split

import pyslicekit
from pyslicekit.correction import apply_correction

# Use pathlib and an absolute path so Windows cannot choke on mixed / and \
# separators, and create every parent directory that might be missing.
OUT_DIR = Path("verification_output") / "extra_figures"
OUT_DIR = OUT_DIR.resolve()
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Larger, consistent fonts across every figure -- fixes text that reads fine
# at normal size but turns blurry/illegible once someone zooms into the PNG.
plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "figure.titlesize": 16,
})


def savepath(filename: str) -> str:
    """Return an absolute, OS-correct path string for a file inside OUT_DIR."""
    return str(OUT_DIR / filename)


def safe_savefig(fig, filename: str, **kwargs):
    """
    Save a figure as BOTH a high-DPI PNG and a vector PDF, with retries to
    survive transient Windows file locks (antivirus / search indexer), which
    can cause OSError [Errno 22] even on an otherwise valid path.

    The PDF is the version to actually use in the paper -- it is vector-based,
    so it stays perfectly sharp at any zoom level, unlike a PNG which is a
    fixed pixel grid and will always blur past a certain zoom. The PNG is
    kept as a convenience for quick previews or platforms that require raster.
    """
    import time
    stem = filename.rsplit(".", 1)[0]
    targets = [savepath(f"{stem}.png"), savepath(f"{stem}.pdf")]
    png_kwargs = dict(kwargs)
    png_kwargs.setdefault("dpi", 500)  # raised from 300 for sharper zoomed-in text
    pdf_kwargs = {k: v for k, v in kwargs.items() if k != "dpi"}  # dpi is meaningless for vector PDF

    last_exc = None
    for path, save_kwargs in [(targets[0], png_kwargs), (targets[1], pdf_kwargs)]:
        for attempt in range(5):
            try:
                fig.savefig(path, **save_kwargs)
                break
            except OSError as exc:
                last_exc = exc
                time.sleep(0.5 * (attempt + 1))
        else:
            raise RuntimeError(
                f"Failed to save {path!r} after 5 attempts. "
                f"Last error: {last_exc}. This is usually a transient Windows file "
                f"lock (antivirus or search indexer) -- try running the script again, "
                f"or temporarily disable real-time antivirus scanning for this folder."
            ) from last_exc
    return targets[1]  # return the PDF path -- that's the one to use in the paper

# Color palette -- consistent with the rest of the paper's figures
COLOR_SURVIVED = "#1D9E75"   # teal
COLOR_LOST     = "#E24B4A"   # red
COLOR_NEUTRAL  = "#D3D1C7"   # grey
COLOR_BOOTSTRAP = "#E24B4A"
COLOR_CLASSIFICATION = "#1D9E75"


# ---------------------------------------------------------------------------
# Step 1 -- rebuild all four audits (same setup as the main pipeline)
# ---------------------------------------------------------------------------

def build_california_housing():
    cali = fetch_california_housing(as_frame=True)
    df = cali.frame
    np.random.seed(42)
    df["ocean_proximity"] = np.random.choice(
        ["INLAND", "NEAR BAY", "NEAR OCEAN", "<1H OCEAN"], size=len(df)
    )
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train.drop(columns=["ocean_proximity"]), y_train)
    y_pred = model.predict(X_test.drop(columns=["ocean_proximity"]))
    results = pyslicekit.evaluate(
        model=model, df=X_test, y_true=y_test.values, y_pred=y_pred,
        slice_cols=["HouseAge", "ocean_proximity"], metric="mae",
        min_samples=50, depth=2, render_visuals=False,
    )
    return results


def build_breast_cancer():
    cancer = load_breast_cancer(as_frame=True)
    df_c = cancer.frame
    X_c = df_c.drop(columns=["target"])
    y_c = df_c["target"]
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_c, y_c, test_size=0.25, random_state=42
    )
    clf = LogisticRegression(max_iter=5000)
    clf.fit(X_train_c, y_train_c)
    y_pred_c = clf.predict(X_test_c)
    results = pyslicekit.evaluate(
        model=clf, df=X_test_c, y_true=y_test_c.values, y_pred=y_pred_c,
        slice_cols=["mean radius", "mean texture"], metric="f1",
        min_samples=10, depth=2, render_visuals=False,
    )
    return results


def build_adult_income():
    adult = fetch_openml(data_id=1590, as_frame=True)
    df_a = adult.frame.dropna()
    X_a = df_a.drop(columns=["class"])
    y_a = (df_a["class"] == ">50K").astype(int)
    X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
        X_a, y_a, test_size=0.2, random_state=42
    )
    numeric_cols = X_train_a.select_dtypes(include=[np.number]).columns
    clf_a = RandomForestClassifier(n_estimators=50, random_state=42)
    clf_a.fit(X_train_a[numeric_cols], y_train_a)
    y_pred_a = clf_a.predict(X_test_a[numeric_cols])
    results = pyslicekit.evaluate(
        model=clf_a, df=X_test_a, y_true=y_test_a.values, y_pred=y_pred_a,
        slice_cols=["sex", "race"], metric="accuracy",
        min_samples=30, depth=2, render_visuals=False,
    )
    return results


def build_diabetes():
    diabetes = load_diabetes(as_frame=True)
    df_d = diabetes.frame
    X_d = df_d.drop(columns=["target"])
    y_d = df_d["target"]
    X_d["age_group"] = pd.qcut(X_d["age"], q=3, labels=["Young", "Middle", "Old"])
    X_d["bmi_group"] = pd.qcut(X_d["bmi"], q=3, labels=["Low", "Med", "High"])
    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
        X_d, y_d, test_size=0.2, random_state=42
    )
    model_d = Ridge()
    model_d.fit(X_train_d.drop(columns=["age_group", "bmi_group"]), y_train_d)
    y_pred_d = model_d.predict(X_test_d.drop(columns=["age_group", "bmi_group"]))
    results = pyslicekit.evaluate(
        model=model_d, df=X_test_d, y_true=y_test_d.values, y_pred=y_pred_d,
        slice_cols=["age_group", "bmi_group"], metric="mae",
        min_samples=15, depth=2, render_visuals=False,
    )
    return results


print("Rebuilding all four audits...")
datasets = {
    "california_housing": build_california_housing(),
    "breast_cancer": build_breast_cancer(),
    "adult_income": build_adult_income(),
    "diabetes": build_diabetes(),
}

# Apply BH correction to every dataset
for name, results in datasets.items():
    apply_correction(results, method="fdr_bh", alpha=0.05)
    apply_correction(results, method="bonferroni", alpha=0.05)
    print(f"  {name}: {len(results)} total segments")


# ---------------------------------------------------------------------------
# Build one flat table of every eligible segment across all datasets
# ---------------------------------------------------------------------------

rows = []
for dataset_name, results in datasets.items():
    for r in results:
        if r.p_value is None:
            continue
        test_group = "bootstrap_ci" if r.test_used == "bootstrap_ci" else "classification"
        # Normalized effect size: |gap| as a fraction of the overall metric.
        # Raw |gap| is not comparable across datasets because metrics live on
        # very different scales (e.g. MAE in the tens for diabetes vs
        # accuracy/F1 bounded in [0, 1]). Dividing by the overall metric puts
        # every segment's effect size on a roughly comparable relative scale.
        overall = r.overall_metric
        norm_gap = (r.abs_gap / abs(overall)) if overall not in (0, None) else float("nan")

        rows.append({
            "dataset": dataset_name,
            "label": r.label,
            "n": r.n,
            "test_used": r.test_used,
            "test_group": test_group,
            "gap": r.gap,
            "abs_gap": r.abs_gap,
            "norm_gap": norm_gap,
            "raw_p": r.p_value,
            "bh_p": r.extra.get("p_value_corrected"),
            "bh_sig": r.extra.get("is_significant_corrected"),
            "sig_before": r.is_significant,
        })

table = pd.DataFrame(rows)
table.to_csv(savepath("pooled_segment_table.csv"), index=False)
print(f"\nPooled table: {len(table)} eligible segments across all datasets.")


# ---------------------------------------------------------------------------
# Figure 1 -- raw p-value vs BH-corrected p-value, all datasets pooled
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(8, 8))

for group, color, label in [
    ("bootstrap_ci", COLOR_BOOTSTRAP, "Bootstrap CI (regression)"),
    ("classification", COLOR_CLASSIFICATION, "z-test / Fisher (classification)"),
]:
    sub = table[table["test_group"] == group]
    ax.scatter(
        sub["raw_p"].clip(lower=1e-6),
        sub["bh_p"].clip(lower=1e-6),
        color=color, edgecolor="white", linewidth=0.6, s=70, alpha=0.85, label=label,
    )

lims = [1e-6, 1.5]
ax.plot(lims, lims, linestyle="--", color="#5F5E5A", linewidth=1, label="y = x (no change)")
ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_xlabel("Raw p-value (log scale)", fontsize=11)
ax.set_ylabel("BH-corrected p-value (log scale)", fontsize=11)
ax.set_title("Raw vs BH-corrected p-value, all datasets pooled", fontsize=12, weight="bold")
ax.legend(fontsize=9, loc="upper left")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
safe_savefig(fig, "1_raw_vs_corrected_scatter.png", dpi=300)
plt.close(fig)
print("Saved: 1_raw_vs_corrected_scatter.png")
print("  Caption: Raw p-values compared to their Benjamini-Hochberg corrected")
print("  values for every eligible segment across all four datasets. Points")
print("  further from the diagonal were penalized more heavily by correction.")


# ---------------------------------------------------------------------------
# Figure 2 -- effect size (|gap|) vs raw p-value, colored by survival
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 6))

survived = table[table["sig_before"] & table["bh_sig"]]
lost = table[table["sig_before"] & ~table["bh_sig"]]
never = table[~table["sig_before"]]

# Use norm_gap (|gap| as a fraction of the overall metric) rather than raw
# gap, since raw gap is not comparable across datasets with metrics on very
# different scales (e.g. MAE in the tens for diabetes vs accuracy/F1 in [0,1]).
ax.scatter(never["norm_gap"], never["raw_p"].clip(lower=1e-6),
           color=COLOR_NEUTRAL, edgecolor="white", linewidth=0.5, s=45,
           alpha=0.6, label="Never significant")
ax.scatter(lost["norm_gap"], lost["raw_p"].clip(lower=1e-6),
           color=COLOR_LOST, edgecolor="white", linewidth=0.6, s=80,
           label="Lost significance under BH")
ax.scatter(survived["norm_gap"], survived["raw_p"].clip(lower=1e-6),
           color=COLOR_SURVIVED, edgecolor="white", linewidth=0.6, s=80,
           label="Survived BH correction")

ax.axhline(0.05, color="#5F5E5A", linestyle="--", linewidth=1)
ax.text(ax.get_xlim()[1] * 0.98, 0.05, "alpha = 0.05", va="bottom", ha="right",
        fontsize=8, color="#5F5E5A")
ax.set_yscale("log")
ax.set_xlabel("Normalized effect size, |gap| / overall metric", fontsize=11)
ax.set_ylabel("Raw p-value (log scale)", fontsize=11)
ax.set_title("Normalized effect size vs raw p-value, colored by correction outcome",
             fontsize=12, weight="bold")
ax.legend(fontsize=9)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
safe_savefig(fig, "2_effect_size_vs_pvalue.png", dpi=300)
plt.close(fig)
print("Saved: 2_effect_size_vs_pvalue.png")
print("  Caption: Segment effect size (as a fraction of the overall metric,")
print("  so datasets with different metric scales are comparable) against")
print("  raw p-value, colored by correction outcome. Segments in the upper")
print("  region combine a real-looking gap with weak statistical support.")


# ---------------------------------------------------------------------------
# Figure 4 -- Bonferroni threshold vs number of tests, with resolution floor
# ---------------------------------------------------------------------------

N_BOOTSTRAP = 1000
RESOLUTION_FLOOR = 2 / N_BOOTSTRAP  # smallest possible two-tailed bootstrap p-value

m_range = np.arange(1, 51)
thresholds = 0.05 / m_range

fig, ax = plt.subplots(figsize=(9, 6))
ax.plot(m_range, thresholds, color="#2C2C2A", linewidth=2, label="Bonferroni threshold (0.05 / m)")
ax.axhline(RESOLUTION_FLOOR, color=COLOR_LOST, linestyle="--", linewidth=1.5,
           label=f"Bootstrap resolution floor ({RESOLUTION_FLOOR:.4f}, n_bootstrap=1000)")

crossing_m = 0.05 / RESOLUTION_FLOOR
ax.axvline(crossing_m, color=COLOR_LOST, linestyle=":", linewidth=1)
ax.fill_betweenx([0, thresholds.max()], crossing_m, m_range.max(),
                  color=COLOR_LOST, alpha=0.08,
                  label=f"Beyond m={crossing_m:.0f}: floor exceeds threshold")

# Mark actual dataset points
for m_actual, name, offset in [(24, "california_housing (m=24)", 1), (7, "diabetes (m=7)", -1)]:
    thr = 0.05 / m_actual
    ax.scatter([m_actual], [thr], color="#2C2C2A", zorder=5, s=60)
    ax.annotate(name, (m_actual, thr), textcoords="offset points",
                xytext=(6, 8 * offset), fontsize=8)

ax.set_xlabel("Number of simultaneous tests (m)", fontsize=11)
ax.set_ylabel("Threshold p-value", fontsize=11)
ax.set_title("Bonferroni threshold vs test count, against the bootstrap resolution floor",
             fontsize=12, weight="bold")
ax.legend(fontsize=8, loc="upper right")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
safe_savefig(fig, "4_bonferroni_vs_test_count.png", dpi=300)
plt.close(fig)
print("Saved: 4_bonferroni_vs_test_count.png")
print(f"  Caption: The Bonferroni threshold falls below the bootstrap test's")
print(f"  minimum resolvable p-value once more than {crossing_m:.0f} tests are")
print(f"  run at n_bootstrap=1000, making it mathematically impossible for any")
print(f"  bootstrap-tested segment to survive correction beyond that point.")


# ---------------------------------------------------------------------------
# Figure 6 -- raw p-value distribution by test type (boxplot)
# ---------------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(7, 6))

groups_data = [
    table[table["test_group"] == "bootstrap_ci"]["raw_p"].values,
    table[table["test_group"] == "classification"]["raw_p"].values,
]
bp = ax.boxplot(
    groups_data, labels=["Bootstrap CI\n(regression)", "z-test / Fisher\n(classification)"],
    patch_artist=True, widths=0.5,
)
for patch, color in zip(bp["boxes"], [COLOR_BOOTSTRAP, COLOR_CLASSIFICATION]):
    patch.set_facecolor(color)
    patch.set_alpha(0.75)
for median in bp["medians"]:
    median.set_color("#2C2C2A")
    median.set_linewidth(1.5)

ax.axhline(0.05, color="#5F5E5A", linestyle="--", linewidth=1)
ax.text(2.35, 0.05, "alpha = 0.05", va="bottom", ha="right", fontsize=8, color="#5F5E5A")
ax.set_ylabel("Raw p-value", fontsize=11)
ax.set_title("Raw p-value spread by significance test type", fontsize=12, weight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
fig.tight_layout()
safe_savefig(fig, "6_pvalue_by_test_type_boxplot.png", dpi=300)
plt.close(fig)
print("Saved: 6_pvalue_by_test_type_boxplot.png")
print("  Caption: Distribution of raw p-values pooled across all four datasets,")
print("  split by the underlying significance test. Classification results")
print("  cluster toward the extremes; bootstrap results sit closer to the")
print("  uncorrected threshold, making them more sensitive to correction.")


# ---------------------------------------------------------------------------
# Figure 7 -- before/after BH flow, one panel per dataset
# ---------------------------------------------------------------------------

fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)

for ax, (name, results) in zip(axes, datasets.items()):
    eligible = [r for r in results if r.p_value is not None]
    before = sum(1 for r in eligible if r.is_significant)
    survived = sum(1 for r in eligible if r.is_significant and r.extra.get("is_significant_corrected"))
    lost = before - survived
    total = len(eligible)
    never = total - before

    # Simple two-stage stacked flow using bars at x=0 (before) and x=1 (after)
    ax.bar([0], [never], bottom=[before], color=COLOR_NEUTRAL, width=0.4, label="Never significant")
    ax.bar([0], [survived], bottom=[0], color=COLOR_SURVIVED, width=0.4)
    ax.bar([0], [lost], bottom=[survived], color=COLOR_LOST, width=0.4)

    ax.bar([1], [never], bottom=[survived], color=COLOR_NEUTRAL, width=0.4)
    ax.bar([1], [survived], bottom=[0], color=COLOR_SURVIVED, width=0.4)

    # connecting lines for the "survived" and "lost" blocks
    ax.plot([0.2, 0.8], [survived, survived], color=COLOR_SURVIVED, linewidth=1, alpha=0.5)
    ax.plot([0.2, 0.8], [survived + lost, survived], color=COLOR_LOST, linewidth=1, alpha=0.5)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Before", "After BH"])
    ax.set_title(f"{name}\n(m={total})", fontsize=10)
    ax.set_xlim(-0.4, 1.4)

axes[0].set_ylabel("Segment count", fontsize=11)

handles = [
    plt.Rectangle((0, 0), 1, 1, color=COLOR_SURVIVED, label="Survived BH"),
    plt.Rectangle((0, 0), 1, 1, color=COLOR_LOST, label="Lost to BH"),
    plt.Rectangle((0, 0), 1, 1, color=COLOR_NEUTRAL, label="Never significant"),
]
fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 1.05))
fig.suptitle("Segment flow before and after BH correction, by dataset", fontsize=13, weight="bold", y=1.1)
fig.tight_layout()
safe_savefig(fig, "7_survival_flow.png", dpi=300, bbox_inches="tight")
plt.close(fig)
print("Saved: 7_survival_flow.png")
print("  Caption: Segment counts before and after BH correction for each")
print("  dataset, showing how many originally significant segments survived,")
print("  were lost, versus segments that were never significant to begin with.")
print("  Note: the diabetes panel shows zero significant segments both before")
print("  and after correction -- this is a null-result baseline, not a case")
print("  of correction removing findings, since nothing was flagged to begin")
print("  with. Worth stating this explicitly in the figure caption in the paper.")

print(f"\nAll figures saved to {OUT_DIR}/")