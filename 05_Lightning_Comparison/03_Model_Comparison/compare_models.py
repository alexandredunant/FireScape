#!/usr/bin/env python
"""
Lightning vs No-Lightning Model Comparison

Compares two Bayesian wildfire models to assess whether including lightning data
improves predictive performance and temporal fit.

Models compared:
1. Baseline: T (temperature) + P (precipitation) only
2. Lightning: T + P + L (lightning flash density)

Metrics assessed:
- Temporal validation (monthly/seasonal correlation)
- ROC-AUC and Precision-Recall
- Feature importance (attention weights)
- Prediction calibration
- Out-of-sample performance

Purpose:
- Determine if lightning significantly improves model performance
- Understand lightning's contribution to fire risk prediction
- Assess cost/benefit of including lightning data in operational model
"""

import pandas as pd
import numpy as np
import joblib
import arviz as az
import matplotlib.pyplot as plt
import scienceplots
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
)
import warnings

warnings.filterwarnings("ignore")

# Use publication-quality settings
plt.style.use(['science', 'no-latex'])
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

print("=" * 80)
print("LIGHTNING vs NO-LIGHTNING MODEL COMPARISON")
print("=" * 80)
print()

# ===================================================================
# CONFIGURATION
# ===================================================================

BASE_DIR = Path("/mnt/CEPH_PROJECTS/Firescape")

# Model directories
BASELINE_MODEL_DIR = BASE_DIR / "output/02_Model_RelativeProbability"
LIGHTNING_MODEL_DIR = BASE_DIR / "output/02_Model_RelativeProbability_Lightning"

# Output
OUTPUT_DIR = BASE_DIR / "output/figures/"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print(f"Configuration:")
print(f"  Baseline model (T+P): {BASELINE_MODEL_DIR}")
print(f"  Lightning model (T+P+L): {LIGHTNING_MODEL_DIR}")
print(f"  Output: {OUTPUT_DIR}")
print()

# ===================================================================
# LOAD MODEL RESULTS
# ===================================================================

print("Loading model results...")

# Load baseline model
try:
    baseline_results = joblib.load(BASELINE_MODEL_DIR / "model_results.joblib")
    baseline_trace = az.from_netcdf(BASELINE_MODEL_DIR / "trace_relative.nc")
    baseline_groups = joblib.load(BASELINE_MODEL_DIR / "temporal_groups.joblib")
    baseline_group_names = joblib.load(BASELINE_MODEL_DIR / "group_names.joblib")
    print("✓ Baseline model loaded (T+P)")
except Exception as e:
    print(f"✗ Error loading baseline model: {e}")
    baseline_results = None

# Load lightning model
try:
    lightning_results = joblib.load(LIGHTNING_MODEL_DIR / "model_results.joblib")
    lightning_trace = az.from_netcdf(LIGHTNING_MODEL_DIR / "trace_relative.nc")
    lightning_groups = joblib.load(LIGHTNING_MODEL_DIR / "temporal_groups.joblib")
    lightning_group_names = joblib.load(LIGHTNING_MODEL_DIR / "group_names.joblib")
    print("✓ Lightning model loaded (T+P+L)")
except Exception as e:
    print(f"✗ Error loading lightning model: {e}")
    lightning_results = None

if baseline_results is None or lightning_results is None:
    print("\n⚠️  Cannot compare models - missing model files")
    print("Please ensure both models have been trained successfully.")
    exit(1)

print()

# ===================================================================
# FILTER TO OVERLAPPING PERIOD
# ===================================================================

print("=" * 80)
print("FILTERING TO OVERLAPPING PERIOD")
print("=" * 80)
print()

# Get dates from both test sets
baseline_dates = pd.to_datetime(baseline_results["test_predictions"]["dates_test"])
lightning_dates = pd.to_datetime(lightning_results["test_predictions"]["dates_test"])

print(f"Original test sets:")
print(f"  Baseline: {baseline_dates.min().date()} to {baseline_dates.max().date()} ({len(baseline_dates)} samples)")
print(f"  Lightning: {lightning_dates.min().date()} to {lightning_dates.max().date()} ({len(lightning_dates)} samples)")

# Determine overlapping period
overlap_start = max(baseline_dates.min(), lightning_dates.min())
overlap_end = min(baseline_dates.max(), lightning_dates.max())

print(f"\nOverlapping period: {overlap_start.date()} to {overlap_end.date()}")

# Filter baseline test set to overlapping period
baseline_overlap_mask = (baseline_dates >= overlap_start) & (baseline_dates <= overlap_end)
baseline_overlap_indices = np.where(baseline_overlap_mask)[0]

baseline_overlap_dates = baseline_dates[baseline_overlap_mask]
baseline_overlap_y = np.array(baseline_results["test_predictions"]["y_test"])[baseline_overlap_indices]
baseline_overlap_pred = np.array(baseline_results["test_predictions"]["mean_prob"])[baseline_overlap_indices]

print(f"  Baseline in overlap: {len(baseline_overlap_dates)} samples, {np.sum(baseline_overlap_y)} fires")

# Filter lightning test set to overlapping period
lightning_overlap_mask = (lightning_dates >= overlap_start) & (lightning_dates <= overlap_end)
lightning_overlap_indices = np.where(lightning_overlap_mask)[0]

lightning_overlap_dates = lightning_dates[lightning_overlap_mask]
lightning_overlap_y = np.array(lightning_results["test_predictions"]["y_test"])[lightning_overlap_indices]
lightning_overlap_pred = np.array(lightning_results["test_predictions"]["mean_prob"])[lightning_overlap_indices]

print(f"  Lightning in overlap: {len(lightning_overlap_dates)} samples, {np.sum(lightning_overlap_y)} fires")

# Recompute validation metrics for overlapping period
print("\nRecomputing metrics for overlapping period...")

# ROC-AUC and PR-AUC
baseline_overlap_roc_auc = roc_auc_score(baseline_overlap_y, baseline_overlap_pred)
baseline_overlap_pr_auc = average_precision_score(baseline_overlap_y, baseline_overlap_pred)

lightning_overlap_roc_auc = roc_auc_score(lightning_overlap_y, lightning_overlap_pred)
lightning_overlap_pr_auc = average_precision_score(lightning_overlap_y, lightning_overlap_pred)

# Create overlap validation metrics
baseline_overlap_validation = {
    "roc_auc": baseline_overlap_roc_auc,
    "pr_auc": baseline_overlap_pr_auc,
}

lightning_overlap_validation = {
    "roc_auc": lightning_overlap_roc_auc,
    "pr_auc": lightning_overlap_pr_auc,
}

# Recompute temporal validation for overlapping period
def compute_temporal_validation_overlap(dates, y_true, y_pred, model_name):
    """Compute temporal validation metrics for overlapping period."""
    df = pd.DataFrame({
        "date": dates,
        "actual": y_true,
        "predicted": y_pred,
    })
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["season"] = df["date"].dt.month.map({
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Fall", 10: "Fall", 11: "Fall",
    })

    # Monthly aggregation
    monthly = df.groupby("month").agg({
        "actual": "sum",
        "predicted": "sum",
    }).reset_index()
    monthly.columns = ["month", "actual_fires", "predicted_fires"]

    # Seasonal aggregation
    season_order = ["Winter", "Spring", "Summer", "Fall"]
    seasonal = df.groupby("season").agg({
        "actual": "sum",
        "predicted": "sum",
    }).reindex(season_order).reset_index()
    seasonal.columns = ["season", "actual_fires", "predicted_fires"]

    # Compute correlations
    monthly_corr = spearmanr(monthly["actual_fires"], monthly["predicted_fires"])[0]
    monthly_r2 = monthly_corr ** 2
    seasonal_corr = spearmanr(seasonal["actual_fires"], seasonal["predicted_fires"])[0]
    seasonal_r2 = seasonal_corr ** 2

    return {
        "monthly": monthly,
        "seasonal": seasonal,
        "monthly_corr": monthly_corr,
        "monthly_r2": monthly_r2,
        "seasonal_corr": seasonal_corr,
        "seasonal_r2": seasonal_r2,
    }

baseline_overlap_temporal = compute_temporal_validation_overlap(
    baseline_overlap_dates, baseline_overlap_y, baseline_overlap_pred, "Baseline"
)

lightning_overlap_temporal = compute_temporal_validation_overlap(
    lightning_overlap_dates, lightning_overlap_y, lightning_overlap_pred, "Lightning"
)

print("✓ Metrics recomputed for overlapping period")
print()

# ===================================================================
# COMPARISON 1: TEMPORAL VALIDATION (OVERLAPPING PERIOD)
# ===================================================================

print("=" * 80)
print("COMPARISON 1: TEMPORAL FIT (OVERLAPPING PERIOD)")
print("=" * 80)
print()

# Use overlapping period metrics
baseline_temporal = baseline_overlap_temporal
lightning_temporal = lightning_overlap_temporal

print("MONTHLY CORRELATION:")
print(
    f"  Baseline (T+P):   R={baseline_temporal['monthly_corr']:.3f}, R²={baseline_temporal['monthly_r2']:.3f}"
)
print(
    f"  Lightning (T+P+L): R={lightning_temporal['monthly_corr']:.3f}, R²={lightning_temporal['monthly_r2']:.3f}"
)
print(f"  Note: Both models evaluated on same overlapping period (2012-2025)")

if lightning_temporal["monthly_r2"] > baseline_temporal["monthly_r2"]:
    improvement = lightning_temporal["monthly_r2"] - baseline_temporal["monthly_r2"]
    pct_improvement = (improvement / baseline_temporal["monthly_r2"]) * 100
    print(f"  → Improvement: +{improvement:.3f} ({pct_improvement:+.1f}%) ✓")
else:
    decline = baseline_temporal["monthly_r2"] - lightning_temporal["monthly_r2"]
    pct_decline = (decline / baseline_temporal["monthly_r2"]) * 100
    print(f"  → Decline: -{decline:.3f} ({pct_decline:.1f}%) ✗")

print("\nSEASONAL CORRELATION:")
print(
    f"  Baseline (T+P):    R={baseline_temporal['seasonal_corr']:.3f}, R²={baseline_temporal['seasonal_r2']:.3f}"
)
print(
    f"  Lightning (T+P+L): R={lightning_temporal['seasonal_corr']:.3f}, R²={lightning_temporal['seasonal_r2']:.3f}"
)

if lightning_temporal["seasonal_r2"] > baseline_temporal["seasonal_r2"]:
    improvement = lightning_temporal["seasonal_r2"] - baseline_temporal["seasonal_r2"]
    pct_improvement = (improvement / baseline_temporal["seasonal_r2"]) * 100
    print(f"  → Improvement: +{improvement:.3f} ({pct_improvement:+.1f}%) ✓")
else:
    decline = baseline_temporal["seasonal_r2"] - lightning_temporal["seasonal_r2"]
    pct_decline = (decline / baseline_temporal["seasonal_r2"]) * 100
    print(f"  → Decline: -{decline:.3f} ({pct_decline:.1f}%) ✗")

print()

# ===================================================================
# COMPARISON 2: ROC-AUC AND PR-AUC
# ===================================================================

print("=" * 80)
print("COMPARISON 2: DISCRIMINATION PERFORMANCE")
print("=" * 80)
print()

# Show both original and overlapping period metrics
baseline_validation_original = baseline_results.get("validation_metrics", baseline_results.get("validation", {}))
lightning_validation_original = lightning_results.get("validation_metrics", lightning_results.get("validation", {}))

print("ORIGINAL TEST SETS (different periods):")
print(f"  Baseline (1999-2025):  ROC-AUC = {baseline_validation_original['roc_auc']:.3f}, PR-AUC = {baseline_validation_original['pr_auc']:.3f}")
print(f"  Lightning (2012-2025): ROC-AUC = {lightning_validation_original['roc_auc']:.3f}, PR-AUC = {lightning_validation_original['pr_auc']:.3f}")

print()
print("OVERLAPPING PERIOD (2012-2025, fair comparison):")

# Use overlapping period validation metrics
baseline_validation = baseline_overlap_validation
lightning_validation = lightning_overlap_validation

print("ROC-AUC:")
print(f"  Baseline (T+P):    {baseline_validation['roc_auc']:.3f}")
print(f"  Lightning (T+P+L): {lightning_validation['roc_auc']:.3f}")

if lightning_validation["roc_auc"] > baseline_validation["roc_auc"]:
    improvement = lightning_validation["roc_auc"] - baseline_validation["roc_auc"]
    print(f"  → Improvement: +{improvement:.3f} ✓")
else:
    decline = baseline_validation["roc_auc"] - lightning_validation["roc_auc"]
    print(f"  → Decline: -{decline:.3f} ✗")

print("\nPRECISION-RECALL AUC:")
print(f"  Baseline (T+P):    {baseline_validation['pr_auc']:.3f}")
print(f"  Lightning (T+P+L): {lightning_validation['pr_auc']:.3f}")

if lightning_validation["pr_auc"] > baseline_validation["pr_auc"]:
    improvement = lightning_validation["pr_auc"] - baseline_validation["pr_auc"]
    print(f"  → Improvement: +{improvement:.3f} ✓")
else:
    decline = baseline_validation["pr_auc"] - lightning_validation["pr_auc"]
    print(f"  → Decline: -{decline:.3f} ✗")

print()

# ===================================================================
# COMPARISON 3: ATTENTION WEIGHTS
# ===================================================================

print("=" * 80)
print("COMPARISON 3: FEATURE IMPORTANCE (ATTENTION WEIGHTS)")
print("=" * 80)
print()

# Extract attention weights
baseline_attention = (
    baseline_trace.posterior["attention_weights"].mean(dim=["chain", "draw"]).values
)
lightning_attention = (
    lightning_trace.posterior["attention_weights"].mean(dim=["chain", "draw"]).values
)

print("BASELINE MODEL (T+P) - Top 5 Feature Groups:")
baseline_attn_df = pd.DataFrame(
    {"group": baseline_group_names, "attention": baseline_attention}
).sort_values("attention", ascending=False)
print(baseline_attn_df.head().to_string(index=False))

print("\nLIGHTNING MODEL (T+P+L) - Top 5 Feature Groups:")
lightning_attn_df = pd.DataFrame(
    {"group": lightning_group_names, "attention": lightning_attention}
).sort_values("attention", ascending=False)
print(lightning_attn_df.head().to_string(index=False))

# Check if lightning groups are in top features
lightning_groups_list = [g for g in lightning_group_names if "light" in g.lower()]
if lightning_groups_list:
    print(f"\nLIGHTNING-SPECIFIC GROUPS:")
    for lg in lightning_groups_list:
        idx = lightning_group_names.index(lg)
        attn = lightning_attention[idx]
        rank = (lightning_attention > attn).sum() + 1
        print(f"  {lg}: {attn:.3f} (rank {rank}/{len(lightning_group_names)})")

print()

# ===================================================================
# VISUALIZATION: COMPARISON PLOTS
# ===================================================================

print("=" * 80)
print("GENERATING COMPARISON PLOTS")
print("=" * 80)
print()

# Plot 1: Temporal fit comparison
fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# Monthly comparison
ax1 = axes[0, 0]
baseline_monthly = baseline_temporal.get("monthly", pd.DataFrame())
lightning_monthly = lightning_temporal.get("monthly", pd.DataFrame())

# Each model has its own actual fires from different test sets
monthly_data = pd.DataFrame(
    {
        "month": range(1, 13),
        "baseline_pred": baseline_monthly["predicted_fires"] if isinstance(baseline_monthly, pd.DataFrame) else np.zeros(12),
        "baseline_actual": baseline_monthly["actual_fires"] if isinstance(baseline_monthly, pd.DataFrame) else np.zeros(12),
        "lightning_pred": lightning_monthly["predicted_fires"] if isinstance(lightning_monthly, pd.DataFrame) else np.zeros(12),
        "lightning_actual": lightning_monthly["actual_fires"] if isinstance(lightning_monthly, pd.DataFrame) else np.zeros(12),
    }
)

x = monthly_data["month"]
width = 0.25
# Use lightning model's actual fires as reference (2012+ period)
ax1.bar(
    x - width, monthly_data["lightning_actual"], width, label="Actual (2012+)", alpha=0.7, color="black"
)
ax1.bar(
    x,
    monthly_data["baseline_pred"],
    width,
    label="Baseline (T+P)",
    alpha=0.7,
    color="steelblue",
)
ax1.bar(
    x + width,
    monthly_data["lightning_pred"],
    width,
    label="Lightning (T+P+L)",
    alpha=0.7,
    color="orange",
)
ax1.set_xlabel("Month", fontsize=12)
ax1.set_ylabel("Fire Count", fontsize=12)
ax1.text(0.02, 0.98, '(a)', transform=ax1.transAxes,
        fontsize=12, fontweight='bold', va='top', ha='left')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Seasonal comparison
ax2 = axes[0, 1]
baseline_seasonal = baseline_temporal.get("seasonal", pd.DataFrame())
lightning_seasonal = lightning_temporal.get("seasonal", pd.DataFrame())

seasonal_data = pd.DataFrame(
    {
        "season": ["Winter", "Spring", "Summer", "Fall"],
        "baseline_pred": baseline_seasonal.get("predicted_fires", np.zeros(4))
        if isinstance(baseline_seasonal, pd.DataFrame)
        else np.zeros(4),
        "lightning_pred": lightning_seasonal.get("predicted_fires", np.zeros(4))
        if isinstance(lightning_seasonal, pd.DataFrame)
        else np.zeros(4),
        # Use lightning model's actual fires as reference (2012+ period)
        "actual": lightning_results["temporal_validation"]["seasonal"]["actual_fires"]
    }
)

x = np.arange(len(seasonal_data))
ax2.bar(
    x - width, seasonal_data["actual"], width, label="Actual (2012+)", alpha=0.7, color="black"
)
ax2.bar(
    x,
    seasonal_data["baseline_pred"],
    width,
    label="Baseline (T+P)",
    alpha=0.7,
    color="steelblue",
)
ax2.bar(
    x + width,
    seasonal_data["lightning_pred"],
    width,
    label="Lightning (T+P+L)",
    alpha=0.7,
    color="orange",
)
ax2.set_xticks(x)
ax2.set_xticklabels(seasonal_data["season"])
ax2.set_ylabel("Fire Count", fontsize=12)
ax2.text(0.02, 0.98, '(b)', transform=ax2.transAxes,
        fontsize=12, fontweight='bold', va='top', ha='left')
ax2.legend()
ax2.grid(True, alpha=0.3)

# ROC comparison - compute from overlapping test predictions
ax3 = axes[1, 0]
try:
    # Use overlapping period data
    baseline_fpr, baseline_tpr, _ = roc_curve(
        baseline_overlap_y, baseline_overlap_pred
    )
    lightning_fpr, lightning_tpr, _ = roc_curve(
        lightning_overlap_y, lightning_overlap_pred
    )

    ax3.plot(
        baseline_fpr,
        baseline_tpr,
        label=f"Baseline (AUC={baseline_validation['roc_auc']:.3f})",
        linewidth=2,
        color="steelblue",
    )
    ax3.plot(
        lightning_fpr,
        lightning_tpr,
        label=f"Lightning (AUC={lightning_validation['roc_auc']:.3f})",
        linewidth=2,
        color="orange",
    )
    ax3.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax3.set_xlabel("False Positive Rate", fontsize=12)
    ax3.set_ylabel("True Positive Rate", fontsize=12)
    ax3.text(0.02, 0.98, '(c)', transform=ax3.transAxes,
            fontsize=12, fontweight='bold', va='top', ha='left')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
except Exception as e:
    ax3.text(0.5, 0.5, f"ROC curves not available\n{str(e)}", ha="center", va="center")
    ax3.text(0.02, 0.98, '(c)', transform=ax3.transAxes,
            fontsize=12, fontweight='bold', va='top', ha='left')

# Attention weights comparison
ax4 = axes[1, 1]
top_n = 10
baseline_top = baseline_attn_df.head(top_n)
lightning_top = lightning_attn_df.head(top_n)

# Create comparison of common groups
all_groups = list(set(baseline_top["group"].tolist() + lightning_top["group"].tolist()))
comparison = (
    pd.DataFrame(
        {
            "group": all_groups,
            "baseline": [
                baseline_attn_df[baseline_attn_df["group"] == g]["attention"].values[0]
                if g in baseline_attn_df["group"].values
                else 0
                for g in all_groups
            ],
            "lightning": [
                lightning_attn_df[lightning_attn_df["group"] == g]["attention"].values[
                    0
                ]
                if g in lightning_attn_df["group"].values
                else 0
                for g in all_groups
            ],
        }
    )
    .sort_values("lightning", ascending=True)
    .tail(top_n)
)

y_pos = np.arange(len(comparison))
ax4.barh(
    y_pos - 0.2,
    comparison["baseline"],
    0.4,
    label="Baseline (T+P)",
    alpha=0.7,
    color="steelblue",
)
ax4.barh(
    y_pos + 0.2,
    comparison["lightning"],
    0.4,
    label="Lightning (T+P+L)",
    alpha=0.7,
    color="orange",
)
ax4.set_yticks(y_pos)
ax4.set_yticklabels(comparison["group"], fontsize=9)
ax4.set_xlabel("Attention Weight", fontsize=12)
ax4.text(0.02, 0.98, '(d)', transform=ax4.transAxes,
        fontsize=12, fontweight='bold', va='top', ha='left')
ax4.legend()
ax4.grid(True, alpha=0.3, axis="x")

plt.tight_layout()
output_plot = OUTPUT_DIR / "model_comparison.png"
plt.savefig(output_plot, bbox_inches="tight")
print(f"✓ Comparison plot saved: {output_plot}")
plt.close()

# ===================================================================
# SUMMARY REPORT
# ===================================================================

print("\n" + "=" * 80)
print("SUMMARY REPORT")
print("=" * 80)
print()

summary = {
    "metric": [],
    "baseline_TP": [],
    "lightning_TPL": [],
    "improvement": [],
    "pct_change": [],
}

# Add metrics
metrics = [
    ("Monthly R²", baseline_temporal["monthly_r2"], lightning_temporal["monthly_r2"]),
    (
        "Seasonal R²",
        baseline_temporal["seasonal_r2"],
        lightning_temporal["seasonal_r2"],
    ),
    ("ROC-AUC", baseline_validation["roc_auc"], lightning_validation["roc_auc"]),
    ("PR-AUC", baseline_validation["pr_auc"], lightning_validation["pr_auc"]),
]

for metric_name, baseline_val, lightning_val in metrics:
    improvement = lightning_val - baseline_val
    pct_change = (improvement / baseline_val) * 100 if baseline_val > 0 else 0

    summary["metric"].append(metric_name)
    summary["baseline_TP"].append(f"{baseline_val:.3f}")
    summary["lightning_TPL"].append(f"{lightning_val:.3f}")
    summary["improvement"].append(f"{improvement:+.3f}")
    summary["pct_change"].append(f"{pct_change:+.1f}%")

summary_df = pd.DataFrame(summary)
print(summary_df.to_string(index=False))

# Save summary
summary_csv = OUTPUT_DIR / "comparison_summary.csv"
summary_df.to_csv(summary_csv, index=False)
print(f"\n✓ Summary saved: {summary_csv}")

# ===================================================================
# RECOMMENDATION
# ===================================================================

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print()

# Calculate overall improvement score
improvements = []
for _, baseline_val, lightning_val in metrics:
    if baseline_val > 0:
        pct_improvement = ((lightning_val - baseline_val) / baseline_val) * 100
        improvements.append(pct_improvement)

avg_improvement = np.mean(improvements)

if avg_improvement > 5:
    print("✓ RECOMMENDATION: USE LIGHTNING MODEL")
    print(f"  Average improvement: {avg_improvement:+.1f}%")
    print(f"  Lightning data provides significant predictive value.")
elif avg_improvement > 1:
    print("? RECOMMENDATION: MARGINAL IMPROVEMENT")
    print(f"  Average improvement: {avg_improvement:+.1f}%")
    print(f"  Small benefit - consider operational costs vs. benefits.")
else:
    print("✗ RECOMMENDATION: LIGHTNING NOT BENEFICIAL")
    print(f"  Average change: {avg_improvement:+.1f}%")
    print(f"  Lightning data does not improve model performance.")

print("\n" + "=" * 80)
print("COMPARISON COMPLETE")
print("=" * 80)
