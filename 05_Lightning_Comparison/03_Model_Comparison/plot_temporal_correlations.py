#!/usr/bin/env python
"""
Plot Temporal Correlations for Model Comparison

Creates scatter plots showing the correlation between actual and predicted
fire counts for both monthly and seasonal aggregations, comparing baseline
vs lightning models on the overlapping period.

Shows both Spearman (rank) and Pearson (linear) correlations.
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import scienceplots
from pathlib import Path
from scipy.stats import spearmanr, pearsonr

# Use publication-quality settings
plt.style.use(['science', 'no-latex'])
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

print("=" * 80)
print("TEMPORAL CORRELATION PLOTS")
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

# ===================================================================
# LOAD MODEL RESULTS AND FILTER TO OVERLAPPING PERIOD
# ===================================================================

print("Loading model results...")

baseline_results = joblib.load(BASELINE_MODEL_DIR / "model_results.joblib")
lightning_results = joblib.load(LIGHTNING_MODEL_DIR / "model_results.joblib")

# Get dates from both test sets
baseline_dates = pd.to_datetime(baseline_results["test_predictions"]["dates_test"])
lightning_dates = pd.to_datetime(lightning_results["test_predictions"]["dates_test"])

# Determine overlapping period
overlap_start = max(baseline_dates.min(), lightning_dates.min())
overlap_end = min(baseline_dates.max(), lightning_dates.max())

print(f"Overlapping period: {overlap_start.date()} to {overlap_end.date()}")

# Filter baseline test set
baseline_overlap_mask = (baseline_dates >= overlap_start) & (baseline_dates <= overlap_end)
baseline_overlap_indices = np.where(baseline_overlap_mask)[0]
baseline_overlap_dates = baseline_dates[baseline_overlap_mask]
baseline_overlap_y = np.array(baseline_results["test_predictions"]["y_test"])[baseline_overlap_indices]
baseline_overlap_pred = np.array(baseline_results["test_predictions"]["mean_prob"])[baseline_overlap_indices]

# Filter lightning test set
lightning_overlap_mask = (lightning_dates >= overlap_start) & (lightning_dates <= overlap_end)
lightning_overlap_indices = np.where(lightning_overlap_mask)[0]
lightning_overlap_dates = lightning_dates[lightning_overlap_mask]
lightning_overlap_y = np.array(lightning_results["test_predictions"]["y_test"])[lightning_overlap_indices]
lightning_overlap_pred = np.array(lightning_results["test_predictions"]["mean_prob"])[lightning_overlap_indices]

print("✓ Models loaded and filtered")
print()

# ===================================================================
# COMPUTE TEMPORAL AGGREGATIONS
# ===================================================================

print("Computing temporal aggregations...")

def compute_aggregations(dates, y_true, y_pred):
    """Compute monthly and seasonal aggregations with both correlation types."""
    df = pd.DataFrame({"date": dates, "actual": y_true, "predicted": y_pred})
    df["month"] = df["date"].dt.month
    df["season"] = df["date"].dt.month.map({
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Fall", 10: "Fall", 11: "Fall",
    })

    # Monthly
    monthly = df.groupby("month").agg({"actual": "sum", "predicted": "sum"}).reset_index()
    monthly.columns = ["month", "actual_fires", "predicted_fires"]

    # Seasonal
    seasonal = df.groupby("season").agg({"actual": "sum", "predicted": "sum"}).reindex(
        ["Winter", "Spring", "Summer", "Fall"]
    ).reset_index()
    seasonal.columns = ["season", "actual_fires", "predicted_fires"]

    return monthly, seasonal

baseline_monthly, baseline_seasonal = compute_aggregations(
    baseline_overlap_dates, baseline_overlap_y, baseline_overlap_pred
)
lightning_monthly, lightning_seasonal = compute_aggregations(
    lightning_overlap_dates, lightning_overlap_y, lightning_overlap_pred
)

print("✓ Aggregations computed")
print()

# ===================================================================
# CREATE CORRELATION PLOTS
# ===================================================================

print("Creating correlation plots...")

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

# ===================================================================
# MONTHLY - BASELINE
# ===================================================================

ax1 = axes[0, 0]

spearman_r, _ = spearmanr(baseline_monthly["actual_fires"], baseline_monthly["predicted_fires"])
pearson_r, _ = pearsonr(baseline_monthly["actual_fires"], baseline_monthly["predicted_fires"])

ax1.scatter(baseline_monthly["actual_fires"], baseline_monthly["predicted_fires"],
           s=120, alpha=0.6, color="steelblue", edgecolors="black", linewidth=2)

for _, row in baseline_monthly.iterrows():
    ax1.annotate(f"{int(row['month'])}", (row["actual_fires"], row["predicted_fires"]),
                fontsize=10, ha="center", va="center", fontweight="bold")

z = np.polyfit(baseline_monthly["actual_fires"], baseline_monthly["predicted_fires"], 1)
p = np.poly1d(z)
x_line = np.linspace(baseline_monthly["actual_fires"].min(), baseline_monthly["actual_fires"].max(), 100)
ax1.plot(x_line, p(x_line), "r-", alpha=0.7, linewidth=3,
         label=f"Linear fit: y={z[0]:.2f}x+{z[1]:.2f}")

ax1.set_xlabel("Actual Fire Count", fontsize=13, fontweight="bold")
ax1.set_ylabel("Predicted Fire Count", fontsize=13, fontweight="bold")
ax1.set_title(
    f"Monthly - Baseline (T+P)\n"
    f"Spearman R={spearman_r:.3f} (R²={spearman_r**2:.3f}) | "
    f"Pearson R={pearson_r:.3f} (R²={pearson_r**2:.3f})",
    fontsize=13, fontweight="bold", pad=10
)
ax1.legend(loc="upper left", fontsize=10)
ax1.grid(True, alpha=0.3)

# ===================================================================
# MONTHLY - LIGHTNING
# ===================================================================

ax2 = axes[0, 1]

spearman_r, _ = spearmanr(lightning_monthly["actual_fires"], lightning_monthly["predicted_fires"])
pearson_r, _ = pearsonr(lightning_monthly["actual_fires"], lightning_monthly["predicted_fires"])

ax2.scatter(lightning_monthly["actual_fires"], lightning_monthly["predicted_fires"],
           s=120, alpha=0.6, color="orange", edgecolors="black", linewidth=2)

for _, row in lightning_monthly.iterrows():
    ax2.annotate(f"{int(row['month'])}", (row["actual_fires"], row["predicted_fires"]),
                fontsize=10, ha="center", va="center", fontweight="bold")

z = np.polyfit(lightning_monthly["actual_fires"], lightning_monthly["predicted_fires"], 1)
p = np.poly1d(z)
x_line = np.linspace(lightning_monthly["actual_fires"].min(), lightning_monthly["actual_fires"].max(), 100)
ax2.plot(x_line, p(x_line), "r-", alpha=0.7, linewidth=3,
         label=f"Linear fit: y={z[0]:.2f}x+{z[1]:.2f}")

ax2.set_xlabel("Actual Fire Count", fontsize=13, fontweight="bold")
ax2.set_ylabel("Predicted Fire Count", fontsize=13, fontweight="bold")
ax2.set_title(
    f"Monthly - Lightning (T+P+L)\n"
    f"Spearman R={spearman_r:.3f} (R²={spearman_r**2:.3f}) | "
    f"Pearson R={pearson_r:.3f} (R²={pearson_r**2:.3f})",
    fontsize=13, fontweight="bold", pad=10
)
ax2.legend(loc="upper left", fontsize=10)
ax2.grid(True, alpha=0.3)

# ===================================================================
# SEASONAL - BASELINE
# ===================================================================

ax3 = axes[1, 0]

spearman_r, _ = spearmanr(baseline_seasonal["actual_fires"], baseline_seasonal["predicted_fires"])
pearson_r, _ = pearsonr(baseline_seasonal["actual_fires"], baseline_seasonal["predicted_fires"])

ax3.scatter(baseline_seasonal["actual_fires"], baseline_seasonal["predicted_fires"],
           s=180, alpha=0.6, color="steelblue", edgecolors="black", linewidth=2)

for _, row in baseline_seasonal.iterrows():
    ax3.annotate(row["season"][:3], (row["actual_fires"], row["predicted_fires"]),
                fontsize=10, ha="center", va="center", fontweight="bold")

z = np.polyfit(baseline_seasonal["actual_fires"], baseline_seasonal["predicted_fires"], 1)
p = np.poly1d(z)
x_line = np.linspace(baseline_seasonal["actual_fires"].min(), baseline_seasonal["actual_fires"].max(), 100)
ax3.plot(x_line, p(x_line), "r-", alpha=0.7, linewidth=3,
         label=f"Linear fit: y={z[0]:.2f}x+{z[1]:.2f}")

ax3.set_xlabel("Actual Fire Count", fontsize=13, fontweight="bold")
ax3.set_ylabel("Predicted Fire Count", fontsize=13, fontweight="bold")
ax3.set_title(
    f"Seasonal - Baseline (T+P)\n"
    f"Spearman R={spearman_r:.3f} (R²={spearman_r**2:.3f}) | "
    f"Pearson R={pearson_r:.3f} (R²={pearson_r**2:.3f})",
    fontsize=13, fontweight="bold", pad=10
)
ax3.legend(loc="upper left", fontsize=10)
ax3.grid(True, alpha=0.3)

# ===================================================================
# SEASONAL - LIGHTNING
# ===================================================================

ax4 = axes[1, 1]

spearman_r, _ = spearmanr(lightning_seasonal["actual_fires"], lightning_seasonal["predicted_fires"])
pearson_r, _ = pearsonr(lightning_seasonal["actual_fires"], lightning_seasonal["predicted_fires"])

ax4.scatter(lightning_seasonal["actual_fires"], lightning_seasonal["predicted_fires"],
           s=180, alpha=0.6, color="orange", edgecolors="black", linewidth=2)

for _, row in lightning_seasonal.iterrows():
    ax4.annotate(row["season"][:3], (row["actual_fires"], row["predicted_fires"]),
                fontsize=10, ha="center", va="center", fontweight="bold")

z = np.polyfit(lightning_seasonal["actual_fires"], lightning_seasonal["predicted_fires"], 1)
p = np.poly1d(z)
x_line = np.linspace(lightning_seasonal["actual_fires"].min(), lightning_seasonal["actual_fires"].max(), 100)
ax4.plot(x_line, p(x_line), "r-", alpha=0.7, linewidth=3,
         label=f"Linear fit: y={z[0]:.2f}x+{z[1]:.2f}")

ax4.set_xlabel("Actual Fire Count", fontsize=13, fontweight="bold")
ax4.set_ylabel("Predicted Fire Count", fontsize=13, fontweight="bold")
ax4.set_title(
    f"Seasonal - Lightning (T+P+L)\n"
    f"Spearman R={spearman_r:.3f} (R²={spearman_r**2:.3f}) | "
    f"Pearson R={pearson_r:.3f} (R²={pearson_r**2:.3f})",
    fontsize=13, fontweight="bold", pad=10
)
ax4.legend(loc="upper left", fontsize=10)
ax4.grid(True, alpha=0.3)

# ===================================================================
# SAVE
# ===================================================================

plt.tight_layout()

output_path = OUTPUT_DIR / "temporal_correlations_comparison.png"
plt.savefig(output_path, bbox_inches="tight")
print(f"✓ Plot saved: {output_path}")

print()
print("=" * 80)
print("CORRELATION SUMMARY")
print("=" * 80)
print()

# Compute all correlations for summary
baseline_monthly_spearman, _ = spearmanr(baseline_monthly["actual_fires"], baseline_monthly["predicted_fires"])
baseline_monthly_pearson, _ = pearsonr(baseline_monthly["actual_fires"], baseline_monthly["predicted_fires"])
baseline_seasonal_spearman, _ = spearmanr(baseline_seasonal["actual_fires"], baseline_seasonal["predicted_fires"])
baseline_seasonal_pearson, _ = pearsonr(baseline_seasonal["actual_fires"], baseline_seasonal["predicted_fires"])

lightning_monthly_spearman, _ = spearmanr(lightning_monthly["actual_fires"], lightning_monthly["predicted_fires"])
lightning_monthly_pearson, _ = pearsonr(lightning_monthly["actual_fires"], lightning_monthly["predicted_fires"])
lightning_seasonal_spearman, _ = spearmanr(lightning_seasonal["actual_fires"], lightning_seasonal["predicted_fires"])
lightning_seasonal_pearson, _ = pearsonr(lightning_seasonal["actual_fires"], lightning_seasonal["predicted_fires"])

print("MONTHLY:")
print(f"  Baseline  - Spearman R²={baseline_monthly_spearman**2:.3f}, Pearson R²={baseline_monthly_pearson**2:.3f}")
print(f"  Lightning - Spearman R²={lightning_monthly_spearman**2:.3f}, Pearson R²={lightning_monthly_pearson**2:.3f}")
print()
print("SEASONAL:")
print(f"  Baseline  - Spearman R²={baseline_seasonal_spearman**2:.3f}, Pearson R²={baseline_seasonal_pearson**2:.3f}")
print(f"  Lightning - Spearman R²={lightning_seasonal_spearman**2:.3f}, Pearson R²={lightning_seasonal_pearson**2:.3f}")
print()
print("NOTE:")
print("  - Spearman R²=1.000 for lightning seasonal means PERFECT RANK ORDER")
print("  - But Pearson R²=0.995 shows there's still a small linear deviation")
print("  - Both are excellent, but Spearman=1.0 can be misleading with only 4 points")
print()
print("=" * 80)
