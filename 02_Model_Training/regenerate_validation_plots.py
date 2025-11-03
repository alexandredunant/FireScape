#!/usr/bin/env python
"""
Regenerate Validation Plots for Trained Models

Loads trained model results and regenerates validation plots,
saving them to output/figures/ for publication.

This script creates:
1. Temporal validation plots (monthly/seasonal bar and scatter plots)
2. Model performance plots (ROC, PR, calibration, lift curves)

For both:
- Baseline model (T+P)
- Lightning model (T+P+L)
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import scienceplots
from pathlib import Path
from scipy.stats import spearmanr
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve,
    average_precision_score
)

# Use publication-quality settings
plt.style.use(['science', 'no-latex'])
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

print("=" * 80)
print("REGENERATING VALIDATION PLOTS FOR PUBLICATION")
print("=" * 80)
print()

# ===================================================================
# CONFIGURATION
# ===================================================================

BASE_DIR = Path("/mnt/CEPH_PROJECTS/Firescape")
OUTPUT_FIGURES = BASE_DIR / "output/figures"
OUTPUT_FIGURES.mkdir(exist_ok=True, parents=True)

MODELS = {
    "baseline": {
        "name": "Baseline (T+P)",
        "path": BASE_DIR / "output/02_Model_RelativeProbability",
        "color": "steelblue",
    },
    "lightning": {
        "name": "Lightning (T+P+L)",
        "path": BASE_DIR / "output/02_Model_RelativeProbability_Lightning",
        "color": "orange",
    }
}

# ===================================================================
# FUNCTIONS
# ===================================================================

def create_temporal_validation_plot(results, model_name, color, output_path):
    """Create temporal validation plots (monthly and seasonal)."""

    temporal = results["temporal_validation"]
    monthly_stats = temporal["monthly"]
    seasonal_stats = temporal["seasonal"]

    monthly_corr = temporal["monthly_corr"]
    monthly_r2 = temporal["monthly_r2"]
    monthly_mae = temporal["monthly_mae"]

    seasonal_corr = temporal["seasonal_corr"]
    seasonal_r2 = temporal["seasonal_r2"]
    seasonal_mae = temporal["seasonal_mae"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # Monthly bar plot
    ax = axes[0, 0]
    x = np.arange(1, 13)
    width = 0.35
    ax.bar(x - width/2, monthly_stats['actual_fires'], width,
           label='Actual', alpha=0.7, color='darkred')
    ax.bar(x + width/2, monthly_stats['predicted_fires'], width,
           label='Predicted (scaled)', alpha=0.7, color=color)
    ax.set_xlabel('Month', fontsize=10)
    ax.set_ylabel('Fire Count', fontsize=10)
    ax.set_title(f'Monthly (R$^2$={monthly_r2:.3f})', fontsize=11, pad=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(x)

    # Monthly scatter plot
    ax = axes[0, 1]
    ax.scatter(monthly_stats['actual_fires'], monthly_stats['predicted_fires'],
              s=100, alpha=0.7, color=color, edgecolors='black', linewidth=1.5)
    max_val = max(monthly_stats['actual_fires'].max(),
                  monthly_stats['predicted_fires'].max())
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=2, alpha=0.4,
           label='Perfect fit')

    # Add month labels
    for _, row in monthly_stats.iterrows():
        ax.annotate(f"{int(row['month'])}",
                   (row['actual_fires'], row['predicted_fires']),
                   xytext=(5, 5), textcoords='offset points', fontsize=8)

    ax.set_xlabel('Actual Fires', fontsize=10)
    ax.set_ylabel('Predicted Fires', fontsize=10)
    ax.set_title(f'Monthly Correlation (R$^2$={monthly_r2:.3f})', fontsize=11, pad=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Seasonal bar plot
    ax = axes[1, 0]
    x_pos = np.arange(len(seasonal_stats))
    ax.bar(x_pos - width/2, seasonal_stats['actual_fires'], width,
           label='Actual', alpha=0.7, color='darkred')
    ax.bar(x_pos + width/2, seasonal_stats['predicted_fires'], width,
           label='Predicted', alpha=0.7, color=color)
    ax.set_xlabel('Season', fontsize=10)
    ax.set_ylabel('Fire Count', fontsize=10)
    ax.set_title(f'Seasonal (R$^2$={seasonal_r2:.3f})', fontsize=11, pad=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(seasonal_stats['season'], fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Seasonal scatter plot
    ax = axes[1, 1]
    ax.scatter(seasonal_stats['actual_fires'], seasonal_stats['predicted_fires'],
              s=150, alpha=0.7, color=color, edgecolors='black', linewidth=1.5)
    max_val = max(seasonal_stats['actual_fires'].max(),
                  seasonal_stats['predicted_fires'].max())
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=2, alpha=0.4,
           label='Perfect fit')

    # Add season labels
    for _, row in seasonal_stats.iterrows():
        ax.annotate(row['season'],
                   (row['actual_fires'], row['predicted_fires']),
                   xytext=(5, 5), textcoords='offset points', fontsize=9)

    ax.set_xlabel('Actual Fires', fontsize=10)
    ax.set_ylabel('Predicted Fires', fontsize=10)
    ax.set_title(f'Seasonal Correlation (R$^2$={seasonal_r2:.3f})', fontsize=11, pad=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(pad=2.0)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path.name}")
    print(f"    Monthly R^2: {monthly_r2:.3f}, Seasonal R^2: {seasonal_r2:.3f}")


def create_performance_validation_plot(results, model_name, color, output_path):
    """Create model performance validation plots."""

    test_preds = results["test_predictions"]
    y_test = np.array(test_preds["y_test"])
    mean_prob = np.array(test_preds["mean_prob"])

    validation = results.get("validation_metrics", results.get("validation", {}))

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    n_positive = int(np.sum(y_test))
    n_negative = len(y_test) - n_positive

    # PLOT 1: ROC CURVE
    ax = axes[0, 0]
    fpr, tpr, _ = roc_curve(y_test, mean_prob)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=3, label=f'Model (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=10)
    ax.set_ylabel('True Positive Rate', fontsize=10)
    ax.set_title(f'ROC Curve (AUC={roc_auc:.3f})', fontsize=11, pad=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # PLOT 2: PRECISION-RECALL CURVE
    ax = axes[0, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, mean_prob)
    pr_auc = average_precision_score(y_test, mean_prob)
    ax.plot(recall, precision, color=color, lw=3, label=f'Model (AP = {pr_auc:.3f})')

    no_skill = n_positive / len(y_test)
    ax.plot([0, 1], [no_skill, no_skill], color='navy', lw=2, linestyle='--',
           label=f'Baseline (Prevalence={no_skill:.3f})')

    # Mark optimal F1 threshold
    f1_scores = np.nan_to_num(2 * recall[:-1] * precision[:-1] /
                               (recall[:-1] + precision[:-1]))
    optimal_idx = np.argmax(f1_scores)
    max_f1 = f1_scores[optimal_idx]
    optimal_threshold = thresholds[optimal_idx]
    ax.scatter(recall[optimal_idx], precision[optimal_idx], marker='*',
              color='red', s=200, zorder=5, edgecolors='black', linewidth=1.5,
              label=f'Max F1 ({max_f1:.3f}) @ {optimal_threshold:.4f}')

    ax.set_xlabel('Recall', fontsize=10)
    ax.set_ylabel('Precision', fontsize=10)
    ax.set_title(f'Precision-Recall (AP={pr_auc:.3f})', fontsize=11, pad=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # PLOT 3: CALIBRATION PLOT
    ax = axes[1, 0]
    n_bins = 10
    bin_edges = np.linspace(0, mean_prob.max(), n_bins + 1)
    observed_freq = []
    predicted_freq = []

    for i in range(n_bins):
        mask = (mean_prob >= bin_edges[i]) & (mean_prob < bin_edges[i+1])
        if i == n_bins - 1:
            mask = (mean_prob >= bin_edges[i]) & (mean_prob <= bin_edges[i+1])

        if np.sum(mask) > 0:
            observed_freq.append(np.mean(y_test[mask]))
            predicted_freq.append(np.mean(mean_prob[mask]))
        else:
            observed_freq.append(np.nan)
            predicted_freq.append((bin_edges[i] + bin_edges[i+1]) / 2)

    valid_mask = ~np.isnan(observed_freq)
    ax.plot(np.array(predicted_freq)[valid_mask],
           np.array(observed_freq)[valid_mask],
           marker='o', color=color, lw=3, markersize=10,
           markeredgecolor='black', markeredgewidth=1.5,
           label='Model Calibration')
    ax.plot([0, mean_prob.max()], [0, mean_prob.max()],
           color='navy', lw=2, linestyle='--', label='Perfect Calibration')
    ax.set_xlabel('Predicted Probability', fontsize=10)
    ax.set_ylabel('Observed Frequency', fontsize=10)
    ax.set_title('Calibration Plot', fontsize=11, pad=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # PLOT 4: LIFT CURVE
    ax = axes[1, 1]
    sorted_indices = np.argsort(mean_prob)[::-1]
    sorted_actuals = y_test[sorted_indices]

    n_samples = len(sorted_actuals)
    percentiles = np.arange(1, 101)
    cumulative_positives = []
    baseline_positives = []

    for pct in percentiles:
        n = int(n_samples * pct / 100)
        cumulative_positives.append(sorted_actuals[:n].sum())
        baseline_positives.append(n_positive * pct / 100)

    lift = np.array(cumulative_positives) / np.array(baseline_positives)

    ax.plot(percentiles, lift, color=color, lw=3, label='Model Lift')
    ax.axhline(y=1, color='navy', lw=2, linestyle='--', label='Baseline (Random)')
    ax.set_xlabel('Percentage of Population Targeted (%)', fontsize=10)
    ax.set_ylabel('Lift', fontsize=10)
    ax.set_title('Lift Curve', fontsize=11, pad=10)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout(pad=2.0)
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    print(f"  ✓ Saved: {output_path.name}")
    print(f"    ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}, Max F1: {max_f1:.4f}")


# ===================================================================
# MAIN EXECUTION
# ===================================================================

for model_key, model_info in MODELS.items():
    print(f"\n{'='*80}")
    print(f"Processing: {model_info['name']}")
    print(f"{'='*80}")

    # Load model results
    model_path = model_info["path"]
    results_file = model_path / "model_results.joblib"

    if not results_file.exists():
        print(f"  ✗ Model results not found: {results_file}")
        continue

    print(f"  Loading: {results_file}")
    results = joblib.load(results_file)

    # Create temporal validation plot
    print(f"\n  Creating temporal validation plot...")
    temporal_output = OUTPUT_FIGURES / f"validation_temporal_{model_key}.png"
    create_temporal_validation_plot(
        results, model_info["name"], model_info["color"], temporal_output
    )

    # Create performance validation plot
    print(f"\n  Creating performance validation plot...")
    performance_output = OUTPUT_FIGURES / f"validation_performance_{model_key}.png"
    create_performance_validation_plot(
        results, model_info["name"], model_info["color"], performance_output
    )

print(f"\n{'='*80}")
print("VALIDATION PLOTS REGENERATION COMPLETE")
print(f"{'='*80}")
print(f"\nAll plots saved to: {OUTPUT_FIGURES}")
print("\nGenerated files:")
print("  • validation_temporal_baseline.png")
print("  • validation_performance_baseline.png")
print("  • validation_temporal_lightning.png")
print("  • validation_performance_lightning.png")
print()
