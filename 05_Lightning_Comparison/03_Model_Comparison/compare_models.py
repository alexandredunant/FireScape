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
import seaborn as sns
from pathlib import Path
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import warnings

warnings.filterwarnings("ignore")

print("="*80)
print("LIGHTNING vs NO-LIGHTNING MODEL COMPARISON")
print("="*80)
print()

# ===================================================================
# CONFIGURATION
# ===================================================================

BASE_DIR = Path("/mnt/CEPH_PROJECTS/Firescape")

# Model directories
BASELINE_MODEL_DIR = BASE_DIR / "Scripts/OUTPUT/02_Model_RelativeProbability"
LIGHTNING_MODEL_DIR = BASE_DIR / "Scripts/OUTPUT/02_Model_RelativeProbability_Lightning"

# Output
OUTPUT_DIR = BASE_DIR / "Scripts/05_Lightning_Comparison/03_Model_Comparison/Results"
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
# COMPARISON 1: TEMPORAL VALIDATION
# ===================================================================

print("="*80)
print("COMPARISON 1: TEMPORAL FIT")
print("="*80)
print()

# Extract temporal validation metrics
baseline_temporal = baseline_results['temporal_validation']
lightning_temporal = lightning_results['temporal_validation']

print("MONTHLY CORRELATION:")
print(f"  Baseline (T+P):   R={baseline_temporal['monthly_pearson_r']:.3f}, R²={baseline_temporal['monthly_r2']:.3f}")
print(f"  Lightning (T+P+L): R={lightning_temporal['monthly_pearson_r']:.3f}, R²={lightning_temporal['monthly_r2']:.3f}")

if lightning_temporal['monthly_r2'] > baseline_temporal['monthly_r2']:
    improvement = lightning_temporal['monthly_r2'] - baseline_temporal['monthly_r2']
    pct_improvement = (improvement / baseline_temporal['monthly_r2']) * 100
    print(f"  → Improvement: +{improvement:.3f} ({pct_improvement:+.1f}%) ✓")
else:
    decline = baseline_temporal['monthly_r2'] - lightning_temporal['monthly_r2']
    pct_decline = (decline / baseline_temporal['monthly_r2']) * 100
    print(f"  → Decline: -{decline:.3f} ({pct_decline:.1f}%) ✗")

print("\nSEASONAL CORRELATION:")
print(f"  Baseline (T+P):    R={baseline_temporal['seasonal_pearson_r']:.3f}, R²={baseline_temporal['seasonal_r2']:.3f}")
print(f"  Lightning (T+P+L): R={lightning_temporal['seasonal_pearson_r']:.3f}, R²={lightning_temporal['seasonal_r2']:.3f}")

if lightning_temporal['seasonal_r2'] > baseline_temporal['seasonal_r2']:
    improvement = lightning_temporal['seasonal_r2'] - baseline_temporal['seasonal_r2']
    pct_improvement = (improvement / baseline_temporal['seasonal_r2']) * 100
    print(f"  → Improvement: +{improvement:.3f} ({pct_improvement:+.1f}%) ✓")
else:
    decline = baseline_temporal['seasonal_r2'] - lightning_temporal['seasonal_r2']
    pct_decline = (decline / baseline_temporal['seasonal_r2']) * 100
    print(f"  → Decline: -{decline:.3f} ({pct_decline:.1f}%) ✗")

print()

# ===================================================================
# COMPARISON 2: ROC-AUC AND PR-AUC
# ===================================================================

print("="*80)
print("COMPARISON 2: DISCRIMINATION PERFORMANCE")
print("="*80)
print()

baseline_validation = baseline_results['validation']
lightning_validation = lightning_results['validation']

print("ROC-AUC:")
print(f"  Baseline (T+P):    {baseline_validation['roc_auc']:.3f}")
print(f"  Lightning (T+P+L): {lightning_validation['roc_auc']:.3f}")

if lightning_validation['roc_auc'] > baseline_validation['roc_auc']:
    improvement = lightning_validation['roc_auc'] - baseline_validation['roc_auc']
    print(f"  → Improvement: +{improvement:.3f} ✓")
else:
    decline = baseline_validation['roc_auc'] - lightning_validation['roc_auc']
    print(f"  → Decline: -{decline:.3f} ✗")

print("\nPRECISION-RECALL AUC:")
print(f"  Baseline (T+P):    {baseline_validation['pr_auc']:.3f}")
print(f"  Lightning (T+P+L): {lightning_validation['pr_auc']:.3f}")

if lightning_validation['pr_auc'] > baseline_validation['pr_auc']:
    improvement = lightning_validation['pr_auc'] - baseline_validation['pr_auc']
    print(f"  → Improvement: +{improvement:.3f} ✓")
else:
    decline = baseline_validation['pr_auc'] - lightning_validation['pr_auc']
    print(f"  → Decline: -{decline:.3f} ✗")

print()

# ===================================================================
# COMPARISON 3: ATTENTION WEIGHTS
# ===================================================================

print("="*80)
print("COMPARISON 3: FEATURE IMPORTANCE (ATTENTION WEIGHTS)")
print("="*80)
print()

# Extract attention weights
baseline_attention = baseline_trace.posterior['attention_weights'].mean(dim=['chain', 'draw']).values
lightning_attention = lightning_trace.posterior['attention_weights'].mean(dim=['chain', 'draw']).values

print("BASELINE MODEL (T+P) - Top 5 Feature Groups:")
baseline_attn_df = pd.DataFrame({
    'group': baseline_group_names,
    'attention': baseline_attention
}).sort_values('attention', ascending=False)
print(baseline_attn_df.head().to_string(index=False))

print("\nLIGHTNING MODEL (T+P+L) - Top 5 Feature Groups:")
lightning_attn_df = pd.DataFrame({
    'group': lightning_group_names,
    'attention': lightning_attention
}).sort_values('attention', ascending=False)
print(lightning_attn_df.head().to_string(index=False))

# Check if lightning groups are in top features
lightning_groups_list = [g for g in lightning_group_names if 'light' in g.lower()]
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

print("="*80)
print("GENERATING COMPARISON PLOTS")
print("="*80)
print()

# Plot 1: Temporal fit comparison
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Monthly comparison
ax1 = axes[0, 0]
monthly_data = pd.DataFrame({
    'month': range(1, 13),
    'baseline': baseline_temporal.get('monthly_predicted', np.zeros(12)),
    'lightning': lightning_temporal.get('monthly_predicted', np.zeros(12)),
    'actual': baseline_temporal.get('monthly_actual', np.zeros(12))
})

x = monthly_data['month']
width = 0.25
ax1.bar(x - width, monthly_data['actual'], width, label='Actual', alpha=0.7, color='black')
ax1.bar(x, monthly_data['baseline'], width, label='Baseline (T+P)', alpha=0.7, color='steelblue')
ax1.bar(x + width, monthly_data['lightning'], width, label='Lightning (T+P+L)', alpha=0.7, color='orange')
ax1.set_xlabel('Month', fontsize=12)
ax1.set_ylabel('Fire Count', fontsize=12)
ax1.set_title(f'Monthly Fit Comparison', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Seasonal comparison
ax2 = axes[0, 1]
seasonal_data = pd.DataFrame({
    'season': ['Winter', 'Spring', 'Summer', 'Fall'],
    'baseline': baseline_temporal.get('seasonal_predicted', np.zeros(4)),
    'lightning': lightning_temporal.get('seasonal_predicted', np.zeros(4)),
    'actual': baseline_temporal.get('seasonal_actual', np.zeros(4))
})

x = np.arange(len(seasonal_data))
ax2.bar(x - width, seasonal_data['actual'], width, label='Actual', alpha=0.7, color='black')
ax2.bar(x, seasonal_data['baseline'], width, label='Baseline (T+P)', alpha=0.7, color='steelblue')
ax2.bar(x + width, seasonal_data['lightning'], width, label='Lightning (T+P+L)', alpha=0.7, color='orange')
ax2.set_xticks(x)
ax2.set_xticklabels(seasonal_data['season'])
ax2.set_ylabel('Fire Count', fontsize=12)
ax2.set_title(f'Seasonal Fit Comparison', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# ROC comparison (if available)
ax3 = axes[1, 0]
if 'fpr' in baseline_validation and 'fpr' in lightning_validation:
    ax3.plot(baseline_validation['fpr'], baseline_validation['tpr'],
             label=f"Baseline (AUC={baseline_validation['roc_auc']:.3f})",
             linewidth=2, color='steelblue')
    ax3.plot(lightning_validation['fpr'], lightning_validation['tpr'],
             label=f"Lightning (AUC={lightning_validation['roc_auc']:.3f})",
             linewidth=2, color='orange')
    ax3.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax3.set_xlabel('False Positive Rate', fontsize=12)
    ax3.set_ylabel('True Positive Rate', fontsize=12)
    ax3.set_title('ROC Curve Comparison', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
else:
    ax3.text(0.5, 0.5, 'ROC curves not available', ha='center', va='center')
    ax3.set_title('ROC Curve Comparison', fontsize=14, fontweight='bold')

# Attention weights comparison
ax4 = axes[1, 1]
top_n = 10
baseline_top = baseline_attn_df.head(top_n)
lightning_top = lightning_attn_df.head(top_n)

# Create comparison of common groups
all_groups = list(set(baseline_top['group'].tolist() + lightning_top['group'].tolist()))
comparison = pd.DataFrame({
    'group': all_groups,
    'baseline': [baseline_attn_df[baseline_attn_df['group']==g]['attention'].values[0]
                 if g in baseline_attn_df['group'].values else 0 for g in all_groups],
    'lightning': [lightning_attn_df[lightning_attn_df['group']==g]['attention'].values[0]
                  if g in lightning_attn_df['group'].values else 0 for g in all_groups]
}).sort_values('lightning', ascending=True).tail(top_n)

y_pos = np.arange(len(comparison))
ax4.barh(y_pos - 0.2, comparison['baseline'], 0.4, label='Baseline (T+P)', alpha=0.7, color='steelblue')
ax4.barh(y_pos + 0.2, comparison['lightning'], 0.4, label='Lightning (T+P+L)', alpha=0.7, color='orange')
ax4.set_yticks(y_pos)
ax4.set_yticklabels(comparison['group'], fontsize=9)
ax4.set_xlabel('Attention Weight', fontsize=12)
ax4.set_title(f'Top {top_n} Feature Group Importance', fontsize=14, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
output_plot = OUTPUT_DIR / "model_comparison.png"
plt.savefig(output_plot, dpi=300, bbox_inches='tight')
print(f"✓ Comparison plot saved: {output_plot}")
plt.close()

# ===================================================================
# SUMMARY REPORT
# ===================================================================

print("\n" + "="*80)
print("SUMMARY REPORT")
print("="*80)
print()

summary = {
    'metric': [],
    'baseline_TP': [],
    'lightning_TPL': [],
    'improvement': [],
    'pct_change': []
}

# Add metrics
metrics = [
    ('Monthly R²', baseline_temporal['monthly_r2'], lightning_temporal['monthly_r2']),
    ('Seasonal R²', baseline_temporal['seasonal_r2'], lightning_temporal['seasonal_r2']),
    ('ROC-AUC', baseline_validation['roc_auc'], lightning_validation['roc_auc']),
    ('PR-AUC', baseline_validation['pr_auc'], lightning_validation['pr_auc'])
]

for metric_name, baseline_val, lightning_val in metrics:
    improvement = lightning_val - baseline_val
    pct_change = (improvement / baseline_val) * 100 if baseline_val > 0 else 0

    summary['metric'].append(metric_name)
    summary['baseline_TP'].append(f"{baseline_val:.3f}")
    summary['lightning_TPL'].append(f"{lightning_val:.3f}")
    summary['improvement'].append(f"{improvement:+.3f}")
    summary['pct_change'].append(f"{pct_change:+.1f}%")

summary_df = pd.DataFrame(summary)
print(summary_df.to_string(index=False))

# Save summary
summary_csv = OUTPUT_DIR / "comparison_summary.csv"
summary_df.to_csv(summary_csv, index=False)
print(f"\n✓ Summary saved: {summary_csv}")

# ===================================================================
# RECOMMENDATION
# ===================================================================

print("\n" + "="*80)
print("RECOMMENDATION")
print("="*80)
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

print("\n" + "="*80)
print("COMPARISON COMPLETE")
print("="*80)
