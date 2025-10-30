#!/usr/bin/env python
"""
Lightning vs No-Lightning Model Comparison (2012-2024 Period)

Compares two Bayesian wildfire models to assess whether including lightning data
improves predictive performance and temporal fit.

Models compared:
1. Baseline: T (temperature) + P (precipitation) only [trained 1999-2024]
2. Lightning: T + P + L (lightning flash density) [trained 2012-2024]

Comparison period: 2012-2024 (when lightning data available)
- Baseline model predictions filtered to 2012-2024 test set
- Lightning model trained and tested on 2012-2024
- Fair apples-to-apples comparison on same time period

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

Key fix: Previous analysis had 48.7% NaN values (1999-2011 had no lightning data)
which were filled with zeros, corrupting the signal. This comparison uses only
the period with complete lightning data.
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

# Use science plots for publication-quality figures
import scienceplots
plt.style.use(['science', 'no-latex'])

warnings.filterwarnings("ignore")

print("="*80)
print("LIGHTNING vs NO-LIGHTNING MODEL COMPARISON (2012-2024)")
print("="*80)
print()
print("NOTE: Fair comparison on 2012-2024 period only")
print("  - Baseline model trained on 1999-2024, evaluated on 2012-2024 test set")
print("  - Lightning model trained on 2012-2024 (complete lightning data)")
print("  - Both evaluated on same time period for fair comparison")
print()

# ===================================================================
# CONFIGURATION
# ===================================================================

BASE_DIR = Path("/mnt/CEPH_PROJECTS/Firescape")

# Model directories
BASELINE_MODEL_DIR = BASE_DIR / "Scripts/OUTPUT/02_Model_RelativeProbability"
LIGHTNING_MODEL_DIR = BASE_DIR / "Scripts/OUTPUT/02_Model_RelativeProbability_Lightning_2012plus"

# Output
OUTPUT_DIR = BASE_DIR / "Scripts/05_Lightning_Comparison/03_Model_Comparison/Results"
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

print(f"Configuration:")
print(f"  Baseline model (T+P): {BASELINE_MODEL_DIR} [1999-2024]")
print(f"  Lightning model (T+P+L): {LIGHTNING_MODEL_DIR} [2012-2024]")
print(f"  Comparison period: 2012-2024 (overlap only)")
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

# Filter baseline predictions to 2012+ for fair comparison
print("Filtering baseline model to 2012-2024 period for fair comparison...")
baseline_test_dates = pd.to_datetime(baseline_results['test_predictions']['dates_test'])
mask_2012plus = baseline_test_dates.year >= 2012

# Filter all baseline test data
baseline_results_filtered = {
    'test_predictions': {
        'mean_prob': baseline_results['test_predictions']['mean_prob'][mask_2012plus],
        'std_prob': baseline_results['test_predictions']['std_prob'][mask_2012plus],
        'y_test': baseline_results['test_predictions']['y_test'][mask_2012plus],
        'dates_test': baseline_results['test_predictions']['dates_test'][mask_2012plus]
    },
    'validation_metrics': baseline_results['validation_metrics'].copy(),
    'temporal_validation': baseline_results['temporal_validation'].copy(),
    'training_fire_rate': baseline_results['training_fire_rate']
}

# Recalculate validation metrics for filtered baseline data
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, r2_score
from scipy.stats import pearsonr

y_test_filtered = baseline_results_filtered['test_predictions']['y_test']
mean_prob_filtered = baseline_results_filtered['test_predictions']['mean_prob']
dates_test_filtered = pd.to_datetime(baseline_results_filtered['test_predictions']['dates_test'])

baseline_results_filtered['validation_metrics']['roc_auc'] = roc_auc_score(y_test_filtered, mean_prob_filtered)
baseline_results_filtered['validation_metrics']['pr_auc'] = average_precision_score(y_test_filtered, mean_prob_filtered)

# Recalculate temporal validation metrics from test set
monthly_data_baseline = pd.DataFrame({
    'month': dates_test_filtered.month,
    'season': dates_test_filtered.month.map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    }),
    'actual': y_test_filtered,
    'predicted_prob': mean_prob_filtered
})

# Monthly aggregation
monthly_agg_baseline = monthly_data_baseline.groupby('month').agg({
    'actual': 'sum',
    'predicted_prob': 'sum'
}).reset_index()
monthly_agg_baseline.columns = ['month', 'actual_fires', 'predicted_fires']

# Seasonal aggregation
seasonal_agg_baseline = monthly_data_baseline.groupby('season').agg({
    'actual': 'sum',
    'predicted_prob': 'sum'
}).reset_index()
seasonal_agg_baseline.columns = ['season', 'actual_fires', 'predicted_fires']

# Calculate temporal metrics
monthly_r2_baseline = r2_score(monthly_agg_baseline['actual_fires'], monthly_agg_baseline['predicted_fires'])
monthly_corr_baseline, _ = pearsonr(monthly_agg_baseline['actual_fires'], monthly_agg_baseline['predicted_fires'])
seasonal_r2_baseline = r2_score(seasonal_agg_baseline['actual_fires'], seasonal_agg_baseline['predicted_fires'])
seasonal_corr_baseline, _ = pearsonr(seasonal_agg_baseline['actual_fires'], seasonal_agg_baseline['predicted_fires'])

# Update temporal validation with recalculated metrics
baseline_results_filtered['temporal_validation'] = {
    'monthly': monthly_agg_baseline,
    'seasonal': seasonal_agg_baseline,
    'monthly_r2': monthly_r2_baseline,
    'monthly_corr': monthly_corr_baseline,
    'seasonal_r2': seasonal_r2_baseline,
    'seasonal_corr': seasonal_corr_baseline
}

print(f"  Original baseline test set: {len(baseline_test_dates)} observations")
print(f"  Filtered to 2012+: {len(y_test_filtered)} observations")
print(f"  Recalculated all metrics for fair comparison (test set only)")

# Replace baseline_results with filtered version
baseline_results = baseline_results_filtered

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
print(f"  Baseline (T+P):   R={baseline_temporal['monthly_corr']:.3f}, R²={baseline_temporal['monthly_r2']:.3f}")
print(f"  Lightning (T+P+L): R={lightning_temporal['monthly_corr']:.3f}, R²={lightning_temporal['monthly_r2']:.3f}")

if lightning_temporal['monthly_r2'] > baseline_temporal['monthly_r2']:
    improvement = lightning_temporal['monthly_r2'] - baseline_temporal['monthly_r2']
    pct_improvement = (improvement / baseline_temporal['monthly_r2']) * 100
    print(f"  → Improvement: +{improvement:.3f} ({pct_improvement:+.1f}%) ✓")
else:
    decline = baseline_temporal['monthly_r2'] - lightning_temporal['monthly_r2']
    pct_decline = (decline / baseline_temporal['monthly_r2']) * 100
    print(f"  → Decline: -{decline:.3f} ({pct_decline:.1f}%) ✗")

print("\nSEASONAL CORRELATION:")
print(f"  Baseline (T+P):    R={baseline_temporal['seasonal_corr']:.3f}, R²={baseline_temporal['seasonal_r2']:.3f}")
print(f"  Lightning (T+P+L): R={lightning_temporal['seasonal_corr']:.3f}, R²={lightning_temporal['seasonal_r2']:.3f}")

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

baseline_validation = baseline_results['validation_metrics']
lightning_validation = lightning_results['validation_metrics']

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
# VISUALIZATION: PUBLICATION-QUALITY COMPARISON PLOTS
# ===================================================================

print("="*80)
print("GENERATING PUBLICATION-QUALITY COMPARISON PLOTS")
print("="*80)
print()

# Define consistent colors
COLOR_ACTUAL = '#2C3E50'
COLOR_BASELINE = '#3498DB'
COLOR_LIGHTNING = '#E67E22'

# ============================================================
# FIGURE 1: Monthly Temporal Fit
# ============================================================
fig1, ax1 = plt.subplots(figsize=(8, 5))

monthly_baseline_df = baseline_temporal['monthly']
monthly_lightning_df = lightning_temporal['monthly']
monthly_data = pd.DataFrame({
    'month': range(1, 13),
    'baseline': monthly_baseline_df['predicted_fires'].values,
    'lightning': monthly_lightning_df['predicted_fires'].values,
    'actual': monthly_baseline_df['actual_fires'].values
})

x = monthly_data['month']
width = 0.25
ax1.bar(x - width, monthly_data['actual'], width, label='Observed', alpha=0.9, color=COLOR_ACTUAL, edgecolor='black', linewidth=0.5)
ax1.bar(x, monthly_data['baseline'], width, label='Baseline (T+P)', alpha=0.9, color=COLOR_BASELINE, edgecolor='black', linewidth=0.5)
ax1.bar(x + width, monthly_data['lightning'], width, label='Lightning (T+P+L)', alpha=0.9, color=COLOR_LIGHTNING, edgecolor='black', linewidth=0.5)
ax1.set_xlabel('Month')
ax1.set_ylabel('Fire Count')
ax1.set_xticks(x)
ax1.legend(frameon=True, loc='upper left')
ax1.text(0.98, 0.98, f"Baseline R² = {baseline_temporal['monthly_r2']:.3f}\nLightning R² = {lightning_temporal['monthly_r2']:.3f}",
         transform=ax1.transAxes, va='top', ha='right', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.tight_layout()
output_fig1 = OUTPUT_DIR / "fig1_monthly_temporal_fit.png"
plt.savefig(output_fig1, dpi=600, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / "fig1_monthly_temporal_fit.pdf", bbox_inches='tight')
print(f"✓ Figure 1 saved: {output_fig1}")
plt.close()

# ============================================================
# FIGURE 2: Seasonal Temporal Fit
# ============================================================
fig2, ax2 = plt.subplots(figsize=(7, 5))

seasonal_baseline_df = baseline_temporal['seasonal']
seasonal_lightning_df = lightning_temporal['seasonal']
season_order = ['Winter', 'Spring', 'Summer', 'Fall']
seasonal_data = pd.DataFrame({
    'season': season_order,
    'baseline': [seasonal_baseline_df[seasonal_baseline_df['season']==s]['predicted_fires'].values[0] for s in season_order],
    'lightning': [seasonal_lightning_df[seasonal_lightning_df['season']==s]['predicted_fires'].values[0] for s in season_order],
    'actual': [seasonal_baseline_df[seasonal_baseline_df['season']==s]['actual_fires'].values[0] for s in season_order]
})

x = np.arange(len(seasonal_data))
width = 0.25
ax2.bar(x - width, seasonal_data['actual'], width, label='Observed', alpha=0.9, color=COLOR_ACTUAL, edgecolor='black', linewidth=0.5)
ax2.bar(x, seasonal_data['baseline'], width, label='Baseline (T+P)', alpha=0.9, color=COLOR_BASELINE, edgecolor='black', linewidth=0.5)
ax2.bar(x + width, seasonal_data['lightning'], width, label='Lightning (T+P+L)', alpha=0.9, color=COLOR_LIGHTNING, edgecolor='black', linewidth=0.5)
ax2.set_xticks(x)
ax2.set_xticklabels(seasonal_data['season'])
ax2.set_ylabel('Fire Count')
ax2.legend(frameon=True, loc='upper left')
ax2.text(0.98, 0.98, f"Baseline R² = {baseline_temporal['seasonal_r2']:.3f}\nLightning R² = {lightning_temporal['seasonal_r2']:.3f}",
         transform=ax2.transAxes, va='top', ha='right', fontsize=9,
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
plt.tight_layout()
output_fig2 = OUTPUT_DIR / "fig2_seasonal_temporal_fit.png"
plt.savefig(output_fig2, dpi=600, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / "fig2_seasonal_temporal_fit.pdf", bbox_inches='tight')
print(f"✓ Figure 2 saved: {output_fig2}")
plt.close()

# ============================================================
# FIGURE 3: ROC Curve Comparison
# ============================================================
fig3, ax3 = plt.subplots(figsize=(6, 6))

baseline_test = baseline_results['test_predictions']
lightning_test = lightning_results['test_predictions']

baseline_fpr, baseline_tpr, _ = roc_curve(baseline_test['y_test'], baseline_test['mean_prob'])
lightning_fpr, lightning_tpr, _ = roc_curve(lightning_test['y_test'], lightning_test['mean_prob'])

ax3.plot(baseline_fpr, baseline_tpr,
         label=f"Baseline (AUC = {baseline_validation['roc_auc']:.3f})",
         linewidth=2, color=COLOR_BASELINE)
ax3.plot(lightning_fpr, lightning_tpr,
         label=f"Lightning (AUC = {lightning_validation['roc_auc']:.3f})",
         linewidth=2, color=COLOR_LIGHTNING)
ax3.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
ax3.set_xlabel('False Positive Rate')
ax3.set_ylabel('True Positive Rate')
ax3.legend(frameon=True, loc='lower right')
ax3.set_xlim([0, 1])
ax3.set_ylim([0, 1])
ax3.set_aspect('equal')
plt.tight_layout()
output_fig3 = OUTPUT_DIR / "fig3_roc_curve.png"
plt.savefig(output_fig3, dpi=600, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / "fig3_roc_curve.pdf", bbox_inches='tight')
print(f"✓ Figure 3 saved: {output_fig3}")
plt.close()

# ============================================================
# FIGURE 4: Precision-Recall Curve Comparison
# ============================================================
fig4, ax4 = plt.subplots(figsize=(6, 6))

baseline_precision, baseline_recall, _ = precision_recall_curve(baseline_test['y_test'], baseline_test['mean_prob'])
lightning_precision, lightning_recall, _ = precision_recall_curve(lightning_test['y_test'], lightning_test['mean_prob'])

ax4.plot(baseline_recall, baseline_precision,
         label=f"Baseline (AUC = {baseline_validation['pr_auc']:.3f})",
         linewidth=2, color=COLOR_BASELINE)
ax4.plot(lightning_recall, lightning_precision,
         label=f"Lightning (AUC = {lightning_validation['pr_auc']:.3f})",
         linewidth=2, color=COLOR_LIGHTNING)
# Add baseline (proportion of positive class)
fire_rate = baseline_test['y_test'].mean()
ax4.axhline(y=fire_rate, color='k', linestyle='--', linewidth=1, alpha=0.5, label=f'Random (p={fire_rate:.3f})')
ax4.set_xlabel('Recall')
ax4.set_ylabel('Precision')
ax4.legend(frameon=True, loc='upper right')
ax4.set_xlim([0, 1])
ax4.set_ylim([0, 1])
ax4.set_aspect('equal')
plt.tight_layout()
output_fig4 = OUTPUT_DIR / "fig4_precision_recall_curve.png"
plt.savefig(output_fig4, dpi=600, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / "fig4_precision_recall_curve.pdf", bbox_inches='tight')
print(f"✓ Figure 4 saved: {output_fig4}")
plt.close()

# ============================================================
# FIGURE 5: Feature Importance (Attention Weights)
# ============================================================
fig5, ax5 = plt.subplots(figsize=(7, 6))

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
width_bar = 0.35
ax5.barh(y_pos - width_bar/2, comparison['baseline'], width_bar, label='Baseline (T+P)', alpha=0.9, color=COLOR_BASELINE, edgecolor='black', linewidth=0.5)
ax5.barh(y_pos + width_bar/2, comparison['lightning'], width_bar, label='Lightning (T+P+L)', alpha=0.9, color=COLOR_LIGHTNING, edgecolor='black', linewidth=0.5)
ax5.set_yticks(y_pos)
ax5.set_yticklabels(comparison['group'])
ax5.set_xlabel('Attention Weight')
ax5.legend(frameon=True, loc='lower right')
plt.tight_layout()
output_fig5 = OUTPUT_DIR / "fig5_feature_importance.png"
plt.savefig(output_fig5, dpi=600, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / "fig5_feature_importance.pdf", bbox_inches='tight')
print(f"✓ Figure 5 saved: {output_fig5}")
plt.close()

# ============================================================
# FIGURE 6: Trade-off Summary
# ============================================================
fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(12, 4.5))

# Panel A: Bar comparison
metrics_data = {
    'Metric': ['Monthly R²', 'Seasonal R²', 'ROC-AUC', 'PR-AUC'],
    'Baseline': [baseline_temporal['monthly_r2'], baseline_temporal['seasonal_r2'],
                 baseline_validation['roc_auc'], baseline_validation['pr_auc']],
    'Lightning': [lightning_temporal['monthly_r2'], lightning_temporal['seasonal_r2'],
                  lightning_validation['roc_auc'], lightning_validation['pr_auc']]
}

x_trade = np.arange(len(metrics_data['Metric']))
width_trade = 0.35

bars1 = ax6a.bar(x_trade - width_trade/2, metrics_data['Baseline'], width_trade,
                  label='Baseline (T+P)', alpha=0.9, color=COLOR_BASELINE, edgecolor='black', linewidth=0.5)
bars2 = ax6a.bar(x_trade + width_trade/2, metrics_data['Lightning'], width_trade,
                  label='Lightning (T+P+L)', alpha=0.9, color=COLOR_LIGHTNING, edgecolor='black', linewidth=0.5)

ax6a.set_xlabel('Metric')
ax6a.set_ylabel('Score')
ax6a.set_xticks(x_trade)
ax6a.set_xticklabels(metrics_data['Metric'], rotation=0)
ax6a.legend(frameon=True, loc='lower left')
ax6a.set_ylim([0, 1])

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax6a.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=8)

# Panel B: Change visualization
changes = [lightning_temporal['monthly_r2'] - baseline_temporal['monthly_r2'],
           lightning_temporal['seasonal_r2'] - baseline_temporal['seasonal_r2'],
           lightning_validation['roc_auc'] - baseline_validation['roc_auc'],
           lightning_validation['pr_auc'] - baseline_validation['pr_auc']]
colors_trade = [COLOR_LIGHTNING if c > 0 else COLOR_BASELINE for c in changes]

ax6b.hlines(y=metrics_data['Metric'], xmin=0, xmax=changes, color=colors_trade,
             alpha=0.6, linewidth=4)
ax6b.scatter(changes, metrics_data['Metric'], color=colors_trade, s=100,
              alpha=0.9, edgecolors='black', linewidth=1, zorder=10)

for i, change in enumerate(changes):
    pct = (change / metrics_data['Baseline'][i]) * 100
    ax6b.text(change, i, f'  {change:+.3f} ({pct:+.1f}%)',
               va='center', ha='left' if change > 0 else 'right',
               fontsize=9)

ax6b.axvline(x=0, color='black', linewidth=1, linestyle='--', alpha=0.5)
ax6b.set_xlabel('Change (Lightning − Baseline)')
ax6b.set_ylabel('')

plt.tight_layout()
output_fig6 = OUTPUT_DIR / "fig6_tradeoff_summary.png"
plt.savefig(output_fig6, dpi=600, bbox_inches='tight')
plt.savefig(OUTPUT_DIR / "fig6_tradeoff_summary.pdf", bbox_inches='tight')
print(f"✓ Figure 6 saved: {output_fig6}")
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

# Old trade-off visualization removed - now using Figure 6 above

print("\n" + "="*80)
print("COMPARISON COMPLETE")
print("="*80)
