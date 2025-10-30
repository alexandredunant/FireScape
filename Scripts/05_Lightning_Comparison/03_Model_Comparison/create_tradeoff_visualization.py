#!/usr/bin/env python
"""
Create visualization highlighting the temporal vs discrimination trade-off
when including lightning data in wildfire prediction models.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

print("Creating trade-off visualization...")

# Data from comparison results
metrics = {
    'Metric': ['Monthly R²', 'Seasonal R²', 'ROC-AUC', 'PR-AUC'],
    'Baseline (T+P)': [0.679, 0.714, 0.835, 0.654],
    'Lightning (T+P+L)': [0.784, 0.838, 0.803, 0.514],
    'Change': [+0.105, +0.124, -0.033, -0.139],
    'Pct Change': [+15.5, +17.3, -3.9, -21.3]
}

df = pd.DataFrame(metrics)

# Create comprehensive figure
fig = plt.figure(figsize=(20, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# ============================================================================
# PLOT 1: Side-by-side bar comparison
# ============================================================================
ax1 = fig.add_subplot(gs[0, :])

x = np.arange(len(df))
width = 0.35

bars1 = ax1.bar(x - width/2, df['Baseline (T+P)'], width,
                label='Baseline (T+P)', alpha=0.8, color='steelblue', edgecolor='black')
bars2 = ax1.bar(x + width/2, df['Lightning (T+P+L)'], width,
                label='Lightning (T+P+L)', alpha=0.8, color='orange', edgecolor='black')

ax1.set_xlabel('Metric', fontsize=14, fontweight='bold')
ax1.set_ylabel('Score', fontsize=14, fontweight='bold')
ax1.set_title('Model Performance Comparison: Temporal Fit vs Discrimination',
              fontsize=16, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(df['Metric'], fontsize=12)
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3, axis='y')
ax1.set_ylim([0, 1])

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=10)

# Add colored background regions
ax1.axvspan(-0.5, 1.5, alpha=0.1, color='green', label='Temporal Metrics')
ax1.axvspan(1.5, 3.5, alpha=0.1, color='red', label='Discrimination Metrics')

# ============================================================================
# PLOT 2: Change visualization (lollipop chart)
# ============================================================================
ax2 = fig.add_subplot(gs[1, :2])

colors = ['green' if c > 0 else 'red' for c in df['Change']]
ax2.hlines(y=df['Metric'], xmin=0, xmax=df['Change'], color=colors, alpha=0.4, linewidth=5)
ax2.scatter(df['Change'], df['Metric'], color=colors, s=200, alpha=0.8, edgecolors='black', linewidth=2)

# Add value labels
for i, (change, pct) in enumerate(zip(df['Change'], df['Pct Change'])):
    ax2.text(change, i, f'  {change:+.3f}\n  ({pct:+.1f}%)',
            va='center', ha='left' if change > 0 else 'right', fontsize=11, fontweight='bold')

ax2.axvline(x=0, color='black', linewidth=2, linestyle='--', alpha=0.5)
ax2.set_xlabel('Change (Lightning - Baseline)', fontsize=14, fontweight='bold')
ax2.set_title('Performance Changes When Adding Lightning', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='x')

# ============================================================================
# PLOT 3: Interpretation box
# ============================================================================
ax3 = fig.add_subplot(gs[1, 2])
ax3.axis('off')

interpretation = """
TRADE-OFF INTERPRETATION

✅ TEMPORAL FIT (improved):
  • Monthly R²: +15.5%
  • Seasonal R²: +17.3%
  • Better capture of seasonal
    patterns and fire timing

⚠️ DISCRIMINATION (declined):
  • ROC-AUC: -3.9%
  • PR-AUC: -21.3%
  • Reduced ability to separate
    fire from non-fire events

🤔 POSSIBLE EXPLANATIONS:
  1. Smaller training set (1556 vs 3035)
  2. Lightning rare → sparse signal
  3. Model overfits temporal patterns
  4. Trade-off: pattern vs prediction

📊 USE CASE DEPENDENT:
  • Climate projections → Lightning
  • Real-time alerts → Baseline
"""

ax3.text(0.05, 0.95, interpretation, transform=ax3.transAxes,
        fontsize=11, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

# ============================================================================
# PLOT 4: Metric categories
# ============================================================================
ax4 = fig.add_subplot(gs[2, 0])

categories = ['Temporal\nFit', 'Discrimination']
baseline_avg = [np.mean([0.679, 0.714]), np.mean([0.835, 0.654])]
lightning_avg = [np.mean([0.784, 0.838]), np.mean([0.803, 0.514])]

x_cat = np.arange(len(categories))
width = 0.35

ax4.bar(x_cat - width/2, baseline_avg, width, label='Baseline (T+P)',
        alpha=0.8, color='steelblue', edgecolor='black')
ax4.bar(x_cat + width/2, lightning_avg, width, label='Lightning (T+P+L)',
        alpha=0.8, color='orange', edgecolor='black')

ax4.set_ylabel('Average Score', fontsize=12, fontweight='bold')
ax4.set_title('Performance by Category', fontsize=13, fontweight='bold')
ax4.set_xticks(x_cat)
ax4.set_xticklabels(categories, fontsize=11)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3, axis='y')
ax4.set_ylim([0, 1])

# Add value labels
for i, (b, l) in enumerate(zip(baseline_avg, lightning_avg)):
    ax4.text(i - width/2, b, f'{b:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax4.text(i + width/2, l, f'{l:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# ============================================================================
# PLOT 5: Percent change by metric
# ============================================================================
ax5 = fig.add_subplot(gs[2, 1])

colors_pct = ['green' if p > 0 else 'red' for p in df['Pct Change']]
bars = ax5.barh(df['Metric'], df['Pct Change'], color=colors_pct, alpha=0.7, edgecolor='black')

ax5.axvline(x=0, color='black', linewidth=2, linestyle='--')
ax5.set_xlabel('Percent Change (%)', fontsize=12, fontweight='bold')
ax5.set_title('Relative Performance Changes', fontsize=13, fontweight='bold')
ax5.grid(True, alpha=0.3, axis='x')

# Add value labels
for bar, pct in zip(bars, df['Pct Change']):
    width = bar.get_width()
    label_x = width + (2 if width > 0 else -2)
    ax5.text(label_x, bar.get_y() + bar.get_height()/2,
            f'{pct:+.1f}%', va='center', ha='left' if width > 0 else 'right',
            fontsize=11, fontweight='bold')

# ============================================================================
# PLOT 6: Decision guide
# ============================================================================
ax6 = fig.add_subplot(gs[2, 2])
ax6.axis('off')

decision = """
RECOMMENDATION

🎯 DEPENDS ON USE CASE:

FOR CLIMATE PROJECTIONS:
  ✅ Use Lightning Model
  • Better temporal patterns
  • Captures seasonal shifts
  • More realistic climate response

FOR OPERATIONAL ALERTS:
  ✅ Use Baseline Model
  • Better discrimination
  • Higher precision
  • Fewer false alarms

COMPROMISE OPTION:
  • Ensemble both models
  • Weight by use case
  • Temporal for trends
  • Discrimination for alerts
"""

ax6.text(0.05, 0.95, decision, transform=ax6.transAxes,
        fontsize=10, verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))

# ============================================================================
# Save figure
# ============================================================================
plt.suptitle('Lightning vs Baseline Model: Temporal Fit vs Discrimination Trade-off',
            fontsize=18, fontweight='bold', y=0.995)

OUTPUT_DIR = Path("/mnt/CEPH_PROJECTS/Firescape/Scripts/05_Lightning_Comparison/03_Model_Comparison/Results")
output_file = OUTPUT_DIR / "tradeoff_analysis.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✓ Saved: {output_file}")

plt.close()

# ============================================================================
# Create summary table
# ============================================================================
print("\nCreating summary table...")

summary_file = OUTPUT_DIR / "tradeoff_summary.txt"
with open(summary_file, 'w') as f:
    f.write("="*80 + "\n")
    f.write("LIGHTNING MODEL: TEMPORAL FIT vs DISCRIMINATION TRADE-OFF\n")
    f.write("="*80 + "\n\n")

    f.write("PERFORMANCE COMPARISON (2012-2024 Period)\n")
    f.write("-"*80 + "\n\n")

    f.write(df.to_string(index=False))
    f.write("\n\n")

    f.write("="*80 + "\n")
    f.write("INTERPRETATION\n")
    f.write("="*80 + "\n\n")

    f.write("✅ TEMPORAL FIT - SIGNIFICANTLY IMPROVED:\n")
    f.write("  • Monthly R² increased by 15.5% (0.679 → 0.784)\n")
    f.write("  • Seasonal R² increased by 17.3% (0.714 → 0.838)\n")
    f.write("  • Lightning model better captures fire seasonality\n")
    f.write("  • More accurate prediction of WHEN fires occur\n\n")

    f.write("⚠️ DISCRIMINATION - DECLINED:\n")
    f.write("  • ROC-AUC decreased by 3.9% (0.835 → 0.803)\n")
    f.write("  • PR-AUC decreased by 21.3% (0.654 → 0.514)\n")
    f.write("  • Reduced ability to distinguish fire from non-fire days\n")
    f.write("  • Lower precision at same recall\n\n")

    f.write("="*80 + "\n")
    f.write("POSSIBLE CAUSES\n")
    f.write("="*80 + "\n\n")

    f.write("1. REDUCED TRAINING DATA:\n")
    f.write("   • Baseline: 3035 observations (1999-2024)\n")
    f.write("   • Lightning: 1556 observations (2012-2024)\n")
    f.write("   • 49% less data may hurt discrimination learning\n\n")

    f.write("2. SPARSE LIGHTNING SIGNAL:\n")
    f.write("   • Only 6.5% of days have lightning\n")
    f.write("   • Rare events harder to learn from\n")
    f.write("   • Strong seasonal pattern but sparse occurrence\n\n")

    f.write("3. OVERFITTING TO TEMPORAL PATTERNS:\n")
    f.write("   • Model may prioritize seasonal fit over discrimination\n")
    f.write("   • Better at 'when' but not 'which specific days'\n\n")

    f.write("="*80 + "\n")
    f.write("RECOMMENDATIONS\n")
    f.write("="*80 + "\n\n")

    f.write("USE CASE: CLIMATE PROJECTIONS\n")
    f.write("  ✅ Prefer Lightning Model\n")
    f.write("  • Temporal accuracy more important than discrimination\n")
    f.write("  • Captures seasonal shifts under climate change\n")
    f.write("  • Better for understanding long-term patterns\n\n")

    f.write("USE CASE: OPERATIONAL FIRE ALERTS\n")
    f.write("  ✅ Prefer Baseline Model\n")
    f.write("  • Discrimination more critical for daily decisions\n")
    f.write("  • Higher precision reduces false alarms\n")
    f.write("  • Better ROC-AUC for threshold setting\n\n")

    f.write("USE CASE: RESEARCH / ENSEMBLE\n")
    f.write("  💡 Use Both Models\n")
    f.write("  • Ensemble predictions for robustness\n")
    f.write("  • Lightning model for temporal trends\n")
    f.write("  • Baseline model for event discrimination\n")
    f.write("  • Combine strengths of both approaches\n\n")

    f.write("="*80 + "\n")

print(f"✓ Saved: {summary_file}")
print("\n✓ Trade-off analysis complete!")
