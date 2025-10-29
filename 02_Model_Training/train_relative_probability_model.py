# ===================================================================
# BAYESIAN ATTENTION WILDFIRE MODEL - RELATIVE PROBABILITY WITH UNCERTAINTY
#
# PURPOSE: Predict relative wildfire risk for Bolzano Province with uncertainty
#
# APPROACH: RELATIVE PROBABILITY ONLY
# 1. Train model on case-control sample with neutral priors
# 2. Learn feature relationships and attention weights
# 3. Output relative risk scores (NOT absolute probabilities)
# 4. Provide uncertainty estimates via Bayesian posterior
#
# INTERPRETATION:
# - Model output: Relative probability scores (0-1 range)
# - Higher scores = higher relative risk compared to baseline
# - Use for ranking days/locations by risk and comparing scenarios
# - Uncertainty quantified via posterior standard deviation
# - NO conversion to absolute fire counts (calibration was unreliable)
# ===================================================================

# ===================================================================
# IMPORTS
# ===================================================================

import xarray as xr
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# --- PyMC & ArviZ for Bayesian Modeling ---
import pymc as pm
import arviz as az
import pytensor.tensor as pt

# --- Standard ML tools ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.cluster import KMeans
import joblib

# --- Visualization tools ---
import seaborn as sns

print(f"Running Bayesian Relative Probability Model on PyMC v{pm.__version__}")

# ===================================================================
# CONFIGURATION
# ===================================================================

# Input data paths
# Note: landcover_fire_risk in the NetCDF is already in ordinal form (0-5 scale)
# representing fire risk from Corine Land Cover classification
NETCDF_PATH = "/mnt/CEPH_PROJECTS/Firescape/Scripts/OUTPUT/01_Training_Data/spacetime_stacks.nc"
WILDFIRE_HISTORY_PATH = "/mnt/CEPH_PROJECTS/Firescape/Data/WILDFIRE_INVENTORY/wildfire_point_Bolzano_Period1999_2025.gpkg"

# Output directory
ML_OUTPUT_DIR = "/mnt/CEPH_PROJECTS/Firescape/Scripts/OUTPUT/02_Model_RelativeProbability/"
os.makedirs(ML_OUTPUT_DIR, exist_ok=True)

# Feature categories
STATIC_VARS = [
    'tri', 'northness', 'slope', 'aspect', 'nasadem',
    'treecoverdensity', 'landcover_fire_risk', 'distroads',
    'eastness', 'flammability', 'walking_time_to_bldg',
    'walking_time_to_elec_infra'
]
DYNAMIC_VARS = ['T', 'P']
TIME_STEPS = 60

# ===================================================================
# FEATURE ENGINEERING (same as relative probability model)
# ===================================================================

def create_cumulative_features(data_for_point):
    """Extract features from spatial-temporal data for a single observation.

    Note: landcover_fire_risk is already in ordinal form (0-5) in the NetCDF file,
    so we can treat it like other continuous variables using mean aggregation.
    """
    features = {}

    center_y, center_x = data_for_point.sizes['y'] // 2, data_for_point.sizes['x'] // 2
    window_size = 4
    half_window = window_size // 2
    y_slice = slice(center_y - half_window, center_y + half_window)
    x_slice = slice(center_x - half_window, center_x + half_window)

    # Extract STATIC features
    # All variables are treated as continuous using mean aggregation
    for var_name in STATIC_VARS:
        if var_name in data_for_point.channel.values:
            window_data = data_for_point.sel(channel=var_name).isel(
                time=0, y=y_slice, x=x_slice
            )
            features[var_name] = window_data.mean().item()

    # Extract DYNAMIC features
    for var_name in DYNAMIC_VARS:
        if var_name in data_for_point.channel.values:
            dynamic_var_data = data_for_point.sel(channel=var_name)
            spatial_mean_series = dynamic_var_data.mean(dim=['x', 'y']).reindex(
                time=list(reversed(range(TIME_STEPS)))
            )

            cumulative_sum = spatial_mean_series.cumsum(dim='time')
            divisor = xr.DataArray(np.arange(1, len(spatial_mean_series.time) + 1), dims='time')
            cumulative_means = cumulative_sum / divisor

            cumulative_maxs = xr.DataArray(
                np.maximum.accumulate(spatial_mean_series.values),
                coords=spatial_mean_series.coords
            )

            day_windows_to_keep = [1, 3, 5, 10, 15, 30, 60]
            for day_window in day_windows_to_keep:
                i = day_window - 1
                features[f"{var_name}_cumulative_mean_{day_window}d"] = cumulative_means.isel(time=i).item()
                features[f"{var_name}_cumulative_max_{day_window}d"] = cumulative_maxs.isel(time=i).item()

    return pd.Series(features)

# ===================================================================
# ATTENTION MECHANISM (same as relative probability model)
# ===================================================================

def create_temporal_attention_groups(feature_names):
    """Group features into logical categories for attention mechanism."""
    temporal_groups = {
        'temp_1d': [], 'temp_short': [], 'temp_medium': [], 'temp_30d': [], 'temp_60d': [],
        'precip_1d': [], 'precip_short': [], 'precip_medium': [], 'precip_30d': [], 'precip_60d': [],
        'static_topo': [], 'static_veg': [], 'static_human': [], 'static_other': []
    }

    for i, name in enumerate(feature_names):
        name_lower = name.lower()

        if 't_' in name_lower:
            if '_60d' in name_lower: temporal_groups['temp_60d'].append(i)
            elif '_30d' in name_lower: temporal_groups['temp_30d'].append(i)
            elif '_1d' in name_lower: temporal_groups['temp_1d'].append(i)
            elif any(x in name_lower for x in ['_3d', '_5d']): temporal_groups['temp_short'].append(i)
            else: temporal_groups['temp_medium'].append(i)
        elif 'p_' in name_lower:
            if '_60d' in name_lower: temporal_groups['precip_60d'].append(i)
            elif '_30d' in name_lower: temporal_groups['precip_30d'].append(i)
            elif '_1d' in name_lower: temporal_groups['precip_1d'].append(i)
            elif any(x in name_lower for x in ['_3d', '_5d']): temporal_groups['precip_short'].append(i)
            else: temporal_groups['precip_medium'].append(i)
        elif any(x in name_lower for x in ['slope', 'aspect', 'northness', 'eastness', 'nasadem', 'tri']):
            temporal_groups['static_topo'].append(i)
        elif any(x in name_lower for x in ['treecoverdensity', 'landcover', 'flammability']):
            temporal_groups['static_veg'].append(i)
        elif any(x in name_lower for x in ['walking_time', 'distroads']):
            temporal_groups['static_human'].append(i)
        else:
            temporal_groups['static_other'].append(i)

    return {k: v for k, v in temporal_groups.items() if v}

# ===================================================================
# BAYESIAN MODEL WITH ABSOLUTE PROBABILITY PRIOR
# ===================================================================

def create_relative_probability_model(X_train_scaled, y_train, feature_names):
    """
    Create Bayesian model for relative probability prediction with uncertainty.

    STRATEGY:
    - Use weakly informative priors
    - Learn feature relationships from case-control data
    - Output relative risk scores (not absolute probabilities)
    - Quantify uncertainty via Bayesian posterior

    Args:
        X_train_scaled: Scaled training features
        y_train: Training labels (from case-control sample)
        feature_names: Feature names

    Returns:
        tuple: (PyMC model, temporal_groups, group_names)
    """
    n_obs, n_features = X_train_scaled.shape

    temporal_groups = create_temporal_attention_groups(feature_names)
    group_names = list(temporal_groups.keys())
    n_groups = len(group_names)

    print(f"\n🎯 MODEL SETUP (Relative Probability with Uncertainty):")
    print(f"  Approach: Bayesian attention mechanism")
    print(f"  Output: Relative risk scores (not absolute probabilities)")
    print(f"  Uncertainty: Quantified via posterior samples")

    # Create the Bayesian model
    with pm.Model() as model:

        # MODEL INTERCEPT - Neutral prior
        alpha = pm.Normal('alpha',
                         mu=0,      # Neutral prior
                         sigma=2.0)  # Weakly informative

        # ATTENTION MECHANISM - Allow model flexibility
        attention_raw = pm.Dirichlet('attention_weights', a=np.ones(n_groups) * 1.5)
        attention_scale = pm.HalfNormal('attention_scale', sigma=4.0)  # Increased to allow stronger effects
        attention_weights = attention_raw * attention_scale

        # Process each feature group with LOOSE priors
        group_contributions = []
        for group_idx, (group_name, feature_indices) in enumerate(temporal_groups.items()):
            n_group_features = len(feature_indices)
            group_features_data = X_train_scaled[:, feature_indices]

            # Loose priors to allow features to have strong effects
            if 'temp' in group_name or 'precip' in group_name:
                group_betas = pm.Normal(f'beta_{group_name}', mu=0, sigma=3.0, shape=n_group_features)  # Increased
            else:
                group_betas = pm.Normal(f'beta_{group_name}', mu=0, sigma=1.5, shape=n_group_features)  # Increased

            group_contrib = pt.dot(group_features_data, group_betas)
            weighted_contrib = attention_weights[group_idx] * group_contrib
            group_contributions.append(weighted_contrib)

        # Combine all contributions
        logit_p = alpha + pt.sum(pt.stack(group_contributions), axis=0)

        # LIKELIHOOD
        pm.Bernoulli('likelihood', logit_p=logit_p, observed=y_train)

        # DERIVED QUANTITIES
        pm.Deterministic('model_baseline_prob', pm.math.sigmoid(alpha))
        pm.Deterministic('most_attended_group', pt.argmax(attention_weights))

    return model, temporal_groups, group_names

# ===================================================================
# PRIOR VISUALIZATION
# ===================================================================

def visualize_priors(model, temporal_groups, group_names, output_dir):
    """
    Visualize the prior distributions used in the Bayesian model.

    Shows:
    1. Prior for intercept (alpha)
    2. Prior for attention mechanism (Dirichlet + scaling)
    3. Priors for feature group coefficients (betas)
    """
    print("\n📊 Visualizing prior assumptions...")

    # Sample from priors only
    with model:
        prior_samples = pm.sample_prior_predictive(samples=5000, random_seed=42)

    # Create comprehensive prior visualization
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # --- ROW 1: Core priors ---

    # 1. Intercept prior (alpha)
    ax1 = fig.add_subplot(gs[0, 0])
    alpha_prior = prior_samples.prior['alpha'].values.flatten()
    ax1.hist(alpha_prior, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(0, color='red', linestyle='--', linewidth=2, label='Prior mean (neutral)')
    ax1.set_xlabel('Intercept (logit scale)', fontsize=11)
    ax1.set_ylabel('Frequency', fontsize=11)
    ax1.set_title('Prior: Intercept α ~ Normal(0, 2)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Show probability scale
    prob_scale = 1 / (1 + np.exp(-alpha_prior))
    ax1_twin = ax1.twiny()
    ax1_twin.hist(prob_scale, bins=50, alpha=0)
    ax1_twin.set_xlabel('Baseline probability', fontsize=10, color='gray')
    ax1_twin.tick_params(axis='x', labelcolor='gray')

    # 2. Attention scale prior
    ax2 = fig.add_subplot(gs[0, 1])
    attn_scale_prior = prior_samples.prior['attention_scale'].values.flatten()
    ax2.hist(attn_scale_prior, bins=50, alpha=0.7, color='coral', edgecolor='black')
    ax2.axvline(attn_scale_prior.mean(), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {attn_scale_prior.mean():.2f}')
    ax2.set_xlabel('Attention scaling factor', fontsize=11)
    ax2.set_ylabel('Frequency', fontsize=11)
    ax2.set_title('Prior: Attention Scale ~ HalfNormal(5)', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 3. Model baseline probability (derived from alpha)
    ax3 = fig.add_subplot(gs[0, 2])
    baseline_prob = prior_samples.prior['model_baseline_prob'].values.flatten()
    ax3.hist(baseline_prob, bins=50, alpha=0.7, color='seagreen', edgecolor='black')
    ax3.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Neutral (50%)')
    ax3.set_xlabel('Baseline probability', fontsize=11)
    ax3.set_ylabel('Frequency', fontsize=11)
    ax3.set_title('Prior: Baseline Fire Probability', fontsize=12, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 1)

    # --- ROW 2: Attention weights ---

    ax4 = fig.add_subplot(gs[1, :])
    attention_prior = prior_samples.prior['attention_weights'].values
    attention_df = pd.DataFrame(attention_prior.reshape(-1, len(group_names)),
                                columns=group_names)

    positions = np.arange(len(group_names))
    bp = ax4.boxplot([attention_df[col] for col in group_names],
                      positions=positions, widths=0.6, patch_artist=True,
                      boxprops=dict(facecolor='lightblue', alpha=0.7),
                      medianprops=dict(color='red', linewidth=2),
                      whiskerprops=dict(linewidth=1.5),
                      capprops=dict(linewidth=1.5))

    ax4.set_xticks(positions)
    ax4.set_xticklabels(group_names, rotation=45, ha='right', fontsize=10)
    ax4.set_ylabel('Attention weight', fontsize=11)
    ax4.set_title('Prior: Feature Group Attention Weights ~ Dirichlet(1.5) × HalfNormal(5)',
                  fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.axhline(1.0, color='gray', linestyle=':', linewidth=1, label='Equal weight')
    ax4.legend()

    # --- ROW 3: Beta coefficient priors by group type ---

    # Collect all beta samples by group type
    temp_precip_betas = []
    static_betas = []

    for group_name in group_names:
        beta_key = f'beta_{group_name}'
        if beta_key in prior_samples.prior:
            beta_samples = prior_samples.prior[beta_key].values.flatten()

            if 'temp' in group_name or 'precip' in group_name:
                temp_precip_betas.extend(beta_samples)
            else:
                static_betas.extend(beta_samples)

    # Plot temperature/precipitation beta priors
    ax5 = fig.add_subplot(gs[2, 0])
    if temp_precip_betas:
        ax5.hist(temp_precip_betas, bins=50, alpha=0.7, color='orange', edgecolor='black')
        ax5.axvline(0, color='red', linestyle='--', linewidth=2, label='Prior mean (0)')
        ax5.set_xlabel('Coefficient value', fontsize=11)
        ax5.set_ylabel('Frequency', fontsize=11)
        ax5.set_title('Prior: Climate Feature Coefficients ~ Normal(0, 2)',
                      fontsize=12, fontweight='bold')
        ax5.legend()
        ax5.grid(True, alpha=0.3)

    # Plot static feature beta priors
    ax6 = fig.add_subplot(gs[2, 1])
    if static_betas:
        ax6.hist(static_betas, bins=50, alpha=0.7, color='purple', edgecolor='black')
        ax6.axvline(0, color='red', linestyle='--', linewidth=2, label='Prior mean (0)')
        ax6.set_xlabel('Coefficient value', fontsize=11)
        ax6.set_ylabel('Frequency', fontsize=11)
        ax6.set_title('Prior: Static Feature Coefficients ~ Normal(0, 1.5)',
                      fontsize=12, fontweight='bold')
        ax6.legend()
        ax6.grid(True, alpha=0.3)

    # Summary statistics table
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')

    summary_text = f"""
    PRIOR ASSUMPTIONS SUMMARY

    Intercept (α):
      Distribution: Normal(0, 2)
      Interpretation: Neutral baseline

    Attention Weights:
      Distribution: Dirichlet(1.5) × HalfNormal(5)
      Interpretation: Flexible weighting
      Expected equal weight: ~{1.0:.2f}

    Climate Coefficients (β):
      Distribution: Normal(0, 2)
      Interpretation: Uninformative, wide

    Static Coefficients (β):
      Distribution: Normal(0, 1.5)
      Interpretation: Uninformative, moderate

    Philosophy:
      • Weakly informative priors
      • Allow data to dominate
      • Regularization without constraint
    """

    ax7.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
             verticalalignment='center', bbox=dict(boxstyle='round',
             facecolor='wheat', alpha=0.3))

    plt.suptitle('Bayesian Model: Prior Distributions',
                 fontsize=16, fontweight='bold', y=0.995)

    # Save figure
    output_path = os.path.join(output_dir, 'prior_distributions.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✓ Prior visualization saved: {output_path}")

    # Print numerical summary
    print("\n📋 PRIOR STATISTICS:")
    print(f"  Intercept (α): mean={alpha_prior.mean():.3f}, std={alpha_prior.std():.3f}")
    print(f"  Baseline prob: mean={baseline_prob.mean():.3f}, median={np.median(baseline_prob):.3f}")
    print(f"  Attention scale: mean={attn_scale_prior.mean():.3f}, std={attn_scale_prior.std():.3f}")
    print(f"  Climate β: mean={np.mean(temp_precip_betas):.3f}, std={np.std(temp_precip_betas):.3f}")
    print(f"  Static β: mean={np.mean(static_betas):.3f}, std={np.std(static_betas):.3f}")

# ===================================================================
# PREDICTION GENERATION (same as relative model)
# ===================================================================

def generate_predictions(trace, temporal_groups, group_names, X_scaled):
    """
    Generate relative probability predictions with uncertainty.

    Args:
        trace: PyMC trace object
        temporal_groups: Dictionary of feature groups
        group_names: List of group names
        X_scaled: Scaled feature matrix

    Returns:
        tuple: (mean_probability, std_probability)
            - mean_probability: Average predicted probability across posterior samples
            - std_probability: Standard deviation (uncertainty) across posterior samples
    """
    print("Generating relative probability predictions with uncertainty...")

    n_samples = 300
    alpha_samples = trace.posterior['alpha'].values.reshape(-1)[:n_samples]
    attention_samples = trace.posterior['attention_weights'].values.reshape(-1, len(group_names))[:n_samples]

    group_betas = {}
    for group_name in temporal_groups.keys():
        beta_key = f'beta_{group_name}'
        if beta_key in trace.posterior:
            beta_samples = trace.posterior[beta_key].values
            beta_flat = beta_samples.reshape(-1, beta_samples.shape[-1])[:n_samples]
            group_betas[group_name] = beta_flat

    n_test = X_scaled.shape[0]
    prob_predictions = []

    for sample_idx in range(n_samples):
        logit_pred = np.full(n_test, alpha_samples[sample_idx])

        for group_idx, (group_name, feature_indices) in enumerate(temporal_groups.items()):
            if group_name in group_betas:
                beta_sample = group_betas[group_name][sample_idx]
                group_features_data = X_scaled[:, feature_indices]
                group_contrib = np.dot(group_features_data, beta_sample)
                attention_weight = attention_samples[sample_idx, group_idx]
                weighted_contrib = attention_weight * group_contrib
                logit_pred += weighted_contrib

        prob_pred = 1 / (1 + np.exp(-logit_pred))
        prob_predictions.append(prob_pred)

    prob_predictions = np.array(prob_predictions)
    return prob_predictions.mean(axis=0), prob_predictions.std(axis=0)

# ===================================================================
# VALIDATION: TEMPORAL (by month/season)
# ===================================================================

def validate_temporal_fire_counts(y_test, mean_prob, test_dates):
    """
    Validate predictions against actual historical fire counts by month/season.

    Compares:
    - Predicted number of fires per month/season
    - Actual observed fire counts
    """
    print("\n=== TEMPORAL VALIDATION: FIRE COUNTS BY MONTH ===")

    # Convert dates to datetime if needed
    if isinstance(test_dates, pd.Series):
        test_dates_dt = pd.to_datetime(test_dates.values)
    else:
        test_dates_dt = pd.to_datetime(test_dates)

    # Create dataframe with predictions and actual outcomes
    df = pd.DataFrame({
        'date': test_dates_dt,
        'actual_fire': y_test,
        'predicted_prob': mean_prob
    })
    df['month'] = df['date'].dt.month
    df['season'] = df['date'].dt.month.map({
        12: 'Winter', 1: 'Winter', 2: 'Winter',
        3: 'Spring', 4: 'Spring', 5: 'Spring',
        6: 'Summer', 7: 'Summer', 8: 'Summer',
        9: 'Fall', 10: 'Fall', 11: 'Fall'
    })

    # Monthly comparison
    monthly_stats = df.groupby('month').agg({
        'actual_fire': 'sum',
        'predicted_prob': 'sum'
    }).reset_index()
    monthly_stats.columns = ['month', 'actual_fires', 'predicted_fires']

    # Seasonal comparison
    seasonal_stats = df.groupby('season').agg({
        'actual_fire': 'sum',
        'predicted_prob': 'sum'
    }).reset_index()
    seasonal_stats.columns = ['season', 'actual_fires', 'predicted_fires']

    print("\n📅 MONTHLY FIRE COUNTS:")
    print(monthly_stats.to_string(index=False))

    print("\n📅 SEASONAL FIRE COUNTS:")
    print(seasonal_stats.to_string(index=False))

    # Calculate correlation (Pearson R)
    monthly_corr = monthly_stats[['actual_fires', 'predicted_fires']].corr().iloc[0, 1]
    seasonal_corr = seasonal_stats[['actual_fires', 'predicted_fires']].corr().iloc[0, 1]

    # Calculate R² (coefficient of determination) - checks magnitude match
    from sklearn.metrics import r2_score
    monthly_r2 = r2_score(monthly_stats['actual_fires'], monthly_stats['predicted_fires'])
    seasonal_r2 = r2_score(seasonal_stats['actual_fires'], seasonal_stats['predicted_fires'])

    # Calculate RMSE (Root Mean Squared Error) - checks absolute error
    from sklearn.metrics import mean_squared_error
    monthly_rmse = np.sqrt(mean_squared_error(monthly_stats['actual_fires'], monthly_stats['predicted_fires']))
    seasonal_rmse = np.sqrt(mean_squared_error(seasonal_stats['actual_fires'], seasonal_stats['predicted_fires']))

    # Calculate MAE (Mean Absolute Error) - easier to interpret
    from sklearn.metrics import mean_absolute_error
    monthly_mae = mean_absolute_error(monthly_stats['actual_fires'], monthly_stats['predicted_fires'])
    seasonal_mae = mean_absolute_error(seasonal_stats['actual_fires'], seasonal_stats['predicted_fires'])

    print(f"\n📊 TEMPORAL VALIDATION METRICS:")
    print(f"\n  MONTHLY:")
    print(f"    Pearson R (trend): {monthly_corr:.3f}")
    print(f"    R² (magnitude fit): {monthly_r2:.3f}")
    print(f"    RMSE: {monthly_rmse:.2f} fires/month")
    print(f"    MAE: {monthly_mae:.2f} fires/month")

    print(f"\n  SEASONAL:")
    print(f"    Pearson R (trend): {seasonal_corr:.3f}")
    print(f"    R² (magnitude fit): {seasonal_r2:.3f}")
    print(f"    RMSE: {seasonal_rmse:.2f} fires/season")
    print(f"    MAE: {seasonal_mae:.2f} fires/season")

    # Check if there's a systematic scaling issue (expected for relative probability)
    print(f"\n  🔍 SCALING ANALYSIS:")

    # Calculate optimal scaling factor
    monthly_scale = monthly_stats['actual_fires'].sum() / monthly_stats['predicted_fires'].sum()
    seasonal_scale = seasonal_stats['actual_fires'].sum() / seasonal_stats['predicted_fires'].sum()

    print(f"    Optimal scaling factor (monthly): {monthly_scale:.3f}")
    print(f"    Optimal scaling factor (seasonal): {seasonal_scale:.3f}")

    # Apply scaling and recalculate R²
    monthly_stats['predicted_fires_scaled'] = monthly_stats['predicted_fires'] * monthly_scale
    seasonal_stats['predicted_fires_scaled'] = seasonal_stats['predicted_fires'] * seasonal_scale

    monthly_r2_scaled = r2_score(monthly_stats['actual_fires'], monthly_stats['predicted_fires_scaled'])
    seasonal_r2_scaled = r2_score(seasonal_stats['actual_fires'], seasonal_stats['predicted_fires_scaled'])

    monthly_mae_scaled = mean_absolute_error(monthly_stats['actual_fires'], monthly_stats['predicted_fires_scaled'])
    seasonal_mae_scaled = mean_absolute_error(seasonal_stats['actual_fires'], seasonal_stats['predicted_fires_scaled'])

    print(f"\n    After optimal scaling:")
    print(f"      Monthly R²: {monthly_r2:.3f} → {monthly_r2_scaled:.3f} (Δ={monthly_r2_scaled-monthly_r2:+.3f})")
    print(f"      Monthly MAE: {monthly_mae:.2f} → {monthly_mae_scaled:.2f} (Δ={monthly_mae_scaled-monthly_mae:+.2f})")
    print(f"      Seasonal R²: {seasonal_r2:.3f} → {seasonal_r2_scaled:.3f} (Δ={seasonal_r2_scaled-seasonal_r2:+.3f})")
    print(f"      Seasonal MAE: {seasonal_mae:.2f} → {seasonal_mae_scaled:.2f} (Δ={seasonal_mae_scaled-seasonal_mae:+.2f})")

    # Interpretation guide
    print(f"\n  📖 INTERPRETATION:")
    print(f"    - Pearson R: Measures correlation (trend match)")
    print(f"    - R²: Measures how well magnitude matches (1.0 = perfect)")
    print(f"    - RMSE/MAE: Average prediction error in fire counts")
    print(f"    - Scaling factor: How much to multiply predictions to match actual counts")
    print(f"    - If R² improves a lot after scaling: Model has RIGHT PATTERN, WRONG SCALE")
    print(f"      → This is EXPECTED for relative probability models on case-control data")

    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Monthly bar plot
    ax = axes[0, 0]
    x = monthly_stats['month']
    width = 0.35
    ax.bar(x - width/2, monthly_stats['actual_fires'], width, label='Actual', alpha=0.7, color='darkred')
    ax.bar(x + width/2, monthly_stats['predicted_fires'], width, label='Relative Risk (scaled)', alpha=0.7, color='steelblue')
    ax.set_xlabel('Month')
    ax.set_ylabel('Fire Count / Relative Risk')
    ax.set_title(f'Monthly Temporal Pattern (R={monthly_corr:.3f}, R²={monthly_r2:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Monthly scatter plot (actual vs predicted)
    ax = axes[0, 1]
    ax.scatter(monthly_stats['actual_fires'], monthly_stats['predicted_fires'], s=100, alpha=0.7, color='steelblue')
    # Add diagonal line (perfect fit)
    max_val = max(monthly_stats['actual_fires'].max(), monthly_stats['predicted_fires'].max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect fit')
    # Add labels for each month
    for idx, row in monthly_stats.iterrows():
        ax.annotate(f"{int(row['month'])}",
                   (row['actual_fires'], row['predicted_fires']),
                   xytext=(5, 5), textcoords='offset points', fontsize=8)
    ax.set_xlabel('Actual Fires')
    ax.set_ylabel('Predicted Fires')
    ax.set_title(f'Monthly: Actual vs Predicted (R²={monthly_r2:.3f}, MAE={monthly_mae:.1f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Seasonal bar plot
    ax = axes[1, 0]
    x_pos = np.arange(len(seasonal_stats))
    ax.bar(x_pos - width/2, seasonal_stats['actual_fires'], width, label='Actual', alpha=0.7, color='darkred')
    ax.bar(x_pos + width/2, seasonal_stats['predicted_fires'], width, label='Predicted', alpha=0.7, color='steelblue')
    ax.set_xlabel('Season')
    ax.set_ylabel('Number of Fires')
    ax.set_title(f'Seasonal Fire Counts (R={seasonal_corr:.3f}, R²={seasonal_r2:.3f})')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(seasonal_stats['season'])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Seasonal scatter plot (actual vs predicted)
    ax = axes[1, 1]
    ax.scatter(seasonal_stats['actual_fires'], seasonal_stats['predicted_fires'], s=150, alpha=0.7, color='steelblue')
    # Add diagonal line (perfect fit)
    max_val = max(seasonal_stats['actual_fires'].max(), seasonal_stats['predicted_fires'].max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect fit')
    # Add labels for each season
    for idx, row in seasonal_stats.iterrows():
        ax.annotate(row['season'],
                   (row['actual_fires'], row['predicted_fires']),
                   xytext=(5, 5), textcoords='offset points', fontsize=9, fontweight='bold')
    ax.set_xlabel('Actual Fires')
    ax.set_ylabel('Predicted Fires')
    ax.set_title(f'Seasonal: Actual vs Predicted (R²={seasonal_r2:.3f}, MAE={seasonal_mae:.1f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(ML_OUTPUT_DIR, "temporal_validation.png"), dpi=300)
    plt.close()

    return {'monthly': monthly_stats, 'seasonal': seasonal_stats,
            'monthly_corr': monthly_corr, 'seasonal_corr': seasonal_corr,
            'monthly_r2': monthly_r2, 'seasonal_r2': seasonal_r2,
            'monthly_rmse': monthly_rmse, 'seasonal_rmse': seasonal_rmse,
            'monthly_mae': monthly_mae, 'seasonal_mae': seasonal_mae,
            'monthly_scale': monthly_scale, 'seasonal_scale': seasonal_scale,
            'monthly_r2_scaled': monthly_r2_scaled, 'seasonal_r2_scaled': seasonal_r2_scaled,
            'monthly_mae_scaled': monthly_mae_scaled, 'seasonal_mae_scaled': seasonal_mae_scaled}

# ===================================================================
# NOTE: Absolute calibration removed - it degraded performance
# The model outputs relative probabilities only, which are reliable for:
# - Ranking fire risk across space and time
# - Comparing scenarios (e.g., climate change impacts)
# - Identifying high-risk periods and locations
# ===================================================================

# ===================================================================
# VALIDATION: COMPREHENSIVE METRICS
# ===================================================================

def create_relative_probability_validation_plots(y_test, mean_prob, test_dates):
    """
    Create comprehensive validation plots for relative probability predictions.

    Includes:
    1. ROC Curve (discrimination ability)
    2. Precision-Recall Curve
    3. Calibration Plot (reliability)
    4. Lift Curve (targeting effectiveness)
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle("Relative Probability Model Validation", fontsize=16)

    n_positive = int(np.sum(y_test))
    n_negative = len(y_test) - n_positive

    # PLOT 1: ROC CURVE
    fpr, tpr, _ = roc_curve(y_test, mean_prob)
    roc_auc = auc(fpr, tpr)
    axes[0, 0].plot(fpr, tpr, color='blue', lw=2, label=f'Model (AUC = {roc_auc:.3f})')
    axes[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    axes[0, 0].set_xlabel('False Positive Rate')
    axes[0, 0].set_ylabel('True Positive Rate')
    axes[0, 0].set_title('ROC Curve (Discrimination)')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # PLOT 2: PRECISION-RECALL CURVE
    precision, recall, thresholds = precision_recall_curve(y_test, mean_prob)
    pr_auc = average_precision_score(y_test, mean_prob)
    axes[0, 1].plot(recall, precision, color='darkorange', lw=2, label=f'Model (AP = {pr_auc:.3f})')
    no_skill = n_positive / len(y_test)
    axes[0, 1].plot([0, 1], [no_skill, no_skill], color='navy', lw=2, linestyle='--',
                    label=f'Baseline (Prevalence={no_skill:.3f})')

    # Mark optimal F1 threshold
    f1_scores = np.nan_to_num(2 * recall[:-1] * precision[:-1] / (recall[:-1] + precision[:-1]))
    optimal_idx = np.argmax(f1_scores)
    max_f1 = f1_scores[optimal_idx]
    optimal_threshold = thresholds[optimal_idx]
    axes[0, 1].scatter(recall[optimal_idx], precision[optimal_idx], marker='*', color='red',
                      s=150, zorder=5, label=f'Max F1 ({max_f1:.3f}) @ {optimal_threshold:.4f}')

    axes[0, 1].set_xlabel('Recall')
    axes[0, 1].set_ylabel('Precision')
    axes[0, 1].set_title('Precision-Recall Curve')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # PLOT 3: CALIBRATION PLOT
    n_bins = 10
    bin_edges = np.linspace(0, mean_prob.max(), n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    observed_freq = []
    predicted_freq = []
    bin_counts = []

    for i in range(n_bins):
        mask = (mean_prob >= bin_edges[i]) & (mean_prob < bin_edges[i+1])
        if i == n_bins - 1:
            mask = (mean_prob >= bin_edges[i]) & (mean_prob <= bin_edges[i+1])

        if np.sum(mask) > 0:
            observed_freq.append(np.mean(y_test[mask]))
            predicted_freq.append(np.mean(mean_prob[mask]))
            bin_counts.append(np.sum(mask))
        else:
            observed_freq.append(np.nan)
            predicted_freq.append(bin_centers[i])
            bin_counts.append(0)

    valid_mask = ~np.isnan(observed_freq)
    axes[1, 0].plot(np.array(predicted_freq)[valid_mask], np.array(observed_freq)[valid_mask],
                   marker='o', color='darkgreen', lw=2, markersize=8, label='Model Calibration')
    axes[1, 0].plot([0, mean_prob.max()], [0, mean_prob.max()], color='navy', lw=2,
                   linestyle='--', label='Perfect Calibration')
    axes[1, 0].set_xlabel('Predicted Probability')
    axes[1, 0].set_ylabel('Observed Frequency')
    axes[1, 0].set_title('Calibration Plot (Reliability)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)

    # PLOT 4: LIFT CURVE
    # Sort predictions descending
    sorted_indices = np.argsort(mean_prob)[::-1]
    sorted_actuals = y_test.values[sorted_indices] if hasattr(y_test, 'values') else y_test[sorted_indices]

    # Calculate cumulative lift
    n_samples = len(sorted_actuals)
    percentiles = np.arange(1, 101)
    cumulative_positives = []
    baseline_positives = []

    for pct in percentiles:
        n = int(n_samples * pct / 100)
        cumulative_positives.append(sorted_actuals[:n].sum())
        baseline_positives.append(n_positive * pct / 100)

    lift = np.array(cumulative_positives) / np.array(baseline_positives)

    axes[1, 1].plot(percentiles, lift, color='purple', lw=2, label='Model Lift')
    axes[1, 1].axhline(y=1, color='navy', lw=2, linestyle='--', label='Baseline (Random)')
    axes[1, 1].set_xlabel('Percentage of Population Targeted (%)')
    axes[1, 1].set_ylabel('Lift')
    axes[1, 1].set_title('Lift Curve (Targeting Effectiveness)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Add annotation for top 10% lift
    lift_10pct = lift[9]  # 10% index
    axes[1, 1].annotate(f'Top 10%: {lift_10pct:.2f}x',
                       xy=(10, lift_10pct), xytext=(25, lift_10pct),
                       arrowprops=dict(arrowstyle='->', color='red'),
                       fontsize=10, color='red')

    plt.tight_layout()
    plt.savefig(os.path.join(ML_OUTPUT_DIR, "relative_probability_validation.png"), dpi=300)
    plt.close()

    print(f"\n📊 VALIDATION METRICS:")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  PR-AUC (Average Precision): {pr_auc:.4f}")
    print(f"  Max F1 Score: {max_f1:.4f} @ threshold {optimal_threshold:.4f}")
    print(f"  Lift at 10%: {lift_10pct:.2f}x baseline")

    return {
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'max_f1': max_f1,
        'optimal_threshold': optimal_threshold,
        'lift_10pct': lift_10pct
    }

# ===================================================================
# MAIN EXECUTION
# ===================================================================

def run_relative_probability_analysis():
    """Execute complete analysis pipeline for relative probability prediction with uncertainty."""

    print("="*70)
    print("BAYESIAN RELATIVE PROBABILITY MODEL FOR BOLZANO WILDFIRE PREDICTION")
    print("="*70)

    # STEP 1: Load training data and extract features
    print("\n=== LOADING TRAINING DATA ===")
    with xr.open_dataset(NETCDF_PATH) as ds:
        main_data_var = list(ds.data_vars)[0]
        y = ds['label'].to_pandas()

        # Get dates for temporal validation
        dates = pd.to_datetime(ds['event_date'].values)

        print(f"Dataset: {len(ds.id_obs)} observations")
        print(f"Fire rate in training: {y.mean():.4f} (case-control sample)")

        print("\nExtracting features...")
        feature_list = []
        for i, obs_id in enumerate(ds['id_obs'].values):
            if i % 100 == 0:
                print(f"  Processing {i+1}/{len(ds.id_obs)}")
            feature_list.append(create_cumulative_features(ds[main_data_var].sel(id_obs=obs_id)))

        X = pd.DataFrame(feature_list, index=y.index).fillna(0)
        print(f"Feature matrix: {X.shape}")

    # STEP 2: Train/test split
    X_train, X_test, y_train, y_test, dates_train, dates_test = train_test_split(
        X, y, dates, test_size=0.3, random_state=42, stratify=y
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    feature_names = X_train.columns

    # STEP 3: Create and train model
    print("\n=== TRAINING BAYESIAN RELATIVE PROBABILITY MODEL ===")
    model, temporal_groups, group_names = create_relative_probability_model(
        X_train_scaled, y_train.values, feature_names
    )

    # Visualize prior distributions before training
    visualize_priors(model, temporal_groups, group_names, ML_OUTPUT_DIR)

    with model:
        trace = pm.sample(draws=2000, tune=1000, chains=4, cores=4,
                         target_accept=0.99, return_inferencedata=True)

    print("✓ Model training complete!")

    # Check what the model learned
    learned_alpha = trace.posterior['alpha'].mean().item()
    learned_prob = 1 / (1 + np.exp(-learned_alpha))

    print(f"\n📊 MODEL LEARNED BASELINE:")
    print(f"  Learned intercept: {learned_alpha:.4f} logit = {learned_prob:.4f} prob")
    print(f"  Training fire rate: {y_train.mean():.4f}")

    # STEP 4: Generate predictions with uncertainty
    mean_prob, std_prob = generate_predictions(
        trace, temporal_groups, group_names, X_test_scaled
    )

    # Check predictions
    print(f"\n📊 PREDICTION SUMMARY:")
    print(f"  Mean predicted probability: {mean_prob.mean():.4f}")
    print(f"  Median predicted probability: {np.median(mean_prob):.4f}")
    print(f"  Prediction range: [{mean_prob.min():.4f}, {mean_prob.max():.4f}]")
    print(f"  Mean uncertainty (std): {std_prob.mean():.4f}")
    print(f"  High uncertainty cases (std > 0.15): {(std_prob > 0.15).sum()}")

    # STEP 5: Comprehensive validation
    print("\n=== VALIDATION ===")

    # Temporal validation (fire counts by month/season)
    # Note: For relative probability, temporal patterns still meaningful
    temporal_results = validate_temporal_fire_counts(
        y_test.values, mean_prob, dates_test
    )

    # Performance metrics (ROC, PR, Calibration, Lift)
    validation_results = create_relative_probability_validation_plots(
        y_test.values, mean_prob, dates_test
    )

    # STEP 6: Save model artifacts
    print("\n=== SAVING RESULTS ===")
    az.to_netcdf(trace, os.path.join(ML_OUTPUT_DIR, "trace_relative.nc"))
    joblib.dump(scaler, os.path.join(ML_OUTPUT_DIR, "scaler_relative.joblib"))
    joblib.dump(temporal_groups, os.path.join(ML_OUTPUT_DIR, "temporal_groups.joblib"))
    joblib.dump(group_names, os.path.join(ML_OUTPUT_DIR, "group_names.joblib"))

    # Save test predictions with uncertainty (relative probabilities only)
    test_predictions = pd.DataFrame({
        'mean_probability': mean_prob,
        'std_probability': std_prob,
        'actual_fire': y_test.values,
        'date': dates_test.values
    })
    test_predictions.to_csv(os.path.join(ML_OUTPUT_DIR, "test_predictions_with_uncertainty.csv"), index=False)
    print(f"✓ Test predictions with uncertainty saved ({len(test_predictions)} observations)")

    # Save comprehensive results
    results = {
        'validation_metrics': validation_results,
        'temporal_validation': temporal_results,
        'training_fire_rate': y_train.mean(),
        'test_predictions': {
            'mean_prob': mean_prob,
            'std_prob': std_prob,
            'y_test': y_test.values,
            'dates_test': dates_test.values
        }
    }
    joblib.dump(results, os.path.join(ML_OUTPUT_DIR, "model_results.joblib"))

    print(f"\n✓ All results saved to: {ML_OUTPUT_DIR}")

    return results

# ===================================================================
# SCRIPT EXECUTION
# ===================================================================

if __name__ == "__main__":
    results = run_relative_probability_analysis()
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)