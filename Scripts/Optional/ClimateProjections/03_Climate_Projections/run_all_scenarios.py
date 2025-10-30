"""
Orchestration Script for Climate Projection Analysis

Runs the complete pipeline for all scenarios:
1. Extract features for each scenario
2. Generate predictions using trained model
3. Create visualizations

This script iterates through all TARGET_SCENARIO configurations.
"""

import os
import sys
import pandas as pd
import numpy as np
import joblib
import xarray as xr
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import scenario configuration
from config_scenarios import SCENARIOS, get_scenario_output_dir, get_projection_dates
from extract_projection_features import create_spatial_grid, extract_features_for_scenario


# Trained model paths
MODEL_DIR = "/mnt/CEPH_PROJECTS/Firescape/Scripts/OUTPUT/02_Model_RelativeProbability/"
TRACE_PATH = os.path.join(MODEL_DIR, "trace_relative.nc")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_relative.joblib")
TEMPORAL_GROUPS_PATH = os.path.join(MODEL_DIR, "temporal_groups.joblib")
GROUP_NAMES_PATH = os.path.join(MODEL_DIR, "group_names.joblib")


def load_trained_model():
    """Load trained PyMC model artifacts."""
    print("\n📦 Loading trained model artifacts...")

    # Import PyMC and ArviZ
    import arviz as az

    # Load trace
    if not os.path.exists(TRACE_PATH):
        raise FileNotFoundError(f"Trace not found: {TRACE_PATH}")

    trace = az.from_netcdf(TRACE_PATH)
    print(f"   ✓ Trace loaded: {TRACE_PATH}")

    # Load scaler
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(f"Scaler not found: {SCALER_PATH}")

    scaler = joblib.load(SCALER_PATH)
    print(f"   ✓ Scaler loaded: {SCALER_PATH}")

    # Load temporal groups
    if not os.path.exists(TEMPORAL_GROUPS_PATH):
        raise FileNotFoundError(f"Temporal groups not found: {TEMPORAL_GROUPS_PATH}")

    temporal_groups = joblib.load(TEMPORAL_GROUPS_PATH)
    group_names = joblib.load(GROUP_NAMES_PATH)
    print(f"   ✓ Temporal groups loaded")

    return trace, scaler, temporal_groups, group_names


def generate_predictions(trace, scaler, temporal_groups, group_names, features_df):
    """
    Generate predictions for new data using trained model.

    Args:
        trace: PyMC trace
        scaler: Fitted StandardScaler
        temporal_groups: Feature group dictionary
        group_names: List of group names
        features_df: DataFrame with extracted features

    Returns:
        DataFrame with predictions (mean and std)
    """
    print("\n🔮 Generating predictions...")

    # Get feature columns (exclude metadata)
    metadata_cols = ['scenario', 'year', 'month', 'day', 'date', 'grid_id', 'x', 'y']
    feature_cols = [col for col in features_df.columns if col not in metadata_cols]

    # Extract features and scale
    X = features_df[feature_cols].values
    X_scaled = scaler.transform(X)

    # Generate predictions using posterior samples
    n_samples = 300  # Number of posterior samples to use
    alpha_samples = trace.posterior['alpha'].values.reshape(-1)[:n_samples]
    attention_samples = trace.posterior['attention_weights'].values.reshape(-1, len(group_names))[:n_samples]

    # Collect beta samples
    group_betas = {}
    for group_name in group_names:
        beta_key = f'beta_{group_name}'
        if beta_key in trace.posterior:
            beta_samples = trace.posterior[beta_key].values
            beta_flat = beta_samples.reshape(-1, beta_samples.shape[-1])[:n_samples]
            group_betas[group_name] = beta_flat

    n_obs = X_scaled.shape[0]
    prob_predictions = []

    print(f"   Processing {n_obs} observations with {n_samples} posterior samples...")

    for sample_idx in tqdm(range(n_samples), desc="Posterior sampling"):
        logit_pred = np.full(n_obs, alpha_samples[sample_idx])

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

    # Compute mean and std across posterior samples
    mean_risk = prob_predictions.mean(axis=0)
    std_risk = prob_predictions.std(axis=0)

    # Add predictions to DataFrame
    predictions_df = features_df[metadata_cols].copy()
    predictions_df['mean_risk'] = mean_risk
    predictions_df['std_risk'] = std_risk

    print(f"   ✓ Predictions generated")
    print(f"      Mean risk: {mean_risk.mean():.4f} ± {mean_risk.std():.4f}")
    print(f"      Risk range: [{mean_risk.min():.4f}, {mean_risk.max():.4f}]")

    return predictions_df


def process_scenario(scenario, trace, scaler, temporal_groups, group_names, spatial_grid):
    """
    Process a single scenario: extract features and generate predictions.

    Args:
        scenario: ClimateScenario object
        trace, scaler, temporal_groups, group_names: Model artifacts
        spatial_grid: GeoDataFrame with grid points

    Returns:
        DataFrame with predictions
    """
    print(f"\n{'='*80}")
    print(f"PROCESSING SCENARIO: {scenario.name}")
    print(f"{'='*80}")
    print(f"Description: {scenario.description}")
    print(f"Period: {scenario.period[0]}-{scenario.period[1]}")

    output_dir = get_scenario_output_dir(scenario.name)

    # Check if predictions already exist
    pred_path = os.path.join(output_dir, f"predictions_{scenario.name}.csv")
    if os.path.exists(pred_path):
        print(f"\n✓ Predictions already exist: {pred_path}")
        print("   Skipping. Delete file to reprocess.")
        return pd.read_csv(pred_path, parse_dates=['date'])

    # Step 1: Extract features
    features_path = os.path.join(output_dir, f"features_{scenario.name}.csv")

    if not os.path.exists(features_path):
        print("\n📊 Extracting features...")
        target_dates = get_projection_dates(scenario, months=[6, 7, 8, 9])

        try:
            features_df = extract_features_for_scenario(
                scenario, spatial_grid, target_dates, features_path
            )
        except Exception as e:
            print(f"\n❌ ERROR extracting features: {e}")
            import traceback
            traceback.print_exc()
            return None
    else:
        print(f"\n✓ Features already extracted: {features_path}")
        features_df = pd.read_csv(features_path, parse_dates=['date'])

    # Step 2: Generate predictions
    print("\n🔮 Generating predictions...")

    try:
        predictions_df = generate_predictions(
            trace, scaler, temporal_groups, group_names, features_df
        )

        # Save predictions
        predictions_df.to_csv(pred_path, index=False)
        print(f"\n✓ Predictions saved: {pred_path}")

        return predictions_df

    except Exception as e:
        print(f"\n❌ ERROR generating predictions: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main execution function."""
    print("="*80)
    print("CLIMATE PROJECTION ANALYSIS - ALL SCENARIOS")
    print("="*80)

    # Load trained model
    try:
        trace, scaler, temporal_groups, group_names = load_trained_model()
    except Exception as e:
        print(f"\n❌ ERROR loading model: {e}")
        print("\nPlease ensure the model has been trained first:")
        print(f"   Run: python ../02_Model_Training/train_relative_probability_model.py")
        sys.exit(1)

    # Create spatial grid (reusable across all scenarios)
    print("\n📍 Creating spatial grid...")
    spatial_grid = create_spatial_grid(bounds=None, resolution=1000)

    # Process each scenario
    results = {}

    for scenario in SCENARIOS:
        # Skip historical (used for training)
        if scenario.name == "historical":
            print(f"\n⏭️  Skipping historical scenario (used for training)")
            continue

        # Process scenario
        predictions_df = process_scenario(
            scenario, trace, scaler, temporal_groups, group_names, spatial_grid
        )

        if predictions_df is not None:
            results[scenario.name] = predictions_df

    # Summary
    print("\n" + "="*80)
    print("SCENARIO PROCESSING COMPLETE")
    print("="*80)

    print(f"\nSuccessfully processed scenarios: {len(results)}/{len(SCENARIOS)-1}")

    for scenario_name in results.keys():
        print(f"   ✓ {scenario_name}")

    if len(results) < len(SCENARIOS) - 1:
        failed = [s.name for s in SCENARIOS if s.name != 'historical' and s.name not in results]
        print(f"\nFailed scenarios: {len(failed)}")
        for scenario_name in failed:
            print(f"   ✗ {scenario_name}")

    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("Run visualization script to create plots:")
    print("   python visualize_risk_evolution.py")
    print("="*80)


if __name__ == "__main__":
    main()
