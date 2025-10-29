# ===================================================================
# BAYESIAN ATTENTION WILDFIRE MODEL - RELATIVE PROBABILITY WITH UNCERTAINTY
# (Revised for Full Time Series with Dask-PyMC)
# ===================================================================

# ===================================================================
# IMPORTS
# ===================================================================

import xarray as xr
import pandas as pd
import numpy as np
import os
import sys # Added for controlled exit
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from scipy import stats
from collections import Counter

# New compute_mode_robust function
def compute_mode_robust(data, **kwargs):
    values = data.flatten()
    values = values[~np.isnan(values)]
    if values.shape[0] == 0:
        return 0.0
    
    # If values is a Dask array, compute it to a NumPy array before bincount
    if isinstance(values, da.Array):
        values = values.compute() # Materialize Dask array to NumPy array
    
    try:
        int_values = values.astype(int)
        counts = np.bincount(int_values)
        mode = np.argmax(counts)
    except ValueError: # This ValueError might be from astype(int) if values are not convertible
        counts = Counter(values)
        mode = counts.most_common(1)[0][0]
        
    return float(mode) # Explicitly return a Python float

# --- PyMC & ArviZ for Bayesian Modeling ---
import pymc as pm
import arviz as az
import pytensor.tensor as pt

# --- Standard ML tools ---
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import joblib

# --- Dask-ML for Dask-compatible preprocessing ---
from dask_ml.preprocessing import StandardScaler as DaskStandardScaler

# --- Visualization tools ---
import seaborn as sns

# --- Dask for parallel computing ---
import dask.array as da
from dask.distributed import Client

print(f"Running Bayesian Relative Probability Model on PyMC v{pm.__version__}")

# ===================================================================
# CONFIGURATION
# ===================================================================

# Input data paths
NETCDF_PATH = "/mnt/CEPH_PROJECTS/Firescape/Scripts/OUTPUT/01_Training_Data/spacetime_stacks.nc"
WILDFIRE_HISTORY_PATH = "/mnt/CEPH_PROJECTS/Firescape/Data/WILDFIRE_INVENTORY/wildfire_point_Bolzano_Period1999_2025.gpkg"

# Output directory
ML_OUTPUT_DIR = "/mnt/CEPH_PROJECTS/Firescape/Scripts/OUTPUT/02_Model_RelativeProbability/DaskPyMC"
os.makedirs(ML_OUTPUT_DIR, exist_ok=True)

# Feature categories
STATIC_VARS = [
    'tri', 'northness', 'slope', 'aspect', 'nasadem',
    'treecoverdensity', 'landcoverfull', 'distroads',
    'eastness', 'flammability', 'walking_time_to_bldg',
    'walking_time_to_elec_infra'
]
DYNAMIC_VARS = ['T', 'P']
TIME_STEPS = 60
CHANNELS = STATIC_VARS + DYNAMIC_VARS

# ===================================================================
# MAIN EXECUTION
# ===================================================================

if __name__ == "__main__":
    # New: Start Dask client for parallel computation
    client = Client(n_workers=2, threads_per_worker=1)  # Adjusted to reduce memory footprint
    print(client) # http://127.0.0.1:8787/status 

    # ===================================================================
    # DATA LOADING WITH DASK
    # ===================================================================
    print("Loading data with Dask...")
    ds = xr.open_dataset(NETCDF_PATH, chunks={'id_obs': 500})  # Adjusted chunk size for memory efficiency
    main_data_var = list(ds.data_vars)[0]
    y = ds['label'].values  # Load labels into memory as they are small
    dates = pd.to_datetime(ds['event_date'].values)

    # Extract features as Dask arrays (lazy loading)
    # Spatial mean for dynamics; replicate statics over time
    dynamic_data = ds[main_data_var].sel(channel=DYNAMIC_VARS).mean(dim=['y', 'x'])  # id_obs x time x dynamic_channels
    static_continuous_vars = [v for v in STATIC_VARS if v != 'landcoverfull']
    static_cont_data = ds[main_data_var].sel(channel=static_continuous_vars).isel(time=0).mean(dim=['y', 'x'])  # id_obs x continuous_static_channels

    # Handle landcoverfull as categorical (majority vote / mode)
    # Handle landcoverfull as categorical (majority vote / mode)
    landcover_data_full = ds[main_data_var].sel(channel='landcoverfull').isel(time=0)  # id_obs x y x x

    # Group by 'id_obs' and apply the reduction
    # This will apply compute_mode_robust to each (y, x) slice for each id_obs
    landcover_mode = landcover_data_full.groupby('id_obs').apply(
        lambda x: xr.DataArray(compute_mode_robust(x.data),             coords={'id_obs': x['id_obs'].item()}, dims=[]) # Return a DataArray
    )
    landcover_mode = landcover_mode.expand_dims(channel=1)
    landcover_mode = landcover_mode.assign_coords(channel=['landcoverfull'])

    # Combine static data: continuous means + landcover mode
    static_data = xr.concat([static_cont_data, landcover_mode], dim='channel')
    
    # Explicitly compute static_data to a NumPy array to isolate potential Dask graph issues
    static_data_np = static_data.compute()

    static_replicated = da.tile(static_data_np.values[:, None, :], (1, TIME_STEPS, 1))  # id_obs x time x static_channels (Dask array)
    dynamic_dask = dynamic_data.values  # Already a Dask array

    # Combine into a single features array: id_obs x time x total_channels
    features_dask = da.concatenate([static_replicated,
                                    dynamic_dask], axis=2)  # Preserve chunks
    y_dask = da.from_array(y, chunks=(100,))  # Chunked labels for consistency

    print(f"Features shape: {features_dask.shape} (Dask chunks: {features_dask.chunks})")

    # Train/test split (using indices, as Dask arrays are lazy)
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=42, stratify=y)
    X_train_dask = features_dask[train_idx]
    y_train_dask = y_dask[train_idx]
    X_test_dask = features_dask[test_idx]
    y_test = y[test_idx]  # Materialize test labels for validation
    dates_test = dates[test_idx]

    # Standardize features (fit on train, apply to both; now fully Dask-compatible)
    scaler = DaskStandardScaler()

    # Rechunk X_train_dask and X_test_dask to have a single chunk along the feature dimension
    # This is required by dask_ml.preprocessing.StandardScaler when reshaping for fitting
    X_train_dask_rechunked = X_train_dask.rechunk({2: -1})
    X_test_dask_rechunked = X_test_dask.rechunk({2: -1})

    scaler.fit(X_train_dask_rechunked.reshape(-1, len(CHANNELS)))  # Flatten time dimension for scaling
    X_train_scaled = scaler.transform(X_train_dask_rechunked.reshape(-1, len(CHANNELS))).reshape(X_train_dask.shape)
    X_test_scaled = scaler.transform(X_test_dask_rechunked.reshape(-1, len(CHANNELS))).reshape(X_test_dask.shape)

    # ===================================================================
    # BAYESIAN MODEL WITH FULL TIME SERIES AND TEMPORAL ATTENTION
    # ===================================================================
    n_obs, n_time, n_features = X_train_scaled.shape

    def create_relative_probability_model(X_scaled, y):
        with pm.Model() as model:
            # MODEL INTERCEPT - Neutral prior
            alpha = pm.Normal('alpha', mu=0, sigma=2.0)

            # BETAS FOR FEATURES - Loose priors
            betas = pm.Normal('betas', mu=0, sigma=3.0, shape=n_features)  # Shared across time

            # TEMPORAL ATTENTION - Weights over time steps
            attention_raw = pm.Dirichlet('time_attention', a=np.ones(n_time) * 1.5)
            attention_scale = pm.HalfNormal('attention_scale', sigma=5.0)
            time_attention = attention_raw * attention_scale

            # COMPUTE LOGIT: Sum over time with attention (PyTensor handles Dask arrays)
            contrib_per_time = pt.einsum('otf,f->ot', X_scaled, betas)  # id_obs x time
            logit_p = alpha + pt.dot(contrib_per_time, time_attention)  # Weighted sum over time

            # LIKELIHOOD
            pm.Bernoulli('likelihood', logit_p=logit_p, observed=y)

            # DERIVED QUANTITIES
            pm.Deterministic('model_baseline_prob', pm.math.sigmoid(alpha))

        return model

    print("\n=== TRAINING BAYESIAN RELATIVE PROBABILITY MODEL ===")
    model = create_relative_probability_model(X_train_scaled, y_train_dask)

    # Visualize priors (similar to original, but simplified for brevity)
    with model:
        prior_samples = pm.sample_prior_predictive(samples=5000, random_seed=42)
        # Add prior visualization code from original script if needed

    with model:
        trace = pm.sample(draws=2000, tune=1000, chains=4, cores=4,
                          target_accept=0.99, return_inferencedata=True)

    print("✓ Model training complete!")

    # ===================================================================
    # GENERATE PREDICTIONS WITH UNCERTAINTY
    # ===================================================================
    with model:
        post_pred = pm.sample_posterior_predictive(trace, predictions=True)
    mean_prob = post_pred.predictions['likelihood'].mean(dim=['chain', 'draw']).values  # Materialize
    std_prob = post_pred.predictions['likelihood'].std(dim=['chain', 'draw']).values

    # ===================================================================
    # VALIDATION: TEMPORAL FIRE COUNTS
    # ===================================================================
    def validate_temporal_fire_counts(y_test, mean_prob, test_dates):
        print("\n=== TEMPORAL VALIDATION (Monthly/Seasonal Fire Patterns) ===")
        
        test_df = pd.DataFrame({
            'actual_fire': y_test,
            'predicted_prob': mean_prob,
            'date': test_dates
        })
        test_df['month'] = test_df['date'].dt.month
        test_df['season'] = pd.cut(test_df['month'], bins=[0, 3, 6, 9, 12], labels=['Winter', 'Spring', 'Summer', 'Fall'])
        
        # Monthly stats
        monthly_stats = test_df.groupby('month').agg(
            actual_fires=('actual_fire', 'sum'),
            predicted_fires=('predicted_prob', 'sum')
        ).reset_index()
        
        # Seasonal stats
        seasonal_stats = test_df.groupby('season').agg(
            actual_fires=('actual_fire', 'sum'),
            predicted_fires=('predicted_prob', 'sum')
        ).reset_index()
        
        # Monthly metrics
        monthly_corr = monthly_stats['actual_fires'].corr(monthly_stats['predicted_fires'])
        monthly_r2 = 1 - np.sum((monthly_stats['actual_fires'] - monthly_stats['predicted_fires'])**2) / np.sum((monthly_stats['actual_fires'] - monthly_stats['actual_fires'].mean())**2)
        monthly_rmse = np.sqrt(np.mean((monthly_stats['actual_fires'] - monthly_stats['predicted_fires'])**2))
        monthly_mae = np.mean(np.abs(monthly_stats['actual_fires'] - monthly_stats['predicted_fires']))
        
        # Seasonal metrics
        seasonal_corr = seasonal_stats['actual_fires'].corr(seasonal_stats['predicted_fires'])
        seasonal_r2 = 1 - np.sum((seasonal_stats['actual_fires'] - seasonal_stats['predicted_fires'])**2) / np.sum((seasonal_stats['actual_fires'] - seasonal_stats['actual_fires'].mean())**2)
        seasonal_rmse = np.sqrt(np.mean((seasonal_stats['actual_fires'] - seasonal_stats['predicted_fires'])**2))
        seasonal_mae = np.mean(np.abs(seasonal_stats['actual_fires'] - seasonal_stats['predicted_fires']))
        
        # Scaling factors
        monthly_scale = monthly_stats['actual_fires'].sum() / monthly_stats['predicted_fires'].sum()
        seasonal_scale = seasonal_stats['actual_fires'].sum() / seasonal_stats['predicted_fires'].sum()
        
        monthly_stats['predicted_fires_scaled'] = monthly_stats['predicted_fires'] * monthly_scale
        seasonal_stats['predicted_fires_scaled'] = seasonal_stats['predicted_fires'] * seasonal_scale
        
        monthly_r2_scaled = 1 - np.sum((monthly_stats['actual_fires'] - monthly_stats['predicted_fires_scaled'])**2) / np.sum((monthly_stats['actual_fires'] - monthly_stats['actual_fires'].mean())**2)
        seasonal_r2_scaled = 1 - np.sum((seasonal_stats['actual_fires'] - seasonal_stats['predicted_fires_scaled'])**2) / np.sum((seasonal_stats['actual_fires'] - seasonal_stats['actual_fires'].mean())**2)
        monthly_mae_scaled = np.mean(np.abs(monthly_stats['actual_fires'] - monthly_stats['predicted_fires_scaled']))
        seasonal_mae_scaled = np.mean(np.abs(seasonal_stats['actual_fires'] - seasonal_stats['predicted_fires_scaled']))
        
        print(f"  Monthly Pearson R: {monthly_corr:.3f}")
        print(f"  Monthly R²: {monthly_r2:.3f} → {monthly_r2_scaled:.3f} (after scaling by {monthly_scale:.2f})")
        print(f"  Monthly MAE: {monthly_mae:.2f} → {monthly_mae_scaled:.2f}")
        print(f"  Seasonal Pearson R: {seasonal_corr:.3f}")
        print(f"  Seasonal R²: {seasonal_r2:.3f} → {seasonal_r2_scaled:.3f} (after scaling by {seasonal_scale:.2f})")
        print(f"  Seasonal MAE: {seasonal_mae:.2f} → {seasonal_mae_scaled:.2f}")
        
        # Plotting
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Monthly bar
        ax = axes[0, 0]
        x = monthly_stats['month']
        width = 0.35
        ax.bar(x - width/2, monthly_stats['actual_fires'], width, label='Actual', alpha=0.7, color='darkred')
        ax.bar(x + width/2, monthly_stats['predicted_fires_scaled'], width, label='Predicted (scaled)', alpha=0.7, color='steelblue')
        ax.set_title(f'Monthly Pattern (R={monthly_corr:.3f}, R²={monthly_r2_scaled:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Monthly scatter
        ax = axes[0, 1]
        ax.scatter(monthly_stats['actual_fires'], monthly_stats['predicted_fires_scaled'], s=100, alpha=0.7, color='steelblue')
        max_val = max(monthly_stats['actual_fires'].max(), monthly_stats['predicted_fires_scaled'].max())
        ax.plot([0, max_val], [0, max_val], 'r--', label='Perfect fit')
        ax.set_title(f'Monthly Actual vs Predicted (R²={monthly_r2_scaled:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Seasonal bar
        ax = axes[1, 0]
        x_pos = np.arange(len(seasonal_stats))
        ax.bar(x_pos - width/2, seasonal_stats['actual_fires'], width, label='Actual', alpha=0.7, color='darkred')
        ax.bar(x_pos + width/2, seasonal_stats['predicted_fires_scaled'], width, label='Predicted (scaled)', alpha=0.7, color='steelblue')
        ax.set_title(f'Seasonal Pattern (R={seasonal_corr:.3f}, R²={seasonal_r2_scaled:.3f})')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(seasonal_stats['season'])
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Seasonal scatter
        ax = axes[1, 1]
        ax.scatter(seasonal_stats['actual_fires'], seasonal_stats['predicted_fires_scaled'], s=150, alpha=0.7, color='steelblue')
        max_val = max(seasonal_stats['actual_fires'].max(), seasonal_stats['predicted_fires_scaled'].max())
        ax.plot([0, max_val], [0, max_val], 'r--', label='Perfect fit')
        ax.set_title(f'Seasonal Actual vs Predicted (R²={seasonal_r2_scaled:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(ML_OUTPUT_DIR, "temporal_validation.png"), dpi=300)
        plt.close()
        
        return {
            'monthly': monthly_stats,
            'seasonal': seasonal_stats,
            'monthly_corr': monthly_corr,
            'seasonal_corr': seasonal_corr,
            'monthly_r2': monthly_r2,
            'seasonal_r2': seasonal_r2,
            'monthly_rmse': monthly_rmse,
            'seasonal_rmse': seasonal_rmse,
            'monthly_mae': monthly_mae,
            'seasonal_mae': seasonal_mae,
            'monthly_scale': monthly_scale,
            'seasonal_scale': seasonal_scale,
            'monthly_r2_scaled': monthly_r2_scaled,
            'seasonal_r2_scaled': seasonal_r2_scaled,
            'monthly_mae_scaled': monthly_mae_scaled,
            'seasonal_mae_scaled': seasonal_mae_scaled
        }

    # ===================================================================
    # VALIDATION: COMPREHENSIVE METRICS
    # ===================================================================
    def create_relative_probability_validation_plots(y_test, mean_prob, test_dates):
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle("Relative Probability Model Validation", fontsize=16)
        
        n_positive = int(np.sum(y_test))
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, mean_prob)
        roc_auc = auc(fpr, tpr)
        axes[0, 0].plot(fpr, tpr, color='blue', lw=2, label=f'Model (AUC = {roc_auc:.3f})')
        axes[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        axes[0, 0].set_title('ROC Curve')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Precision-Recall Curve
        precision, recall, thresholds = precision_recall_curve(y_test, mean_prob)
        pr_auc = average_precision_score(y_test, mean_prob)
        axes[0, 1].plot(recall, precision, color='darkorange', lw=2, label=f'Model (AP = {pr_auc:.3f})')
        no_skill = n_positive / len(y_test)
        axes[0, 1].plot([0, 1], [no_skill, no_skill], color='navy', lw=2, linestyle='--', label=f'Baseline ({no_skill:.3f})')
        f1_scores = np.nan_to_num(2 * recall[:-1] * precision[:-1] / (recall[:-1] + precision[:-1]))
        optimal_idx = np.argmax(f1_scores)
        max_f1 = f1_scores[optimal_idx]
        optimal_threshold = thresholds[optimal_idx]
        axes[0, 1].scatter(recall[optimal_idx], precision[optimal_idx], marker='*', color='red', s=150, label=f'Max F1 ({max_f1:.3f})')
        axes[0, 1].set_title('Precision-Recall Curve')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Calibration Plot
        n_bins = 10
        bin_edges = np.linspace(0, mean_prob.max(), n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        observed_freq = []
        predicted_freq = []
        for i in range(n_bins):
            mask = (mean_prob >= bin_edges[i]) & (mean_prob < bin_edges[i+1]) if i < n_bins - 1 else (mean_prob >= bin_edges[i]) & (mean_prob <= bin_edges[i+1])
            if np.sum(mask) > 0:
                observed_freq.append(np.mean(y_test[mask]))
                predicted_freq.append(np.mean(mean_prob[mask]))
            else:
                observed_freq.append(np.nan)
                predicted_freq.append(bin_centers[i])
        valid_mask = ~np.isnan(observed_freq)
        axes[1, 0].plot(np.array(predicted_freq)[valid_mask], np.array(observed_freq)[valid_mask], marker='o', color='darkgreen', lw=2, label='Model')
        axes[1, 0].plot([0, mean_prob.max()], [0, mean_prob.max()], color='navy', lw=2, linestyle='--', label='Perfect')
        axes[1, 0].set_title('Calibration Plot')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Lift Curve
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
        axes[1, 1].plot(percentiles, lift, color='purple', lw=2, label='Model Lift')
        axes[1, 1].axhline(y=1, color='navy', lw=2, linestyle='--', label='Baseline')
        axes[1, 1].set_title('Lift Curve')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        lift_10pct = lift[9]
        axes[1, 1].annotate(f'Top 10%: {lift_10pct:.2f}x', xy=(10, lift_10pct), xytext=(25, lift_10pct), arrowprops=dict(arrowstyle='->', color='red'))
        
        plt.tight_layout()
        plt.savefig(os.path.join(ML_OUTPUT_DIR, "relative_probability_validation.png"), dpi=300)
        plt.close()
        
        print(f"\nVALIDATION METRICS:")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  PR-AUC: {pr_auc:.4f}")
        print(f"  Max F1: {max_f1:.4f} @ threshold {optimal_threshold:.4f}")
        print(f"  Lift at 10%: {lift_10pct:.2f}x")
        
        return {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'max_f1': max_f1,
            'optimal_threshold': optimal_threshold,
            'lift_10pct': lift_10pct
        }

    # Call validation functions
    temporal_results = validate_temporal_fire_counts(y_test, mean_prob, dates_test)
    validation_results = create_relative_probability_validation_plots(y_test, mean_prob, dates_test)

    # ===================================================================
    # SAVE MODEL ARTIFACTS
    # ===================================================================
    az.to_netcdf(trace, os.path.join(ML_OUTPUT_DIR, "trace_relative.nc"))
    joblib.dump(scaler, os.path.join(ML_OUTPUT_DIR, "scaler_relative.joblib"))

    test_predictions = pd.DataFrame({
        'mean_probability': mean_prob,
        'std_probability': std_prob,
        'actual_fire': y_test,
        'date': dates_test
    })
    test_predictions.to_csv(os.path.join(ML_OUTPUT_DIR, "test_predictions_with_uncertainty.csv"), index=False)

    results = {
        'validation_metrics': validation_results,
        'temporal_validation': temporal_results,
    }
    joblib.dump(results, os.path.join(ML_OUTPUT_DIR, "model_results.joblib"))

    # Close Dask client
    client.close()

    print("\nANALYSIS COMPLETE!")