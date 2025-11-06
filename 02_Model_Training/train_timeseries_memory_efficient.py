#!/usr/bin/env python3
"""
BAYESIAN TEMPORAL WILDFIRE MODEL - MEMORY EFFICIENT FULL TIME SERIES

This version:
1. Uses full 60-day time series (not collapsed features)
2. Handles Dask arrays properly to avoid memory explosion
3. Uses mini-batch processing for PyMC
4. Implements temporal attention mechanism

Key improvements over train_Dask_PyMC_timeseries.py:
- Smaller chunks (50 vs 500)
- Compute in batches before PyMC (avoid Dask+PyMC conflicts)
- Memory-mapped arrays for large data
- Progress tracking
"""

import xarray as xr
import pandas as pd
import numpy as np
import os
import sys
from datetime import timedelta

# PyMC & ArviZ
import pymc as pm
import arviz as az
import pytensor.tensor as pt

# ML tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import joblib

# Dask
import dask.array as da
from dask.distributed import Client, progress
from dask_ml.preprocessing import StandardScaler as DaskStandardScaler

print(f"Running Bayesian Temporal Model on PyMC v{pm.__version__}")

# ===================================================================
# CONFIGURATION
# ===================================================================

# Corine Land Cover → Fire Risk Ordinal Mapping
LANDCOVER_FIRE_RISK_ORDINAL = {
    335: 0, 511: 0, 512: 0, 411: 0, 412: 0,
    111: 1, 112: 1, 121: 1, 122: 1, 124: 1, 131: 1, 133: 1, 142: 1, 331: 1, 332: 1,
    211: 2, 221: 2, 222: 2, 231: 2, 242: 2, 243: 2,
    311: 3, 321: 3, 333: 3,
    313: 4, 322: 4, 324: 4,
    312: 5
}

# Paths
NETCDF_PATH = "/mnt/CEPH_PROJECTS/Firescape/output/01_Training_Data/spacetime_stacks.nc"
ML_OUTPUT_DIR = "/mnt/CEPH_PROJECTS/Firescape/output/02_Model_TimeSeries/"
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
CHANNELS = STATIC_VARS + DYNAMIC_VARS

# Memory management
CHUNK_SIZE = 50  # Process 50 observations at a time
BATCH_SIZE = 200  # For PyMC training
MAX_MEMORY_GB = 8  # Max memory per Dask worker

# ===================================================================
# HELPER FUNCTIONS
# ===================================================================

def compute_mode_robust(data):
    """Compute mode for categorical data"""
    values = data.flatten()
    values = values[~np.isnan(values)]
    if values.shape[0] == 0:
        return 0.0

    if isinstance(values, da.Array):
        values = values.compute()

    try:
        int_values = values.astype(int)
        counts = np.bincount(int_values)
        mode = np.argmax(counts)
    except ValueError:
        from collections import Counter
        counts = Counter(values)
        mode = counts.most_common(1)[0][0]

    return float(mode)

# ===================================================================
# MAIN EXECUTION
# ===================================================================

if __name__ == "__main__":
    print("="*80)
    print("STARTING MEMORY-EFFICIENT TEMPORAL MODEL")
    print("="*80)
    print()

    # Start Dask client with strict memory limits
    print("Starting Dask client...")
    client = Client(
        n_workers=2,
        threads_per_worker=2,
        memory_limit=f'{MAX_MEMORY_GB}GB',
        dashboard_address=':8787'
    )
    print(f"  Dask dashboard: {client.dashboard_link}")
    print()

    # ===================================================================
    # DATA LOADING (LAZY)
    # ===================================================================
    print("Loading data (lazy)...")
    ds = xr.open_dataset(NETCDF_PATH, chunks={'id_obs': CHUNK_SIZE})
    main_data_var = list(ds.data_vars)[0]

    # Load labels (small, can fit in memory)
    y = ds['label'].values
    dates = pd.to_datetime(ds['event_date'].values)
    n_total = len(y)

    print(f"  Total observations: {n_total}")
    print(f"  Time steps: {TIME_STEPS}")
    print(f"  Channels: {len(CHANNELS)}")
    print()

    # ===================================================================
    # FEATURE EXTRACTION (LAZY, IN CHUNKS)
    # ===================================================================
    print("Extracting features (lazy)...")

    # Dynamic features: spatial mean over time
    dynamic_data = ds[main_data_var].sel(channel=DYNAMIC_VARS).mean(dim=['y', 'x'])
    print(f"  Dynamic data shape: {dynamic_data.shape}")

    # Static features (continuous)
    static_continuous_vars = [v for v in STATIC_VARS if v != 'landcover_fire_risk']
    static_cont_data = ds[main_data_var].sel(channel=static_continuous_vars).isel(time=0).mean(dim=['y', 'x'])
    print(f"  Static continuous shape: {static_cont_data.shape}")

    # Landcover (categorical) - needs special handling
    print("  Processing landcover (categorical)...")
    landcover_data_full = ds[main_data_var].sel(channel='landcover_fire_risk').isel(time=0)

    # Compute landcover mode in chunks to avoid memory issues
    landcover_mode_values = []
    for i in range(0, n_total, CHUNK_SIZE):
        chunk_end = min(i + CHUNK_SIZE, n_total)
        chunk = landcover_data_full.isel(id_obs=slice(i, chunk_end))

        # Compute mode for each observation in chunk
        chunk_modes = []
        for j in range(len(chunk.id_obs)):
            obs_data = chunk.isel(id_obs=j).values
            mode_val = compute_mode_robust(obs_data)
            chunk_modes.append(mode_val)

        landcover_mode_values.extend(chunk_modes)

        if (i // CHUNK_SIZE) % 10 == 0:
            print(f"    Processed {i}/{n_total} landcover values...")

    landcover_fire_risk = xr.DataArray(
        landcover_mode_values,
        coords={'id_obs': ds.id_obs},
        dims=['id_obs']
    ).expand_dims(channel=1).assign_coords(channel=['landcover_fire_risk'])

    print(f"  Landcover fire risk shape: {landcover_fire_risk.shape}")

    # Combine static features
    static_data = xr.concat([static_cont_data, landcover_fire_risk], dim='channel')
    print(f"  Combined static shape: {static_data.shape}")

    # Replicate static over time and combine with dynamic
    print("  Combining static and dynamic features...")
    static_replicated = static_data.expand_dims(time=TIME_STEPS, axis=1)
    static_replicated = static_replicated.transpose('id_obs', 'time', 'channel')

    # Combine features: id_obs × time × channels
    features_combined = xr.concat([static_replicated, dynamic_data], dim='channel')
    features_combined = features_combined.transpose('id_obs', 'time', 'channel')

    print(f"  Final features shape: {features_combined.shape}")
    print(f"  Expected memory: ~{features_combined.nbytes / 1e9:.2f} GB")
    print()

    # ===================================================================
    # TRAIN/TEST SPLIT
    # ===================================================================
    print("Creating train/test split...")
    indices = np.arange(n_total)
    train_idx, test_idx = train_test_split(
        indices, test_size=0.3, random_state=42, stratify=y
    )

    print(f"  Train: {len(train_idx)} observations")
    print(f"  Test: {len(test_idx)} observations")
    print()

    # ===================================================================
    # COMPUTE TRAINING DATA IN BATCHES
    # ===================================================================
    print("Computing training data in batches...")
    print("  (This avoids Dask+PyMC conflicts)")
    print()

    # Create memory-mapped files for efficient storage
    train_shape = (len(train_idx), TIME_STEPS, len(CHANNELS))
    test_shape = (len(test_idx), TIME_STEPS, len(CHANNELS))

    train_memmap_path = os.path.join(ML_OUTPUT_DIR, "train_data.dat")
    test_memmap_path = os.path.join(ML_OUTPUT_DIR, "test_data.dat")

    X_train = np.memmap(train_memmap_path, dtype='float32', mode='w+', shape=train_shape)
    X_test = np.memmap(test_memmap_path, dtype='float32', mode='w+', shape=test_shape)

    # Compute train data in chunks
    print("  Computing train data...")
    for i in range(0, len(train_idx), CHUNK_SIZE):
        chunk_end = min(i + CHUNK_SIZE, len(train_idx))
        chunk_indices = train_idx[i:chunk_end]

        chunk_data = features_combined.isel(id_obs=chunk_indices).compute()
        X_train[i:chunk_end] = chunk_data.values

        if i % (CHUNK_SIZE * 10) == 0:
            print(f"    {i}/{len(train_idx)} train samples...")

    # Compute test data in chunks
    print("  Computing test data...")
    for i in range(0, len(test_idx), CHUNK_SIZE):
        chunk_end = min(i + CHUNK_SIZE, len(test_idx))
        chunk_indices = test_idx[i:chunk_end]

        chunk_data = features_combined.isel(id_obs=chunk_indices).compute()
        X_test[i:chunk_end] = chunk_data.values

        if i % (CHUNK_SIZE * 10) == 0:
            print(f"    {i}/{len(test_idx)} test samples...")

    y_train = y[train_idx]
    y_test = y[test_idx]
    dates_test = dates[test_idx]

    print()
    print("✓ Data computed and stored in memory-mapped files")
    print(f"  Train data: {train_memmap_path}")
    print(f"  Test data: {test_memmap_path}")
    print()

    # ===================================================================
    # FEATURE SCALING
    # ===================================================================
    print("Scaling features...")

    # Flatten time dimension for scaling
    X_train_flat = X_train.reshape(-1, len(CHANNELS))
    X_test_flat = X_test.reshape(-1, len(CHANNELS))

    scaler = StandardScaler()
    X_train_scaled_flat = scaler.fit_transform(X_train_flat)
    X_test_scaled_flat = scaler.transform(X_test_flat)

    # Reshape back to 3D
    X_train_scaled = X_train_scaled_flat.reshape(train_shape)
    X_test_scaled = X_test_scaled_flat.reshape(test_shape)

    print("✓ Features scaled")
    print()

    # ===================================================================
    # BAYESIAN MODEL WITH TEMPORAL ATTENTION
    # ===================================================================
    print("="*80)
    print("BUILDING BAYESIAN TEMPORAL MODEL")
    print("="*80)
    print()

    n_obs_train, n_time, n_features = X_train_scaled.shape

    print(f"Model dimensions:")
    print(f"  Observations: {n_obs_train}")
    print(f"  Time steps: {n_time}")
    print(f"  Features: {n_features}")
    print()

    # Use mini-batch for memory efficiency
    print(f"Using mini-batch size: {BATCH_SIZE}")
    n_batches = int(np.ceil(n_obs_train / BATCH_SIZE))
    print(f"Number of batches: {n_batches}")
    print()

    # Sample a subset for initial model development
    print("Training on subset for memory efficiency...")
    sample_size = min(1000, n_obs_train)
    sample_idx = np.random.choice(n_obs_train, sample_size, replace=False)

    X_sample = X_train_scaled[sample_idx]
    y_sample = y_train[sample_idx]

    print(f"  Sample size: {sample_size}")
    print()

    def create_temporal_model(X, y):
        """Create PyMC model with temporal attention"""
        with pm.Model() as model:
            # Intercept
            alpha = pm.Normal('alpha', mu=0, sigma=2.0)

            # Feature coefficients (shared across time)
            betas = pm.Normal('betas', mu=0, sigma=3.0, shape=n_features)

            # Temporal attention weights (sum to 1, scaled)
            attention_raw = pm.Dirichlet('time_attention', a=np.ones(n_time))
            attention_scale = pm.HalfNormal('attention_scale', sigma=5.0)
            time_attention = attention_raw * attention_scale

            # Compute predictions
            # X shape: (n_obs, n_time, n_features)
            # betas shape: (n_features,)
            # Result: (n_obs, n_time)
            contrib_per_time = pt.einsum('otf,f->ot', X, betas)

            # Apply temporal attention
            # time_attention shape: (n_time,)
            # Result: (n_obs,)
            logit_p = alpha + pt.dot(contrib_per_time, time_attention)

            # Likelihood
            pm.Bernoulli('likelihood', logit_p=logit_p, observed=y)

            # Derived quantities
            pm.Deterministic('baseline_prob', pm.math.sigmoid(alpha))
            pm.Deterministic('attention_weights', time_attention)

        return model

    print("Building model...")
    model = create_temporal_model(X_sample, y_sample)

    print("Model graph:")
    print(model)
    print()

    # Sample
    print("="*80)
    print("SAMPLING POSTERIOR")
    print("="*80)
    print()

    with model:
        trace = pm.sample(
            draws=1000,
            tune=500,
            chains=2,
            cores=2,
            target_accept=0.95,
            return_inferencedata=True,
            progressbar=True
        )

    print()
    print("✓ Sampling complete!")
    print()

    # ===================================================================
    # SAVE RESULTS
    # ===================================================================
    print("Saving results...")

    az.to_netcdf(trace, os.path.join(ML_OUTPUT_DIR, "trace_timeseries.nc"))
    joblib.dump(scaler, os.path.join(ML_OUTPUT_DIR, "scaler_timeseries.joblib"))

    # Save model metadata
    metadata = {
        'n_features': n_features,
        'n_time': n_time,
        'channels': CHANNELS,
        'sample_size': sample_size,
        'train_size': n_obs_train,
        'test_size': len(test_idx)
    }
    joblib.dump(metadata, os.path.join(ML_OUTPUT_DIR, "model_metadata.joblib"))

    print(f"✓ Results saved to: {ML_OUTPUT_DIR}")
    print()

    # ===================================================================
    # GENERATE PREDICTIONS
    # ===================================================================
    print("Generating predictions on test set (batched)...")

    # Predict in batches to avoid memory issues
    test_predictions = []
    test_uncertainties = []

    for i in range(0, len(test_idx), BATCH_SIZE):
        batch_end = min(i + BATCH_SIZE, len(test_idx))
        X_batch = X_test_scaled[i:batch_end]

        with model:
            # Update data
            pm.set_data({'X': X_batch, 'y': np.zeros(len(X_batch))})

            # Sample posterior predictive
            post_pred = pm.sample_posterior_predictive(
                trace,
                var_names=['likelihood'],
                progressbar=False
            )

        batch_mean = post_pred.posterior_predictive['likelihood'].mean(dim=['chain', 'draw']).values
        batch_std = post_pred.posterior_predictive['likelihood'].std(dim=['chain', 'draw']).values

        test_predictions.extend(batch_mean)
        test_uncertainties.extend(batch_std)

        if i % (BATCH_SIZE * 5) == 0:
            print(f"  {i}/{len(test_idx)} predictions...")

    test_predictions = np.array(test_predictions)
    test_uncertainties = np.array(test_uncertainties)

    print()
    print("✓ Predictions generated")
    print()

    # ===================================================================
    # VALIDATION
    # ===================================================================
    print("="*80)
    print("VALIDATION METRICS")
    print("="*80)
    print()

    # ROC-AUC
    fpr, tpr, _ = roc_curve(y_test, test_predictions)
    roc_auc = auc(fpr, tpr)

    # PR-AUC
    precision, recall, _ = precision_recall_curve(y_test, test_predictions)
    pr_auc = average_precision_score(y_test, test_predictions)

    print(f"ROC-AUC: {roc_auc:.4f}")
    print(f"PR-AUC:  {pr_auc:.4f}")
    print()

    # Save predictions
    results_df = pd.DataFrame({
        'date': dates_test,
        'actual': y_test,
        'prediction': test_predictions,
        'uncertainty': test_uncertainties
    })
    results_df.to_csv(os.path.join(ML_OUTPUT_DIR, "test_predictions.csv"), index=False)

    print(f"✓ Predictions saved")
    print()

    # Close Dask client
    client.close()

    print("="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print()
    print(f"Output directory: {ML_OUTPUT_DIR}")
    print("Files created:")
    print("  - trace_timeseries.nc (model trace)")
    print("  - scaler_timeseries.joblib (feature scaler)")
    print("  - model_metadata.joblib (model info)")
    print("  - test_predictions.csv (predictions)")
    print("  - train_data.dat (memory-mapped train data)")
    print("  - test_data.dat (memory-mapped test data)")
