#!/usr/bin/env python
# %%
"""
EXPLAINABLE BOOSTING MACHINE (EBM) FOR FIRE RISK PREDICTION

This script trains an Explainable Boosting Machine using the InterpretML library.
EBMs are interpretable models that automatically learn feature interactions while
maintaining transparency through additive structure and visualization tools.

Key advantages:
- High predictive accuracy (often comparable to XGBoost)
- Full interpretability with feature importance and shape functions
- Automatic interaction detection
- No need for manual feature engineering
"""

import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys

# InterpretML for Explainable Boosting Machine
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show, set_visualize_provider
from interpret.provider import InlineProvider

# Standard ML tools
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
import joblib

# Visualization
import seaborn as sns

print("=" * 80)
print("EXPLAINABLE BOOSTING MACHINE (EBM) FOR FIRE RISK")
print("=" * 80)
print()

# ===================================================================
# CONFIGURATION
# ===================================================================

# Get resolution from command line argument (default to 250m)
if len(sys.argv) > 1:
    RESOLUTION = sys.argv[1]
    if RESOLUTION not in ["50m", "250m"]:
        print(f"ERROR: Invalid resolution '{RESOLUTION}'. Must be '50m' or '250m'")
        sys.exit(1)
else:
    RESOLUTION = "250m"  # Default to 250m

print(f"Training with {RESOLUTION} resolution features")
print()

# Set FIRESCAPE_ROOT when the data live outside this repository.
BASE_DIR = Path(os.environ.get("FIRESCAPE_ROOT", Path(__file__).resolve().parents[1]))
INPUT_DIR = BASE_DIR / "output/01_Training_Data"
OUTPUT_DIR = BASE_DIR / "output/02_Model_Training/EBM_SPEI"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Input file (created by prepare_ebm_training.py)
FEATURES_PARQUET = INPUT_DIR / f"training_features_ebm_{RESOLUTION}.parquet"

# Model configuration
RANDOM_SEED = 42
TEST_SIZE = 0.2

# Define configurations for both models
CONFIGS = {
    "with_lightning": {
        "use_lightning": True,
        "min_year": 2012,
        "description": "With Lightning (2012-2024)",
    },
    "without_lightning": {
        "use_lightning": False,
        "min_year": 1999,
        "description": "Without Lightning (1999-2024)",
    },
}

# EBM hyperparameters
EBM_CONFIG = {
    "max_bins": 256,  # Discretization bins
    "max_interaction_bins": 48,  # Bins for interactions
    "interactions": 20,  # Number of pairwise interactions to detect
    "outer_bags": 12,  # Bagging rounds (moderate, was 8 originally)
    "inner_bags": 3,  # Inner bagging (0 = disabled)
    "learning_rate": 0.02,  # Learning rate
    "validation_size": 0.2,  # Validation split for early stopping
    "early_stopping_rounds": 75,  # Early stopping patience
    "n_jobs": 4,  # Multi-threaded but not -1 (safer than all cores)
    "random_state": RANDOM_SEED,
    "feature_names": None,  # Will be set after loading data
}

print(f"Configuration:")
print(f"  Input: {FEATURES_PARQUET}")
print(f"  Output: {OUTPUT_DIR}")
print(f"  EBM interactions: {EBM_CONFIG['interactions']}")
print(f"  EBM bags: {EBM_CONFIG['outer_bags']}")
print()

# ===================================================================
# LOAD DATA
# ===================================================================

print("Loading training features...")
gdf = gpd.read_parquet(FEATURES_PARQUET)
gdf["date"] = pd.to_datetime(gdf["date"])

print(f"✓ Loaded {len(gdf)} observations")
print(f"  Fires (bin=1): {(gdf['bin'] == 1).sum()}")
print(f"  Non-fires (bin=0): {(gdf['bin'] == 0).sum()}")
print(f"  Date range: {gdf['date'].min()} to {gdf['date'].max()}")
print()

# ===================================================================
# TRAIN BOTH MODELS
# ===================================================================

# Store results for final comparison
all_results = []

for suffix, config in CONFIGS.items():
    print()
    print("=" * 80)
    print(f"TRAINING MODEL: {config['description']}")
    print("=" * 80)
    print()

    # Make a copy of the original data for this configuration
    gdf_config = gdf.copy()

    # ===================================================================
    # PREPARE FEATURES
    # ===================================================================

    print("Preparing feature matrix...")

    # Get all feature columns (exclude metadata)
    feature_cols = [
        col
        for col in gdf_config.columns
        if col not in ["id_obs", "geometry", "date", "bin"]
    ]

    # Handle lightning feature based on configuration
    if config["use_lightning"]:
        print("Lightning feature: ENABLED")
        print("  Using only 2012+ data (when lightning data is available)")
        # Filter by year (lightning data available from 2012 onwards)
        gdf_config = gdf_config[gdf_config["date"].dt.year >= config["min_year"]]
        print(f"  Filtered to {len(gdf_config)} observations (2012+)")

        # Verify no NaN values in lightning_density for this period
        n_nan_lightning = gdf_config["lightning_density"].isna().sum()
        if n_nan_lightning > 0:
            print(
                f"  WARNING: {n_nan_lightning} NaN values in lightning_density for 2012+ data"
            )
            print(f"  Removing rows with NaN lightning values...")
            gdf_config = gdf_config[gdf_config["lightning_density"].notna()]
            print(f"  Final observations: {len(gdf_config)}")
    else:
        print("Lightning feature: DISABLED")
        print("  Using all data (1999-2024)")
        # Remove lightning feature from feature list
        if "lightning_density" in feature_cols:
            feature_cols.remove("lightning_density")
            print(f"  Removed lightning_density from features")

    print(f"Total features: {len(feature_cols)}")
    print(f"Total observations: {len(gdf_config)}")
    print()

    # Create feature matrix
    X = gdf_config[feature_cols].values
    y = gdf_config["bin"].values
    dates = gdf_config["date"].values

    # Remove rows with any NaN or Inf
    valid_mask = ~np.isnan(X).any(axis=1) & ~np.isinf(X).any(axis=1)
    X = X[valid_mask]
    y = y[valid_mask]
    dates = dates[valid_mask]

    print(f"Observations after removing NaN/Inf: {len(X)}")
    print(f"  Fires: {(y == 1).sum()}")
    print(f"  Non-fires: {(y == 0).sum()}")
    print()

    # Verify no remaining NaN/Inf values
    if np.isnan(X).any():
        print("WARNING: NaN values still present in data!")
        print(f"  Total NaN values: {np.isnan(X).sum()}")
    if np.isinf(X).any():
        print("WARNING: Inf values present in data!")
        print(f"  Total Inf values: {np.isinf(X).sum()}")
    print()

    # ===================================================================
    # TRAIN/TEST SPLITS (TEMPORAL AND SPATIAL VALIDATION)
    # ===================================================================

    print("=" * 80)
    print("CREATING TRAIN/TEST SPLITS FOR GENERALIZATION TESTING")
    print("=" * 80)
    print()

    # Get geometries for spatial split
    geometries = gdf_config[valid_mask]["geometry"].values
    coords = np.array([[geom.x, geom.y] for geom in geometries])

    # ---------------------------------------------------------------------
    # 1. TEMPORAL SPLIT (Test on future years)
    # ---------------------------------------------------------------------
    print("1. TEMPORAL SPLIT (Train on old data, test on new)")
    TEMPORAL_CUTOFF = pd.Timestamp("2020-01-01")

    temporal_train_mask = dates < TEMPORAL_CUTOFF
    temporal_test_mask = dates >= TEMPORAL_CUTOFF

    X_train_temporal = X[temporal_train_mask]
    y_train_temporal = y[temporal_train_mask]
    dates_train_temporal = dates[temporal_train_mask]

    X_test_temporal = X[temporal_test_mask]
    y_test_temporal = y[temporal_test_mask]
    dates_test_temporal = dates[temporal_test_mask]

    print(
        f"  Train: {len(X_train_temporal)} obs (1999-2019), {(y_train_temporal == 1).sum()} fires"
    )
    print(
        f"  Test:  {len(X_test_temporal)} obs (2020+), {(y_test_temporal == 1).sum()} fires"
    )
    print()

    # ---------------------------------------------------------------------
    # 2. SPATIAL SPLIT (Test on held-out locations)
    # ---------------------------------------------------------------------
    print("2. SPATIAL SPLIT (Train on some zones, test on others)")

    from sklearn.cluster import KMeans

    # Create spatial zones using K-means clustering
    n_spatial_zones = 5
    kmeans = KMeans(n_clusters=n_spatial_zones, random_state=RANDOM_SEED, n_init=10)
    spatial_zones = kmeans.fit_predict(coords)

    # Assign 20% of zones to test set
    test_zones = [0, 1]  # Hold out zones 0 and 1 for testing
    train_zones = [z for z in range(n_spatial_zones) if z not in test_zones]

    spatial_train_mask = np.isin(spatial_zones, train_zones)
    spatial_test_mask = np.isin(spatial_zones, test_zones)

    X_train_spatial = X[spatial_train_mask]
    y_train_spatial = y[spatial_train_mask]
    dates_train_spatial = dates[spatial_train_mask]
    zones_train_spatial = spatial_zones[spatial_train_mask]

    X_test_spatial = X[spatial_test_mask]
    y_test_spatial = y[spatial_test_mask]
    dates_test_spatial = dates[spatial_test_mask]
    zones_test_spatial = spatial_zones[spatial_test_mask]

    print(
        f"  Train zones: {train_zones} → {len(X_train_spatial)} obs, {(y_train_spatial == 1).sum()} fires"
    )
    print(
        f"  Test zones:  {test_zones} → {len(X_test_spatial)} obs, {(y_test_spatial == 1).sum()} fires"
    )
    print()

    # ---------------------------------------------------------------------
    # 3. RANDOM SPLIT (Baseline comparison)
    # ---------------------------------------------------------------------
    print("3. RANDOM SPLIT (Baseline - IID assumption)")

    (
        X_train_random,
        X_test_random,
        y_train_random,
        y_test_random,
        dates_train_random,
        dates_test_random,
    ) = train_test_split(
        X, y, dates, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y
    )

    print(f"  Train: {len(X_train_random)} obs, {(y_train_random == 1).sum()} fires")
    print(f"  Test:  {len(X_test_random)} obs, {(y_test_random == 1).sum()} fires")
    print()

    # ---------------------------------------------------------------------
    # Use TEMPORAL split as primary (most realistic for deployment)
    # ---------------------------------------------------------------------
    print("Using TEMPORAL split as primary validation strategy")
    print("(Testing generalization to future years)")
    print()

    X_train = X_train_temporal
    y_train = y_train_temporal
    dates_train = dates_train_temporal

    X_test = X_test_temporal
    y_test = y_test_temporal
    dates_test = dates_test_temporal

    print(f"Primary train set: {len(X_train)} ({(y_train == 1).sum()} fires)")
    print(f"Primary test set:  {len(X_test)} ({(y_test == 1).sum()} fires)")
    print()

    # ===================================================================
    # TRAIN EBM MODEL
    # ===================================================================

    print("=" * 80)
    print("TRAINING EXPLAINABLE BOOSTING MACHINE")
    print("=" * 80)
    print()

    print("EBM hyperparameters:")
    for key, value in EBM_CONFIG.items():
        print(f"  {key}: {value}")
    print()

    print("Training EBM...")
    print()

    # Set feature names in config
    EBM_CONFIG["feature_names"] = feature_cols

    ebm = ExplainableBoostingClassifier(**EBM_CONFIG)
    ebm.fit(X_train, y_train)

    print()
    print("✓ EBM training complete!")
    print()

    # Extract feature importance
    print("Extracting feature importance...")
    feature_importance_data = []
    for i, term_name in enumerate(ebm.term_names_):
        if term_name in feature_cols: # Check if it's a main effect
            feature_importance_data.append(
                {"feature": term_name, "importance": np.abs(ebm.term_importances()[i])}
            )
    importance_df = (
        pd.DataFrame(feature_importance_data)
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    print(f"✓ Feature importance extracted")
    print()

    # ===================================================================
    # SAVE MODEL
    # ===================================================================

    print("Saving EBM model...")
    joblib.dump(ebm, OUTPUT_DIR / f"ebm_model_{suffix}_{RESOLUTION}.joblib")
    joblib.dump(feature_cols, OUTPUT_DIR / f"feature_names_{suffix}_{RESOLUTION}.joblib")
    importance_df.to_csv(OUTPUT_DIR / f"feature_importance_{suffix}_{RESOLUTION}.csv", index=False)

    print(f"✓ Model saved to {OUTPUT_DIR}")
    print()

    # ===================================================================
    # GENERATE PREDICTIONS
    # ===================================================================

    print("=" * 80)
    print("GENERATING PREDICTIONS")
    print("=" * 80)
    print()

    print("Generating predictions on test set...")
    y_pred_proba = ebm.predict_proba(X_test)[:, 1]
    y_pred_class = ebm.predict(X_test)

    print("✓ Predictions generated")
    print()

    # ===================================================================
    # SAVE PREDICTIONS
    # ===================================================================

    print("Saving predictions...")

    predictions_df = pd.DataFrame(
        {
            "date": pd.to_datetime(dates_test),
            "y_true": y_test,
            "y_pred_proba": y_pred_proba,
            "y_pred_class": y_pred_class,
        }
    )

    predictions_df.to_csv(OUTPUT_DIR / f"test_predictions_{suffix}_{RESOLUTION}.csv", index=False)
    print(f"✓ Predictions saved to {OUTPUT_DIR / f'test_predictions_{suffix}_{RESOLUTION}.csv'}")
    print()

    # ===================================================================
    # MODEL EVALUATION (ALL SPLITS)
    # ===================================================================

    print("=" * 80)
    print("MODEL EVALUATION - GENERALIZATION TESTING")
    print("=" * 80)
    print()

    # Evaluate on all three test sets
    test_sets = {
        "Temporal (2020+)": (X_test_temporal, y_test_temporal),
        "Spatial (held-out zones)": (X_test_spatial, y_test_spatial),
        "Random (IID)": (X_test_random, y_test_random),
    }

    results_comparison = []

    for split_name, (X_test_split, y_test_split) in test_sets.items():
        print(f"\n{split_name}:")
        print("-" * 60)

        # Generate predictions
        y_pred_proba_split = ebm.predict_proba(X_test_split)[:, 1]
        y_pred_class_split = ebm.predict(X_test_split)

        # Metrics
        roc_auc_split = roc_auc_score(y_test_split, y_pred_proba_split)
        pr_auc_split = average_precision_score(y_test_split, y_pred_proba_split)

        print(f"  ROC AUC: {roc_auc_split:.4f}")
        print(f"  PR AUC:  {pr_auc_split:.4f}")

        # Store results
        results_comparison.append(
            {
                "split": split_name,
                "n_samples": len(X_test_split),
                "n_fires": (y_test_split == 1).sum(),
                "roc_auc": roc_auc_split,
                "pr_auc": pr_auc_split,
            }
        )

    # Create comparison dataframe
    comparison_df = pd.DataFrame(results_comparison)
    comparison_df.to_csv(
        OUTPUT_DIR / f"generalization_comparison_{suffix}_{RESOLUTION}.csv", index=False
    )

    print()
    print("=" * 80)
    print("GENERALIZATION COMPARISON SUMMARY")
    print("=" * 80)
    print(comparison_df.to_string(index=False))
    print()

    # Detailed evaluation on primary test set (temporal)
    print("=" * 80)
    print("DETAILED EVALUATION - PRIMARY TEST SET (Temporal)")
    print("=" * 80)
    print()

    # ROC AUC
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    print(f"ROC AUC: {roc_auc:.4f}")

    # PR AUC
    pr_auc = average_precision_score(y_test, y_pred_proba)
    print(f"PR AUC: {pr_auc:.4f}")
    print()

    # Classification Report
    print("Classification Report:")
    print(
        classification_report(y_test, y_pred_class, target_names=["Non-fire", "Fire"])
    )

    # Confusion Matrix
    print("Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_class)
    print(cm)
    print()

    # ===================================================================
    # INTERACTION IMPORTANCE
    # ===================================================================

    print("=" * 80)
    print("INTERACTION IMPORTANCE")
    print("=" * 80)
    print()

    # Extract interaction terms if they exist
    if hasattr(ebm, "term_names_"):
        interaction_terms = [name for name in ebm.term_names_ if " x " in name]

        if interaction_terms:
            print(f"Detected {len(interaction_terms)} interaction terms:")
            for term in interaction_terms[:10]:  # Show top 10
                print(f"  {term}")
            print()
        else:
            print("No significant interactions detected")
            print()

    # ===================================================================
    # VISUALIZATION
    # ===================================================================

    print("Creating visualizations...")

    # Figure 1: ROC and PR curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    axes[0].plot(
        fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})", linewidth=2, color="green"
    )
    axes[0].plot([0, 1], [0, 1], "k--", label="Random", linewidth=1)
    axes[0].set_xlabel("False Positive Rate", fontsize=12)
    axes[0].set_ylabel("True Positive Rate", fontsize=12)
    axes[0].set_title(
        f"ROC Curve - EBM Model ({config['description']})",
        fontsize=14,
        fontweight="bold",
    )
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    axes[1].plot(
        recall,
        precision,
        label=f"PR curve (AUC = {pr_auc:.3f})",
        linewidth=2,
        color="green",
    )
    axes[1].set_xlabel("Recall", fontsize=12)
    axes[1].set_ylabel("Precision", fontsize=12)
    axes[1].set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"model_performance_{suffix}_{RESOLUTION}.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Figure 2: Feature importance
    fig, ax = plt.subplots(figsize=(10, 8))

    top_n = 15
    top_features = importance_df.head(top_n)

    ax.barh(
        range(top_n),
        top_features["importance"].values,
        color="green",
        alpha=0.7,
        edgecolor="black",
    )
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_features["feature"].values)
    ax.invert_yaxis()
    ax.set_xlabel("Average Absolute Contribution", fontsize=12)
    ax.set_title(
        f"Top {top_n} Feature Importance - EBM Model ({config['description']}) [{RESOLUTION}]",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"feature_importance_{suffix}_{RESOLUTION}.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    print(f"✓ Visualizations saved to {OUTPUT_DIR}")
    print()

    # Store results for final comparison
    all_results.append(
        {
            "model": config["description"],
            "suffix": suffix,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_features": len(feature_cols),
            "n_fires_train": (y_train == 1).sum(),
            "n_fires_test": (y_test == 1).sum(),
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
        }
    )

# ===================================================================
# FINAL COMPARISON SUMMARY
# ===================================================================

print()
print("=" * 80)
print("ALL MODELS TRAINING COMPLETE!")
print("=" * 80)
print()

# Create comparison dataframe
comparison_summary_df = pd.DataFrame(all_results)
print("Model Comparison Summary:")
print("=" * 80)
print(comparison_summary_df.to_string(index=False))
print()

# Save comparison summary
comparison_summary_df.to_csv(OUTPUT_DIR / f"models_comparison_summary_{RESOLUTION}.csv", index=False)
print(f"✓ Comparison summary saved to {OUTPUT_DIR / f'models_comparison_summary_{RESOLUTION}.csv'}")
print()

print("Saved artifacts:")
print("=" * 80)
for suffix in CONFIGS.keys():
    print(f"\n{CONFIGS[suffix]['description']} [{RESOLUTION}]:")
    print(f"  Model: {OUTPUT_DIR / f'ebm_model_{suffix}_{RESOLUTION}.joblib'}")
    print(f"  Features: {OUTPUT_DIR / f'feature_names_{suffix}_{RESOLUTION}.joblib'}")
    print(f"  Feature importance: {OUTPUT_DIR / f'feature_importance_{suffix}_{RESOLUTION}.csv'}")
    print(f"  Predictions: {OUTPUT_DIR / f'test_predictions_{suffix}_{RESOLUTION}.csv'}")
    print(f"  Performance plot: {OUTPUT_DIR / f'model_performance_{suffix}_{RESOLUTION}.png'}")
    print(
        f"  Feature importance plot: {OUTPUT_DIR / f'feature_importance_{suffix}_{RESOLUTION}.png'}"
    )
    print(
        f"  Generalization comparison: {OUTPUT_DIR / f'generalization_comparison_{suffix}_{RESOLUTION}.csv'}"
    )

print()
print("Next steps:")
print("=" * 80)
print("  1. Review the saved validation metrics and feature-importance plots")
print("  2. Explore feature interactions using ebm.explain_global()")
print("  3. Run src/03_generate_seasonal_risk.py")
print()
print("To explore models interactively:")
print("  from interpret import show")
print("  import joblib")
print(f"  ebm = joblib.load('{OUTPUT_DIR / 'ebm_model_<suffix>.joblib'}')")
print("  show(ebm.explain_global())")
print()
