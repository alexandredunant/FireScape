"""
Shared Prediction Utilities for Firescape Model

This module provides standardized functions for making predictions with the trained
Bayesian wildfire model. It ensures consistency across all scripts (hindcasting,
climate projections, validation, etc.).

Key principle: Match the training data structure exactly
- Training used 32x32 pixel chips with 4x4 window averaging
- Predictions should use the same spatial scale and feature extraction

Usage:
    from shared_prediction_utils import load_model, predict_at_points, predict_grid_to_zones
"""

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray
from pathlib import Path
from datetime import timedelta
import joblib
import arviz as az
from shapely.geometry import Point
from typing import Dict, List, Tuple, Optional
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


class FirescapeModel:
    """
    Wrapper for the trained Bayesian wildfire model.
    Handles loading, feature extraction, and prediction.
    """

    def __init__(self, model_dir: Path):
        """
        Initialize model from saved artifacts.

        Args:
            model_dir: Directory containing trace, scaler, and temporal groups
        """
        self.model_dir = Path(model_dir)

        # Load model artifacts
        print("Loading model artifacts...")
        self.trace = az.from_netcdf(self.model_dir / "trace_relative.nc")
        self.scaler = joblib.load(self.model_dir / "scaler_relative.joblib")
        self.temporal_groups = joblib.load(self.model_dir / "temporal_groups.joblib")
        self.group_names = joblib.load(self.model_dir / "group_names.joblib")

        self.feature_names = self.scaler.feature_names_in_
        self.n_features = len(self.feature_names)

        # Pre-compute posterior means for faster prediction
        self._precompute_posterior_means()

        print(f"✓ Model loaded: {self.n_features} features, {len(self.group_names)} attention groups")

    def _precompute_posterior_means(self):
        """Pre-compute posterior means for faster prediction."""
        # Get intercept mean
        self.intercept_mean = self.trace.posterior['alpha'].values.mean()

        # Get attention means
        # The variable named 'attention_weights' in the trace is the raw Dirichlet output
        attention_raw_samples = self.trace.posterior['attention_weights'].values.reshape(-1, len(self.group_names))
        attention_scale_samples = self.trace.posterior['attention_scale'].values.reshape(-1, 1)
        
        # Combine raw weights with the scaling factor
        attention_samples = attention_raw_samples * attention_scale_samples
        self.attention_mean = attention_samples.mean(axis=0)

        # Reconstruct full beta vector from grouped betas
        self.beta_full = np.zeros(self.n_features)

        for g_idx, group_name in enumerate(self.group_names):
            group_indices = self.temporal_groups[group_name]
            beta_var_name = f'beta_{group_name}'

            if beta_var_name in self.trace.posterior:
                beta_group_samples = self.trace.posterior[beta_var_name].values
                beta_group_samples = beta_group_samples.reshape(-1, len(group_indices))
                beta_group_mean = beta_group_samples.mean(axis=0)

                for i, feature_idx in enumerate(group_indices):
                    self.beta_full[feature_idx] = beta_group_mean[i]

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Make predictions given a DataFrame of features.

        Args:
            features_df: DataFrame with features matching training (columns = feature names)

        Returns:
            Array of predicted probabilities
        """
        # Ensure features are in correct order
        X = features_df[self.feature_names].values

        # Handle missing values
        X = np.nan_to_num(X, nan=0.0)

        # Scale features
        X_scaled = self.scaler.transform(X)

        # Compute predictions by applying attention to group contributions
        logits = np.full(X_scaled.shape[0], self.intercept_mean)

        for g_idx, group_name in enumerate(self.group_names):
            group_indices = self.temporal_groups[group_name]
            if not group_indices:
                continue

            group_features_data = X_scaled[:, group_indices]
            
            # Get pre-computed beta means for this group
            beta_group_mean = self.beta_full[group_indices]
            
            group_contrib = np.dot(group_features_data, beta_group_mean)
            
            weighted_contrib = self.attention_mean[g_idx] * group_contrib
            logits += weighted_contrib

        predictions = 1 / (1 + np.exp(-logits))

        return predictions


def load_static_rasters(static_raster_dir: Path,
                         static_vars: List[str],
                         target_resolution: float = None,
                         template_raster: str = "nasadem.tif") -> Dict[str, xr.DataArray]:
    """
    Load and align static rasters to a common grid.

    Args:
        static_raster_dir: Directory containing static rasters
        static_vars: List of variable names to load
        target_resolution: Target resolution in meters (None = keep original)
        template_raster: Name of raster to use as template for alignment

    Returns:
        Dictionary of aligned raster grids
    """
    print("Loading static rasters...")

    # Load template
    template_path = Path(static_raster_dir) / template_raster
    template_grid = rioxarray.open_rasterio(template_path, masked=True).squeeze('band', drop=True)

    # Resample template if needed
    if target_resolution is not None:
        scale_factor = target_resolution / abs(template_grid.rio.resolution()[0])
        if abs(scale_factor - 1.0) > 0.01:
            new_width = int(template_grid.rio.width / scale_factor)
            new_height = int(template_grid.rio.height / scale_factor)
            template_grid = template_grid.rio.reproject(
                template_grid.rio.crs,
                shape=(new_height, new_width),
                resampling=5  # Average
            )

    print(f"  Template grid: {template_grid.shape}")

    # Load and align all static grids
    static_grids = {}
    for var in static_vars:
        raster_path = Path(static_raster_dir) / f"{var}.tif"
        if raster_path.exists():
            grid = rioxarray.open_rasterio(raster_path, masked=True).squeeze('band', drop=True)
            grid = grid.rio.reproject_match(template_grid)

            if grid.shape != template_grid.shape:
                print(f"  WARNING: {var} shape mismatch, skipping")
                continue

            static_grids[var] = grid
            print(f"  ✓ {var}: {grid.shape}")

    print(f"✓ Loaded {len(static_grids)} static variables")
    return static_grids


def load_climate_timeseries(target_date: pd.Timestamp,
                             lookback_days: int,
                             temp_dir: Path,
                             precip_dir: Path,
                             template_grid: xr.DataArray,
                             cache: Optional[dict] = None) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Load temperature and precipitation time series for a date.

    Args:
        target_date: Target date for prediction
        lookback_days: Number of days to look back
        temp_dir: Directory with temperature rasters (tavg_YYYYMMDD.tif)
        precip_dir: Directory with precipitation rasters (pr_YYYYMMDD.tif)
        template_grid: Template grid for alignment
        cache: Optional cache dictionary for loaded data

    Returns:
        Tuple of (temp_timeseries, precip_timeseries) as lists of arrays
    """
    start_date = target_date - timedelta(days=lookback_days - 1)

    temp_list = []
    precip_list = []

    for i in range(lookback_days):
        current_date = start_date + timedelta(days=i)
        date_str = current_date.strftime('%Y%m%d')

        # Temperature
        if cache is not None:
            cache_key = f"temp_{date_str}"
            temp_data = cache.get(cache_key)
        else:
            temp_data = None

        if temp_data is None:
            temp_file = Path(temp_dir) / f"tavg_{date_str}.tif"
            if temp_file.exists():
                try:
                    temp_grid = rioxarray.open_rasterio(temp_file, masked=True).squeeze('band', drop=True)
                    temp_grid = temp_grid.rio.reproject_match(template_grid)
                    temp_data = temp_grid.values
                    if temp_data.shape != template_grid.shape:
                        temp_data = np.full(template_grid.shape, np.nan)
                    if cache is not None:
                        cache[cache_key] = temp_data
                except Exception:
                    temp_data = np.full(template_grid.shape, np.nan)
            else:
                temp_data = np.full(template_grid.shape, np.nan)

        temp_list.append(temp_data)

        # Precipitation
        if cache is not None:
            cache_key = f"precip_{date_str}"
            precip_data = cache.get(cache_key)
        else:
            precip_data = None

        if precip_data is None:
            precip_file = Path(precip_dir) / f"pr_{date_str}.tif"
            if precip_file.exists():
                try:
                    precip_grid = rioxarray.open_rasterio(precip_file, masked=True).squeeze('band', drop=True)
                    precip_grid = precip_grid.rio.reproject_match(template_grid)
                    precip_data = precip_grid.values
                    if precip_data.shape != template_grid.shape:
                        precip_data = np.full(template_grid.shape, np.nan)
                    if cache is not None:
                        cache[cache_key] = precip_data
                except Exception:
                    precip_data = np.full(template_grid.shape, np.nan)
            else:
                precip_data = np.full(template_grid.shape, np.nan)

        precip_list.append(precip_data)

    # Reverse to have most recent first (for cumulative calculations)
    temp_list.reverse()
    precip_list.reverse()

    return temp_list, precip_list


def extract_features_grid(static_grids: Dict[str, xr.DataArray],
                          temp_timeseries: List[np.ndarray],
                          precip_timeseries: List[np.ndarray],
                          static_vars: List[str],
                          day_windows: List[int] = [1, 3, 5, 10, 15, 30, 60]) -> pd.DataFrame:
    """
    Extract features for all grid cells.

    IMPORTANT: This matches the training feature extraction:
    - Static features: Grid cell values (or small window averages)
    - Dynamic features: Cumulative mean/max over time windows

    Args:
        static_grids: Dictionary of static variable grids
        temp_timeseries: Temperature time series (time, rows, cols)
        precip_timeseries: Precipitation time series (time, rows, cols)
        static_vars: List of static variable names
        day_windows: List of day windows for cumulative features

    Returns:
        DataFrame with features for each grid cell (rows × cols)
    """
    template_grid = list(static_grids.values())[0]
    n_rows, n_cols = template_grid.shape
    expected_size = n_rows * n_cols

    features_dict = {}

    # Extract static features
    for var_name in static_vars:
        if var_name in static_grids:
            flattened = static_grids[var_name].values.flatten()
            if len(flattened) != expected_size:
                raise ValueError(f"Static grid {var_name} size mismatch: {len(flattened)} vs {expected_size}")
            features_dict[var_name] = flattened

    # Extract dynamic features (temperature)
    if temp_timeseries and len(temp_timeseries) > 0:
        temp_array = np.array(temp_timeseries)

        if temp_array.shape[1:] != (n_rows, n_cols):
            raise ValueError(f"Temperature array shape mismatch: {temp_array.shape[1:]} vs {(n_rows, n_cols)}")

        # Calculate cumulative statistics
        cumsum = np.cumsum(temp_array, axis=0)
        divisor = np.arange(1, len(temp_array) + 1).reshape(-1, 1, 1)
        cum_means = cumsum / divisor
        cum_maxs = np.maximum.accumulate(temp_array, axis=0)

        for day_window in day_windows:
            if day_window <= len(temp_array):
                idx = day_window - 1
                features_dict[f'T_cumulative_mean_{day_window}d'] = cum_means[idx].flatten()
                features_dict[f'T_cumulative_max_{day_window}d'] = cum_maxs[idx].flatten()

    # Extract dynamic features (precipitation)
    if precip_timeseries and len(precip_timeseries) > 0:
        precip_array = np.array(precip_timeseries)

        if precip_array.shape[1:] != (n_rows, n_cols):
            raise ValueError(f"Precipitation array shape mismatch: {precip_array.shape[1:]} vs {(n_rows, n_cols)}")

        cumsum = np.cumsum(precip_array, axis=0)
        divisor = np.arange(1, len(precip_array) + 1).reshape(-1, 1, 1)
        cum_means = cumsum / divisor
        cum_maxs = np.maximum.accumulate(precip_array, axis=0)

        for day_window in day_windows:
            if day_window <= len(precip_array):
                idx = day_window - 1
                features_dict[f'P_cumulative_mean_{day_window}d'] = cum_means[idx].flatten()
                features_dict[f'P_cumulative_max_{day_window}d'] = cum_maxs[idx].flatten()

    # Create DataFrame
    df = pd.DataFrame(features_dict)

    # Add spatial coordinates
    y_coords, x_coords = np.meshgrid(template_grid.y.values, template_grid.x.values, indexing='ij')
    df['x'] = x_coords.flatten()
    df['y'] = y_coords.flatten()

    return df


def aggregate_predictions_to_zones(predictions_df: pd.DataFrame,
                                   zones_gdf: gpd.GeoDataFrame,
                                   zone_id_col: str = 'ID') -> pd.DataFrame:
    """
    Aggregate grid-level predictions to zone level.

    Args:
        predictions_df: DataFrame with columns ['x', 'y', 'predicted_risk', ...]
        zones_gdf: GeoDataFrame with zone boundaries
        zone_id_col: Column name for zone identifier

    Returns:
        DataFrame with zone-level aggregated predictions
    """
    # Create point geometries
    predictions_gdf = gpd.GeoDataFrame(
        predictions_df,
        geometry=[Point(x, y) for x, y in zip(predictions_df['x'], predictions_df['y'])],
        crs=zones_gdf.crs
    )

    # Spatial join
    joined = gpd.sjoin(predictions_gdf, zones_gdf[['geometry', zone_id_col]], how='left', predicate='within')

    # Aggregate by zone
    agg_cols = {
        'predicted_risk': ['mean', 'max', 'std', 'count']
    }

    if 'date' in joined.columns:
        zone_aggregated = joined.groupby(['date', zone_id_col]).agg(agg_cols).reset_index()
        zone_aggregated.columns = ['date', 'zone_id', 'mean_risk', 'max_risk', 'std_risk', 'n_cells']
    else:
        zone_aggregated = joined.groupby(zone_id_col).agg(agg_cols).reset_index()
        zone_aggregated.columns = ['zone_id', 'mean_risk', 'max_risk', 'std_risk', 'n_cells']

    return zone_aggregated


def predict_grid_to_zones(model: FirescapeModel,
                          target_date: pd.Timestamp,
                          static_grids: Dict[str, xr.DataArray],
                          temp_dir: Path,
                          precip_dir: Path,
                          zones_gdf: gpd.GeoDataFrame,
                          lookback_days: int = 60,
                          cache: Optional[dict] = None) -> pd.DataFrame:
    """
    Complete workflow: Load climate data → Extract features → Predict → Aggregate to zones.

    Args:
        model: Loaded FirescapeModel
        target_date: Date to predict
        static_grids: Dictionary of static rasters
        temp_dir: Temperature data directory
        precip_dir: Precipitation data directory
        zones_gdf: Fire brigade zones
        lookback_days: Days of climate history to use
        cache: Optional cache for climate data

    Returns:
        DataFrame with zone-level predictions
    """
    # Load climate data
    template_grid = list(static_grids.values())[0]
    temp_ts, precip_ts = load_climate_timeseries(
        target_date, lookback_days, temp_dir, precip_dir, template_grid, cache
    )

    # Extract features
    static_vars = list(static_grids.keys())
    features_df = extract_features_grid(static_grids, temp_ts, precip_ts, static_vars)

    # Predict
    predictions = model.predict(features_df)
    features_df['predicted_risk'] = predictions
    features_df['date'] = target_date

    # Aggregate to zones
    zone_predictions = aggregate_predictions_to_zones(
        features_df[['x', 'y', 'predicted_risk', 'date']],
        zones_gdf
    )

    return zone_predictions


# Simple LRU cache for climate data
class ClimateCache:
    """Simple LRU cache for climate data."""

    def __init__(self, max_size: int = 120):
        self.cache = {}
        self.access_order = []
        self.max_size = max_size

    def get(self, key):
        if key in self.cache:
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None

    def __setitem__(self, key, value):
        if key in self.cache:
            self.access_order.remove(key)
        self.cache[key] = value
        self.access_order.append(key)
        while len(self.cache) > self.max_size:
            oldest = self.access_order.pop(0)
            del self.cache[oldest]

    def __getitem__(self, key):
        return self.get(key)

    def __contains__(self, key):
        return key in self.cache
