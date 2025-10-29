"""
Fix Landcover Categorical Encoding

Currently, 'landcoverfull' is treated as a continuous numeric variable,
which is incorrect. This script provides two solutions:

1. One-Hot Encoding: Convert landcover classes to binary features
2. Ordinal Encoding: Map landcover classes to meaningful ordered values

RECOMMENDATION: Use ordinal encoding based on flammability:
- Higher values = more flammable vegetation types
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder


# Landcover class definitions (adjust based on your actual data)
# These are typical Corine Land Cover classes - adjust as needed
LANDCOVER_CLASSES = {
    1: 'Urban areas',
    2: 'Agriculture',
    3: 'Grassland',
    4: 'Shrubland',
    5: 'Broadleaf forest',
    6: 'Coniferous forest',
    7: 'Mixed forest',
    8: 'Bare rock',
    9: 'Water',
    10: 'Snow/Ice'
}

# Ordinal mapping based on fire risk (0 = no risk, 5 = high risk)
LANDCOVER_FIRE_RISK_ORDINAL = {
    9: 0,   # Water - no fire risk
    10: 0,  # Snow/Ice - no fire risk
    1: 1,   # Urban - low risk (but human ignition)
    8: 1,   # Bare rock - low risk
    2: 2,   # Agriculture - moderate (seasonal)
    3: 3,   # Grassland - moderate-high
    5: 3,   # Broadleaf - moderate-high
    7: 4,   # Mixed forest - high
    4: 4,   # Shrubland - high
    6: 5,   # Coniferous - very high (resinous, dry)
}


def encode_landcover_onehot(landcover_series: pd.Series) -> pd.DataFrame:
    """
    Convert landcover to one-hot encoded features.

    Args:
        landcover_series: Series of landcover class IDs

    Returns:
        DataFrame with one-hot encoded columns
    """
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    # Fit and transform
    landcover_encoded = encoder.fit_transform(landcover_series.values.reshape(-1, 1))

    # Create column names
    col_names = [f'landcover_{int(cat)}' for cat in encoder.categories_[0]]

    return pd.DataFrame(landcover_encoded, columns=col_names, index=landcover_series.index)


def encode_landcover_ordinal(landcover_series: pd.Series, mapping: dict = None) -> pd.Series:
    """
    Convert landcover to ordinal values based on fire risk.

    Args:
        landcover_series: Series of landcover class IDs
        mapping: Custom mapping dict (default: LANDCOVER_FIRE_RISK_ORDINAL)

    Returns:
        Series with ordinal fire risk values
    """
    if mapping is None:
        mapping = LANDCOVER_FIRE_RISK_ORDINAL

    # Map landcover classes to ordinal values
    ordinal_values = landcover_series.map(mapping)

    # Fill unmapped values with median
    median_value = np.median(list(mapping.values()))
    ordinal_values = ordinal_values.fillna(median_value)

    return ordinal_values


def update_feature_matrix_with_landcover(X: pd.DataFrame, method: str = 'ordinal') -> pd.DataFrame:
    """
    Update feature matrix with properly encoded landcover.

    Args:
        X: Original feature DataFrame (with 'landcoverfull' column)
        method: 'ordinal' or 'onehot'

    Returns:
        Updated feature DataFrame
    """
    if 'landcoverfull' not in X.columns:
        print("⚠️  'landcoverfull' column not found in feature matrix")
        return X

    print(f"\n🔧 Encoding landcover using method: {method}")

    if method == 'ordinal':
        # Replace landcoverfull with ordinal fire risk values
        X['landcover_fire_risk'] = encode_landcover_ordinal(X['landcoverfull'])
        X = X.drop(columns=['landcoverfull'])
        print(f"   ✓ Replaced 'landcoverfull' with 'landcover_fire_risk' (ordinal)")

    elif method == 'onehot':
        # Add one-hot encoded columns
        landcover_onehot = encode_landcover_onehot(X['landcoverfull'])
        X = pd.concat([X.drop(columns=['landcoverfull']), landcover_onehot], axis=1)
        print(f"   ✓ Replaced 'landcoverfull' with {len(landcover_onehot.columns)} one-hot features")

    else:
        raise ValueError(f"Unknown encoding method: {method}")

    return X


def visualize_landcover_distribution(landcover_series: pd.Series, output_path: str = None):
    """
    Visualize the distribution of landcover classes.

    Args:
        landcover_series: Series of landcover class IDs
        output_path: Path to save plot (optional)
    """
    import matplotlib.pyplot as plt

    # Count occurrences
    counts = landcover_series.value_counts().sort_index()

    # Map to class names
    class_names = [LANDCOVER_CLASSES.get(int(lc), f'Class {int(lc)}') for lc in counts.index]
    fire_risk = [LANDCOVER_FIRE_RISK_ORDINAL.get(int(lc), np.nan) for lc in counts.index]

    # Create plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Frequency distribution
    ax1.bar(range(len(counts)), counts.values, color='steelblue', alpha=0.7)
    ax1.set_xticks(range(len(counts)))
    ax1.set_xticklabels(class_names, rotation=45, ha='right')
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Landcover Class Distribution', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Fire risk mapping
    ax2.bar(range(len(counts)), fire_risk, color='orangered', alpha=0.7)
    ax2.set_xticks(range(len(counts)))
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    ax2.set_ylabel('Fire Risk (Ordinal)', fontsize=12)
    ax2.set_title('Landcover Fire Risk Encoding', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 5.5)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ Plot saved: {output_path}")
    else:
        plt.show()

    plt.close()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("="*80)
    print("LANDCOVER CATEGORICAL ENCODING FIX")
    print("="*80)

    # Example: Load your feature data
    # X = pd.read_csv("your_features.csv")

    # Create synthetic example data
    print("\n📊 Creating example data...")
    n_samples = 1000
    landcover_example = pd.Series(
        np.random.choice([1, 2, 3, 4, 5, 6, 7, 8], size=n_samples, p=[0.05, 0.15, 0.10, 0.15, 0.15, 0.20, 0.15, 0.05])
    )

    # Visualize distribution
    print("\n📈 Visualizing landcover distribution...")
    visualize_landcover_distribution(landcover_example, output_path="landcover_distribution.png")

    # Example feature matrix
    X_example = pd.DataFrame({
        'landcoverfull': landcover_example,
        'temperature': np.random.randn(n_samples),
        'precipitation': np.random.randn(n_samples)
    })

    print("\n🔧 Original feature matrix:")
    print(X_example.head())

    # Apply ordinal encoding
    print("\n" + "="*80)
    print("METHOD 1: ORDINAL ENCODING (RECOMMENDED)")
    print("="*80)
    X_ordinal = update_feature_matrix_with_landcover(X_example.copy(), method='ordinal')
    print("\n✓ Updated feature matrix:")
    print(X_ordinal.head())
    print(f"\n   Feature count: {len(X_example.columns)} → {len(X_ordinal.columns)}")

    # Apply one-hot encoding
    print("\n" + "="*80)
    print("METHOD 2: ONE-HOT ENCODING")
    print("="*80)
    X_onehot = update_feature_matrix_with_landcover(X_example.copy(), method='onehot')
    print("\n✓ Updated feature matrix:")
    print(X_onehot.head())
    print(f"\n   Feature count: {len(X_example.columns)} → {len(X_onehot.columns)}")

    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    print("""
    USE ORDINAL ENCODING because:
    1. Preserves single-feature simplicity (no dimensionality explosion)
    2. Encodes domain knowledge about fire risk
    3. Compatible with existing model structure
    4. Interpretable coefficients

    TO APPLY TO YOUR MODEL:
    1. Update create_cumulative_features() in training script
    2. Replace landcoverfull extraction with:
       features['landcover_fire_risk'] = ordinal_value (from mapping)
    3. Retrain model with updated features
    """)

    print("\n✓ Example complete!")
