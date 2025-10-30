"""
Corine Land Cover Fire Risk Mapping for Bolzano Province

Based on actual land cover codes found in landcoverfull_onehotencoder.csv
Maps Corine Land Cover Level 3 codes (3-digit) to fire risk ordinal values (0-5)

Fire Risk Scale:
0 = No fire risk (water, snow/ice)
1 = Very low (urban, bare soil)
2 = Low (agriculture, intensive farming)
3 = Moderate (grassland, broadleaf forest)
4 = High (shrubland, mixed forest)
5 = Very high (coniferous forest, dry vegetation)
"""

import pandas as pd
from pathlib import Path

# Corine Land Cover codes found in Bolzano Province dataset
# Based on landcoverfull_onehotencoder.csv
CORINE_CODES_IN_DATASET = {
    # Urban fabric
    111: "Continuous urban fabric",
    112: "Discontinuous urban fabric",
    121: "Industrial or commercial units",
    122: "Road and rail networks",
    124: "Airports",

    # Mining, construction, artificial green
    131: "Mineral extraction sites",
    133: "Construction sites",
    142: "Sport and leisure facilities",

    # Agricultural areas
    211: "Non-irrigated arable land",
    221: "Vineyards",
    222: "Fruit trees and berry plantations",
    231: "Pastures",
    242: "Complex cultivation patterns",
    243: "Land principally occupied by agriculture with significant natural vegetation",

    # Forest and semi-natural areas
    311: "Broad-leaved forest",
    312: "Coniferous forest",
    313: "Mixed forest",
    321: "Natural grasslands",
    322: "Moors and heathland",
    324: "Transitional woodland-shrub",
    331: "Beaches, dunes, sands",
    332: "Bare rocks",
    333: "Sparsely vegetated areas",
    335: "Glaciers and perpetual snow",

    # Wetlands and water bodies
    411: "Inland marshes",
    412: "Peat bogs",
    511: "Water courses",
    512: "Water bodies"
}


# Fire risk ordinal mapping (0-5 scale)
LANDCOVER_FIRE_RISK_ORDINAL = {
    # 0 = No fire risk
    335: 0,  # Glaciers and perpetual snow
    511: 0,  # Water courses
    512: 0,  # Water bodies
    411: 0,  # Inland marshes (wet)
    412: 0,  # Peat bogs (wet, but can burn if dried - keeping at 0 for now)

    # 1 = Very low fire risk
    111: 1,  # Continuous urban fabric
    112: 1,  # Discontinuous urban fabric
    121: 1,  # Industrial or commercial units
    122: 1,  # Road and rail networks
    124: 1,  # Airports
    131: 1,  # Mineral extraction sites
    133: 1,  # Construction sites
    142: 1,  # Sport and leisure facilities
    331: 1,  # Beaches, dunes, sands
    332: 1,  # Bare rocks

    # 2 = Low fire risk
    211: 2,  # Non-irrigated arable land (seasonal, managed)
    221: 2,  # Vineyards (managed, low fuel load)
    222: 2,  # Fruit trees (managed, low understory)
    231: 2,  # Pastures (grazed, low fuel)
    242: 2,  # Complex cultivation patterns (managed)
    243: 2,  # Agriculture with natural vegetation (mixed, managed)

    # 3 = Moderate fire risk
    311: 3,  # Broad-leaved forest (deciduous, higher moisture)
    321: 3,  # Natural grasslands (moderate fuel load)
    333: 3,  # Sparsely vegetated areas (low fuel but dry)

    # 4 = High fire risk
    313: 4,  # Mixed forest (combination of fuel types)
    322: 4,  # Moors and heathland (dense shrubs, dry)
    324: 4,  # Transitional woodland-shrub (dense understory)

    # 5 = Very high fire risk
    312: 5,  # Coniferous forest (resinous, very flammable, especially Alpine species)
}


def validate_mapping():
    """Validate that all codes in dataset are mapped."""
    unmapped = []
    for code in CORINE_CODES_IN_DATASET.keys():
        if code not in LANDCOVER_FIRE_RISK_ORDINAL:
            unmapped.append(code)

    if unmapped:
        print("⚠️  WARNING: Unmapped codes found:")
        for code in unmapped:
            print(f"    {code}: {CORINE_CODES_IN_DATASET[code]}")
        return False

    print("✓ All codes in dataset are mapped")
    return True


def get_fire_risk(corine_code):
    """
    Get fire risk value for a Corine Land Cover code.

    Args:
        corine_code: Integer Corine code (e.g., 312)

    Returns:
        Fire risk ordinal value (0-5), or 2 (moderate) if code not found
    """
    # Convert to int if needed
    if isinstance(corine_code, float):
        corine_code = int(corine_code)

    return LANDCOVER_FIRE_RISK_ORDINAL.get(corine_code, 2)  # Default: moderate risk


def create_mapping_table():
    """Create a DataFrame with the mapping for reference."""
    data = []

    for code in sorted(CORINE_CODES_IN_DATASET.keys()):
        data.append({
            'corine_code': code,
            'description': CORINE_CODES_IN_DATASET[code],
            'fire_risk': LANDCOVER_FIRE_RISK_ORDINAL.get(code, -1)
        })

    df = pd.DataFrame(data)

    # Add risk category labels
    risk_labels = {
        0: "No risk",
        1: "Very low",
        2: "Low",
        3: "Moderate",
        4: "High",
        5: "Very high"
    }
    df['risk_category'] = df['fire_risk'].map(risk_labels)

    return df


def print_mapping_summary():
    """Print a summary of the fire risk mapping."""
    print("="*80)
    print("CORINE LAND COVER FIRE RISK MAPPING - BOLZANO PROVINCE")
    print("="*80)
    print()

    # Group by risk level
    risk_groups = {}
    for code, risk in LANDCOVER_FIRE_RISK_ORDINAL.items():
        if risk not in risk_groups:
            risk_groups[risk] = []
        risk_groups[risk].append((code, CORINE_CODES_IN_DATASET.get(code, "Unknown")))

    risk_labels = {
        0: "NO RISK (Water, Snow/Ice)",
        1: "VERY LOW (Urban, Bare soil)",
        2: "LOW (Agriculture, Managed land)",
        3: "MODERATE (Grassland, Broadleaf forest)",
        4: "HIGH (Shrubland, Mixed forest)",
        5: "VERY HIGH (Coniferous forest)"
    }

    for risk_level in sorted(risk_groups.keys()):
        print(f"\n{risk_labels[risk_level]}")
        print("-" * 80)
        for code, description in sorted(risk_groups[risk_level]):
            print(f"  {code}: {description}")

    print("\n" + "="*80)
    print(f"Total codes mapped: {len(LANDCOVER_FIRE_RISK_ORDINAL)}")
    print(f"Total codes in dataset: {len(CORINE_CODES_IN_DATASET)}")
    print("="*80)


if __name__ == "__main__":
    print_mapping_summary()
    print()

    # Validate mapping
    validate_mapping()
    print()

    # Create and display mapping table
    df = create_mapping_table()

    print("\nMapping Table (sorted by fire risk):")
    print("="*80)
    print(df.sort_values('fire_risk').to_string(index=False))

    # Save to CSV
    output_path = Path(__file__).parent / "CORINE_fire_risk_mapping.csv"
    df.to_csv(output_path, index=False)
    print(f"\n✓ Mapping table saved to: {output_path}")

    # Statistics
    print("\n" + "="*80)
    print("FIRE RISK DISTRIBUTION")
    print("="*80)
    print(df['risk_category'].value_counts().sort_index())
