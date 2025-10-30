"""
Configuration for Climate Projection Scenarios

Defines all RCP scenarios and time periods for wildfire risk modeling.
"""

import os
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class ClimateScenario:
    """Configuration for a single climate scenario."""
    name: str
    rcp: str  # e.g., 'rcp45', 'rcp85'
    period: Tuple[int, int]  # (start_year, end_year)
    description: str
    temp_dir: str
    precip_dir: str

    def __post_init__(self):
        """Validate paths after initialization."""
        if not os.path.exists(self.temp_dir):
            print(f"⚠️  WARNING: Temperature directory not found: {self.temp_dir}")
        if not os.path.exists(self.precip_dir):
            print(f"⚠️  WARNING: Precipitation directory not found: {self.precip_dir}")


# Base directory for climate projections
CLIMATE_BASE_DIR = "/mnt/CEPH_PROJECTS/FACT_CLIMAX/tmp_data_Firescape"

# Define all scenarios
SCENARIOS = [
    # Historical baseline (1999-2025)
    ClimateScenario(
        name="historical",
        rcp="historical",
        period=(1999, 2025),
        description="Historical observations (baseline)",
        temp_dir="/mnt/CEPH_PROJECTS/CLIMATE/GRIDS/TEMPERATURE/TIME_SERIES/UPLOAD/",
        precip_dir="/mnt/CEPH_PROJECTS/CLIMATE/GRIDS/PRECIPITATION/TIME_SERIES/UPLOAD/"
    ),

    # RCP 4.5 scenarios
    ClimateScenario(
        name="rcp45_2030",
        rcp="rcp45",
        period=(2021, 2040),
        description="RCP 4.5: Near-term (2030s) - Moderate emissions",
        temp_dir=os.path.join(CLIMATE_BASE_DIR, "tas/rcp45"),
        precip_dir=os.path.join(CLIMATE_BASE_DIR, "pr/rcp45")
    ),
    ClimateScenario(
        name="rcp45_2050",
        rcp="rcp45",
        period=(2041, 2060),
        description="RCP 4.5: Mid-century (2050s) - Moderate emissions",
        temp_dir=os.path.join(CLIMATE_BASE_DIR, "tas/rcp45"),
        precip_dir=os.path.join(CLIMATE_BASE_DIR, "pr/rcp45")
    ),
    ClimateScenario(
        name="rcp45_2070",
        rcp="rcp45",
        period=(2061, 2080),
        description="RCP 4.5: Late-century (2070s) - Moderate emissions",
        temp_dir=os.path.join(CLIMATE_BASE_DIR, "tas/rcp45"),
        precip_dir=os.path.join(CLIMATE_BASE_DIR, "pr/rcp45")
    ),

    # RCP 8.5 scenarios
    ClimateScenario(
        name="rcp85_2030",
        rcp="rcp85",
        period=(2021, 2040),
        description="RCP 8.5: Near-term (2030s) - High emissions",
        temp_dir=os.path.join(CLIMATE_BASE_DIR, "tas/rcp85"),
        precip_dir=os.path.join(CLIMATE_BASE_DIR, "pr/rcp85")
    ),
    ClimateScenario(
        name="rcp85_2050",
        rcp="rcp85",
        period=(2041, 2060),
        description="RCP 8.5: Mid-century (2050s) - High emissions",
        temp_dir=os.path.join(CLIMATE_BASE_DIR, "tas/rcp85"),
        precip_dir=os.path.join(CLIMATE_BASE_DIR, "pr/rcp85")
    ),
    ClimateScenario(
        name="rcp85_2070",
        rcp="rcp85",
        period=(2061, 2080),
        description="RCP 8.5: Late-century (2070s) - High emissions",
        temp_dir=os.path.join(CLIMATE_BASE_DIR, "tas/rcp85"),
        precip_dir=os.path.join(CLIMATE_BASE_DIR, "pr/rcp85")
    ),
]

# Regional subsets for Bolzano Province
REGIONS = {
    'full_province': {
        'name': 'Full Province',
        'bounds': None,  # Use all data
        'description': 'Entire Bolzano Province'
    },
    'alta_val_venosta': {
        'name': 'Alta Val Venosta',
        'bounds': (10.4, 46.6, 10.9, 46.9),  # (xmin, ymin, xmax, ymax) in EPSG:4326
        'description': 'Upper Venosta Valley - High elevation forests'
    },
    'val_badia': {
        'name': 'Val Badia',
        'bounds': (11.8, 46.5, 12.2, 46.7),
        'description': 'Badia Valley - Dolomites region'
    },
    'oltradige': {
        'name': 'Oltradige/Bassa Atesina',
        'bounds': (11.2, 46.3, 11.5, 46.5),
        'description': 'Lower Adige Valley - Wine-growing region'
    },
    'alta_pusteria': {
        'name': 'Alta Pusteria',
        'bounds': (12.0, 46.7, 12.5, 47.0),
        'description': 'Upper Puster Valley - Alpine forests'
    }
}

# Time periods for aggregation
TIME_AGGREGATION_WINDOWS = [1, 3, 5, 10, 15, 30, 60]  # days

# Mid-month dates for projection (one per month, per year)
def get_projection_dates(scenario: ClimateScenario, months: List[int] = None) -> List[Tuple[int, int, int]]:
    """
    Generate mid-month dates for a scenario.

    Args:
        scenario: Climate scenario configuration
        months: List of months to include (default: fire season Jun-Sep)

    Returns:
        List of (year, month, day) tuples
    """
    if months is None:
        months = [6, 7, 8, 9]  # Fire season: Jun-Sep

    dates = []
    start_year, end_year = scenario.period

    for year in range(start_year, end_year + 1):
        for month in months:
            # Use mid-month (day 15) as reference
            dates.append((year, month, 15))

    return dates


# Output directories
OUTPUT_BASE_DIR = "/mnt/CEPH_PROJECTS/Firescape/Scripts/OUTPUT/03_Climate_Projections"

def get_scenario_output_dir(scenario_name: str) -> str:
    """Get output directory for a specific scenario."""
    output_dir = os.path.join(OUTPUT_BASE_DIR, scenario_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_region_output_dir(scenario_name: str, region_name: str) -> str:
    """Get output directory for a specific scenario and region."""
    output_dir = os.path.join(OUTPUT_BASE_DIR, scenario_name, region_name)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


# Print scenario summary
if __name__ == "__main__":
    print("="*80)
    print("CLIMATE PROJECTION SCENARIOS CONFIGURATION")
    print("="*80)

    print(f"\nTotal scenarios: {len(SCENARIOS)}")
    print(f"Total regions: {len(REGIONS)}")

    for i, scenario in enumerate(SCENARIOS, 1):
        print(f"\n{i}. {scenario.name}")
        print(f"   RCP: {scenario.rcp}")
        print(f"   Period: {scenario.period[0]}-{scenario.period[1]}")
        print(f"   Description: {scenario.description}")
        print(f"   Temperature data: {scenario.temp_dir}")
        print(f"   Precipitation data: {scenario.precip_dir}")

        # Check data availability
        dates = get_projection_dates(scenario)
        print(f"   Projection dates: {len(dates)} (fire season only)")

    print("\n" + "="*80)
    print("REGIONAL SUBSETS")
    print("="*80)

    for region_key, region_info in REGIONS.items():
        print(f"\n• {region_info['name']} ({region_key})")
        print(f"  {region_info['description']}")
        if region_info['bounds']:
            print(f"  Bounds: {region_info['bounds']}")
