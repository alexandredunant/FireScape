# Climate Percentile Interpretation Issue

## Problem Identified

The 99th percentile climate projections show **erratic, non-monotonic patterns** in fire risk predictions:

### Example (Spring 2020-2080):
```
Year     Mean Risk    Change vs 2020
------------------------------------
2020     0.009424     baseline
2030     0.008874     -5.8%
2040     0.012603     +33.7%
2050     0.025576     +171.4%  ← spike
2060     0.014271     +51.4%
2070     0.025377     +169.3%  ← spike
2080     0.006143     -34.8%   ← collapse
```

## Root Cause

**The 99th percentile has opposite meanings for temperature vs precipitation:**

| Variable | 99th Percentile Means | Fire Risk Impact |
|----------|----------------------|------------------|
| Temperature | **HOT** (extreme heat) | ✓ Increases fire risk |
| Precipitation | **WET** (extreme rainfall) | ✗ **DECREASES** fire risk |

### Evidence from March Data:

**Temperature** (generally increasing = expected):
```
Year    99th pctl    50th pctl    Difference
2020    5.04°C       0.17°C       +4.87°C
2030    5.74°C       1.16°C       +4.57°C
2080    9.22°C       3.77°C       +5.45°C
```

**Precipitation** (erratic, extremely high):
```
Year    99th pctl       50th pctl       Ratio
2020    18.76 mm/day    0.55 mm/day     34x wetter
2030    13.61 mm/day    0.08 mm/day     171x wetter  ← extreme
2040    11.22 mm/day    0.03 mm/day     410x wetter  ← extreme
2050    15.16 mm/day    0.21 mm/day     71x wetter
2060    19.47 mm/day    0.47 mm/day     41x wetter
2070    10.92 mm/day    0.06 mm/day     186x wetter  ← extreme
2080    19.67 mm/day    0.61 mm/day     32x wetter
```

**Key Issue:** Years with extremely high 99th percentile precipitation (170x-410x normal) create model extrapolation problems. The model was trained on typical conditions and cannot reliably predict fire risk under such extreme wetness.

## Why This Matters

The current "pctl99" scenario is **NOT** representing "worst-case fire conditions". Instead, it represents:
- ✓ Extreme heat (fire-prone)
- ✗ Extreme wetness (fire-resistant)

This creates a **contradictory climate scenario** that doesn't correspond to any real fire danger situation.

## Recommended Solution

Use **asymmetric percentiles** to create realistic fire-prone scenarios:

### Scenario Definitions:

| Scenario Name | Temperature | Precipitation | Interpretation |
|---------------|-------------|---------------|----------------|
| **Baseline** | 50th | 50th | Median conditions |
| **Optimistic** | 25th | 75th | Cooler + wetter (low fire risk) |
| **Fire-Prone** | 99th | **1st or 5th** | Hot + **dry** (high fire risk) |

### Alternative Approach:

If only symmetric percentiles are available (pctl25, pctl50, pctl99):

1. **Keep pctl25 and pctl50** as-is (both variables same percentile)
2. **Replace "pctl99" with custom fire-prone scenario:**
   - Temperature: 99th percentile (hot)
   - Precipitation: **25th or lower** (dry)
3. **Rename scenarios** to reflect actual fire danger:
   - `pctl25` → "Low Fire Risk" (cool + wet)
   - `pctl50` → "Median Conditions"
   - ~~`pctl99`~~ → "High Fire Risk" (hot + dry)

## Data Requirements

Check if the following percentile files exist:

```bash
# For fire-prone scenarios, we need:
/mnt/CEPH_PROJECTS/FACT_CLIMAX/tmp_data_Firescape/tas/rcp85/tas_EUR-11_pctl99_rcp85.nc  # ✓ exists
/mnt/CEPH_PROJECTS/FACT_CLIMAX/tmp_data_Firescape/pr/rcp85/pr_EUR-11_pctl01_rcp85.nc   # ? check
# OR
/mnt/CEPH_PROJECTS/FACT_CLIMAX/tmp_data_Firescape/pr/rcp85/pr_EUR-11_pctl05_rcp85.nc   # ? check
# OR
/mnt/CEPH_PROJECTS/FACT_CLIMAX/tmp_data_Firescape/pr/rcp85/pr_EUR-11_pctl25_rcp85.nc   # ✓ exists
```

## Implementation Steps

1. **Check available precipitation percentiles**
   ```bash
   ls /mnt/CEPH_PROJECTS/FACT_CLIMAX/tmp_data_Firescape/pr/rcp85/
   ```

2. **Modify projection scripts** to use asymmetric percentiles:
   - `Scripts/04_Zone_Climate_Projections/project_zone_fire_risk.py`
   - `Scripts/04_Zone_Climate_Projections/analyze_warning_level_evolution.py`

3. **Create new scenario configuration:**
   ```python
   CLIMATE_SCENARIOS = {
       'low_risk': {'T': 'pctl25', 'P': 'pctl75'},  # Cool + wet
       'median': {'T': 'pctl50', 'P': 'pctl50'},     # Median
       'high_risk': {'T': 'pctl99', 'P': 'pctl25'},  # Hot + dry
   }
   ```

4. **Re-run projections** with corrected scenarios

5. **Update documentation** to reflect proper scenario interpretations

## Scientific Rationale

Fire risk depends on **compound extremes** in the same direction:
- High fire risk = Hot + **Dry** + Low humidity + High wind
- Low fire risk = Cool + **Wet** + High humidity + Low wind

Using 99th percentile for both T and P creates a physically inconsistent scenario (hot but extremely wet) that:
1. Doesn't represent realistic fire danger conditions
2. Forces the model to extrapolate beyond training data
3. Produces unreliable, erratic predictions

The correct approach is to use **compound fire danger indices** or construct scenarios with consistent fire-promoting conditions across all variables.

---

**Date:** 2025-10-28
**Status:** Issue identified, solution proposed, awaiting implementation
