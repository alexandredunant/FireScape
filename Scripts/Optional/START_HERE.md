# START HERE - Simple Guide

## What You Need to Know

Today I fixed **one issue**: your landcover data (Corine codes 111-512) was treated as continuous numbers instead of categories.

## What You Need to Run

### Your Existing 3-Step Pipeline:

```bash
# Step 1: Generate training data (FIXED!)
cd /mnt/CEPH_PROJECTS/Firescape/Scripts/01_Data_Preparation
python create_raster_stacks.py

# Step 2: Train model
cd ../02_Model_Training
python train_relative_probability_model.py

# Step 3: Your zone projections
cd ../04_Zone_Climate_Projections
python project_zone_fire_risk.py
```

**That's it!** Those 3 scripts are your complete pipeline.

## What I Fixed

**File**: `01_Data_Preparation/create_raster_stacks.py`

**Change**: Now loads `landcoverfull.tif` (Corine codes) and maps them to fire risk (0-5):
- 312 (Coniferous) → 5 (very high)
- 324 (Woodland-shrub) → 4 (high)
- 231 (Pastures) → 2 (low)
- etc.

**Result**: Your model will now correctly understand that landcover classes are categories, not continuous numbers.

## What to Ignore

All the other stuff created today:
- ❌ `00_Documentation/` - Just reference docs
- ❌ `00_Utilities/` - Helper scripts you don't need
- ❌ `03_Climate_Projections/` - Optional new pipeline I created
- ❌ All the README.md files

**You already had a working pipeline**. I just fixed the landcover issue in your existing scripts.

## Next Steps

Just run your 3-step pipeline above. If you get errors, let me know!

## Questions Your Original Pipeline Answers

1. ✅ **Mid-month dates**: Your `project_zone_fire_risk.py` already handles this (line 417: uses mid-month with 60-day lookback)
2. ✅ **Landcover**: Now fixed (Corine codes → fire risk mapping)
3. ✅ **Visualization**: Your `analyze_warning_level_evolution.py` already creates plots
4. ✅ **Scenarios**: Your `project_zone_fire_risk.py` already iterates scenarios (TARGET_SCENARIO variable)

**You already had everything!** I just fixed the landcover encoding bug.

---

**If confused, just run the 3 commands above and ignore everything else.**
