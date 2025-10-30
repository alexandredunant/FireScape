# Firescape Project - Complete Execution Plan
## RAM-Optimized Workflow for 30GB System

**Status:** Raster generation in progress (25% complete, ~20 min remaining)

---

## Pipeline Overview

```
[CURRENT] Raster Generation (25% done) →
[NEXT] Model Training (~1.5 hrs) →
[THEN] Climate Projections (~3-4 hrs) →
[FINALLY] Fire Brigade Analysis (~30 min)

Total Estimated Time: ~6-7 hours
```

---

## Step 1: Raster Stack Generation ⏳ IN PROGRESS

**Status:** 440/1,781 complete (24.7%)
**Remaining:** ~20 minutes
**Memory:** ~500 MB (well within limits)

**Output:**
- `/mnt/CEPH_PROJECTS/Firescape/Data/OUTPUT/spacetime_stacks.nc`

---

## Step 2: Bayesian Model Training (Next)

### Configuration
```python
# In 04_Bayesian_pyMCLogisticRegression_Linear_Attention_commented.py
RUN_CV = False  # Skip cross-validation to save time (optional)
draws = 1000    # Reduce from 2000 (saves RAM and time)
chains = 2      # Reduce from 4 (saves RAM)
```

### Expected Performance
- **RAM Usage:** 8-10 GB peak (well within 30GB)
- **Runtime:** 60-90 minutes
- **CPU Cores:** Will use 2 cores for parallel chains

### Outputs
```
Uncertainty_Attention/model_plots_bayesian_linear/
├── trace.nc                    # Bayesian posterior samples
├── scaler.joblib               # Feature scaler
├── baseline_stats.joblib       # Historical fire statistics
├── validation_plots.png        # ROC curves, etc.
├── attention_weights.png       # Feature group importance
├── feature_importance.png      # Individual feature importance
└── corner_plot.png             # Parameter relationships
```

### Memory Breakdown
| Component | RAM | Cumulative |
|-----------|-----|------------|
| Data loading | 2 GB | 2 GB |
| Feature extraction | 1 GB | 3 GB |
| PyMC sampling (2 chains) | 4 GB | 7 GB |
| ArviZ diagnostics | 2 GB | 9 GB |
| **Peak Total** | | **9 GB** ✓ |

---

## Step 3: Optimized Climate Projections

### Recommended Configuration (RAM-Efficient)

```python
# In 05_Bayesian_Climate_Projection_OPTIMIZED.py

SPATIAL_DOWNSAMPLE = 3  # 150m resolution (329,628 points)
PROJECTION_DATES = [2020, 2050, 2080]  # 3 key decades

# This uses:
# - Memory: ~1.2 GB peak
# - Runtime: ~3-4 hours for all 3 dates
# - Quality: Excellent for decision-making
```

### Why These Settings?

**Spatial Resolution: 150m**
- Original: 2.96M points, ~7GB RAM needed
- Downsampled: 330K points, ~1.2GB RAM needed
- **Quality loss:** Minimal (~10% reduction in detail)
- **Speed gain:** 9× faster
- **RAM savings:** 83% less memory

**Temporal Coverage: 3 Key Decades**
- **2020:** Current baseline
- **2050:** Mid-century projection
- **2080:** End-century scenario
- Captures full climate change trajectory

### Expected Performance Per Date
| Task | Time | Memory |
|------|------|--------|
| Load model | 30s | 50 MB |
| Extract static features | 10 min | 300 MB |
| Extract climate features | 30 min | 500 MB |
| Generate predictions | 30 min | 800 MB |
| Save outputs | 5 min | 200 MB |
| **Total per date** | **~75 min** | **1.2 GB peak** |

### Total for 3 Dates
- **Runtime:** 3.5-4 hours
- **Peak RAM:** 1.2 GB
- **Outputs:** 3 fire risk maps (2020, 2050, 2080)

### Alternative Options

**If you want faster results (1.5 hours):**
```python
SPATIAL_DOWNSAMPLE = 5  # 250m resolution
PROJECTION_DATES = [2050, 2080]  # Just future
```

**If you want more detail (6-8 hours):**
```python
SPATIAL_DOWNSAMPLE = 2  # 100m resolution
PROJECTION_DATES = [2020, 2030, 2040, 2050, 2060, 2070, 2080]  # All future decades
```

---

## Step 4: Fire Brigade Activity Analysis

### What This Does

1. **Current Baseline (2020)**
   - Calculate mean fire risk per fire brigade zone
   - Compare with historical fire frequencies
   - Identify high-risk zones

2. **Future Projections (2050, 2080)**
   - Calculate projected fire risk per zone
   - Compute % increase from baseline
   - Rank zones by projected increase

3. **Resource Planning**
   - Identify zones needing increased capacity
   - Calculate expected activity increase
   - Create priority maps

### Configuration
```python
# Will use clipped shapefile (481 zones, Bolzano only)
FIRE_BRIGADE_SHP = "FireBrigade_ResponsibilityAreas_Bolzano_clipped.gpkg"

# Will process 3 climate scenarios
SCENARIOS = [2020, 2050, 2080]
```

### Expected Performance
- **RAM Usage:** <1 GB
- **Runtime:** 20-30 minutes
- **CPU:** Single core sufficient

### Outputs
```
fire_brigade_analysis/
├── current_risk_by_zone_2020.png      # Current baseline map
├── projected_risk_2050.png            # Mid-century projection
├── projected_risk_2080.png            # End-century projection
├── risk_increase_2020_to_2050.png     # Change map
├── risk_increase_2020_to_2080.png     # Change map
├── top_20_zones_risk_increase.csv     # Priority zones
├── zone_statistics_all_scenarios.csv  # Complete data
└── summary_report.pdf                 # Final report
```

---

## Complete Execution Sequence

### Automated Run (Recommended)

```bash
cd /mnt/CEPH_PROJECTS/Firescape/Scripts

# Wait for raster generation to complete, then run:
python 04_Bayesian_pyMCLogisticRegression_Linear_Attention_commented.py && \
python 05_Bayesian_Climate_Projection_OPTIMIZED.py && \
python 07_Fire_Brigade_Zone_Analysis.py

# Or run with logging:
python 04_Bayesian_pyMCLogisticRegression_Linear_Attention_commented.py 2>&1 | tee model_training.log && \
python 05_Bayesian_Climate_Projection_OPTIMIZED.py 2>&1 | tee climate_projection.log && \
python 07_Fire_Brigade_Zone_Analysis.py 2>&1 | tee brigade_analysis.log
```

### Manual Step-by-Step

```bash
# Step 1: Wait for raster generation (check with monitor_progress.py)
python monitor_progress.py

# Step 2: Train model (1-1.5 hours)
python 04_Bayesian_pyMCLogisticRegression_Linear_Attention_commented.py

# Step 3: Run climate projections (3-4 hours)
python 05_Bayesian_Climate_Projection_OPTIMIZED.py

# Step 4: Analyze fire brigade zones (30 min)
python 07_Fire_Brigade_Zone_Analysis.py
```

---

## Timeline Summary

| Task | Duration | RAM | When |
|------|----------|-----|------|
| **Raster Generation** | 40 min total | 0.5 GB | Now (25% done) |
| **Model Training** | 1-1.5 hours | 9 GB | After rasters |
| **Climate Projections** | 3-4 hours | 1.2 GB | After model |
| **Brigade Analysis** | 30 min | <1 GB | After projections |
| **TOTAL** | **~6-7 hours** | **9 GB peak** | |

**All within your 30GB RAM capacity! ✓**

---

## Monitoring Commands

### Check Progress
```bash
# Raster generation
python monitor_progress.py

# Model training (check log file)
tail -f model_training.log

# Climate projections
ls -lh climate_projections_rcp85_optimized/

# RAM usage
watch -n 5 free -h
```

### Troubleshooting

**If RAM gets too high (>25GB):**
```bash
# Stop current process
pkill -f python

# Increase downsampling
# Edit 05_Bayesian_Climate_Projection_OPTIMIZED.py:
# Change: SPATIAL_DOWNSAMPLE = 3
# To: SPATIAL_DOWNSAMPLE = 5
```

**If process seems stuck:**
```bash
# Check if still running
ps aux | grep python

# Check latest output
tail -100 <logfile>

# Check disk space
df -h /mnt/CEPH_PROJECTS/
```

---

## Expected Final Outputs

### Maps
- Current fire risk map (2020, 150m resolution)
- Mid-century projection (2050)
- End-century projection (2080)
- Risk change maps (2020→2050, 2020→2080)
- Fire brigade zone maps with overlays

### Data Files
- Zone-level risk statistics (481 zones × 3 scenarios)
- Top priority zones for each scenario
- Temporal risk trends
- Validation statistics

### Reports
- Climate projection summary
- Fire brigade activity projections
- Resource allocation recommendations
- Summary visualizations

---

## Success Criteria

✓ Model training completes without errors
✓ ROC-AUC > 0.70 (validation metric)
✓ Climate projections generate for all 3 dates
✓ No NaN/Inf values in outputs
✓ Fire brigade analysis completes for all zones
✓ Final reports generated

---

## Next Steps After Completion

1. **Review Outputs**
   - Check validation metrics
   - Inspect risk maps visually
   - Verify zone statistics

2. **Quality Control**
   - Compare 2020 projection with historical data
   - Check for spatial artifacts
   - Validate extreme values

3. **Interpretation**
   - Identify highest-risk zones
   - Calculate % increases
   - Prioritize intervention zones

4. **Reporting**
   - Generate summary presentations
   - Create stakeholder reports
   - Archive results

---

*Last updated: 2025-10-19 23:50*
*Raster generation: 25% complete*
