# Storage Requirements for Firescape Pipeline

**Date:** October 20, 2025
**Analysis:** Complete pipeline storage calculation

---

## Summary

| Component | Size | Files |
|-----------|------|-------|
| **Model Training** | 0.02 GB | ~30 files |
| **2022 Lookback** | 0.15 GB | 25 files |
| **Climate Projections** | 12.1 GB | 2,736 files |
| **Logs** | 0.5 GB | Variable |
| **TOTAL** | **~12.8 GB** | **~2,800 files** |

**Recommended free space:** 19.1 GB (with 50% buffer)

---

## Detailed Breakdown

### Grid Specifications
- **Dimensions:** 1,990 × 3,206 pixels
- **Valid pixels:** 2,965,655 (46.5% coverage of bounding box)
- **Resolution:** 100m × 100m
- **CRS:** EPSG:32632 (UTM Zone 32N)
- **Area:** Bolzano/Südtirol region, Italy

### File Size Calculation
- **Data type:** float32 (4 bytes per pixel)
- **Uncompressed size per map:** 11.3 MB
- **Compressed size (LZW):** 4.5 MB per GeoTIFF
- **Compression ratio:** ~2.5× (typical for LZW on float data)

---

## Stage 1: Model Training

**Output:** `Data/OUTPUT/02_Model/`

| Item | Size | Description |
|------|------|-------------|
| Main trace.nc | ~2 MB | Posterior samples (4 chains × 2000 samples) |
| scaler.joblib | ~3 KB | Feature standardization parameters |
| baseline_stats.joblib | ~2 KB | Baseline statistics |
| validation_plots.png | ~200 KB | Enhanced validation plots |
| cross_validation_results.json | ~66 KB | CV metrics and fold results |
| CV fold artifacts | ~10 MB | Individual fold models (12 folds) |
| **Subtotal** | **~17 MB** | **~0.02 GB** |

---

## Stage 2: 2022 Historical Lookback

**Output:** `Data/OUTPUT/07_Historical_Analysis/lookback_2022_monthly/`

| Item | Quantity | Size per file | Total Size |
|------|----------|---------------|------------|
| Fire risk GeoTIFFs | 12 months | 4.5 MB | 54 MB |
| Visualization PNGs | 12 months | 4.5 MB | 54 MB |
| Animated GIF | 1 | ~50 MB | 50 MB |
| **Subtotal** | **25 files** | - | **~159 MB (0.15 GB)** |

**Months included:** Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec

---

## Stage 3: Climate Projections (RCP 8.5)

**Output:** `Data/OUTPUT/04_Climate_Projections/rcp85/`

### Configuration
- **Time period:** 2020-2100 (9 decades)
- **Seasons:** February + July
- **Temporal resolution:** Every 3 days
- **Total dates:** 684
- **Quantiles:** 4 (pctl25, pctl50, pctl75, pctl99)

### Storage by Quantile

| Quantile | Dates | Size per file | Total Size |
|----------|-------|---------------|------------|
| **pctl25** (25th percentile) | 684 | 4.5 MB | 3.0 GB |
| **pctl50** (50th percentile - median) | 684 | 4.5 MB | 3.0 GB |
| **pctl75** (75th percentile) | 684 | 4.5 MB | 3.0 GB |
| **pctl99** (99th percentile - extreme) | 684 | 4.5 MB | 3.0 GB |
| **Total** | **2,736 maps** | 4.5 MB | **12.1 GB** |

### Directory Structure
```
04_Climate_Projections/rcp85/
├── pctl25/     # 3.0 GB (684 files) - Conservative scenario
├── pctl50/     # 3.0 GB (684 files) - Most likely scenario
├── pctl75/     # 3.0 GB (684 files) - Elevated risk scenario
└── pctl99/     # 3.0 GB (684 files) - Extreme risk scenario
```

### Temporal Coverage
- **Per decade:** 76 dates (38 Feb + 38 Jul)
- **Per quantile per decade:** 76 × 4.5 MB = 342 MB
- **All quantiles per decade:** 1.3 GB

| Decade | Years | Files | Size |
|--------|-------|-------|------|
| 2020s | 2020-2029 | 304 | 1.3 GB |
| 2030s | 2030-2039 | 304 | 1.3 GB |
| 2040s | 2040-2049 | 304 | 1.3 GB |
| 2050s | 2050-2059 | 304 | 1.3 GB |
| 2060s | 2060-2069 | 304 | 1.3 GB |
| 2070s | 2070-2079 | 304 | 1.3 GB |
| 2080s | 2080-2089 | 304 | 1.3 GB |
| 2090s | 2090-2099 | 304 | 1.3 GB |
| 2100 | 2100 only | 40 | 0.2 GB |
| **Total** | **2020-2100** | **2,736** | **12.1 GB** |

---

## Stage 4: Logs

**Output:** `Data/OUTPUT/00_Logs/`

| Log Type | Estimated Size | Description |
|----------|----------------|-------------|
| Model training | 50-100 MB | Sampling diagnostics, CV progress |
| Lookback GIF | 50-100 MB | Monthly processing logs |
| Climate projections | 300-400 MB | Progress for 2,736 maps |
| Master pipeline log | 10-20 MB | Combined output |
| **Total** | **~500 MB (0.5 GB)** | All logs combined |

---

## Total Storage Summary

```
┌─────────────────────────────────────────────────────────┐
│                 STORAGE REQUIREMENTS                     │
├─────────────────────────────────────────────────────────┤
│  Model artifacts:              0.02 GB                   │
│  2022 Lookback:                0.15 GB                   │
│  Climate Projections:         12.10 GB                   │
│  Logs:                         0.50 GB                   │
│  ───────────────────────────────────────────────────     │
│  TOTAL:                       12.77 GB                   │
│                                                           │
│  WITH 20% BUFFER:             15.32 GB (minimum)         │
│  WITH 50% BUFFER:             19.16 GB (recommended)     │
└─────────────────────────────────────────────────────────┘
```

---

## Disk Space Check

Before running the pipeline, verify available space:

```bash
# Check available space on Firescape directory
df -h /mnt/CEPH_PROJECTS/Firescape

# Check specific output directory
du -sh /mnt/CEPH_PROJECTS/Firescape/Data/OUTPUT
```

**Minimum requirement:** 15.3 GB free
**Recommended:** 19.1 GB free (allows room for temporary files and growth)

---

## Storage Optimization Options

If storage is constrained, consider these options:

### Option 1: Reduce Quantiles (Save 9.1 GB)
Keep only median (pctl50) and extreme (pctl99):
```python
# In 05_Bayesian_Climate_Projection_MultiQuantile_Seasonal.py
# Line 41-43
QUANTILES = {
    'pctl50': 50,  # Median scenario
    'pctl99': 99   # Extreme scenario
}
```
**Savings:** 6.0 GB (25th and 75th percentiles removed)
**New total:** 6.1 GB

### Option 2: Reduce Temporal Resolution (Save 8.1 GB)
Generate every 6 days instead of 3:
```python
# In script, change:
date_range_step = 6  # Instead of 3
```
**Savings:** 6.0 GB (half the maps)
**New total:** 6.8 GB

### Option 3: Single Season Only (Save 6.1 GB)
Generate July only (peak fire season):
```python
# In script configuration:
TARGET_MONTHS = [7]  # July only
```
**Savings:** 6.0 GB (February removed)
**New total:** 6.8 GB

### Option 4: Compress After Generation
Use maximum compression (slower writes, smaller files):
```python
# In raster saving code, change:
prob_raster.rio.to_raster(
    output_path,
    compress='DEFLATE',  # Instead of 'LZW'
    predictor=3,         # For floating point
    zlevel=9             # Maximum compression
)
```
**Savings:** Additional 20-30% compression (2-3 GB)
**Trade-off:** 30-50% slower write times

---

## File Count Considerations

### Total Files Generated
- Model: ~30 files
- Lookback: 25 files
- Climate: 2,736 files
- Logs: 4-10 files
- **Total: ~2,800 files**

### Filesystem Notes
- **Inode usage:** Minimal (2,800 files is small)
- **Directory listing:** May be slow with 684 files per quantile directory
- **Backup impact:** Consider tar/zip for archival

---

## Growth Projections

If extending the analysis in the future:

| Extension | Additional Storage |
|-----------|-------------------|
| Add RCP 4.5 scenario | +12.1 GB |
| Add monthly resolution (12 months/year) | +72.6 GB |
| Add 50m resolution | +48.4 GB (4× pixels) |
| Add ensemble members (5) | +60.5 GB (5× quantiles) |

---

## Data Archival Recommendations

### Short-term (Active Analysis)
Keep all files uncompressed for fast access.

### Medium-term (Post-Analysis)
Compress each quantile directory:
```bash
cd Data/OUTPUT/04_Climate_Projections/rcp85
tar -czf pctl25.tar.gz pctl25/
tar -czf pctl50.tar.gz pctl50/
tar -czf pctl75.tar.gz pctl75/
tar -czf pctl99.tar.gz pctl99/
```
**Expected compressed size:** ~8-9 GB (25-30% reduction)

### Long-term (Archival)
- Keep only pctl50 (median) and pctl99 (extreme): ~6 GB
- Archive training data to cold storage
- Document random seeds for reproducibility

---

## Monitoring Storage During Execution

### Real-time Monitoring
```bash
# Watch output directory size
watch -n 60 'du -sh /mnt/CEPH_PROJECTS/Firescape/Data/OUTPUT'

# Monitor specific stage
watch -n 60 'du -sh /mnt/CEPH_PROJECTS/Firescape/Data/OUTPUT/04_Climate_Projections'

# Track disk usage
df -h /mnt/CEPH_PROJECTS/Firescape
```

### Progress Estimation
Climate projections (largest component):
- **Per hour:** ~2 GB (assuming 5.7 days for 12.1 GB)
- **Per day:** ~2.1 GB
- **Progress check:** Count files in quantile directories

```bash
# Count completed maps per quantile
for q in pctl25 pctl50 pctl75 pctl99; do
  count=$(find Data/OUTPUT/04_Climate_Projections/rcp85/$q -name "*.tif" 2>/dev/null | wc -l)
  echo "$q: $count / 684 maps ($(echo "scale=1; $count/684*100" | bc)%)"
done
```

---

## Disk Space Troubleshooting

### If Running Low on Space

1. **Check for temporary files:**
   ```bash
   find /tmp -name "*climate*" -o -name "*pymc*"
   ```

2. **Clear old logs:**
   ```bash
   find Data/OUTPUT/00_Logs -name "*.log" -mtime +30 -delete
   ```

3. **Compress completed quantiles:**
   ```bash
   cd Data/OUTPUT/04_Climate_Projections/rcp85
   tar -czf completed_pctl25.tar.gz pctl25/ && rm -rf pctl25/
   ```

4. **Move to external storage:**
   ```bash
   rsync -avz Data/OUTPUT/04_Climate_Projections /backup/location/
   ```

---

## Summary

The complete Firescape pipeline will generate **approximately 12.8 GB** of output data across ~2,800 files. This is manageable for most modern storage systems.

**Key takeaway:** Ensure at least **19 GB of free space** before starting the pipeline to account for temporary files and growth.

**Most storage-intensive stage:** Climate projections (12.1 GB, 94% of total)

---

**Last Updated:** October 20, 2025
