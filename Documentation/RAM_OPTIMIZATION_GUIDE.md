# RAM Optimization Guide for Climate Projections
## System: 30GB RAM Available

---

## Memory Budget Analysis

### Current Configuration (30GB RAM)

| Component | Memory Usage | Status |
|-----------|--------------|--------|
| **Operating System** | ~2-4 GB | Reserved |
| **Available for Analysis** | ~26-28 GB | **✓ Sufficient** |

---

## Optimized Climate Projection Settings

### With Spatial Downsampling (Every 5th pixel = 250m)

| Component | Memory | Notes |
|-----------|--------|-------|
| **Prediction Grid** | ~0.9 MB | 118,626 points × 8 bytes |
| **Static Features** | ~11 MB | 118,626 × 12 vars × 8 bytes |
| **Climate Features** | ~26 MB | 118,626 × 28 features × 8 bytes |
| **Model Posterior** | ~50 MB | Cached samples |
| **Predictions** | ~0.9 MB | Output array |
| **Working Memory** | ~200 MB | Buffers, temporary arrays |
| **Peak Usage** | **~289 MB** | **✓ Very Safe** |

**Safety Margin:** 99% of RAM remains free

---

## Alternative Configurations

### Option 1: Every 3rd Pixel (150m) - RECOMMENDED FOR 30GB RAM

| Metric | Value |
|--------|-------|
| **Points** | 329,628 |
| **Peak RAM** | ~800 MB |
| **Runtime** | ~2 hours (all 3 decades) |
| **Quality** | Excellent |
| **Safety** | ✓ 97% RAM free |

### Option 2: Every 2nd Pixel (100m) - Feasible

| Metric | Value |
|--------|-------|
| **Points** | 741,414 |
| **Peak RAM** | ~1.8 GB |
| **Runtime** | ~4-5 hours (all 3 decades) |
| **Quality** | Near-original |
| **Safety** | ✓ 94% RAM free |

### Option 3: Full Resolution (50m) - Use with Chunking

| Metric | Value |
|--------|-------|
| **Points** | 2,965,655 |
| **Peak RAM** | ~7 GB per chunk |
| **Chunks** | 4-5 spatial chunks |
| **Runtime** | ~12-16 hours (all 3 decades) |
| **Quality** | Original |
| **Safety** | ✓ Requires spatial chunking |

---

## Recommended Workflow for 30GB RAM

### Phase 1: Testing (Quick - 1 hour)
```python
# Settings:
SPATIAL_DOWNSAMPLE = 5  # 250m resolution
PROJECTION_DATES = ['2020-07-15']  # Single date
```

**Memory:** <300 MB
**Purpose:** Verify everything works

---

### Phase 2: Key Decades (Recommended - 4 hours)
```python
# Settings:
SPATIAL_DOWNSAMPLE = 3  # 150m resolution
PROJECTION_DATES = [2020, 2050, 2080]  # 3 key decades
```

**Memory:** ~800 MB
**Purpose:** Generate actionable climate projections
**Output:** Fire risk maps for current, mid-century, end-century

---

### Phase 3: Full Analysis (Optional - 1-2 days)
```python
# Settings:
SPATIAL_DOWNSAMPLE = 2  # 100m resolution
PROJECTION_DATES = range(2020, 2101, 10)  # All future decades
```

**Memory:** ~1.8 GB
**Purpose:** Complete temporal analysis
**Output:** Full time series 2020-2100

---

## Model Training Memory Requirements

### Bayesian Model Training

**Expected Peak Memory:** ~8-12 GB

| Component | Memory |
|-----------|--------|
| **Training Data** | ~2-3 GB (1,781 observations) |
| **PyMC Sampling** | ~4-6 GB (MCMC chains) |
| **ArviZ Diagnostics** | ~2-3 GB (posterior analysis) |
| **Peak Total** | **8-12 GB** |

**Status:** ✓ Well within 30GB limit

**Optimization Tips:**
- Use `chains=2` instead of `chains=4` (saves ~3GB)
- Use `draws=1000` instead of `draws=2000` (saves ~2GB)
- Run validation plots separately (saves ~1GB during training)

---

## Fire Brigade Analysis Memory

### Zone-Based Risk Calculation

**Expected Memory:** <2 GB

| Component | Memory |
|-----------|--------|
| **481 Fire Brigade Zones** | <10 MB (geometries) |
| **Fire History** | <5 MB (636 events) |
| **Zonal Statistics** | <100 MB (per projection) |
| **Visualization** | <500 MB (plotting buffers) |
| **Total** | **<615 MB** |

**Status:** ✓ Very safe with 30GB

---

## Monitoring RAM Usage

### During Execution

```bash
# Monitor RAM in real-time
watch -n 5 free -h

# Or use htop for detailed view
htop

# Check Python process specifically
ps aux | grep python | awk '{print $6/1024 "MB", $11}'
```

### Warning Signs
- RAM usage > 25 GB: Consider stopping and using more downsampling
- Swap usage increasing: Reduce spatial resolution immediately
- System becomes slow: Kill process, increase downsampling

---

## Emergency RAM Management

### If Running Out of Memory

1. **Stop Current Process**
   ```bash
   # Find Python process ID
   ps aux | grep python

   # Kill it
   kill -9 <PID>
   ```

2. **Increase Downsampling**
   - Change from `SPATIAL_DOWNSAMPLE = 3` to `SPATIAL_DOWNSAMPLE = 5`
   - Or use `SPATIAL_DOWNSAMPLE = 10` (500m, ultra-fast)

3. **Reduce Posterior Samples**
   ```python
   # In prediction function
   n_samples = 100  # Instead of 300-500
   ```

4. **Process Fewer Dates**
   ```python
   PROJECTION_DATES = ['2050-07-15']  # Just one date
   ```

---

## Recommended Configuration for Your 30GB System

### Conservative (Guaranteed Success)
```python
SPATIAL_DOWNSAMPLE = 5  # 250m
PROJECTION_DATES = [2020, 2050, 2080]  # 3 dates
```
- **Memory:** <500 MB
- **Runtime:** ~2 hours
- **Quality:** Good for decision-making

### Balanced (Recommended)
```python
SPATIAL_DOWNSAMPLE = 3  # 150m
PROJECTION_DATES = [2020, 2030, 2040, 2050, 2060, 2070, 2080]  # 7 dates
```
- **Memory:** ~1.2 GB
- **Runtime:** ~6 hours
- **Quality:** Excellent

### Aggressive (Maximum Quality)
```python
SPATIAL_DOWNSAMPLE = 2  # 100m
PROJECTION_DATES = range(2020, 2101, 10)  # All dates
USE_CHUNKING = True  # Process in spatial chunks
```
- **Memory:** ~2-3 GB per chunk
- **Runtime:** ~1 day
- **Quality:** Near-original

---

## Summary

✅ **Your 30GB RAM is MORE than sufficient** for all analyses

**Recommended Settings:**
- Model Training: Default settings (uses ~10GB peak)
- Climate Projection: 150m resolution, 3-7 key decades (uses ~1GB)
- Fire Brigade Analysis: Full resolution (uses <1GB)

**Expected Total Runtime:**
- Model Training: 1-2 hours
- Climate Projections: 2-6 hours (depending on dates)
- Fire Brigade Analysis: 30 minutes
- **Total: 4-9 hours**

No special RAM optimizations needed - you have plenty of headroom!

---

*Last updated: 2025-10-19*
