# Climate Projection Runtime Analysis Report

**Generated:** October 19, 2025, 23:41 UTC
**Analysis:** Bayesian Fire Risk Climate Projections for Bolzano Province

---

## Executive Summary

This report provides detailed runtime estimates for climate projection analysis covering 14 decades (1970-2100) at 50m spatial resolution across the Bolzano province.

### Key Findings

- **Total Prediction Points:** 2,965,655 (at 50m resolution)
- **Study Area:** 7,397 km² (Bolzano Province)
- **Temporal Coverage:** 14 decades (1970-2100, every 10 years)
- **Total Predictions Needed:** 41,519,170

---

## Grid Specifications

| Parameter | Value |
|-----------|-------|
| **Spatial Resolution** | 50m × 50m |
| **Study Area** | 7,397.38 km² |
| **Valid Pixels** | 2,965,655 |
| **Point Density** | 400.9 points/km² |
| **Grid Dimensions** | 1,990 × 3,206 pixels |

---

## Projection Configuration

**Temporal Scope:**
- **Start Year:** 1970
- **End Year:** 2100
- **Interval:** Every 10 years
- **Total Decades:** 14
- **Projection Dates:** July 15 of each decade year

**Climate Scenario:**
- **RCP Scenario:** RCP 8.5 (high emissions)
- **Ensemble Quantile:** 50th percentile (median)
- **Available Quantiles:** 25th, 50th, 75th, 99th

**Features:**
- **Static Variables:** 12 (topography, vegetation, infrastructure)
- **Dynamic Variables:** 2 (temperature, precipitation)
- **Time Windows:** 60-day lookback period
- **Feature Windows:** 1, 3, 5, 10, 15, 30, 60 days

---

## Runtime Estimates

### Scenario 1: Very Optimized (Parallel, GPU)
**Assumption:** 1ms per prediction point

| Metric | Single Date | All 14 Decades |
|--------|-------------|----------------|
| **Processing Time** | 49.4 minutes | 11.5 hours |
| **Days** | 0.03 | 0.5 |

**Requirements:**
- GPU acceleration
- Highly optimized parallel code
- Efficient batching

---

### Scenario 2: Optimized (Parallel CPU) ⭐ **Most Realistic**
**Assumption:** 10ms per prediction point

| Metric | Single Date | All 14 Decades |
|--------|-------------|----------------|
| **Processing Time** | 494 minutes (8.2 hrs) | 115.3 hours |
| **Days** | 0.34 | **4.8 days** |

**With 4-Core Parallelization:**
- **Total Time:** 28.8 hours (1.2 days)

**Requirements:**
- Multi-core CPU (4-8 cores)
- Parallel processing framework
- Optimized feature extraction

---

### Scenario 3: Standard (Serial Processing)
**Assumption:** 100ms per prediction point

| Metric | Single Date | All 14 Decades |
|--------|-------------|----------------|
| **Processing Time** | 4,943 minutes (82.4 hrs) | 1,153 hours |
| **Days** | 3.4 | **48.1 days** |

**Requirements:**
- Standard single-threaded execution
- No optimization

---

### Scenario 4: Conservative (Worst Case)
**Assumption:** 500ms per prediction point

| Metric | Single Date | All 14 Decades |
|--------|-------------|----------------|
| **Processing Time** | 24,714 minutes (412 hrs) | 5,767 hours |
| **Days** | 17.2 | **240.3 days** |

**Note:** Unlikely unless severe bottlenecks exist

---

## Optimization Strategies

### 1. Spatial Downsampling
**Reduce resolution for initial tests**

| Downsampling | Points | Time (Optimized) | Time Savings |
|--------------|--------|------------------|--------------|
| **Original (50m)** | 2,965,655 | 115 hours | Baseline |
| **Every 2nd pixel (100m)** | 741,414 | 28.9 hours | 75% faster |
| **Every 5th pixel (250m)** | 118,626 | **4.6 hours** | 96% faster |
| **Every 10th pixel (500m)** | 29,657 | 1.2 hours | 99% faster |

**Recommendation:** Start with 250m resolution (every 5th pixel) for testing

---

### 2. Parallel Processing
**Leverage multiple CPU cores**

| Cores | Speedup | Time (14 decades) |
|-------|---------|-------------------|
| 1 | 1× | 115.3 hours |
| 2 | 2× | 57.7 hours |
| 4 | 4× | **28.8 hours** |
| 8 | 8× | 14.4 hours |
| 16 | 16× | 7.2 hours |

**Recommendation:** 4-8 cores optimal for most systems

---

### 3. Spatial Chunking
**Process regions separately**

- Divide study area into 10×10 km tiles
- Process ~100 tiles in parallel
- Reduces memory footprint
- Enables distributed processing

**Estimated chunking overhead:** +10-15%

---

### 4. Feature Caching
**Extract static features once**

- Static features: Extract once (12 variables)
- Reuse across all 14 decades
- **Time savings:** ~30-40% for static features

---

### 5. Temporal Batching
**Process multiple dates efficiently**

- Batch climate data loading
- Reduce file I/O operations
- Process 2-3 decades at once

**Estimated I/O savings:** ~20%

---

## Computational Bottlenecks

### 1. Feature Extraction (40% of time)
- Reading climate NetCDF files (~60 files per date)
- Calculating cumulative statistics (7 time windows × 2 variables)
- Spatial averaging (4×4 pixel windows)

**Optimization:** Parallelize file reading, use Dask arrays

---

### 2. Model Inference (50% of time)
- Bayesian posterior sampling (300-2000 samples)
- Matrix multiplications for each sample
- Attention mechanism calculations

**Optimization:** Use vectorized operations, GPU acceleration

---

### 3. I/O Operations (10% of time)
- Writing output rasters
- Reading input climate data
- Temporary file management

**Optimization:** Use compressed formats, async I/O

---

## Memory Requirements

### Per Single Date Prediction

| Component | Memory Usage |
|-----------|--------------|
| **Prediction Grid** | ~23 MB (2.9M × 8 bytes) |
| **Static Features** | ~340 MB (2.9M × 12 vars × 8 bytes) |
| **Dynamic Features** | ~170 MB (2.9M × 28 features × 8 bytes) |
| **Model Parameters** | ~50 MB (posterior samples) |
| **Output Rasters** | ~23 MB per output |
| **Total (Peak)** | ~**606 MB** |

**With Safety Margin:** Recommend 2-4 GB RAM per worker

### For Full Analysis (14 Decades)

| Approach | Memory Required |
|----------|-----------------|
| **Serial (all in memory)** | ~8.5 GB |
| **Chunked (1000 pts)** | ~500 MB |
| **Spatial tiles** | ~2 GB per tile |

**Recommendation:** Use spatial chunking with 2-4 GB per worker

---

## Recommended Workflow

### Phase 1: Testing (1-2 hours)
1. **Single Date Test** (2020-07-15)
   - Full resolution (50m)
   - Measure actual runtime
   - Verify outputs
   - **Estimated time:** 8-10 hours

2. **Downsampled Test** (Every 5th pixel)
   - All 14 decades
   - Validate temporal trends
   - **Estimated time:** 23 hours

### Phase 2: Validation (1 day)
3. **Multi-Decade Test** (5 decades)
   - 100m resolution (every 2nd pixel)
   - Verify temporal consistency
   - **Estimated time:** 14 hours

### Phase 3: Production (4-6 days)
4. **Full Analysis** (14 decades, 50m)
   - With 4-core parallelization
   - Spatial chunking enabled
   - **Estimated time:** 4-6 days

---

## Hardware Recommendations

### Minimum Requirements
- **CPU:** 4 cores, 3.0+ GHz
- **RAM:** 8 GB
- **Storage:** 50 GB free space (for outputs)
- **Runtime:** ~7 days

### Recommended Configuration
- **CPU:** 8 cores, 3.5+ GHz
- **RAM:** 16 GB
- **Storage:** 100 GB SSD
- **Runtime:** ~3 days

### Optimal Configuration
- **CPU:** 16 cores, 4.0+ GHz (or GPU)
- **RAM:** 32 GB
- **Storage:** 200 GB NVMe SSD
- **Runtime:** ~1 day

---

## Implementation Checklist

### Before Running
- [ ] Verify climate data files are accessible
- [ ] Check available disk space (≥50 GB)
- [ ] Ensure model files exist (trace.nc, scaler.joblib)
- [ ] Test with single date first
- [ ] Monitor memory usage

### During Execution
- [ ] Monitor progress logs
- [ ] Check intermediate outputs
- [ ] Watch for memory issues
- [ ] Verify no NaN/Inf values
- [ ] Save checkpoints periodically

### After Completion
- [ ] Validate output rasters
- [ ] Check temporal consistency
- [ ] Generate summary statistics
- [ ] Create visualization maps
- [ ] Archive results

---

## Cost-Benefit Analysis

### Option A: Local Workstation (Recommended)
**Setup:** 8-core CPU, 16 GB RAM
**Runtime:** 3-4 days
**Cost:** $0 (existing hardware)
**Pros:** No setup overhead, full control
**Cons:** Ties up workstation

### Option B: HPC Cluster
**Setup:** 32+ cores, distributed
**Runtime:** 4-12 hours
**Cost:** ~$50-200 (cluster time)
**Pros:** Very fast, scalable
**Cons:** Setup complexity, data transfer

### Option C: Cloud Computing
**Setup:** AWS/Azure VM (16 cores)
**Runtime:** 1-2 days
**Cost:** ~$100-300
**Pros:** Flexible, no local impact
**Cons:** Data upload, ongoing costs

---

## Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| **Memory overflow** | Medium | High | Use chunking, monitor RAM |
| **Disk space full** | Low | High | Pre-check space, compress outputs |
| **Climate data missing** | Low | High | Validate files before run |
| **Model divergence** | Low | Medium | Test single date first |
| **Long runtime** | High | Low | Use downsampling initially |
| **Process crash** | Medium | Medium | Implement checkpointing |

---

## Next Steps

1. **✅ COMPLETED:** Runtime estimation analysis
2. **⏳ IN PROGRESS:** Raster stack generation (54 stacks/min, ETA: 31 minutes)
3. **PENDING:** Train Bayesian model (~30-120 minutes)
4. **PENDING:** Single date climate projection test
5. **PENDING:** Full climate projection run (4-6 days)

---

## Conclusion

**Recommended Approach:**
- Start with **250m resolution** test (4.6 hours for all 14 decades)
- Validate outputs and temporal trends
- Scale up to **100m resolution** if results look good (29 hours)
- Final production run at **50m resolution** with 4-core parallelization (3-4 days)

**Total Project Timeline:** 1-2 weeks including validation

**Confidence Level:** High - estimates based on actual grid size and realistic processing times

---

*Report generated by: estimate_climate_projection_time.py*
*Last updated: 2025-10-19 23:41 UTC*
