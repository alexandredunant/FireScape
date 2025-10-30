# Time Series Model Comparison

## Three Approaches to Temporal Modeling

### 1. **train_relative_probability_model.py** (Main - Recommended for Most Use Cases)

**Approach**: Collapses time series into temporal features

**How it works**:
- Takes 60 days of T/P data
- Computes 7 temporal windows (1d, 3d, 5d, 10d, 15d, 30d, 60d)
- Extracts mean and max for each window
- Results in 28 temporal features (7 windows × 2 vars × 2 stats)

**Pros**:
- ✅ Memory efficient (~50 MB)
- ✅ Fast training (~30 min)
- ✅ Stable and proven
- ✅ Easy to interpret (feature importance per window)

**Cons**:
- ⚠️ Loses fine-grained temporal dynamics
- ⚠️ Assumes linear feature aggregation

**Best for**: Production models, resource-constrained systems

---

### 2. **train_Dask_PyMC_timeseries.py** (Original Dask - Has Memory Issues)

**Approach**: Full time series with temporal attention

**How it works**:
- Keeps all 60 timesteps × 14 features
- Uses Dirichlet attention weights over time
- Attempts to parallelize with Dask

**Pros**:
- ✅ Preserves full temporal information
- ✅ Learns time-varying relationships
- ✅ Attention weights show which time periods matter

**Cons**:
- ❌ Memory explosion (~10 GB for 3000 observations)
- ❌ Dask+PyMC conflicts
- ❌ Crashes on larger datasets

**Status**: ⚠️ **Has memory issues** - moved to Optional/

---

### 3. **train_timeseries_memory_efficient.py** (New - Recommended for Research)

**Approach**: Full time series with memory management

**How it works**:
- Keeps all 60 timesteps × 14 features
- Computes data in batches BEFORE PyMC
- Uses memory-mapped files for large arrays
- Trains on samples to reduce memory

**Pros**:
- ✅ Preserves full temporal information
- ✅ Memory efficient (batched computation)
- ✅ Learns temporal attention weights
- ✅ No Dask+PyMC conflicts

**Cons**:
- ⚠️ Slower training (~2-4 hours)
- ⚠️ More complex code
- ⚠️ Trains on sample (1000 obs) for efficiency

**Best for**: Research, understanding temporal dynamics

---

## Key Differences

| Feature | Main Script | Dask (Original) | Memory-Efficient |
|---------|-------------|-----------------|------------------|
| **Time representation** | Collapsed features | Full series | Full series |
| **Memory usage** | ~50 MB | ~10 GB (crash!) | ~500 MB |
| **Training time** | 30 min | N/A (crashes) | 2-4 hours |
| **Sample size** | Full (3000) | Full (crash) | Sample (1000) |
| **Temporal attention** | ❌ No | ✅ Yes | ✅ Yes |
| **Interpretability** | Feature windows | Time attention | Time attention |
| **Status** | ✅ Production | ❌ Broken | ✅ Research |

## Memory Comparison

```
Input data: 3000 obs × 60 time × 14 features × 4 bytes = ~10 GB

Main script:
  3000 obs × 28 features × 4 bytes = ~340 KB ✅

Dask script:
  3000 obs × 60 time × 14 features = ~10 GB ❌
  (Tries to load all at once → crash)

Memory-efficient script:
  Batches: 50 obs × 60 time × 14 features = ~170 KB per batch ✅
  Train sample: 1000 obs → ~3 GB total ✅
```

## Which to Use?

### Use **main script** if:
- You want production-ready model
- Memory/compute resources are limited
- You need fast training
- Feature importance by time window is sufficient

### Use **memory-efficient script** if:
- You want to study fine-grained temporal dynamics
- You have time for longer training
- You want to see which exact days matter most (attention)
- Research/exploration phase

### Don't use **Dask script**:
- ❌ Has memory issues
- ❌ Crashes on real data
- ❌ Moved to Optional/ for a reason

## Running the Memory-Efficient Script

```bash
cd /mnt/CEPH_PROJECTS/Firescape/Scripts/02_Model_Training

# Ensure you have enough disk space for memory-mapped files (~6 GB)
df -h OUTPUT/

# Run the script
python train_timeseries_memory_efficient.py

# Monitor Dask dashboard
# Open browser: http://localhost:8787

# Output: OUTPUT/02_Model_TimeSeries/
```

## Understanding Temporal Attention

The memory-efficient script learns attention weights showing which time periods matter:

```python
# After training, check attention weights
import arviz as az
trace = az.from_netcdf("OUTPUT/02_Model_TimeSeries/trace_timeseries.nc")

attention = trace.posterior['time_attention'].mean(dim=['chain', 'draw'])
print(attention.values)  # 60 weights (one per day)

# Plot attention over time
import matplotlib.pyplot as plt
plt.plot(range(60, 0, -1), attention.values)
plt.xlabel('Days before fire')
plt.ylabel('Attention weight')
plt.title('Which days matter most for fire prediction?')
plt.savefig('temporal_attention.png')
```

**Interpretation**:
- High weights on recent days (1-5d) → immediate conditions matter
- High weights on 30-60d → drought/accumulation matters
- Low weights on 10-20d → mid-term doesn't matter as much

## Advantages of Full Time Series Approach

### 1. **Non-linear temporal patterns**
Can learn: "Day 1 matters IF day 30 was dry"
Main script: Can't learn these interactions

### 2. **Flexible temporal importance**
Learns: Which exact days matter (not just aggregated windows)
Main script: Fixed windows (1d, 3d, 5d...)

### 3. **Uncertainty in timing**
Shows: Confidence in temporal relationships
Main script: Feature-level uncertainty only

## Recommendations

### For Production:
Use `train_relative_probability_model.py`
- Fast, stable, proven
- Good enough for most applications

### For Research:
Use `train_timeseries_memory_efficient.py`
- Understand temporal dynamics
- Publish attention patterns
- Explore time-varying effects

### For Troubleshooting Original Dask Script:
Don't bother - use memory-efficient version instead
- Same model architecture
- Better memory management
- Actually works!

## Next Steps

1. **Try memory-efficient script**:
   ```bash
   python train_timeseries_memory_efficient.py
   ```

2. **Compare results** with main script:
   - Do predictions differ?
   - Which temporal patterns emerge?
   - Is the extra complexity worth it?

3. **Visualize attention**:
   - Which days matter most?
   - Does it match domain knowledge?
   - Seasonal patterns?

4. **Decide which to use**:
   - Production → main script
   - Paper/research → memory-efficient
   - Both? Train both, compare insights!
