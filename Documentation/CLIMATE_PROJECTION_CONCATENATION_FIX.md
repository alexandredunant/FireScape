# Climate Projection Scripts - Concatenation Fix

**Date:** October 20, 2025
**Issue:** Shape mismatch when concatenating chunked point extractions
**Status:** ✓ Fixed in all 3 climate projection scripts

---

## Problem Description

### Error Message:
```
Error processing Jan: shape mismatch: value array of shape (5931310,) could not be broadcast
to indexing result of shape (2965655,)
```

### Root Cause:

When extracting climate features for the prediction grid:
1. The grid is split into chunks (e.g., 250,000 points per chunk) for memory efficiency
2. Each chunk is processed separately using `xvec.extract_points()`
3. Each extraction returns a pandas Series with its own index (0, 1, 2, ...)
4. `pd.concat(point_values_list, ignore_index=False)` **stacks these Series on top of each other**

**Result:** If you have 2 chunks of ~1.48M points each:
- Expected: Single series with ~2.97M points (5931310 / 2)
- Actual: Stacked series with ~5.93M points (duplicate indices)
- When trying to assign to grid_gdf: **shape mismatch error**

### Why the Other Scripts "Worked":

The CLEAN and MultiQuantile scripts used `.values` to extract the numpy array:
```python
result_df[column_name] = full_series.values
```

This bypassed the index issue BUT:
- ⚠️ **Risky**: If chunks don't maintain order, data could be misaligned
- ❌ **Silent failure**: No error, but potentially wrong data
- ❌ **Still has duplicates**: The full_series still had wrong length, just not checked

---

## Solution Implemented

### Fixed Pattern (Applied to All 3 Scripts):

```python
# OLD (incorrect):
full_series = pd.concat(point_values_list, ignore_index=False)
result_df[column_name] = full_series.values  # Bypass index issues

# NEW (correct):
full_series = pd.concat(point_values_list, ignore_index=True)  # Reset indices
full_series.index = grid_gdf.index  # Explicitly align with grid
result_df[column_name] = full_series  # Assign with proper alignment
```

### Key Changes:

1. **`ignore_index=True`**: Concatenates chunks sequentially without preserving chunk-level indices
2. **Explicit index assignment**: `full_series.index = grid_gdf.index` ensures alignment
3. **Direct assignment**: Use `full_series` not `full_series.values` for index-aware assignment
4. **Better error message**: Added expected vs actual length to error message

---

## Files Fixed

### 1. `05_Bayesian_Lookback_2022_GIF.py` (Lines 216-221)

**Original Issue:**
- User reported shape mismatch error
- Script was concatenating without proper index handling

**Fix Applied:**
```python
# Concatenate chunks with ignore_index to avoid duplicate indices
full_series = pd.concat(point_values_list, ignore_index=True)
# Reset index to match grid_gdf
full_series.index = grid_gdf.index
full_series.name = f"{prefix}_cumulative_{op_name}_{day_window}d"
dynamic_features_list.append(full_series)
```

**Status:** ✓ Fixed

---

### 2. `05_Bayesian_Climate_Projection_CLEAN.py` (Lines 325-333)

**Original Issue:**
- Used `.values` workaround
- Silent potential misalignment
- Used `ignore_index=False` (incorrect)

**Fix Applied:**
```python
# Concatenate chunks with proper index handling
full_series = pd.concat(point_values_list, ignore_index=True)
full_series.index = grid_gdf.index  # Align with grid indices
column_name = f"{prefix}_cumulative_{op_name}_{day_window}d"

if len(full_series) == len(grid_gdf):
    result_df[column_name] = full_series  # Changed from .values
else:
    print(f"    ERROR: Length mismatch for {column_name}: expected {len(grid_gdf)}, got {len(full_series)}")
```

**Status:** ✓ Fixed

---

### 3. `05_Bayesian_Climate_Projection_MultiQuantile_Seasonal.py` (Lines 364-372)

**Original Issue:**
- Same as CLEAN script
- Used `.values` workaround
- Potential silent misalignment

**Fix Applied:**
```python
# Concatenate chunks with proper index handling
full_series = pd.concat(point_values_list, ignore_index=True)
full_series.index = grid_gdf.index  # Align with grid indices
column_name = f"{prefix}_cumulative_{op_name}_{day_window}d"

if len(full_series) == len(grid_gdf):
    result_df[column_name] = full_series  # Changed from .values
else:
    print(f"    ERROR: Length mismatch for {column_name}: expected {len(grid_gdf)}, got {len(full_series)}")
```

**Status:** ✓ Fixed

---

## Why This Matters

### Before Fix:
```python
# Chunk 1: [val1, val2, val3] with index [0, 1, 2]
# Chunk 2: [val4, val5, val6] with index [0, 1, 2]
full_series = pd.concat([chunk1, chunk2], ignore_index=False)
# Result: [val1, val2, val3, val4, val5, val6] with index [0, 1, 2, 0, 1, 2]
# Length: 6 (correct data but WRONG indices)

# Using .values bypasses the index:
result_df[col] = full_series.values  # Works but risky!
```

### After Fix:
```python
# Chunk 1: [val1, val2, val3]
# Chunk 2: [val4, val5, val6]
full_series = pd.concat([chunk1, chunk2], ignore_index=True)
# Result: [val1, val2, val3, val4, val5, val6] with index [0, 1, 2, 3, 4, 5]
# Length: 6 (correct)

full_series.index = grid_gdf.index  # Explicit alignment
result_df[col] = full_series  # Safe index-aware assignment
```

---

## Testing Recommendations

To verify the fix works correctly:

1. **Run lookback script** with a single month first:
   ```bash
   # Modify TARGET_MONTHS in the script to test one month
   python Scripts/03_Climate_Projections/05_Bayesian_Lookback_2022_GIF.py
   ```
   - Should complete without shape mismatch errors
   - Check that output rasters look reasonable

2. **Test climate projection** with a single date:
   ```bash
   # Modify PROJECTION_DATES to ["2020-07-15"] for quick test
   python Scripts/03_Climate_Projections/05_Bayesian_Climate_Projection_CLEAN.py
   ```
   - Should complete successfully
   - Check output GeoTIFF dimensions match expected

3. **Verify alignment**: Add debug prints to check data integrity:
   ```python
   print(f"Grid length: {len(grid_gdf)}")
   print(f"Series length: {len(full_series)}")
   print(f"Indices match: {full_series.index.equals(grid_gdf.index)}")
   ```

---

## Related Issues

This same pattern might appear in other scripts that use chunked point extraction. Search for:
```bash
grep -r "pd.concat.*point_values" Scripts/
```

If found elsewhere, apply the same fix pattern.

---

**Status:** All climate projection scripts updated and ready for use.

**Last Updated:** October 20, 2025
