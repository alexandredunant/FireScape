# Lightning Comparison - Ready to Train (2012-2024)

## ✅ Setup Complete!

All scripts have been updated to use the filtered 2012-2024 data where lightning measurements are complete.

---

## 📊 Data Status

### Filtered Datasets Created:
- ✅ **Baseline (T+P)**: `01_Training_Data_Baseline_2012plus/spacetime_stacks_baseline_2012plus.nc`
  - 1556 observations (2012-2024)
  - Fire rate: 0.237

- ✅ **Lightning (T+P+L)**: `01_Training_Data_Lightning_2012plus/spacetime_stacks_lightning_2012plus.nc`
  - 1556 observations (2012-2024)
  - Fire rate: 0.237
  - **Zero NaN values** - all lightning data valid!
  - **Fire days have 7.6x more lightning!**

### Key Finding:
Lightning signal is STRONG in the 2012-2024 data:
- Fire days: 16.3% have lightning, mean = 0.379
- Non-fire days: 3.5% have lightning, mean = 0.050
- Summer: 15.5% have lightning (as expected!)

---

## 🚀 Next Steps - Run Training

### Step 1: Train Lightning Model (T+P+L) on 2012-2024

```bash
cd /mnt/CEPH_PROJECTS/Firescape/Scripts/05_Lightning_Comparison/02_Model_Training
/home/adunant/miniconda3/envs/dask-geo/bin/python train_relative_probability_model_with_lightning.py
```

**What it does:**
- Loads: `spacetime_stacks_lightning_2012plus.nc` (1556 obs, 2012-2024)
- Trains: Bayesian model with lightning attention groups
- Outputs: `OUTPUT/02_Model_RelativeProbability_Lightning_2012plus/`
- Time: ~10-15 minutes

**Expected:**
- Lightning features should get significant attention weights
- `light_60d`, `light_30d`, `light_short` should rank high
- Model should show improved performance over T+P only

---

### Step 2: Compare Models on Same Period

```bash
cd /mnt/CEPH_PROJECTS/Firescape/Scripts/05_Lightning_Comparison/03_Model_Comparison
/home/adunant/miniconda3/envs/dask-geo/bin/python compare_models.py
```

**What it does:**
- Loads baseline model (T+P, trained 1999-2024)
- Filters baseline test set to 2012-2024 for fair comparison
- Loads lightning model (T+P+L, trained 2012-2024)
- Compares both on **identical time period**
- Outputs: `Results/model_comparison.png` and CSV

**Expected Results:**
With clean data (no NaN→0 corruption), lightning should show:
- ✅ Improved ROC-AUC and PR-AUC
- ✅ Better summer fire prediction
- ✅ Higher attention weights on lightning features
- ✅ Clear evidence lightning is valuable predictor

---

## 🔍 What Changed from Original Analysis?

### Problem Identified:
- Original analysis used 1999-2024 data (3035 obs)
- Lightning data only exists from 2012 onwards
- 48.7% of observations had NaN lightning values
- NaNs were filled with **zeros**, corrupting the signal
- Model learned to ignore lightning

### Solution Applied:
- Filtered both datasets to 2012-2024 (1556 obs)
- **Zero NaN values** - complete lightning coverage
- Fair comparison on identical time period
- No artificial zeros masking the true signal

### Evidence Signal Exists:
Fire days show **7.6x more lightning** in clean data!

---

## 📁 Files Modified

### Data Preparation:
- ✅ `01_Data_Preparation/filter_baseline_stack_2012plus.py` - Created
- ✅ `01_Data_Preparation/filter_lightning_stack_2012plus.py` - Created

### Model Training:
- ✅ `02_Model_Training/train_relative_probability_model_with_lightning.py` - Updated
  - Now uses: `spacetime_stacks_lightning_2012plus.nc`
  - Output dir: `02_Model_RelativeProbability_Lightning_2012plus/`

### Comparison:
- ✅ `03_Model_Comparison/compare_models.py` - Updated
  - Filters baseline test set to 2012-2024
  - Compares on same period
  - Recalculates metrics for fair comparison

### Not Modified (working well):
- ✅ `Scripts/02_Model_Training/train_relative_probability_model.py`
  - Baseline model performs well on 1999-2024 data
  - Will be evaluated on 2012-2024 subset for comparison

---

## 💡 Expected Outcomes

### If Lightning is Valuable (expected):
- ROC-AUC improvement: +2-5%
- PR-AUC improvement: +3-8%
- Lightning attention weights: Top 5-7 features
- Summer prediction: Substantial improvement
- Conclusion: **Include lightning in operational model**

### Key Metrics to Watch:
1. **Attention weights** - Should see:
   - `light_60d` in top 5
   - `light_30d` in top 7
   - Not at rank 18/18 like before!

2. **Summer performance** - Should see:
   - Better temporal correlation
   - Higher lift in summer months
   - Lightning capturing ignition events

3. **Overall metrics** - Should see:
   - Positive improvement, not decline
   - Clearer separation in ROC curves
   - Higher precision at same recall

---

## 🎯 Why This Will Work Now

1. **Complete data**: No missing values
2. **True signal**: 7.6x more lightning on fire days
3. **Fair comparison**: Same time period
4. **No corruption**: No zeros from missing data
5. **Known domain**: Lightning IS an ignition source in Alps

---

## 📞 Questions?

If you encounter issues:
1. Check that both filtered datasets exist
2. Verify no convergence warnings (r_hat < 1.01)
3. Confirm output directories are created
4. Check that test set sizes match between models

Good luck with training! The data looks very promising. 🔥⚡
