# Firescape Utility Scripts

Reusable utility scripts for the Firescape wildfire risk modeling pipeline.

## 📄 Utilities

### 1. apply_landcover_fix.py

**Purpose**: Automatically fix landcover categorical encoding across all pipeline scripts.

**What it does**:
- Backs up all affected files
- Replaces `'landcoverfull'` with `'landcover_fire_risk'` in STATIC_VARS lists
- Adds Corine Land Cover → fire risk ordinal mapping (0-5 scale)
- Updates feature extraction code to use ordinal encoding
- Generates summary report

**Affected files** (7 scripts):
1. `01_Data_Preparation/create_raster_stacks.py`
2. `02_Model_Training/train_relative_probability_model.py`
3. `02_Model_Training/train_Dask_PyMC_timeseries.py`
4. `03_Climate_Projections/extract_projection_features.py`
5. `04_Zone_Climate_Projections/project_zone_fire_risk.py`
6. `05_Lightning_Comparison/01_Data_Preparation/create_raster_stacks_with_lightning.py`
7. `05_Lightning_Comparison/02_Model_Training/train_relative_probability_model_with_lightning.py`

**Usage**:
```bash
cd /mnt/CEPH_PROJECTS/Firescape/Scripts/00_Utilities
python apply_landcover_fix.py
```

**What happens**:
1. Shows list of files to update
2. Prompts for confirmation
3. Creates backup files (`*_backup_YYYYMMDD_HHMMSS.py`)
4. Applies fixes
5. Generates summary report

**After running**:
- Review changes in each file
- Retrain model: `python ../02_Model_Training/train_relative_probability_model.py`
- Re-run climate projections if needed

**Rollback**:
```bash
# If issues occur, restore from backup
cp 01_Data_Preparation/create_raster_stacks_backup_*.py 01_Data_Preparation/create_raster_stacks.py
```

**Documentation**: [../00_Documentation/Landcover_Fix/02_SOLUTION.md](../00_Documentation/Landcover_Fix/02_SOLUTION.md)

---

### 2. shared_prediction_utils.py

**Purpose**: Shared utility functions for generating predictions across different prediction scripts.

**Functions** (to be added as needed):
- Feature extraction helpers
- Prediction generation functions
- Uncertainty quantification utilities
- Common validation metrics

**Status**: Template/placeholder - populate as needed when functions are extracted from main scripts.

**Usage**:
```python
# In your prediction script
import sys
sys.path.append('/mnt/CEPH_PROJECTS/Firescape/Scripts/00_Utilities')
from shared_prediction_utils import generate_predictions, compute_uncertainty
```

**Best Practices**:
- Add well-documented, reusable functions here
- Include type hints and docstrings
- Keep functions small and focused
- Add unit tests if possible

---

## 🛠️ Adding New Utilities

When creating new utility scripts:

### 1. Purpose
- Should be **reusable** across multiple pipeline components
- Should **not** contain hardcoded paths (use parameters)
- Should have **clear, single responsibility**

### 2. Structure
```python
#!/usr/bin/env python3
"""
Brief description of utility.

Purpose:
- List main purposes

Functions:
- function_name(): Brief description

Usage:
    python utility_name.py [args]
"""

import necessary_modules

def main_function(param1, param2):
    """
    Description.

    Args:
        param1: Description
        param2: Description

    Returns:
        Description
    """
    pass

if __name__ == "__main__":
    # CLI interface if applicable
    pass
```

### 3. Documentation
- Add docstrings for all functions
- Include usage examples
- Update this README with new utility

### 4. Testing
```python
# Test your utility
python utility_name.py --test
# Or
pytest test_utility_name.py
```

## 📦 Dependencies

Utilities should minimize dependencies to stay portable. Common imports:
- Standard library: `os`, `sys`, `pathlib`, `re`, `datetime`
- Data: `pandas`, `numpy`, `xarray`
- ML: `joblib`, `arviz` (if needed)

## 🔧 Maintenance

### Updating Utilities

When updating a utility:
1. **Test first**: Ensure changes don't break existing scripts
2. **Version**: Add version comments if making significant changes
3. **Document**: Update docstrings and this README
4. **Notify**: If breaking changes, notify pipeline users

### Deprecation

When deprecating a utility:
1. Add deprecation warning in code
2. Update README with "DEPRECATED" notice
3. Provide migration path to new utility/approach
4. Remove after sufficient notice period

## 🐛 Troubleshooting

### ImportError: No module named 'shared_prediction_utils'
**Solution**: Add utilities directory to Python path:
```python
import sys
sys.path.append('/mnt/CEPH_PROJECTS/Firescape/Scripts/00_Utilities')
```

### apply_landcover_fix.py errors
**Common issues**:
- **File not found**: Check that you're in Scripts directory
- **Permission denied**: Ensure files are writable
- **Backup collision**: Delete old backup files first

**Detailed troubleshooting**: [../00_Documentation/Landcover_Fix/03_CHECKLIST.md](../00_Documentation/Landcover_Fix/03_CHECKLIST.md#troubleshooting)

## 📚 Related Documentation

- **Landcover Fix**: [../00_Documentation/Landcover_Fix/](../00_Documentation/Landcover_Fix/)
- **Pipeline Documentation**: [../00_Documentation/Pipeline_Improvements/](../00_Documentation/Pipeline_Improvements/)
- **Main README**: [../README.md](../README.md)

---

**Last Updated**: 2025-10-28
**Utilities Count**: 2 (1 active, 1 template)
