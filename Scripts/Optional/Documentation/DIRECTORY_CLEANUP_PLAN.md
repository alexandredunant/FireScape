# Scripts Directory Cleanup Plan

## Current State (Messy)

```
Scripts/
├── FINAL_LANDCOVER_FIX.md              ← Root (messy!)
├── IMPROVEMENTS_SUMMARY.md             ← Root
├── LANDCOVER_ENCODING_ISSUE.md         ← Root
├── LANDCOVER_FIX_CHECKLIST.md          ← Root
├── README_COMPLETE_SOLUTION.md         ← Root
├── STANDARDIZATION_GUIDE.md            ← Root
├── TEMPORAL_VALIDATION_GUIDE.md        ← Root
├── WORKFLOW_SUMMARY.md                 ← Root
├── apply_landcover_fix.py              ← Root
├── shared_prediction_utils.py          ← Root
├── 01_Data_Preparation/
├── 02_Model_Training/
├── 03_Climate_Projections/
├── 04_Zone_Climate_Projections/
├── 05_Lightning_Comparison/
└── Archive/
```

**Problem**: 8 markdown docs + 2 Python scripts scattered at root level!

## Proposed Structure (Clean)

```
Scripts/
├── README.md                           ← Main entry point (NEW)
├── 00_Documentation/                   ← NEW directory
│   ├── README.md                       ← Documentation index
│   ├── Landcover_Fix/                  ← Landcover-specific docs
│   │   ├── 01_PROBLEM.md               ← Renamed from LANDCOVER_ENCODING_ISSUE.md
│   │   ├── 02_SOLUTION.md              ← Renamed from FINAL_LANDCOVER_FIX.md
│   │   └── 03_CHECKLIST.md             ← Renamed from LANDCOVER_FIX_CHECKLIST.md
│   ├── Pipeline_Improvements/          ← General improvements
│   │   ├── IMPROVEMENTS_SUMMARY.md     ← Moved here
│   │   └── COMPLETE_SOLUTION.md        ← Renamed from README_COMPLETE_SOLUTION.md
│   └── Workflow_Guides/                ← How-to guides
│       ├── STANDARDIZATION_GUIDE.md    ← Moved here
│       ├── TEMPORAL_VALIDATION_GUIDE.md← Moved here
│       └── WORKFLOW_SUMMARY.md         ← Moved here
├── 00_Utilities/                       ← NEW directory
│   ├── apply_landcover_fix.py          ← Moved here
│   ├── shared_prediction_utils.py      ← Moved here
│   └── README.md                       ← Utilities index (NEW)
├── 01_Data_Preparation/
│   ├── CORINE_LANDCOVER_FIRE_RISK_MAPPING.py
│   ├── create_raster_stacks.py
│   └── ...
├── 02_Model_Training/
│   ├── train_relative_probability_model.py
│   ├── fix_landcover_encoding.py
│   └── ...
├── 03_Climate_Projections/
│   ├── README.md                       ← Already exists
│   └── ...
├── 04_Zone_Climate_Projections/
├── 05_Lightning_Comparison/
└── Archive/
```

## Migration Commands

### Step 1: Create new directories
```bash
cd /mnt/CEPH_PROJECTS/Firescape/Scripts

mkdir -p 00_Documentation/Landcover_Fix
mkdir -p 00_Documentation/Pipeline_Improvements
mkdir -p 00_Documentation/Workflow_Guides
mkdir -p 00_Utilities
```

### Step 2: Move and rename landcover docs
```bash
# Landcover fix documentation
mv LANDCOVER_ENCODING_ISSUE.md 00_Documentation/Landcover_Fix/01_PROBLEM.md
mv FINAL_LANDCOVER_FIX.md 00_Documentation/Landcover_Fix/02_SOLUTION.md
mv LANDCOVER_FIX_CHECKLIST.md 00_Documentation/Landcover_Fix/03_CHECKLIST.md
```

### Step 3: Move general improvement docs
```bash
mv IMPROVEMENTS_SUMMARY.md 00_Documentation/Pipeline_Improvements/
mv README_COMPLETE_SOLUTION.md 00_Documentation/Pipeline_Improvements/COMPLETE_SOLUTION.md
```

### Step 4: Move workflow guides
```bash
mv STANDARDIZATION_GUIDE.md 00_Documentation/Workflow_Guides/
mv TEMPORAL_VALIDATION_GUIDE.md 00_Documentation/Workflow_Guides/
mv WORKFLOW_SUMMARY.md 00_Documentation/Workflow_Guides/
```

### Step 5: Move utility scripts
```bash
mv apply_landcover_fix.py 00_Utilities/
mv shared_prediction_utils.py 00_Utilities/
```

### Step 6: Update symlinks/imports (if needed)
```bash
# If any scripts import these utilities, update paths
# Most likely only apply_landcover_fix.py needs path adjustments
```

## New README Files to Create

### 1. Scripts/README.md (Main entry point)
Provides:
- Overview of entire pipeline
- Quick start guide
- Directory structure
- Links to detailed documentation

### 2. 00_Documentation/README.md
Index of all documentation with descriptions:
- Landcover fix documentation
- Pipeline improvements
- Workflow guides

### 3. 00_Utilities/README.md
Description of utility scripts:
- apply_landcover_fix.py - Automated landcover encoding fix
- shared_prediction_utils.py - Shared prediction functions

## Benefits

✅ **Cleaner root**: Only numbered directories + Archive
✅ **Logical grouping**: Related docs together
✅ **Easy navigation**: Clear directory names
✅ **Numbered order**: 00_ ensures docs/utilities appear first
✅ **Better naming**: Descriptive filenames (01_PROBLEM.md vs LANDCOVER_ENCODING_ISSUE.md)
✅ **Scalable**: Easy to add new docs without cluttering root

## What to Keep at Root

Only these should remain at root level:
- **README.md** (main entry point - NEW)
- **Numbered directories** (01_, 02_, 03_, etc.)
- **Archive/** (historical files)

## Implementation Priority

### High Priority (Do first)
1. Create new directories
2. Move documentation files (no code changes needed)
3. Create main README.md

### Medium Priority
4. Move utility scripts
5. Update apply_landcover_fix.py paths (if broken)
6. Create documentation indices

### Low Priority
7. Update any references in other docs (links)
8. Test that utilities still work from new location

## Backward Compatibility

For any external references to old paths, create symlinks:
```bash
# Example: If something references old path
ln -s 00_Documentation/Landcover_Fix/02_SOLUTION.md FINAL_LANDCOVER_FIX.md
```

## Rollback Plan

If issues occur:
```bash
# Restore original structure
mv 00_Documentation/Landcover_Fix/* .
mv 00_Documentation/Pipeline_Improvements/* .
mv 00_Documentation/Workflow_Guides/* .
mv 00_Utilities/* .
rm -r 00_Documentation 00_Utilities
```

## Validation

After cleanup, verify:
```bash
# Root should be clean
ls -1 | grep -E "^\." | wc -l  # Should be small

# All docs in 00_Documentation
find 00_Documentation -name "*.md" | wc -l  # Should be 8

# All utilities in 00_Utilities
find 00_Utilities -name "*.py" | wc -l  # Should be 2
```

## Next Steps

1. Review this plan
2. Run cleanup script (I'll create this)
3. Test that everything still works
4. Update any broken references
5. Create new README files
