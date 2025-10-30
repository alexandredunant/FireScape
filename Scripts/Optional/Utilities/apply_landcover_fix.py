#!/usr/bin/env python3
"""
Apply Landcover Encoding Fix Across All Scripts

This script systematically updates all files that use 'landcoverfull'
to use proper ordinal encoding based on fire risk.

WHAT IT DOES:
1. Identifies all Python files using landcoverfull
2. Creates backup copies
3. Applies ordinal encoding fix
4. Generates a report of changes

FILES TO UPDATE (non-archived):
- 01_Data_Preparation/create_raster_stacks.py
- 02_Model_Training/train_relative_probability_model.py
- 02_Model_Training/train_Dask_PyMC_timeseries.py
- 03_Climate_Projections/extract_projection_features.py
- 04_Zone_Climate_Projections/project_zone_fire_risk.py
- 05_Lightning_Comparison/01_Data_Preparation/create_raster_stacks_with_lightning.py
- 05_Lightning_Comparison/02_Model_Training/train_relative_probability_model_with_lightning.py
"""

import os
import re
from pathlib import Path
from datetime import datetime
import shutil

# Base directory
BASE_DIR = Path("/mnt/CEPH_PROJECTS/Firescape/Scripts")

# Ordinal mapping (consistent across all files)
# CORRECTED: Uses actual Corine Land Cover codes from dataset
LANDCOVER_ORDINAL_MAPPING = """
# Corine Land Cover → Fire Risk Ordinal Mapping
# Based on actual 3-digit Corine codes in Bolzano Province dataset
LANDCOVER_FIRE_RISK_ORDINAL = {
    # 0 = No fire risk (water, snow/ice, wetlands)
    335: 0, 511: 0, 512: 0, 411: 0, 412: 0,

    # 1 = Very low fire risk (urban, bare soil)
    111: 1, 112: 1, 121: 1, 122: 1, 124: 1, 131: 1, 133: 1, 142: 1, 331: 1, 332: 1,

    # 2 = Low fire risk (agriculture, managed land)
    211: 2, 221: 2, 222: 2, 231: 2, 242: 2, 243: 2,

    # 3 = Moderate fire risk (grassland, broadleaf forest)
    311: 3, 321: 3, 333: 3,

    # 4 = High fire risk (shrubland, mixed forest)
    313: 4, 322: 4, 324: 4,

    # 5 = Very high fire risk (coniferous forest)
    312: 5
}
"""

# Files to update (relative to BASE_DIR)
FILES_TO_UPDATE = [
    "01_Data_Preparation/create_raster_stacks.py",
    "02_Model_Training/train_relative_probability_model.py",
    "02_Model_Training/train_Dask_PyMC_timeseries.py",
    "03_Climate_Projections/extract_projection_features.py",
    "04_Zone_Climate_Projections/project_zone_fire_risk.py",
    "05_Lightning_Comparison/01_Data_Preparation/create_raster_stacks_with_lightning.py",
    "05_Lightning_Comparison/02_Model_Training/train_relative_probability_model_with_lightning.py",
]


def backup_file(filepath):
    """Create a backup of the file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = filepath.parent / f"{filepath.stem}_backup_{timestamp}{filepath.suffix}"
    shutil.copy2(filepath, backup_path)
    return backup_path


def update_static_vars_list(content):
    """Update STATIC_VARS list to replace 'landcoverfull' with 'landcover_fire_risk'."""
    # Pattern: 'landcoverfull' in a list
    pattern = r"'landcoverfull'"
    replacement = "'landcover_fire_risk'"

    updated = re.sub(pattern, replacement, content)
    return updated


def add_ordinal_mapping(content):
    """Add ordinal mapping if not already present."""
    if "LANDCOVER_FIRE_RISK_ORDINAL" in content:
        return content, False  # Already has mapping

    # Find a good place to add it (after imports, before main code)
    # Look for common patterns like "# Configuration" or "# ===== CONFIGURATION"
    patterns = [
        r"(# ={3,}\n# CONFIGURATION\n# ={3,}\n)",
        r"(# Configuration\n)",
        r"(# Feature categories\n)",
    ]

    for pattern in patterns:
        match = re.search(pattern, content)
        if match:
            insert_pos = match.end()
            updated = content[:insert_pos] + "\n" + LANDCOVER_ORDINAL_MAPPING + "\n" + content[insert_pos:]
            return updated, True

    # If no good pattern found, add after imports (before first non-comment, non-import line)
    lines = content.split('\n')
    insert_idx = 0
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped and not stripped.startswith('#') and not stripped.startswith('import') and not stripped.startswith('from'):
            insert_idx = i
            break

    lines.insert(insert_idx, LANDCOVER_ORDINAL_MAPPING)
    updated = '\n'.join(lines)
    return updated, True


def update_landcover_extraction(content):
    """Update landcover extraction to use ordinal encoding."""

    # Pattern 1: create_raster_stacks.py style (mode extraction)
    # Replace the part that assigns mode_result.mode to features[var_name]
    pattern1 = r"(mode_result = stats\.mode\(values.*?\n\s+features\[)var_name\] = (float|int)\(mode_result\.mode\)"

    replacement1 = r"""\1'landcover_fire_risk'] = float(
                        LANDCOVER_FIRE_RISK_ORDINAL.get(int(mode_result.mode), 2)  # Default: moderate
                    )"""

    updated = re.sub(pattern1, replacement1, content, flags=re.DOTALL)

    # Pattern 2: extract_projection_features.py style
    pattern2 = r"features\[var_name\] = int\(mode_result\.mode\)"
    replacement2 = """landcover_class = int(mode_result.mode)
                features['landcover_fire_risk'] = LANDCOVER_FIRE_RISK_ORDINAL.get(
                    landcover_class, 2
                )"""

    updated = re.sub(pattern2, replacement2, updated)

    # Pattern 3: project_zone_fire_risk.py - just extracting from raster
    # This one doesn't do mode, it directly extracts. Need to add post-processing
    # Look for: static_features_df[var_name] = extracted.values
    # We'll handle this separately below

    return updated


def add_landcover_postprocessing(content, filename):
    """Add post-processing for landcover in project_zone_fire_risk.py."""

    if "project_zone_fire_risk" not in filename:
        return content

    # Find the static features extraction loop
    pattern = r"(for var_name in .*?STATIC_VARS.*?:\n.*?static_features_df\[var_name\] = extracted\.values\n)"

    match = re.search(pattern, content, flags=re.DOTALL)
    if not match:
        return content

    # Add post-processing after the loop
    postprocess_code = """
# Post-process landcover: convert to fire risk ordinal
if 'landcoverfull' in static_features_df.columns:
    static_features_df['landcover_fire_risk'] = static_features_df['landcoverfull'].apply(
        lambda x: LANDCOVER_FIRE_RISK_ORDINAL.get(int(x) if not pd.isna(x) else 0, 2)
    )
    static_features_df = static_features_df.drop(columns=['landcoverfull'])
"""

    insert_pos = match.end()
    updated = content[:insert_pos] + postprocess_code + content[insert_pos:]

    return updated


def remove_categorical_vars_list(content):
    """Remove CATEGORICAL_VARS list if it only contains landcoverfull."""
    # Pattern: CATEGORICAL_VARS = ['landcoverfull']
    pattern = r"CATEGORICAL_VARS\s*=\s*\['landcoverfull'\]\s*\n"
    updated = re.sub(pattern, "", content)

    # Also remove references to CATEGORICAL_VARS in conditionals
    pattern2 = r"if var_name in CATEGORICAL_VARS:"
    replacement2 = "if var_name == 'landcoverfull':  # Categorical - encode as ordinal fire risk"
    updated = re.sub(pattern2, replacement2, updated)

    return updated


def process_file(filepath):
    """Process a single file to apply landcover fix."""
    print(f"\n{'='*80}")
    print(f"Processing: {filepath.relative_to(BASE_DIR)}")
    print(f"{'='*80}")

    if not filepath.exists():
        print(f"  ⚠️  File not found, skipping")
        return False

    # Read content
    with open(filepath, 'r', encoding='utf-8') as f:
        original_content = f.read()

    # Backup
    backup_path = backup_file(filepath)
    print(f"  ✓ Backup created: {backup_path.name}")

    # Apply fixes
    content = original_content
    changes_made = []

    # 1. Update STATIC_VARS list
    new_content = update_static_vars_list(content)
    if new_content != content:
        changes_made.append("Updated STATIC_VARS list")
        content = new_content

    # 2. Add ordinal mapping
    new_content, mapping_added = add_ordinal_mapping(content)
    if mapping_added:
        changes_made.append("Added LANDCOVER_FIRE_RISK_ORDINAL mapping")
        content = new_content

    # 3. Update landcover extraction
    new_content = update_landcover_extraction(content)
    if new_content != content:
        changes_made.append("Updated landcover extraction to use ordinal encoding")
        content = new_content

    # 4. Add post-processing (for project_zone_fire_risk.py)
    new_content = add_landcover_postprocessing(content, filepath.name)
    if new_content != content:
        changes_made.append("Added landcover post-processing")
        content = new_content

    # 5. Remove CATEGORICAL_VARS
    new_content = remove_categorical_vars_list(content)
    if new_content != content:
        changes_made.append("Removed/updated CATEGORICAL_VARS references")
        content = new_content

    if changes_made:
        # Write updated content
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)

        print(f"\n  ✓ Changes applied:")
        for change in changes_made:
            print(f"    - {change}")

        return True
    else:
        print(f"  ℹ️  No changes needed (may already be fixed)")
        return False


def main():
    """Main execution."""
    print("="*80)
    print("LANDCOVER ENCODING FIX - BATCH APPLICATION")
    print("="*80)
    print()
    print("This script will:")
    print("  1. Create backups of all files")
    print("  2. Replace 'landcoverfull' with 'landcover_fire_risk' in STATIC_VARS")
    print("  3. Add ordinal fire risk mapping")
    print("  4. Update feature extraction code")
    print()

    input("Press Enter to continue (or Ctrl+C to cancel)...")
    print()

    updated_files = []
    skipped_files = []

    for rel_path in FILES_TO_UPDATE:
        filepath = BASE_DIR / rel_path
        success = process_file(filepath)

        if success:
            updated_files.append(rel_path)
        else:
            skipped_files.append(rel_path)

    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nUpdated files: {len(updated_files)}")
    for path in updated_files:
        print(f"  ✓ {path}")

    if skipped_files:
        print(f"\nSkipped files: {len(skipped_files)}")
        for path in skipped_files:
            print(f"  ○ {path}")

    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("""
1. Review the changes in each file (backup files created)
2. Test each updated script:
   - Run create_raster_stacks.py to regenerate training data
   - Run train_relative_probability_model.py to retrain model
   - Run climate projection scripts

3. If issues occur:
   - Restore from backup files (*_backup_*.py)
   - Check LANDCOVER_ENCODING_ISSUE.md for manual fix instructions

4. After successful testing:
   - Delete backup files
   - Commit changes to version control
    """)


if __name__ == "__main__":
    main()
