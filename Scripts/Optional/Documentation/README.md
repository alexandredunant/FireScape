# Firescape Documentation Index

This directory contains all project documentation organized by topic.

## 📚 Documentation Structure

```
00_Documentation/
├── Landcover_Fix/              # Landcover categorical encoding issue
│   ├── 01_PROBLEM.md           # What was wrong
│   ├── 02_SOLUTION.md          # How to fix it
│   └── 03_CHECKLIST.md         # Implementation checklist
├── Pipeline_Improvements/      # Overall project improvements
│   ├── COMPLETE_SOLUTION.md    # Main reference (START HERE!)
│   └── IMPROVEMENTS_SUMMARY.md # High-level summary
├── Workflow_Guides/            # How-to guides
│   ├── STANDARDIZATION_GUIDE.md
│   ├── TEMPORAL_VALIDATION_GUIDE.md
│   └── WORKFLOW_SUMMARY.md
└── DIRECTORY_CLEANUP_PLAN.md   # How docs were organized
```

## 🎯 Start Here

### New to the Project?
**→ [Pipeline_Improvements/COMPLETE_SOLUTION.md](Pipeline_Improvements/COMPLETE_SOLUTION.md)**

This is the **main entry point** covering:
- All 4 questions addressed (mid-month dates, landcover, visualization, scenarios)
- Complete setup instructions
- Landcover encoding fix (with correct Corine codes)
- Climate projection pipeline
- Troubleshooting guide

## 📖 Documentation by Topic

### 1. Landcover Encoding Fix (CRITICAL)

**Background**: The `landcoverfull` variable contains Corine Land Cover codes (3-digit: 111-512) but was incorrectly treated as a continuous numeric variable throughout the pipeline.

#### Files
1. **[Landcover_Fix/01_PROBLEM.md](Landcover_Fix/01_PROBLEM.md)**
   - What: Explains why treating categorical data as continuous is wrong
   - Where: Which files are affected
   - Impact: What this means for model predictions

2. **[Landcover_Fix/02_SOLUTION.md](Landcover_Fix/02_SOLUTION.md)** ⭐
   - What: Correct Corine → fire risk ordinal mapping
   - How: Code examples for all affected files
   - Mapping: All 28 Corine classes in Bolzano dataset

3. **[Landcover_Fix/03_CHECKLIST.md](Landcover_Fix/03_CHECKLIST.md)**
   - Step-by-step: Manual fix instructions for each file
   - Verification: How to test the fix
   - Troubleshooting: Common issues

**Quick Fix**:
```bash
cd ../00_Utilities
python apply_landcover_fix.py  # Automated fix
```

### 2. Pipeline Improvements

Complete documentation of all improvements made to the modeling pipeline.

#### Files
1. **[Pipeline_Improvements/COMPLETE_SOLUTION.md](Pipeline_Improvements/COMPLETE_SOLUTION.md)** ⭐ **MAIN REFERENCE**
   - Questions addressed: All 4 original questions
   - Landcover fix: Detailed with Corine codes
   - Climate projections: Complete pipeline
   - Visualization: All plot types
   - Quick start: Step-by-step workflow

2. **[Pipeline_Improvements/IMPROVEMENTS_SUMMARY.md](Pipeline_Improvements/IMPROVEMENTS_SUMMARY.md)**
   - High-level: Overview of improvements
   - Key features: What was added/fixed
   - File structure: What files were created
   - Next steps: Actionable checklist

### 3. Workflow Guides

Detailed guides for specific tasks and methodologies.

#### Files
1. **[Workflow_Guides/STANDARDIZATION_GUIDE.md](Workflow_Guides/STANDARDIZATION_GUIDE.md)**
   - Code standards: Naming conventions, structure
   - Best practices: How to write clean pipeline code
   - File organization: Where things should go

2. **[Workflow_Guides/TEMPORAL_VALIDATION_GUIDE.md](Workflow_Guides/TEMPORAL_VALIDATION_GUIDE.md)**
   - Validation methods: How to validate temporal predictions
   - Metrics: What to measure
   - Interpretation: How to interpret results

3. **[Workflow_Guides/WORKFLOW_SUMMARY.md](Workflow_Guides/WORKFLOW_SUMMARY.md)**
   - Pipeline overview: End-to-end workflow
   - Data flow: How data moves through pipeline
   - Dependencies: What depends on what

## 🔍 Find What You Need

### By Use Case

| I want to... | Read this |
|--------------|-----------|
| **Get started** | [COMPLETE_SOLUTION.md](Pipeline_Improvements/COMPLETE_SOLUTION.md) |
| **Fix landcover encoding** | [Landcover_Fix/02_SOLUTION.md](Landcover_Fix/02_SOLUTION.md) |
| **Understand the problem** | [Landcover_Fix/01_PROBLEM.md](Landcover_Fix/01_PROBLEM.md) |
| **Run climate projections** | [../03_Climate_Projections/README.md](../03_Climate_Projections/README.md) |
| **Check implementation steps** | [Landcover_Fix/03_CHECKLIST.md](Landcover_Fix/03_CHECKLIST.md) |
| **See all improvements** | [Pipeline_Improvements/IMPROVEMENTS_SUMMARY.md](Pipeline_Improvements/IMPROVEMENTS_SUMMARY.md) |
| **Validate results** | [Workflow_Guides/TEMPORAL_VALIDATION_GUIDE.md](Workflow_Guides/TEMPORAL_VALIDATION_GUIDE.md) |

### By Topic

| Topic | Primary Document |
|-------|------------------|
| **Corine Land Cover codes** | [Landcover_Fix/02_SOLUTION.md](Landcover_Fix/02_SOLUTION.md) |
| **Fire risk mapping** | [../01_Data_Preparation/CORINE_LANDCOVER_FIRE_RISK_MAPPING.py](../01_Data_Preparation/CORINE_LANDCOVER_FIRE_RISK_MAPPING.py) |
| **Temporal aggregation** | [COMPLETE_SOLUTION.md § Temporal Aggregation](Pipeline_Improvements/COMPLETE_SOLUTION.md#key-improvements) |
| **Climate scenarios** | [../03_Climate_Projections/README.md](../03_Climate_Projections/README.md) |
| **Visualization** | [COMPLETE_SOLUTION.md § Visualization](Pipeline_Improvements/COMPLETE_SOLUTION.md#spatial-and-temporal-visualization) |

## 📝 Document Summaries

### Landcover Fix Documents

**01_PROBLEM.md** (11 KB)
- Explains categorical vs continuous data
- Lists all affected files (7 scripts)
- Shows current incorrect behavior
- Demonstrates impact on model

**02_SOLUTION.md** (14 KB) ⭐ **Essential reading**
- Actual Corine codes in dataset (28 classes)
- Fire risk ordinal mapping (0-5 scale)
- Code examples for all files
- Implementation options (import vs copy)

**03_CHECKLIST.md** (14 KB)
- File-by-file instructions
- Before/after code comparisons
- Verification steps
- Troubleshooting common issues

### Pipeline Improvement Documents

**COMPLETE_SOLUTION.md** (14 KB) ⭐ **Main reference**
- Complete overview (most comprehensive)
- All 4 questions addressed
- Quick start guide
- Landcover data analysis (28 Corine classes)
- Troubleshooting section

**IMPROVEMENTS_SUMMARY.md** (12 KB)
- High-level overview
- File structure (what was created)
- Usage workflow
- Next actions checklist

### Workflow Guide Documents

**STANDARDIZATION_GUIDE.md** (6 KB)
- Coding standards
- File naming conventions
- Documentation practices

**TEMPORAL_VALIDATION_GUIDE.md** (8 KB)
- Validation methodologies
- Temporal metrics
- Interpretation guidelines

**WORKFLOW_SUMMARY.md** (8 KB)
- Pipeline overview
- Component descriptions
- Data flow diagrams

## 🔧 Meta Documentation

**DIRECTORY_CLEANUP_PLAN.md**
- How documentation was organized
- Migration from messy root structure
- New directory layout rationale

## 🆕 Adding New Documentation

When adding new documentation:

1. **Choose appropriate directory**:
   - `Landcover_Fix/` - Landcover encoding related
   - `Pipeline_Improvements/` - General pipeline improvements
   - `Workflow_Guides/` - How-to guides and methods

2. **Follow naming conventions**:
   - Use descriptive names in UPPERCASE with underscores
   - Add numbers for ordered sequences (01_, 02_, etc.)
   - Use .md extension for all documentation

3. **Update this index**:
   - Add entry in structure diagram
   - Add to "Find What You Need" tables
   - Add to document summaries

4. **Cross-reference**:
   - Link to related documents
   - Update main README.md if needed

## 📚 External Documentation

### Pipeline Component READMEs
- [03_Climate_Projections/README.md](../03_Climate_Projections/README.md) - Climate projection pipeline
- [Main Scripts README.md](../README.md) - Overall pipeline documentation

### Reference Files
- [CORINE_LANDCOVER_FIRE_RISK_MAPPING.py](../01_Data_Preparation/CORINE_LANDCOVER_FIRE_RISK_MAPPING.py) - Landcover mapping code
- [CORINE_fire_risk_mapping.csv](../01_Data_Preparation/CORINE_fire_risk_mapping.csv) - Human-readable table

---

**Documentation Status**: ✅ Complete and organized
**Last Updated**: 2025-10-28
**Total Documents**: 9 markdown files
