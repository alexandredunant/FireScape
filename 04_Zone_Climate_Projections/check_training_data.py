#!/usr/bin/env python
import pandas as pd
from pathlib import Path

BASE_DIR = Path("/mnt/CEPH_PROJECTS/Firescape")
training_data = pd.read_parquet(BASE_DIR / "output/01_Training_Data/spacetime_dataset.parquet")

print("Columns in training data:")
print(training_data.columns.tolist())
print()
print("First few rows:")
print(training_data.head())
print()
print("Data types:")
print(training_data.dtypes)
