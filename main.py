import os
import sys
import pandas as pd

# Include src folder for imports
sys.path.append(os.path.abspath("src"))

from ofi_features import (
    compute_best_level_ofi,
    compute_multi_level_ofi,
    compute_integrated_ofi
)

# Load data
df = pd.read_csv("data/first_25000_rows.csv")

# Compute features
ofi_best = compute_best_level_ofi(df)
ofi_multi = compute_multi_level_ofi(df)
ofi_integrated = compute_integrated_ofi(df)

# Print results
print("Best-Level OFI:\n", ofi_best.head(), "\n")
print("Multi-Level OFI:\n", ofi_multi.head(), "\n")
print("Integrated OFI:\n", ofi_integrated.head())