import os

import numpy as np
import pandas as pd

from config import CLASS_FILES, FILE_TO_CATEGORY, MAX_ROWS_PER_FILE


print("\n" + "=" * 72)
print("  MULTICLASS DATASET DIAGNOSIS")
print("=" * 72)

category_counts = {}
total_rows = 0
feature_count = None

for file_name in CLASS_FILES:
    path = os.path.join("data", file_name)
    if not os.path.exists(path):
        print(f"Missing: {file_name}")
        continue

    read_kwargs = {}
    if MAX_ROWS_PER_FILE is not None:
        read_kwargs["nrows"] = MAX_ROWS_PER_FILE
    df = pd.read_csv(path, **read_kwargs)
    num_df = df.replace([np.inf, -np.inf], np.nan).select_dtypes(include=[np.number]).dropna()
    feature_count = len(num_df.columns)
    total_rows += len(num_df)
    category = FILE_TO_CATEGORY[file_name]
    category_counts[category] = category_counts.get(category, 0) + len(num_df)
    print(f"{file_name:<36} category={category:<16} rows={len(num_df):>8,} features={feature_count}")

print("\nCategory row totals:")
for category, count in sorted(category_counts.items()):
    print(f"  {category:<16} {count:>10,}")

print(f"\nTotal cleaned rows: {total_rows:,}")
print(f"Feature count    : {feature_count}")
print("=" * 72)
