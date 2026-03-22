# diagnose.py
import pandas as pd
import numpy as np
import glob
import os
from config import DATA_DIR, BENIGN_FILES, ATTACK_FILES, MAX_ROWS_PER_FILE

print("\n" + "="*60)
print("   DATASET DIAGNOSIS REPORT")
print("="*60)

all_configured = BENIGN_FILES + ATTACK_FILES
total_rows     = 0
total_benign   = 0
total_attack   = 0
missing_files  = []
found_files    = []

for fname in all_configured:
    fpath = os.path.join(DATA_DIR, fname)

    # ── Check file exists ──────────────────────────────────
    if not os.path.exists(fpath):
        print(f"\n  MISSING : {fname}")
        missing_files.append(fname)
        continue

    # ── Load file ──────────────────────────────────────────
    try:
        df = pd.read_csv(fpath, nrows=MAX_ROWS_PER_FILE)
    except Exception as e:
        print(f"\n  ERROR reading {fname}: {e}")
        continue

    found_files.append(fname)
    n_rows = len(df)
    n_cols = len(df.columns)

    # ── Find label column ──────────────────────────────────
    label_col = None
    for col in df.columns:
        if col.strip().lower() in ['label', 'class', 'attack', 'category']:
            label_col = col
            break

    # ── Check for NaN and Inf ──────────────────────────────
    df_num     = df.select_dtypes(include=[np.number])
    nan_count  = df_num.isnull().sum().sum()
    inf_count  = np.isinf(df_num.values).sum()
    is_benign  = fname in BENIGN_FILES

    print(f"\n  File    : {fname}")
    print(f"  Rows    : {n_rows:,}  |  Columns : {n_cols}")
    print(f"  NaN     : {nan_count:,}  |  Inf     : {inf_count:,}")
    print(f"  Type    : {'BENIGN (label=0)' if is_benign else 'ATTACK (label=1)'}")

    if label_col:
        print(f"  Labels  : {df[label_col].unique()[:5]}")

    total_rows += n_rows
    if is_benign:
        total_benign += n_rows
    else:
        total_attack += n_rows

    print(f"  {'─'*48}")

# ── Summary ────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"   SUMMARY")
print(f"{'='*60}")
print(f"  Files found   : {len(found_files)} / {len(all_configured)}")
print(f"  Files missing : {len(missing_files)}")
if missing_files:
    for f in missing_files:
        print(f"    - {f}")

print(f"\n  Total rows    : {total_rows:,}")
print(f"  Benign rows   : {total_benign:,}  ({total_benign/max(total_rows,1)*100:.1f}%)")
print(f"  Attack rows   : {total_attack:,}  ({total_attack/max(total_rows,1)*100:.1f}%)")

if total_benign > 0 and total_attack > 0:
    ratio = total_attack / total_benign
    print(f"  Imbalance     : {ratio:.1f}x more attack than benign")

    if ratio > 5:
        print(f"\n  STATUS : SEVERELY IMBALANCED")
        print(f"  FIX    : SMOTE will fix this automatically")
    elif ratio > 2:
        print(f"\n  STATUS : MODERATELY IMBALANCED")
        print(f"  FIX    : SMOTE will fix this")
    else:
        print(f"\n  STATUS : REASONABLY BALANCED — good!")

print(f"\n  Data size     : {total_rows:,} rows")
if total_rows < 50000:
    print(f"  WARNING : Too little data — increase MAX_ROWS_PER_FILE")
elif total_rows < 200000:
    print(f"  STATUS  : Acceptable amount of data")
else:
    print(f"  STATUS  : Good amount of data")

print(f"\n  Numeric cols  : {len(df_num.columns)} features detected")
print(f"{'='*60}\n")