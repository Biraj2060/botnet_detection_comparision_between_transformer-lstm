# src/preprocess.py
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from config import (DATA_DIR, BENIGN_FILES, ATTACK_FILES,
                    MAX_ROWS_PER_FILE, TEST_SIZE,
                    RANDOM_STATE, SEQUENCE_LEN)


def load_data():
    print("\n  Loading files...")
    dfs = []

    for fname in BENIGN_FILES:
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  WARNING: {fname} not found — skipping")
            continue
        df         = pd.read_csv(fpath, nrows=MAX_ROWS_PER_FILE)
        df['label'] = 0
        dfs.append(df)
        print(f"  Loaded  {fname}: {len(df):,} rows (benign)")

    for fname in ATTACK_FILES:
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            print(f"  WARNING: {fname} not found — skipping")
            continue
        df          = pd.read_csv(fpath, nrows=MAX_ROWS_PER_FILE)
        df['label'] = 1
        dfs.append(df)
        print(f"  Loaded  {fname}: {len(df):,} rows (attack)")

    combined = pd.concat(dfs, ignore_index=True)

    # Shuffle — critical to prevent temporal leakage
    combined = combined.sample(frac=1, random_state=RANDOM_STATE)
    combined = combined.reset_index(drop=True)

    n_benign = (combined['label'] == 0).sum()
    n_attack = (combined['label'] == 1).sum()
    print(f"\n  Total rows : {len(combined):,}")
    print(f"  Benign (0) : {n_benign:,}  ({n_benign/len(combined)*100:.1f}%)")
    print(f"  Attack (1) : {n_attack:,}  ({n_attack/len(combined)*100:.1f}%)")
    return combined


def clean_data(df):
    print("\n  Cleaning data...")
    before = len(df)

    # Remove infinite and NaN values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    print(f"  Removed {before - len(df):,} bad rows")

    # Separate label before selecting numeric columns
    label     = df['label'].copy()
    df        = df.drop(columns=['label'])
    df        = df.select_dtypes(include=[np.number])
    df['label'] = label

    print(f"  Features   : {len(df.columns) - 1}")
    print(f"  Rows kept  : {len(df):,}")
    return df


def preprocess(df):
    print("\n  Preprocessing...")

    X          = df.drop('label', axis=1).values.astype(np.float32)
    y          = df['label'].values.astype(np.float32)
    n_features = X.shape[1]
    print(f"  Feature matrix : {X.shape}")

    # ── Step 1: Split BEFORE scaling to prevent leakage ───
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size    = TEST_SIZE,
        stratify     = y,
        random_state = RANDOM_STATE
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size    = 0.176,   # 0.176 of 85% ≈ 15% of total
        stratify     = y_temp,
        random_state = RANDOM_STATE
    )

    print(f"\n  Before SMOTE:")
    print(f"  Train : {len(X_train):,} rows  ({y_train.mean()*100:.1f}% attack)")
    print(f"  Val   : {len(X_val):,} rows  ({y_val.mean()*100:.1f}% attack)")
    print(f"  Test  : {len(X_test):,} rows  ({y_test.mean()*100:.1f}% attack)")

    # ── Step 2: Scale AFTER split, fit on train only ──────
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)
    print(f"\n  Scaling done (fit on train only)")

    # ── Step 3: SMOTE on training data only ───────────────
    print(f"\n  Applying SMOTE...")
    sm           = SMOTE(random_state=RANDOM_STATE, k_neighbors=5)
    X_train, y_train = sm.fit_resample(X_train, y_train)
    print(f"  After SMOTE:")
    print(f"  Train : {len(X_train):,} rows  ({y_train.mean()*100:.1f}% attack)")
    print(f"  Balance achieved!")

    return (X_train, X_val, X_test,
            y_train, y_val, y_test,
            scaler, n_features)