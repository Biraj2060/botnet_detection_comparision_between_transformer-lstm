import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import (
    ATTACK_FILES,
    BENIGN_FILES,
    CATEGORY_NAMES,
    CLASS_FILES,
    CLASS_NAMES,
    CLASS_TO_CATEGORY,
    FILE_TO_CATEGORY,
    FILE_TO_CLASS,
    FILE_TO_CLASS_INDEX,
    MAX_ROWS_PER_FILE,
    METADATA_PATH,
    RANDOM_STATE,
    SCALER_PATH,
    SEQUENCE_LEN,
    SPLIT_SUMMARY_PATH,
    TEST_SIZE,
    TRAIN_SIZE,
    VAL_SIZE,
    ensure_directories,
)


def _read_csv(path: str) -> pd.DataFrame:
    read_kwargs = {}
    if MAX_ROWS_PER_FILE is not None:
        read_kwargs["nrows"] = MAX_ROWS_PER_FILE
    return pd.read_csv(path, **read_kwargs)


def _clean_numeric_frame(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.replace([np.inf, -np.inf], np.nan)
    cleaned = cleaned.select_dtypes(include=[np.number]).dropna()
    return cleaned


def _split_rows_by_ratio(n_rows: int) -> Tuple[slice, slice, slice]:
    train_end = int(n_rows * TRAIN_SIZE)
    val_end = train_end + int(n_rows * VAL_SIZE)
    if val_end >= n_rows:
        val_end = max(train_end, n_rows - 1)
    return slice(0, train_end), slice(train_end, val_end), slice(val_end, n_rows)


def _sequence_count(array: np.ndarray, seq_len: int = SEQUENCE_LEN) -> int:
    return max(0, len(array) - seq_len + 1)


def _transform_in_chunks(scaler: StandardScaler, array: np.ndarray, chunk_size: int = 50000) -> np.ndarray:
    chunks = []
    for start in range(0, len(array), chunk_size):
        stop = start + chunk_size
        chunk = scaler.transform(array[start:stop]).astype(np.float32, copy=False)
        chunks.append(chunk)
    if not chunks:
        return np.empty((0, array.shape[1]), dtype=np.float32)
    return np.vstack(chunks)


def load_multiclass_rows(data_dir: str) -> Tuple[Dict[str, Dict[str, np.ndarray]], List[str]]:
    print("\nLoading multiclass dataset files...")
    datasets: Dict[str, Dict[str, np.ndarray]] = {}
    feature_names: List[str] = []

    for file_name in CLASS_FILES:
        path = os.path.join(data_dir, file_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required dataset file not found: {path}")

        raw_df = _read_csv(path)
        cleaned_df = _clean_numeric_frame(raw_df)
        if cleaned_df.empty:
            raise ValueError(f"No valid numeric rows remained after cleaning: {file_name}")

        if not feature_names:
            feature_names = cleaned_df.columns.tolist()
        else:
            if cleaned_df.columns.tolist() != feature_names:
                missing = [column for column in feature_names if column not in cleaned_df.columns]
                extra = [column for column in cleaned_df.columns if column not in feature_names]
                if extra:
                    cleaned_df = cleaned_df.drop(columns=extra)
                if missing:
                    for column in missing:
                        cleaned_df[column] = 0.0
                cleaned_df = cleaned_df.loc[:, feature_names]

        datasets[file_name] = {
            "rows": cleaned_df.to_numpy(dtype=np.float32),
            "class_name": FILE_TO_CLASS[file_name],
            "category_name": FILE_TO_CATEGORY[file_name],
            "class_index": FILE_TO_CLASS_INDEX[file_name],
        }
        print(
            f"  Loaded {file_name:<36} rows={len(cleaned_df):>8,} "
            f"class={FILE_TO_CLASS[file_name]}"
        )

    return datasets, feature_names


def build_sequence_splits(data_dir: str):
    ensure_directories()
    datasets, feature_names = load_multiclass_rows(data_dir)

    train_parts: List[np.ndarray] = []
    for payload in datasets.values():
        train_slice, _, _ = _split_rows_by_ratio(len(payload["rows"]))
        train_parts.append(payload["rows"][train_slice])

    train_matrix = np.vstack(train_parts)
    scaler = StandardScaler()
    scaler.fit(train_matrix)

    split_arrays: Dict[str, List[Dict[str, object]]] = {"train": [], "val": [], "test": []}
    split_summary = {
        "feature_count": len(feature_names),
        "class_names": CLASS_NAMES,
        "category_names": CATEGORY_NAMES,
        "per_file": {},
        "per_split_class_counts": {"train": {}, "val": {}, "test": {}},
        "per_split_sequence_total": {"train": 0, "val": 0, "test": 0},
    }

    for file_name, payload in datasets.items():
        rows = payload["rows"]
        class_index = payload["class_index"]
        class_name = payload["class_name"]
        category_name = payload["category_name"]
        train_slice, val_slice, test_slice = _split_rows_by_ratio(len(rows))

        row_splits = {
            "train": _transform_in_chunks(scaler, rows[train_slice]),
            "val": _transform_in_chunks(scaler, rows[val_slice]),
            "test": _transform_in_chunks(scaler, rows[test_slice]),
        }

        split_summary["per_file"][file_name] = {
            "class_name": class_name,
            "category_name": category_name,
            "row_count": int(len(rows)),
            "train_rows": int(len(row_splits["train"])),
            "val_rows": int(len(row_splits["val"])),
            "test_rows": int(len(row_splits["test"])),
            "train_sequences": 0,
            "val_sequences": 0,
            "test_sequences": 0,
        }

        for split_name, split_rows in row_splits.items():
            sequence_count = _sequence_count(split_rows)
            split_arrays[split_name].append(
                {
                    "file_name": file_name,
                    "class_name": class_name,
                    "category_name": category_name,
                    "class_index": class_index,
                    "rows": split_rows.astype(np.float32, copy=False),
                    "sequence_count": sequence_count,
                }
            )
            split_summary["per_file"][file_name][f"{split_name}_sequences"] = int(sequence_count)
            split_summary["per_split_class_counts"][split_name][class_name] = int(sequence_count)
            split_summary["per_split_sequence_total"][split_name] += int(sequence_count)

    metadata = {
        "random_state": RANDOM_STATE,
        "sequence_length": SEQUENCE_LEN,
        "feature_names": feature_names,
        "feature_count": len(feature_names),
        "class_names": CLASS_NAMES,
        "class_to_category": CLASS_TO_CATEGORY,
        "class_files": CLASS_FILES,
        "attack_files": ATTACK_FILES,
        "benign_files": BENIGN_FILES,
        "train_size": TRAIN_SIZE,
        "val_size": VAL_SIZE,
        "test_size": TEST_SIZE,
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    with open(SPLIT_SUMMARY_PATH, "w", encoding="utf-8") as handle:
        json.dump(split_summary, handle, indent=2)

    print("\nSequence split summary:")
    print(f"  Train sequences : {split_summary['per_split_sequence_total']['train']:,}")
    print(f"  Val sequences   : {split_summary['per_split_sequence_total']['val']:,}")
    print(f"  Test sequences  : {split_summary['per_split_sequence_total']['test']:,}")
    print(f"  Features        : {len(feature_names)}")
    print(f"  Classes         : {len(CLASS_NAMES)}")

    return {
        "train_parts": split_arrays["train"],
        "val_parts": split_arrays["val"],
        "test_parts": split_arrays["test"],
        "scaler": scaler,
        "feature_names": feature_names,
        "class_names": CLASS_NAMES,
        "metadata": metadata,
        "split_summary": split_summary,
    }
