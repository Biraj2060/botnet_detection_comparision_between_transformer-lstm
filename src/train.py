import json
import os
import time
from bisect import bisect_right
from typing import Dict, List, Tuple

import joblib
import matplotlib
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)
from sklearn.preprocessing import label_binarize
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, Dataset, TensorDataset

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    BATCH_SIZE,
    CLASS_NAMES,
    CLASS_WEIGHT_PATH,
    CNNLSTM_HISTORY_PATH,
    CNNLSTM_METRICS_PATH,
    CNNLSTM_PATH,
    CNNLSTM_PRED_PATH,
    D_MODEL,
    DROPOUT,
    EPOCHS,
    GRAD_CLIP_NORM,
    LEARNING_RATE,
    MODEL_DIR,
    N_HEADS,
    N_LAYERS,
    PATIENCE,
    RESULTS_PATH,
    SVM_HISTORY_PATH,
    SVM_MAX_SAMPLES,
    SVM_METRICS_PATH,
    SVM_PATH,
    SVM_PRED_PATH,
    TOP_K,
    TRAINING_SUMMARY_PATH,
    TRANSFORMER_HISTORY_PATH,
    TRANSFORMER_METRICS_PATH,
    TRANSFORMER_PATH,
    TRANSFORMER_PRED_PATH,
    WEIGHT_DECAY,
    ensure_directories,
)
from src.cnn_lstm import BotnetCNNLSTM
from src.model import BotnetTransformer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SequenceDataset(Dataset):
    def __init__(self, parts: List[Dict[str, object]]):
        self.parts = [part for part in parts if int(part["sequence_count"]) > 0]
        self.cumulative = []
        running = 0
        self.labels = []
        for part in self.parts:
            count = int(part["sequence_count"])
            running += count
            self.cumulative.append(running)
            self.labels.extend([int(part["class_index"])] * count)

    def __len__(self) -> int:
        return self.cumulative[-1] if self.cumulative else 0

    def __getitem__(self, index: int):
        part_idx = bisect_right(self.cumulative, index)
        prev_total = 0 if part_idx == 0 else self.cumulative[part_idx - 1]
        offset = index - prev_total
        part = self.parts[part_idx]
        rows = part["rows"]
        seq_len = rows.shape[0] - int(part["sequence_count"]) + 1
        if seq_len <= 0:
            seq_len = rows.shape[0]
        sequence = rows[offset : offset + seq_len]
        label = int(part["class_index"])
        return torch.FloatTensor(sequence), torch.LongTensor([label]).squeeze(0)


def _sequence_len_from_parts(parts: List[Dict[str, object]]) -> int:
    for part in parts:
        count = int(part["sequence_count"])
        rows = part["rows"]
        if count > 0:
            return rows.shape[0] - count + 1
    return 0


def _numpy_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def _compute_class_weights(y_train: np.ndarray, num_classes: int) -> np.ndarray:
    counts = np.bincount(y_train, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return weights


def make_loader(X: np.ndarray, y: np.ndarray, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(torch.FloatTensor(X), torch.LongTensor(y))
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


def make_loader_from_parts(parts: List[Dict[str, object]], shuffle: bool) -> DataLoader:
    dataset = SequenceDataset(parts)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=shuffle)


def labels_from_parts(parts: List[Dict[str, object]]) -> np.ndarray:
    labels = []
    for part in parts:
        labels.extend([int(part["class_index"])] * int(part["sequence_count"]))
    return np.asarray(labels, dtype=np.int64)


def sample_sequences_from_parts(parts: List[Dict[str, object]], sample_limit: int = None) -> Tuple[np.ndarray, np.ndarray]:
    sequences = []
    labels = []
    rng = np.random.default_rng(42)
    for part in parts:
        rows = part["rows"]
        count = int(part["sequence_count"])
        if count <= 0:
            continue
        seq_len = rows.shape[0] - count + 1
        candidate_indices = np.arange(count)
        if sample_limit is not None and count > sample_limit:
            candidate_indices = np.sort(rng.choice(candidate_indices, size=sample_limit, replace=False))
        for idx in candidate_indices:
            sequences.append(rows[idx : idx + seq_len])
        labels.extend([int(part["class_index"])] * len(candidate_indices))
    return np.asarray(sequences, dtype=np.float32), np.asarray(labels, dtype=np.int64)


def _per_class_curve_scores(y_true: np.ndarray, scores: np.ndarray) -> Tuple[Dict[str, float], Dict[str, float]]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    y_bin = label_binarize(y_true, classes=np.arange(len(CLASS_NAMES)))
    roc_auc = {}
    pr_ap = {}
    for class_index, class_name in enumerate(CLASS_NAMES):
        try:
            roc_auc[class_name] = float(roc_auc_score(y_bin[:, class_index], scores[:, class_index]))
        except ValueError:
            roc_auc[class_name] = None
        try:
            pr_ap[class_name] = float(average_precision_score(y_bin[:, class_index], scores[:, class_index]))
        except ValueError:
            pr_ap[class_name] = None
    return roc_auc, pr_ap


def evaluate_predictions(
    model_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    score_matrix: np.ndarray,
    train_time_s: float,
    inference_ms_per_sample: float,
    param_count: int = 0,
) -> Dict:
    probabilities = _numpy_softmax(score_matrix)
    report = classification_report(
        y_true,
        y_pred,
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True,
    )
    roc_auc_per_class, pr_ap_per_class = _per_class_curve_scores(y_true, score_matrix)
    macro_roc_auc = np.mean([v for v in roc_auc_per_class.values() if v is not None])
    macro_pr_ap = np.mean([v for v in pr_ap_per_class.values() if v is not None])

    metrics = {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "top3_accuracy": float(
            top_k_accuracy_score(
                y_true,
                probabilities,
                k=min(TOP_K, len(CLASS_NAMES)),
                labels=np.arange(len(CLASS_NAMES)),
            )
        ),
        "macro_roc_auc_ovr": float(macro_roc_auc),
        "macro_pr_ap": float(macro_pr_ap),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "classification_report": report,
        "roc_auc_per_class": roc_auc_per_class,
        "pr_ap_per_class": pr_ap_per_class,
        "train_time_s": float(train_time_s),
        "inference_ms_per_sample": float(inference_ms_per_sample),
        "parameter_count": int(param_count),
    }
    return metrics


def _save_predictions(path: str, y_true: np.ndarray, y_pred: np.ndarray, score_matrix: np.ndarray) -> None:
    np.savez_compressed(
        path,
        y_true=y_true.astype(np.int64),
        y_pred=y_pred.astype(np.int64),
        scores=score_matrix.astype(np.float32),
        probabilities=_numpy_softmax(score_matrix).astype(np.float32),
        class_names=np.asarray(CLASS_NAMES),
    )


def _save_json(path: str, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def train_deep_model(
    model: nn.Module,
    model_name: str,
    save_path: str,
    history_path: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: np.ndarray,
) -> Tuple[Dict, float]:
    print(f"\nTraining {model_name} on {device}...")
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)

    history = {
        "train_loss": [],
        "val_loss": [],
        "val_macro_f1": [],
        "val_weighted_f1": [],
        "val_accuracy": [],
    }
    best_macro_f1 = -1.0
    patience_ctr = 0
    train_start = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            train_losses.append(loss.item())

        model.eval()
        val_losses = []
        all_logits = []
        all_labels = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                val_losses.append(loss.item())
                all_logits.append(logits.cpu().numpy())
                all_labels.append(yb.cpu().numpy())

        logits_np = np.concatenate(all_logits, axis=0)
        labels_np = np.concatenate(all_labels, axis=0)
        preds_np = logits_np.argmax(axis=1)
        val_macro_f1 = f1_score(labels_np, preds_np, average="macro", zero_division=0)
        val_weighted_f1 = f1_score(labels_np, preds_np, average="weighted", zero_division=0)
        val_accuracy = accuracy_score(labels_np, preds_np)

        history["train_loss"].append(float(np.mean(train_losses)))
        history["val_loss"].append(float(np.mean(val_losses)))
        history["val_macro_f1"].append(float(val_macro_f1))
        history["val_weighted_f1"].append(float(val_weighted_f1))
        history["val_accuracy"].append(float(val_accuracy))

        print(
            f"  Epoch {epoch:03d}/{EPOCHS} "
            f"train_loss={history['train_loss'][-1]:.4f} "
            f"val_loss={history['val_loss'][-1]:.4f} "
            f"macro_f1={val_macro_f1:.4f} "
            f"weighted_f1={val_weighted_f1:.4f} "
            f"acc={val_accuracy:.4f}"
        )

        scheduler.step(val_macro_f1)
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            patience_ctr = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"  Early stopping {model_name} at epoch {epoch} (best macro-F1={best_macro_f1:.4f})")
                break

    train_time_s = time.time() - train_start
    history["train_time_s"] = float(train_time_s)
    history["best_val_macro_f1"] = float(best_macro_f1)
    _save_json(history_path, history)
    return history, train_time_s


def predict_deep_model(model: nn.Module, state_path: str, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, float]:
    model.load_state_dict(torch.load(state_path, map_location=device))
    model = model.to(device)
    model.eval()
    logits_all = []
    labels_all = []
    start = time.time()
    with torch.no_grad():
        for xb, yb in loader:
            logits = model(xb.to(device)).cpu().numpy()
            logits_all.append(logits)
            labels_all.append(yb.numpy())
    elapsed = time.time() - start
    logits_np = np.concatenate(logits_all, axis=0)
    labels_np = np.concatenate(labels_all, axis=0)
    inf_ms = elapsed / max(len(labels_np), 1) * 1000.0
    preds_np = logits_np.argmax(axis=1)
    return logits_np, preds_np, inf_ms


def train_svm(train_parts: List[Dict[str, object]], test_parts: List[Dict[str, object]], history_path: str):
    print("\nTraining SVM baseline...")
    train_sequences, y_train = sample_sequences_from_parts(train_parts)
    test_sequences, y_test = sample_sequences_from_parts(test_parts)
    train_rows = train_sequences.reshape(len(train_sequences), -1)
    test_rows = test_sequences.reshape(len(test_sequences), -1)

    if len(train_rows) > SVM_MAX_SAMPLES:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(train_rows), size=SVM_MAX_SAMPLES, replace=False)
        train_rows = train_rows[indices]
        y_train = y_train[indices]

    svm = LinearSVC(class_weight="balanced", dual="auto", max_iter=5000)
    start = time.time()
    svm.fit(train_rows, y_train)
    train_time_s = time.time() - start
    decision_scores = svm.decision_function(test_rows)
    if decision_scores.ndim == 1:
        decision_scores = np.column_stack([-decision_scores, decision_scores])
    preds = svm.predict(test_rows)
    history = {
        "train_time_s": float(train_time_s),
        "train_samples": int(len(train_rows)),
        "feature_dim": int(train_rows.shape[1]),
    }
    _save_json(history_path, history)
    joblib.dump(svm, SVM_PATH)
    return svm, decision_scores, preds, y_test, train_time_s


def save_training_charts(histories: Dict[str, Dict], metrics_by_model: Dict[str, Dict]) -> None:
    ensure_directories()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Multiclass Model Comparison", fontsize=15)

    chart_models = ["Transformer", "CNN-LSTM", "SVM"]
    colors = {"Transformer": "#1565C0", "CNN-LSTM": "#C62828", "SVM": "#2E7D32"}

    metric_names = ["accuracy", "macro_f1", "weighted_f1", "top3_accuracy", "macro_roc_auc_ovr"]
    x = np.arange(len(metric_names))
    width = 0.25
    for idx, model_name in enumerate(chart_models):
        values = [metrics_by_model[model_name][metric] for metric in metric_names]
        axes[0, 0].bar(x + (idx - 1) * width, values, width=width, label=model_name, color=colors[model_name], alpha=0.85)
    axes[0, 0].set_xticks(x)
    axes[0, 0].set_xticklabels(metric_names, rotation=20)
    axes[0, 0].set_ylim(0, 1.05)
    axes[0, 0].set_title("Evaluation Metrics")
    axes[0, 0].legend()
    axes[0, 0].grid(axis="y", alpha=0.3)

    for model_name in ("Transformer", "CNN-LSTM"):
        history = histories[model_name]
        axes[0, 1].plot(history["val_macro_f1"], label=f"{model_name} val macro-F1", color=colors[model_name], linewidth=2)
    axes[0, 1].set_title("Validation Macro-F1")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Macro-F1")
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    for model_name in ("Transformer", "CNN-LSTM"):
        history = histories[model_name]
        axes[1, 0].plot(history["train_loss"], label=f"{model_name} train", color=colors[model_name], linewidth=2)
        axes[1, 0].plot(history["val_loss"], label=f"{model_name} val", color=colors[model_name], linestyle="--", linewidth=2)
    axes[1, 0].set_title("Training and Validation Loss")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend(fontsize=8)

    inf_values = [metrics_by_model[m]["inference_ms_per_sample"] for m in chart_models]
    train_values = [metrics_by_model[m]["train_time_s"] for m in chart_models]
    axes[1, 1].bar(chart_models, inf_values, color=[colors[m] for m in chart_models], alpha=0.85)
    axes[1, 1].set_title("Inference Latency (ms/sample)")
    axes[1, 1].grid(axis="y", alpha=0.3)
    for idx, value in enumerate(inf_values):
        axes[1, 1].text(idx, value, f"{value:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(MODEL_DIR, "comparison_multiclass.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(chart_models, train_values, color=[colors[m] for m in chart_models], alpha=0.85)
    ax.set_title("Training Time Comparison")
    ax.set_ylabel("Seconds")
    ax.grid(axis="y", alpha=0.3)
    for idx, value in enumerate(train_values):
        ax.text(idx, value, f"{value:.1f}s", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(MODEL_DIR, "training_time_multiclass.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)


def train_all_models(
    train_parts: List[Dict[str, object]],
    val_parts: List[Dict[str, object]],
    test_parts: List[Dict[str, object]],
    feature_count: int,
) -> Dict[str, Dict]:
    ensure_directories()
    num_classes = len(CLASS_NAMES)
    y_train = labels_from_parts(train_parts)
    y_val = labels_from_parts(val_parts)
    y_test = labels_from_parts(test_parts)
    class_weights = _compute_class_weights(y_train, num_classes)
    np.save(CLASS_WEIGHT_PATH, class_weights)

    train_loader = make_loader_from_parts(train_parts, shuffle=True)
    val_loader = make_loader_from_parts(val_parts, shuffle=False)
    test_loader = make_loader_from_parts(test_parts, shuffle=False)

    transformer = BotnetTransformer(
        input_dim=feature_count,
        num_classes=num_classes,
        d_model=D_MODEL,
        n_heads=N_HEADS,
        n_layers=N_LAYERS,
        dropout=DROPOUT,
    )
    cnn_lstm = BotnetCNNLSTM(
        input_dim=feature_count,
        num_classes=num_classes,
        dropout=DROPOUT,
    )

    transformer_history, transformer_train_time = train_deep_model(
        transformer,
        "Transformer",
        TRANSFORMER_PATH,
        TRANSFORMER_HISTORY_PATH,
        train_loader,
        val_loader,
        class_weights,
    )
    cnn_history, cnn_train_time = train_deep_model(
        cnn_lstm,
        "CNN-LSTM",
        CNNLSTM_PATH,
        CNNLSTM_HISTORY_PATH,
        train_loader,
        val_loader,
        class_weights,
    )

    transformer_scores, transformer_preds, transformer_inf_ms = predict_deep_model(transformer, TRANSFORMER_PATH, test_loader)
    cnn_scores, cnn_preds, cnn_inf_ms = predict_deep_model(cnn_lstm, CNNLSTM_PATH, test_loader)
    _, svm_scores, svm_preds, svm_y_test, svm_train_time = train_svm(train_parts, test_parts, SVM_HISTORY_PATH)

    transformer_metrics = evaluate_predictions(
        "Transformer",
        y_test,
        transformer_preds,
        transformer_scores,
        train_time_s=transformer_train_time,
        inference_ms_per_sample=transformer_inf_ms,
        param_count=sum(p.numel() for p in transformer.parameters()),
    )
    cnn_metrics = evaluate_predictions(
        "CNN-LSTM",
        y_test,
        cnn_preds,
        cnn_scores,
        train_time_s=cnn_train_time,
        inference_ms_per_sample=cnn_inf_ms,
        param_count=sum(p.numel() for p in cnn_lstm.parameters()),
    )
    svm_metrics = evaluate_predictions(
        "SVM",
        svm_y_test,
        svm_preds,
        svm_scores,
        train_time_s=svm_train_time,
        inference_ms_per_sample=0.0,
        param_count=0,
    )

    svm = joblib.load(SVM_PATH)
    svm_eval_sequences, _ = sample_sequences_from_parts(test_parts, sample_limit=10000)
    start = time.time()
    _ = svm.decision_function(svm_eval_sequences.reshape(len(svm_eval_sequences), -1))
    svm_metrics["inference_ms_per_sample"] = (time.time() - start) / max(len(svm_eval_sequences), 1) * 1000.0

    _save_predictions(TRANSFORMER_PRED_PATH, y_test, transformer_preds, transformer_scores)
    _save_predictions(CNNLSTM_PRED_PATH, y_test, cnn_preds, cnn_scores)
    _save_predictions(SVM_PRED_PATH, svm_y_test, svm_preds, svm_scores)

    _save_json(TRANSFORMER_METRICS_PATH, transformer_metrics)
    _save_json(CNNLSTM_METRICS_PATH, cnn_metrics)
    _save_json(SVM_METRICS_PATH, svm_metrics)

    results = {
        "Transformer": transformer_metrics,
        "CNN-LSTM": cnn_metrics,
        "SVM": svm_metrics,
    }
    _save_json(RESULTS_PATH, results)
    _save_json(
        TRAINING_SUMMARY_PATH,
        {
            "device": str(device),
            "class_count": len(CLASS_NAMES),
            "feature_count": int(feature_count),
            "sequence_length": int(_sequence_len_from_parts(train_parts)),
            "train_sequences": int(len(y_train)),
            "val_sequences": int(len(y_val)),
            "test_sequences": int(len(y_test)),
        },
    )

    save_training_charts(
        {"Transformer": transformer_history, "CNN-LSTM": cnn_history, "SVM": {"train_time_s": svm_train_time}},
        results,
    )
    return results
