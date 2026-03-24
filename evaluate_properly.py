import json
import os

import numpy as np
from sklearn.metrics import classification_report

from config import (
    CLASS_NAMES,
    CNNLSTM_METRICS_PATH,
    CNNLSTM_PRED_PATH,
    RESULTS_PATH,
    SVM_METRICS_PATH,
    SVM_PRED_PATH,
    TRANSFORMER_METRICS_PATH,
    TRANSFORMER_PRED_PATH,
)


MODEL_FILES = {
    "Transformer": (TRANSFORMER_METRICS_PATH, TRANSFORMER_PRED_PATH),
    "CNN-LSTM": (CNNLSTM_METRICS_PATH, CNNLSTM_PRED_PATH),
    "SVM": (SVM_METRICS_PATH, SVM_PRED_PATH),
}


def print_model_report(model_name: str, metrics_path: str, pred_path: str) -> None:
    if not os.path.exists(metrics_path) or not os.path.exists(pred_path):
        print(f"{model_name}: missing artifacts, train the model first.")
        return

    with open(metrics_path, "r", encoding="utf-8") as handle:
        metrics = json.load(handle)
    preds = np.load(pred_path, allow_pickle=True)
    y_true = preds["y_true"]
    y_pred = preds["y_pred"]

    print("\n" + "=" * 72)
    print(f"  {model_name} - Detailed Multiclass Evaluation")
    print("=" * 72)
    print(
        f"  Accuracy={metrics['accuracy']:.4f}  "
        f"Macro-F1={metrics['macro_f1']:.4f}  "
        f"Weighted-F1={metrics['weighted_f1']:.4f}  "
        f"Top-3={metrics['top3_accuracy']:.4f}"
    )
    print(
        f"  Macro ROC-AUC={metrics['macro_roc_auc_ovr']:.4f}  "
        f"Macro PR-AP={metrics['macro_pr_ap']:.4f}  "
        f"Inference={metrics['inference_ms_per_sample']:.4f} ms/sample"
    )
    print("\nPer-class report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))


def main() -> None:
    for model_name, (metrics_path, pred_path) in MODEL_FILES.items():
        print_model_report(model_name, metrics_path, pred_path)

    if os.path.exists(RESULTS_PATH):
        print("\nSaved summary:", RESULTS_PATH)


if __name__ == "__main__":
    main()
