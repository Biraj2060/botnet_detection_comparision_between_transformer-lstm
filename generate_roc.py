import json
import os

import matplotlib
import numpy as np
from sklearn.metrics import roc_curve
from sklearn.preprocessing import label_binarize

from config import CLASS_NAMES, MODEL_DIR, ROC_DATA_PATH, ensure_directories

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PREDICTION_FILES = {
    "Transformer": os.path.join(MODEL_DIR, "predictions_transformer.npz"),
    "CNN-LSTM": os.path.join(MODEL_DIR, "predictions_cnn_lstm.npz"),
    "SVM": os.path.join(MODEL_DIR, "predictions_svm.npz"),
}


def main() -> None:
    ensure_directories()
    roc_payload = {}
    class_indices = np.arange(len(CLASS_NAMES))

    for model_name, pred_path in PREDICTION_FILES.items():
        if not os.path.exists(pred_path):
            print(f"Skipping {model_name}: predictions not found.")
            continue

        data = np.load(pred_path, allow_pickle=True)
        y_true = data["y_true"]
        scores = data["scores"]
        y_bin = label_binarize(y_true, classes=class_indices)

        roc_payload[model_name] = {}
        for class_index, class_name in enumerate(CLASS_NAMES):
            try:
                fpr, tpr, _ = roc_curve(y_bin[:, class_index], scores[:, class_index])
                roc_payload[model_name][class_name] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
            except ValueError:
                roc_payload[model_name][class_name] = {"fpr": [0.0, 1.0], "tpr": [0.0, 1.0]}

    with open(ROC_DATA_PATH, "w", encoding="utf-8") as handle:
        json.dump(roc_payload, handle, indent=2)

    focus_classes = ["Benign", "DDoS-ICMP_Flood", "Mirai-greeth_flood", "DNS_Spoofing"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    colors = {"Transformer": "#1565C0", "CNN-LSTM": "#C62828", "SVM": "#2E7D32"}

    for ax, class_name in zip(axes, focus_classes):
        for model_name, model_curves in roc_payload.items():
            curve = model_curves.get(class_name)
            if not curve:
                continue
            ax.plot(curve["fpr"], curve["tpr"], label=model_name, color=colors[model_name], linewidth=2)
        ax.plot([0, 1], [0, 1], linestyle="--", color="#999999", linewidth=1)
        ax.set_title(f"ROC: {class_name}")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(MODEL_DIR, "roc_curves_multiclass.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved multiclass ROC data to {ROC_DATA_PATH}")


if __name__ == "__main__":
    main()
