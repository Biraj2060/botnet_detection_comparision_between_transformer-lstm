import json
import os

import matplotlib
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.preprocessing import label_binarize

from config import CLASS_NAMES, MODEL_DIR, PR_DATA_PATH, ensure_directories

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PREDICTION_FILES = {
    "Transformer": os.path.join(MODEL_DIR, "predictions_transformer.npz"),
    "CNN-LSTM": os.path.join(MODEL_DIR, "predictions_cnn_lstm.npz"),
    "SVM": os.path.join(MODEL_DIR, "predictions_svm.npz"),
}


def main() -> None:
    ensure_directories()
    pr_payload = {}
    class_indices = np.arange(len(CLASS_NAMES))

    for model_name, pred_path in PREDICTION_FILES.items():
        if not os.path.exists(pred_path):
            print(f"Skipping {model_name}: predictions not found.")
            continue

        data = np.load(pred_path, allow_pickle=True)
        y_true = data["y_true"]
        scores = data["scores"]
        y_bin = label_binarize(y_true, classes=class_indices)

        pr_payload[model_name] = {}
        for class_index, class_name in enumerate(CLASS_NAMES):
            try:
                precision, recall, _ = precision_recall_curve(y_bin[:, class_index], scores[:, class_index])
                ap = average_precision_score(y_bin[:, class_index], scores[:, class_index])
                pr_payload[model_name][class_name] = {
                    "precision": precision.tolist(),
                    "recall": recall.tolist(),
                    "average_precision": float(ap),
                }
            except ValueError:
                pr_payload[model_name][class_name] = {
                    "precision": [1.0, 0.0],
                    "recall": [0.0, 1.0],
                    "average_precision": None,
                }

    with open(PR_DATA_PATH, "w", encoding="utf-8") as handle:
        json.dump(pr_payload, handle, indent=2)

    focus_classes = ["Benign", "DDoS-ICMP_Flood", "Mirai-greeth_flood", "DNS_Spoofing"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()
    colors = {"Transformer": "#1565C0", "CNN-LSTM": "#C62828", "SVM": "#2E7D32"}

    for ax, class_name in zip(axes, focus_classes):
        for model_name, model_curves in pr_payload.items():
            curve = model_curves.get(class_name)
            if not curve:
                continue
            label = model_name
            if curve["average_precision"] is not None:
                label = f"{model_name} (AP={curve['average_precision']:.3f})"
            ax.plot(curve["recall"], curve["precision"], label=label, color=colors[model_name], linewidth=2)
        ax.set_title(f"PR: {class_name}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(MODEL_DIR, "pr_curves_multiclass.png"), dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved multiclass PR data to {PR_DATA_PATH}")


if __name__ == "__main__":
    main()
