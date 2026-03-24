import json
import os

import matplotlib
import numpy as np

from config import REPORT_DIR, RESULTS_PATH, SPLIT_SUMMARY_PATH, ensure_directories

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    ensure_directories()
    if not os.path.exists(RESULTS_PATH) or not os.path.exists(SPLIT_SUMMARY_PATH):
        raise FileNotFoundError("Train the multiclass pipeline before generating report figures.")

    results = load_json(RESULTS_PATH)
    split_summary = load_json(SPLIT_SUMMARY_PATH)
    model_names = list(results.keys())
    colors = {"Transformer": "#1565C0", "CNN-LSTM": "#C62828", "SVM": "#2E7D32"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Multiclass CICIoT2023 Report Figures", fontsize=15)

    categories = list(split_summary["per_split_class_counts"]["train"].keys())
    counts = list(split_summary["per_split_class_counts"]["train"].values())
    axes[0, 0].bar(range(len(categories)), counts, color="#546E7A", alpha=0.85)
    axes[0, 0].set_title("Training Sequence Count per Class")
    axes[0, 0].set_xticks(range(len(categories)))
    axes[0, 0].set_xticklabels(categories, rotation=90, fontsize=7)
    axes[0, 0].grid(axis="y", alpha=0.3)

    metric_names = ["accuracy", "macro_f1", "weighted_f1", "top3_accuracy", "macro_roc_auc_ovr"]
    x = np.arange(len(metric_names))
    width = 0.25
    for idx, model_name in enumerate(model_names):
        values = [results[model_name][metric] for metric in metric_names]
        axes[0, 1].bar(x + (idx - 1) * width, values, width=width, label=model_name, color=colors[model_name], alpha=0.85)
    axes[0, 1].set_title("Model Metric Comparison")
    axes[0, 1].set_xticks(x)
    axes[0, 1].set_xticklabels(metric_names, rotation=20)
    axes[0, 1].set_ylim(0, 1.05)
    axes[0, 1].legend()
    axes[0, 1].grid(axis="y", alpha=0.3)

    train_times = [results[name]["train_time_s"] for name in model_names]
    axes[1, 0].bar(model_names, train_times, color=[colors[name] for name in model_names], alpha=0.85)
    axes[1, 0].set_title("Training Time")
    axes[1, 0].set_ylabel("Seconds")
    axes[1, 0].grid(axis="y", alpha=0.3)

    inf_times = [results[name]["inference_ms_per_sample"] for name in model_names]
    axes[1, 1].bar(model_names, inf_times, color=[colors[name] for name in model_names], alpha=0.85)
    axes[1, 1].set_title("Inference Latency")
    axes[1, 1].set_ylabel("ms/sample")
    axes[1, 1].grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(REPORT_DIR, "figure_multiclass_dashboard.png")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved report figure to {out_path}")


if __name__ == "__main__":
    main()
