import json
import os
import pickle

import gradio as gr
import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from config import (
    APP_HOST,
    APP_PORT,
    CLASS_NAMES,
    CNNLSTM_PATH,
    CONFIDENCE_THRESHOLD,
    D_MODEL,
    DROPOUT,
    METADATA_PATH,
    N_HEADS,
    N_LAYERS,
    RESULTS_PATH,
    SCALER_PATH,
    SVM_PATH,
    TRANSFORMER_PATH,
)
from src.cnn_lstm import BotnetCNNLSTM
from src.model import BotnetTransformer
from src.svm import load_svm, predict_svm_scores


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _numpy_softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_scores = np.exp(shifted)
    return exp_scores / exp_scores.sum(axis=1, keepdims=True)


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


metadata = _load_json(METADATA_PATH)
results = _load_json(RESULTS_PATH)
feature_names = metadata["feature_names"]
class_to_category = metadata["class_to_category"]

with open(SCALER_PATH, "rb") as handle:
    scaler = pickle.load(handle)

transformer = BotnetTransformer(
    input_dim=len(feature_names),
    num_classes=len(CLASS_NAMES),
    d_model=D_MODEL,
    n_heads=N_HEADS,
    n_layers=N_LAYERS,
    dropout=DROPOUT,
).to(device)
transformer.load_state_dict(torch.load(TRANSFORMER_PATH, map_location=device))
transformer.eval()

cnn_lstm = BotnetCNNLSTM(
    input_dim=len(feature_names),
    num_classes=len(CLASS_NAMES),
    dropout=DROPOUT,
).to(device)
cnn_lstm.load_state_dict(torch.load(CNNLSTM_PATH, map_location=device))
cnn_lstm.eval()

svm_model = load_svm(SVM_PATH)


def preprocess_file(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    df = df.replace([np.inf, -np.inf], np.nan)
    if "label" in df.columns:
        df = df.drop(columns=["label"])
    numeric = df.select_dtypes(include=[np.number]).dropna()
    numeric = numeric.reindex(columns=feature_names, fill_value=0.0)
    return numeric.to_numpy(dtype=np.float32)


def make_sequences(rows: np.ndarray, seq_len: int) -> np.ndarray:
    if len(rows) < seq_len:
        return np.empty((0, seq_len, rows.shape[1]), dtype=np.float32)
    return np.asarray([rows[i : i + seq_len] for i in range(len(rows) - seq_len + 1)], dtype=np.float32)


def run_deep_model(model, sequences: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        logits = model(torch.FloatTensor(sequences).to(device)).cpu().numpy()
    return _numpy_softmax(logits)


def summarize_predictions(probabilities: np.ndarray):
    mean_probs = probabilities.mean(axis=0)
    top_indices = np.argsort(mean_probs)[::-1][:5]
    summary = []
    for index in top_indices:
        class_name = CLASS_NAMES[index]
        category = class_to_category[class_name]
        summary.append(
            {
                "class_name": class_name,
                "category": category,
                "probability": float(mean_probs[index]),
            }
        )
    best = summary[0]
    best["status"] = "CONFIDENT" if best["probability"] >= CONFIDENCE_THRESHOLD else "UNCERTAIN"
    return best, summary


def aggregate_categories(summary_rows):
    bucket = {}
    for row in summary_rows:
        bucket[row["category"]] = bucket.get(row["category"], 0.0) + row["probability"]
    return dict(sorted(bucket.items(), key=lambda item: item[1], reverse=True))


def analyse_file(file_obj):
    if file_obj is None:
        return "Upload a CICIoT2023 CSV file first.", None, None

    rows = preprocess_file(file_obj.name)
    sequences = make_sequences(scaler.transform(rows), metadata["sequence_length"])
    if len(sequences) == 0:
        return "Need at least 10 cleaned numeric rows to form one sequence.", None, None

    transformer_probs = run_deep_model(transformer, sequences)
    cnn_probs = run_deep_model(cnn_lstm, sequences)
    svm_probs = _numpy_softmax(predict_svm_scores(svm_model, sequences))

    model_outputs = {
        "Transformer": summarize_predictions(transformer_probs),
        "CNN-LSTM": summarize_predictions(cnn_probs),
        "SVM": summarize_predictions(svm_probs),
    }

    lines = [
        "=" * 72,
        "  MULTICLASS TRAFFIC ANALYSIS",
        "=" * 72,
        f"  Sequences analysed : {len(sequences):,}",
        f"  Confidence gate    : {CONFIDENCE_THRESHOLD:.2f}",
        "-" * 72,
    ]
    for model_name, (best, top_rows) in model_outputs.items():
        lines.append(
            f"  {model_name:<11} top_class={best['class_name']} "
            f"category={best['category']} confidence={best['probability']:.4f} "
            f"status={best['status']}"
        )
        for row in top_rows[1:4]:
            lines.append(
                f"               alt={row['class_name']} "
                f"({row['category']}) prob={row['probability']:.4f}"
            )
        lines.append("")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    model_names = list(model_outputs.keys())
    top_classes = [model_outputs[name][0]["class_name"] for name in model_names]
    top_scores = [model_outputs[name][0]["probability"] for name in model_names]
    axes[0].bar(model_names, top_scores, color=["#1565C0", "#C62828", "#2E7D32"], alpha=0.85)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Top-class confidence by model")
    axes[0].grid(axis="y", alpha=0.3)
    for idx, label in enumerate(top_classes):
        axes[0].text(idx, top_scores[idx], label, ha="center", va="bottom", rotation=20, fontsize=8)

    category_mix = aggregate_categories(model_outputs["Transformer"][1])
    axes[1].bar(list(category_mix.keys()), list(category_mix.values()), color="#546E7A", alpha=0.85)
    axes[1].set_title("Transformer category distribution")
    axes[1].tick_params(axis="x", rotation=25)
    axes[1].grid(axis="y", alpha=0.3)
    plt.tight_layout()

    compare_fig, compare_ax = plt.subplots(figsize=(12, 5))
    compare_ax.set_title("Validated model comparison")
    metrics = ["accuracy", "macro_f1", "weighted_f1", "top3_accuracy", "macro_roc_auc_ovr"]
    x = np.arange(len(metrics))
    width = 0.25
    colors = {"Transformer": "#1565C0", "CNN-LSTM": "#C62828", "SVM": "#2E7D32"}
    for idx, model_name in enumerate(model_names):
        values = [results[model_name][metric] for metric in metrics]
        compare_ax.bar(x + (idx - 1) * width, values, width=width, label=model_name, color=colors[model_name], alpha=0.85)
    compare_ax.set_xticks(x)
    compare_ax.set_xticklabels(metrics, rotation=20)
    compare_ax.set_ylim(0, 1.05)
    compare_ax.legend()
    compare_ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    return "\n".join(lines), fig, compare_fig


def show_methodology():
    lines = [
        "=" * 72,
        "  MULTICLASS PROJECT METHODOLOGY",
        "=" * 72,
        f"  Classes            : {len(CLASS_NAMES)} (1 benign + 33 attack types)",
        f"  Attack categories  : 7",
        f"  Sequence length    : {metadata['sequence_length']}",
        "  Class balancing    : None (original CICIoT2023 distribution retained)",
        "  Loss function      : CrossEntropyLoss with class weights",
        "  Baselines          : Transformer, CNN-LSTM, SVM",
        "  Evaluation         : per-class confusion matrix, ROC/PR, macro and weighted metrics",
        "  Deployment gate    : use confidence threshold plus external benign validation before live use",
        "",
        "  Lab-to-deployment gate:",
        "  1. Weighted-F1 >= 0.90",
        "  2. Macro-F1 >= 0.80",
        "  3. No critical class recall below 0.60",
        "  4. External benign false positive rate <= 5%",
        "  5. Confidence-based abstention enabled",
        "=" * 72,
    ]
    return "\n".join(lines)


CSS = """
.gradio-container { max-width: 1200px !important; margin: auto !important; }
"""

with gr.Blocks(css=CSS, title="Multiclass Botnet Detection System") as app:
    gr.Markdown(
        """
        # Multiclass IoT Botnet Detection
        Transformer, CNN-LSTM, and SVM trained on CICIoT2023 with full 33-attack coverage.
        """
    )

    with gr.Tabs():
        with gr.Tab("Traffic Analysis"):
            file_input = gr.File(label="Upload CICIoT2023 CSV", file_types=[".csv"])
            run_button = gr.Button("Analyse Traffic", variant="primary")
            report_box = gr.Textbox(label="Per-class report", lines=24)
            fig_analysis = gr.Plot(label="Per-file inference summary")
            fig_compare = gr.Plot(label="Validated model comparison")
            run_button.click(analyse_file, inputs=[file_input], outputs=[report_box, fig_analysis, fig_compare])

        with gr.Tab("Methodology"):
            gr.Textbox(value=show_methodology(), lines=20, interactive=False, label="System methodology")


if __name__ == "__main__":
    app.launch(server_name=APP_HOST, server_port=APP_PORT, share=False)
