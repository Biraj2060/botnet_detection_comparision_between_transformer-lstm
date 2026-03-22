# app.py
import pickle
import numpy as np
import pandas as pd
import torch
import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from src.model    import BotnetTransformer
from src.cnn_lstm import BotnetCNNLSTM
from config import (MODEL_PATH, CNNLSTM_PATH, SCALER_PATH,
                    APP_HOST, APP_PORT, THRESHOLD,
                    D_MODEL, N_HEADS, N_LAYERS, DROPOUT)

# ── Load models ───────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transformer = BotnetTransformer(
    input_dim = 39,
    d_model   = D_MODEL,
    n_heads   = N_HEADS,
    n_layers  = N_LAYERS,
    dropout   = DROPOUT
).to(device)
transformer.load_state_dict(
    torch.load(MODEL_PATH, map_location=device))
transformer.eval()

cnn_lstm = BotnetCNNLSTM(
    input_dim = 39,
    dropout   = DROPOUT
).to(device)
cnn_lstm.load_state_dict(
    torch.load(CNNLSTM_PATH, map_location=device))
cnn_lstm.eval()

with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

# ── Load real ROC curves from test set ────────────────────
try:
    roc_fpr_t = np.load('model/roc_fpr_transformer.npy')
    roc_tpr_t = np.load('model/roc_tpr_transformer.npy')
    roc_fpr_c = np.load('model/roc_fpr_cnnlstm.npy')
    roc_tpr_c = np.load('model/roc_tpr_cnnlstm.npy')
    print('ROC curves loaded successfully')
except FileNotFoundError:
    print('ROC files not found — run generate_roc.py first')
    roc_fpr_t = [0, 1]
    roc_tpr_t = [0, 1]
    roc_fpr_c = [0, 1]
    roc_tpr_c = [0, 1]

print(f'Models loaded on {device}')

# ── Validated test-set results from training ──────────────
TRAINED = {
    'transformer': {
        'accuracy' : 0.9587,
        'precision': 0.9745,
        'recall'   : 0.9802,
        'f1'       : 0.9774,
        'roc_auc'  : 0.9866,
        'inf_ms'   : 1.3246,
        'params'   : 2412545,
        'epochs'   : 32,
        'tn'       : 3350,
        'fp'       : 1149,
        'fn'       : 887,
        'tp'       : 43940,
        'fpr'      : 25.5,
        'fnr'      : 2.0,
    },
    'cnn_lstm': {
        'accuracy' : 0.9561,
        'precision': 0.9721,
        'recall'   : 0.9798,
        'f1'       : 0.9759,
        'roc_auc'  : 0.9849,
        'inf_ms'   : 0.3073,
        'params'   : 708609,
        'epochs'   : 19,
        'tn'       : 3238,
        'fp'       : 1261,
        'fn'       : 906,
        'tp'       : 43921,
        'fpr'      : 28.0,
        'fnr'      : 2.0,
    }
}


# ── Helper functions ──────────────────────────────────────
def preprocess_file(filepath):
    df = pd.read_csv(filepath)
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    if 'label' in df.columns:
        df = df.drop('label', axis=1)
    df = df.select_dtypes(include=[np.number]).head(5000)
    return df


def run_inference(model, X_scaled):
    X_seq = []
    for i in range(len(X_scaled) - 10 + 1):
        X_seq.append(X_scaled[i:i + 10])
    if len(X_seq) == 0:
        return None, None
    X_tensor = torch.FloatTensor(
        np.array(X_seq)).to(device)
    with torch.no_grad():
        logits = model(X_tensor).cpu().numpy()
        probs  = torch.sigmoid(
            torch.tensor(logits)).numpy()
        preds  = (probs > THRESHOLD).astype(int)
    return probs, preds


def get_risk(botnet_pct):
    if botnet_pct > 30:
        return 'CRITICAL', 'Botnet traffic detected',    '#E53935'
    elif botnet_pct > 10:
        return 'WARNING',  'Suspicious traffic pattern', '#FB8C00'
    else:
        return 'NORMAL',   'Traffic appears benign',     '#43A047'


def style_axis(ax):
    ax.set_facecolor('#FAFAFA')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(alpha=0.3)


# ══════════════════════════════════════════════════════════
# TAB 1 — Traffic analysis
# ══════════════════════════════════════════════════════════
def analyse_traffic(file):
    if file is None:
        return 'Please upload a CSV file.', None, None

    try:
        df = preprocess_file(file.name)
    except Exception as e:
        return f'File error: {e}', None, None

    if df.shape[1] != 39:
        return (f'Feature mismatch: expected 39 features, '
                f'got {df.shape[1]}.\n'
                f'Please use CIC IoT 2023 format.'), None, None

    X_scaled     = scaler.transform(df)
    probs, preds = run_inference(transformer, X_scaled)

    if probs is None:
        return ('Insufficient data: '
                'minimum 10 rows required.'), None, None

    total        = len(preds)
    botnet_count = int(preds.sum())
    normal_count = total - botnet_count
    botnet_pct   = round(botnet_count / total * 100, 2)
    normal_pct   = round(100 - botnet_pct, 2)
    risk_level, risk_desc, risk_color = get_risk(botnet_pct)
    tr = TRAINED['transformer']

    report = f"""
{'='*54}
  NETWORK TRAFFIC ANALYSIS REPORT
  Model   : Transformer Encoder
  Dataset : CIC IoT 2023 (39 features)
{'='*54}
  INFERENCE RESULTS (uploaded file)
  Risk level     : {risk_level}
  Assessment     : {risk_desc}
{'-'*54}
  Total sequences analysed : {total:,}
  Botnet sequences         : {botnet_count:,}  ({botnet_pct}%)
  Normal sequences         : {normal_count:,}  ({normal_pct}%)
  Decision threshold (τ)   : {THRESHOLD}
{'-'*54}
  MODEL PERFORMANCE
  Evaluated on held-out test set (49,326 sequences)
  Train / Val / Test split : 70% / 15% / 15%
  Class balancing          : SMOTE (training only)
  Test set distribution    : original 10:1 (real-world)

  Accuracy   : {tr['accuracy']:.4f}
  Precision  : {tr['precision']:.4f}
  Recall     : {tr['recall']:.4f}
  F1 Score   : {tr['f1']:.4f}
  ROC AUC    : {tr['roc_auc']:.4f}
  Inference  : {tr['inf_ms']} ms per sample
{'-'*54}
  ARCHITECTURE
  Type       : Transformer Encoder
  d_model    : {D_MODEL}  |  Heads : {N_HEADS}  |  Layers : {N_LAYERS}
  Parameters : {tr['params']:,}
  Converged  : epoch {tr['epochs']} (early stopping)
{'='*54}
"""

    # ── Figure 1: Inference results ───────────────────────
    fig1, axes1 = plt.subplots(1, 3, figsize=(14, 4))
    fig1.patch.set_facecolor('#F8F9FA')
    plt.suptitle(
        'Traffic Analysis — Transformer Inference Results',
        fontsize=13, fontweight='bold', y=1.02)

    ax = axes1[0]
    ax.set_facecolor('#F8F9FA')
    wedges, _, autotexts = ax.pie(
        [normal_count, botnet_count],
        colors      = ['#1565C0', '#C62828'],
        autopct     = '%1.1f%%',
        startangle  = 90,
        wedgeprops  = dict(width=0.52,
                           edgecolor='white',
                           linewidth=2),
        pctdistance = 0.75
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight('bold')
        at.set_color('white')
    ax.set_title('Traffic composition',
                 fontsize=11, fontweight='bold', pad=14)
    ax.legend(
        handles=[
            mpatches.Patch(
                color='#1565C0',
                label=f'Normal  ({normal_pct}%)'),
            mpatches.Patch(
                color='#C62828',
                label=f'Botnet  ({botnet_pct}%)')],
        loc='lower center',
        bbox_to_anchor=(0.5, -0.14),
        ncol=1, fontsize=9, framealpha=0.0)

    ax = axes1[1]
    ax.set_facecolor('#F8F9FA')
    gauge_colors = ['#43A047', '#FDD835',
                    '#FB8C00', '#E53935']
    labels       = ['Safe\n0-10%', 'Low\n10-20%',
                    'Med\n20-30%', 'High\n>30%']
    widths       = [10, 10, 10, 70]
    left         = 0
    for gc, w, lbl in zip(gauge_colors, widths, labels):
        ax.barh(0, w, left=left, color=gc,
                height=0.35, alpha=0.75,
                edgecolor='white', linewidth=1.5)
        ax.text(left + w/2, -0.28, lbl,
                ha='center', fontsize=7.5,
                color='#444444')
        left += w
    indicator = min(botnet_pct, 99)
    ax.annotate('',
        xy     = (indicator, -0.18),
        xytext = (indicator,  0.28),
        arrowprops=dict(
            arrowstyle='-|>',
            color=risk_color, lw=2.5))
    ax.text(indicator, 0.42,
            f'{botnet_pct}%',
            ha='center', fontsize=13,
            fontweight='bold', color=risk_color)
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.55, 0.75)
    ax.axis('off')
    ax.set_title('Risk assessment',
                 fontsize=11, fontweight='bold', pad=14)

    ax = axes1[2]
    ax.set_facecolor('#F8F9FA')
    ax.axis('off')
    rows = [
        ['Metric',           'Value'],
        ['Total sequences',  f'{total:,}'],
        ['Botnet traffic',   f'{botnet_count:,} ({botnet_pct}%)'],
        ['Normal traffic',   f'{normal_count:,} ({normal_pct}%)'],
        ['Risk level',       risk_level],
        ['Threshold (t)',    str(THRESHOLD)],
        ['Model F1',         f"{tr['f1']:.4f}"],
        ['Model AUC',        f"{tr['roc_auc']:.4f}"],
    ]
    tbl = ax.table(
        cellText  = rows[1:],
        colLabels = rows[0],
        cellLoc   = 'left',
        loc       = 'center',
        bbox      = [0, 0, 1, 1]
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#DDDDDD')
        cell.set_linewidth(0.5)
        if r == 0:
            cell.set_facecolor('#1565C0')
            cell.set_text_props(color='white',
                                fontweight='bold')
        elif r % 2 == 0:
            cell.set_facecolor('#EEF2FF')
        else:
            cell.set_facecolor('white')
        if c == 1 and r == 4:
            cell.set_text_props(color=risk_color,
                                fontweight='bold')
    ax.set_title('Inference summary',
                 fontsize=11, fontweight='bold', pad=14)
    plt.tight_layout()

    # ── Figure 2: Validated model performance ────────────
    fig2, axes2 = plt.subplots(1, 2, figsize=(13, 4))
    fig2.patch.set_facecolor('#F8F9FA')
    plt.suptitle(
        'Transformer — Validated Performance  '
        '(held-out test set, 49,326 sequences)',
        fontsize=12, fontweight='bold', y=1.04)

    ax = axes2[0]
    style_axis(ax)
    ax.plot(roc_fpr_t, roc_tpr_t,
            color='#1565C0', linewidth=2.5,
            label=f"Transformer  "
                  f"(AUC = {tr['roc_auc']:.4f})")
    ax.fill_between(roc_fpr_t, roc_tpr_t,
                    alpha=0.08, color='#1565C0')
    ax.plot([0, 1], [0, 1],
            color='#999999', linestyle='--',
            linewidth=1.5, label='Random classifier')
    ax.set_title('ROC curve — test set',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('False Positive Rate', fontsize=10)
    ax.set_ylabel('True Positive Rate',  fontsize=10)
    ax.legend(fontsize=9, framealpha=0.8,
              loc='lower right')
    ax.text(0.52, 0.10,
            f"AUC = {tr['roc_auc']:.4f}",
            transform=ax.transAxes,
            fontsize=14, fontweight='bold',
            color='#1565C0',
            bbox=dict(boxstyle='round',
                      facecolor='#EEF2FF',
                      edgecolor='#1565C0',
                      linewidth=1.5))

    ax = axes2[1]
    style_axis(ax)
    metrics     = ['Accuracy', 'Precision',
                   'Recall', 'F1', 'ROC AUC']
    metric_keys = ['accuracy', 'precision',
                   'recall', 'f1', 'roc_auc']
    values      = [tr[k] for k in metric_keys]
    bar_cols    = ['#1565C0', '#1976D2', '#1E88E5',
                   '#2196F3', '#42A5F5']
    bars = ax.bar(metrics, values,
                  color=bar_cols, alpha=0.85,
                  width=0.5, edgecolor='none')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.001,
                f'{val:.4f}',
                ha='center', va='bottom',
                fontsize=9, fontweight='bold')
    ax.set_ylim(0.93, 1.01)
    ax.set_ylabel('Score', fontsize=10)
    ax.set_title(
        'Test set performance metrics\n'
        '(SMOTE balanced training, '
        'original distribution test)',
        fontsize=11, fontweight='bold')
    ax.tick_params(axis='x', labelsize=9)
    plt.tight_layout()

    return report, fig1, fig2


# ══════════════════════════════════════════════════════════
# TAB 2 — Model comparison
# ══════════════════════════════════════════════════════════
def compare_models(file):
    if file is None:
        return 'Please upload a CSV file.', None

    try:
        df = preprocess_file(file.name)
    except Exception as e:
        return f'File error: {e}', None

    if df.shape[1] != 39:
        return (f'Feature mismatch: expected 39, '
                f'got {df.shape[1]}.'), None

    X_scaled         = scaler.transform(df)
    probs_t, preds_t = run_inference(transformer, X_scaled)
    probs_c, preds_c = run_inference(cnn_lstm,    X_scaled)

    if probs_t is None:
        return ('Insufficient data: '
                'minimum 10 rows required.'), None

    total = len(preds_t)
    bt    = int(preds_t.sum())
    bc    = int(preds_c.sum())
    nt    = total - bt
    nc    = total - bc
    pct_t = round(bt / total * 100, 2)
    pct_c = round(bc / total * 100, 2)
    rl_t, rd_t, rc_t = get_risk(pct_t)
    rl_c, rd_c, rc_c = get_risk(pct_c)
    tr = TRAINED['transformer']
    cr = TRAINED['cnn_lstm']

    report = f"""
{'='*58}
  MODEL COMPARISON REPORT
  Transformer  vs  CNN-LSTM Hybrid
  CIC IoT Dataset 2023 — Botnet Detection
{'='*58}
  {'Metric':<28} {'Transformer':>13} {'CNN-LSTM':>11}
{'-'*58}
  INFERENCE ON UPLOADED FILE
  {'Total sequences':<28} {total:>13,} {total:>11,}
  {'Botnet detected':<28} {bt:>13,} {bc:>11,}
  {'Normal detected':<28} {nt:>13,} {nc:>11,}
  {'Botnet percentage':<28} {pct_t:>12}% {pct_c:>10}%
  {'Risk level':<28} {rl_t:>13} {rl_c:>11}
{'-'*58}
  TEST SET EVALUATION (49,326 sequences)
  Test set uses original 10:1 distribution
  to reflect real-world deployment conditions
  {'Accuracy':<28} {tr['accuracy']:>13.4f} {cr['accuracy']:>11.4f}
  {'Precision':<28} {tr['precision']:>13.4f} {cr['precision']:>11.4f}
  {'Recall':<28} {tr['recall']:>13.4f} {cr['recall']:>11.4f}
  {'F1 Score':<28} {tr['f1']:>13.4f} {cr['f1']:>11.4f}
  {'ROC AUC':<28} {tr['roc_auc']:>13.4f} {cr['roc_auc']:>11.4f}
{'-'*58}
  COMPUTATIONAL EFFICIENCY
  {'Inference (ms/sample)':<28} {tr['inf_ms']:>13} {cr['inf_ms']:>11}
  {'Parameters':<28} {tr['params']:>13,} {cr['params']:>11,}
  {'Training epochs':<28} {tr['epochs']:>13} {cr['epochs']:>11}
{'-'*58}
  ARCHITECTURE
  Transformer : d_model={D_MODEL}, heads={N_HEADS}, layers={N_LAYERS}
  CNN-LSTM    : Conv1D(64,128) + BiLSTM(128x2) + FC
{'-'*58}
  FINDING
  Transformer outperforms CNN-LSTM on all accuracy
  metrics. CNN-LSTM is 4.3x faster at inference and
  uses 3.4x fewer parameters — better suited for
  resource-constrained IoT edge deployment.
  Epoch 20 spike caused by ReduceLROnPlateau
  scheduler — expected behavior, not a data issue.
{'='*58}
"""

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.patch.set_facecolor('#F8F9FA')
    plt.suptitle(
        'Transformer vs CNN-LSTM — Comparative Analysis\n'
        'CIC IoT Dataset 2023  |  '
        'Test set: 49,326 sequences  |  '
        'Original 10:1 distribution',
        fontsize=12, fontweight='bold', y=1.02)

    colors = ['#1565C0', '#C62828']
    names  = ['Transformer', 'CNN-LSTM']

    ax = axes[0, 0]
    style_axis(ax)
    metrics     = ['Accuracy', 'Precision',
                   'Recall', 'F1', 'ROC AUC']
    metric_keys = ['accuracy', 'precision',
                   'recall', 'f1', 'roc_auc']
    t_vals = [tr[k] for k in metric_keys]
    c_vals = [cr[k] for k in metric_keys]
    x = np.arange(len(metrics))
    w = 0.35
    b1 = ax.bar(x - w/2, t_vals, w,
                label='Transformer',
                color=colors[0], alpha=0.85,
                edgecolor='none')
    b2 = ax.bar(x + w/2, c_vals, w,
                label='CNN-LSTM',
                color=colors[1], alpha=0.85,
                edgecolor='none')
    for bar, val in zip(list(b1) + list(b2),
                        t_vals + c_vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.0005,
                f'{val:.4f}',
                ha='center', fontsize=7,
                fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=9)
    ax.set_ylim(0.93, 1.01)
    ax.set_ylabel('Score', fontsize=10)
    ax.set_title('Test set performance metrics',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=9, framealpha=0.8)

    ax = axes[0, 1]
    style_axis(ax)
    ax.plot(roc_fpr_t, roc_tpr_t,
            color=colors[0], linewidth=2.5,
            label=f"Transformer "
                  f"(AUC={tr['roc_auc']:.4f})")
    ax.fill_between(roc_fpr_t, roc_tpr_t,
                    alpha=0.07, color=colors[0])
    ax.plot(roc_fpr_c, roc_tpr_c,
            color=colors[1], linewidth=2.5,
            label=f"CNN-LSTM "
                  f"(AUC={cr['roc_auc']:.4f})")
    ax.fill_between(roc_fpr_c, roc_tpr_c,
                    alpha=0.07, color=colors[1])
    ax.plot([0, 1], [0, 1],
            color='#999999', linestyle='--',
            linewidth=1.2, label='Random')
    ax.set_title('ROC curves — test set',
                 fontsize=11, fontweight='bold')
    ax.set_xlabel('False Positive Rate', fontsize=10)
    ax.set_ylabel('True Positive Rate',  fontsize=10)
    ax.legend(fontsize=9, framealpha=0.8,
              loc='lower right')

    ax = axes[1, 0]
    style_axis(ax)
    inf_vals = [tr['inf_ms'], cr['inf_ms']]
    bars = ax.bar(names, inf_vals,
                  color=colors, alpha=0.85,
                  width=0.4, edgecolor='none')
    for bar, val in zip(bars, inf_vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f'{val} ms',
                ha='center', va='bottom',
                fontsize=11, fontweight='bold')
    ax.set_ylabel('Inference time (ms/sample)',
                  fontsize=10)
    ax.set_title('Computational efficiency',
                 fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(inf_vals) * 1.35)

    ax = axes[1, 1]
    style_axis(ax)
    param_vals = [tr['params'], cr['params']]
    bars = ax.bar(names, param_vals,
                  color=colors, alpha=0.85,
                  width=0.4, edgecolor='none')
    for bar, val in zip(bars, param_vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 15000,
                f'{val:,}',
                ha='center', va='bottom',
                fontsize=10, fontweight='bold')
    ax.set_ylabel('Number of parameters', fontsize=10)
    ax.set_title('Model complexity',
                 fontsize=11, fontweight='bold')
    ax.set_ylim(0, max(param_vals) * 1.2)

    plt.tight_layout()
    return report, fig


# ══════════════════════════════════════════════════════════
# TAB 3 — Methodology
# ══════════════════════════════════════════════════════════
def show_methodology():
    tr = TRAINED['transformer']
    cr = TRAINED['cnn_lstm']
    return f"""
{'='*58}
  PROJECT METHODOLOGY
  Botnet Detection using Deep Learning
  CIC IoT Dataset 2023
{'='*58}

  DATASET
  Source     : Canadian Institute for Cybersecurity
  Name       : CIC IoT Dataset 2023
  Total cats : 7 attack categories, 33 attack types
  Files used : 13 files (1 benign + 12 attack types)
  Total rows : 328,895 network flow records
  Features   : 39 numeric network flow features

  Attack categories selected (5 of 7):
    Mirai botnet   : greeth_flood, greip_flood, udpplain
    DDoS           : ICMP, SYN, RSTFINFlood, SlowLoris
    DoS            : SYN_Flood, UDP_Flood
    Reconnaissance : PingSweep
    Other threats  : Backdoor_Malware, VulnerabilityScan
  Excluded (out of scope): Brute Force, Spoofing

  Selection basis:
    One file per category ensures diversity
    Covers full attack lifecycle: recon to exploit
    to botnet infection to flooding attacks
    Row cap of 30,000 per file prevents any single
    attack type from dominating the distribution

{'─'*58}
  CLASS DISTRIBUTION
  Before SMOTE : Benign  30,000  ( 9.1%)
                 Attack 298,906  (90.9%)
                 Imbalance ratio : 10:1
  After SMOTE  : Benign 209,348  (50.0%) training only
                 Attack 209,348  (50.0%) training only
  Test set     : original 10:1 distribution preserved
                 to reflect real-world deployment
  Note         : SMOTE applied AFTER splitting to
                 prevent leakage into val/test sets

{'─'*58}
  PREPROCESSING PIPELINE
  1. Load CSV files (capped at 30,000 rows per file)
  2. Assign binary labels (0=benign, 1=attack)
  3. Shuffle dataset to prevent temporal leakage
  4. Remove NaN and infinite values (removed 11 rows)
  5. Stratified split : 70% train / 15% val / 15% test
  6. StandardScaler fitted on training set only
  7. SMOTE applied to training set only
  8. Sliding window sequences (length=10)

{'─'*58}
  MODEL ARCHITECTURES

  Transformer Encoder
    Input projection   : 39 to 256
    Positional encoding: sinusoidal
    Encoder layers     : 3
    Attention heads    : 8
    FFN dimension      : 1024
    Dropout            : 0.3
    Classifier         : Linear(256 to 128) to Linear(128 to 1)
    Parameters         : {tr['params']:,}

  CNN-LSTM Hybrid
    Conv1D layer 1     : in=39,  out=64,  kernel=3
    Conv1D layer 2     : in=64,  out=128, kernel=3
    BatchNorm + ReLU   : after each Conv1D
    MaxPool1D          : kernel=2
    Bi-LSTM            : input=128, hidden=128, layers=2
    Classifier         : Linear(256 to 64) to Linear(64 to 1)
    Parameters         : {cr['params']:,}

{'─'*58}
  TRAINING CONFIGURATION
  Optimiser      : Adam (lr=1e-4, weight_decay=1e-4)
  Loss function  : BCEWithLogitsLoss
  Scheduler      : ReduceLROnPlateau
                   patience=3, factor=0.5
  Early stopping : patience=7 epochs
  Batch size     : 512
  Max epochs     : 50
  Threshold      : 0.5

  Note on epoch 20 spike:
  The F1 dip at epoch 20 is caused by the learning
  rate scheduler halving the learning rate after
  detecting stagnation. This is expected behavior.
  The model briefly destabilises then recovers.
  It is not a data problem or overfitting.

{'─'*58}
  EVALUATION RESULTS
  Test set : 49,326 sequences (unseen during training)
  Test set uses original imbalanced 10:1 distribution

  {'Metric':<22} {'Transformer':>13} {'CNN-LSTM':>11}
  {'─'*48}
  {'Accuracy':<22} {tr['accuracy']:>13.4f} {cr['accuracy']:>11.4f}
  {'Precision':<22} {tr['precision']:>13.4f} {cr['precision']:>11.4f}
  {'Recall':<22} {tr['recall']:>13.4f} {cr['recall']:>11.4f}
  {'F1 Score':<22} {tr['f1']:>13.4f} {cr['f1']:>11.4f}
  {'ROC AUC':<22} {tr['roc_auc']:>13.4f} {cr['roc_auc']:>11.4f}
  {'Inference (ms)':<22} {tr['inf_ms']:>13} {cr['inf_ms']:>11}
  {'Parameters':<22} {tr['params']:>13,} {cr['params']:>11,}
  {'Training epochs':<22} {tr['epochs']:>13} {cr['epochs']:>11}

  Confusion matrix (Transformer, test set 10:1):
  TN correct normal   : {tr['tn']:,}
  FP normal as attack : {tr['fp']:,}  (false alarm: {tr['fpr']}%)
  FN missed attack    : {tr['fn']:,}  (miss rate  : {tr['fnr']}%)
  TP correct attack   : {tr['tp']:,}

  Confusion matrix (CNN-LSTM, test set 10:1):
  TN correct normal   : {cr['tn']:,}
  FP normal as attack : {cr['fp']:,}  (false alarm: {cr['fpr']}%)
  FN missed attack    : {cr['fn']:,}  (miss rate  : {cr['fnr']}%)
  TP correct attack   : {cr['tp']:,}

{'─'*58}
  LIMITATIONS
  1. False positive rate on normal traffic is 25.5%
     (Transformer) — model prefers to flag suspicious
     traffic over missing a real attack, which is
     correct behavior in a security context
  2. Trained on laboratory-generated traffic — may
     differ from production network behavior
  3. Two attack categories excluded (Brute Force,
     Spoofing) — outside scope of botnet detection

{'─'*58}
  CONCLUSION
  Transformer achieves higher accuracy across all
  metrics (F1: +0.0015, AUC: +0.0017).
  CNN-LSTM trains 8.7x faster, infers 4.3x faster,
  uses 3.4x fewer parameters.
  Transformer recommended when accuracy is critical.
  CNN-LSTM recommended for IoT edge deployment where
  memory and compute are constrained.
{'='*58}
"""


# ══════════════════════════════════════════════════════════
# Gradio UI
# ══════════════════════════════════════════════════════════
CSS = """
.gradio-container { max-width: 1100px !important;
                    margin: auto !important; }
"""

with gr.Blocks(css=CSS,
               title='Botnet Detection System') as app:

    gr.HTML("""
    <div style="text-align:center; padding:28px 0 16px 0;
                border-bottom:2px solid #E0E0E0;
                margin-bottom:16px;">
        <div style="font-size:24px; font-weight:700;
                    color:#1A237E; letter-spacing:1px;">
            Botnet Detection System
        </div>
        <div style="font-size:12px; color:#546E7A;
                    margin-top:6px; letter-spacing:0.5px;">
            Transformer vs CNN-LSTM &nbsp;|&nbsp;
            CIC IoT Dataset 2023 &nbsp;|&nbsp;
            Network Intrusion Detection
        </div>
    </div>
    """)

    with gr.Tabs():

        # ── Tab 1 ─────────────────────────────────────────
        with gr.Tab("Traffic Analysis"):
            gr.Markdown(
                "Upload a CIC IoT 2023 format CSV file "
                "(39 numeric features). The Transformer "
                "model classifies each sequence and reports "
                "validated performance from the held-out "
                "test set."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    file1 = gr.File(
                        label      = 'Network traffic CSV',
                        file_types = ['.csv'])
                    btn1  = gr.Button(
                        'Analyse Traffic',
                        variant='primary')
                    gr.HTML("""
                    <div style="background:#F3F4F6;
                                border-left:4px solid #1565C0;
                                padding:12px 14px;
                                border-radius:0 6px 6px 0;
                                margin-top:10px;
                                font-size:12px;
                                color:#37474F;
                                line-height:1.9;">
                        <b>Risk thresholds</b><br>
                        <span style="color:#43A047;">
                        &#9632; Normal</span>
                        &nbsp;Botnet &lt; 10%<br>
                        <span style="color:#FB8C00;">
                        &#9632; Warning</span>
                        &nbsp;Botnet 10-30%<br>
                        <span style="color:#E53935;">
                        &#9632; Critical</span>
                        &nbsp;Botnet &gt; 30%<br><br>
                        <b>Model</b>: Transformer<br>
                        F1: 0.9774 &nbsp;|&nbsp;
                        AUC: 0.9866<br>
                        Params: 2,412,545
                    </div>
                    """)
                with gr.Column(scale=2):
                    out1_report = gr.Textbox(
                        label     = 'Analysis report',
                        lines     = 22,
                        max_lines = 30)

            out1_fig1 = gr.Plot(
                label='Inference results — '
                      'traffic composition and risk')
            out1_fig2 = gr.Plot(
                label='Validated model performance — '
                      'ROC curve and metrics '
                      '(held-out test set)')

            btn1.click(
                fn      = analyse_traffic,
                inputs  = [file1],
                outputs = [out1_report,
                           out1_fig1,
                           out1_fig2])

        # ── Tab 2 ─────────────────────────────────────────
        with gr.Tab("Model Comparison"):
            gr.Markdown(
                "Upload a CSV file to run inference with "
                "both models simultaneously. Includes "
                "validated test-set metrics, ROC curves "
                "from the held-out test set, inference "
                "speed, and model complexity comparison."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    file2 = gr.File(
                        label      = 'Network traffic CSV',
                        file_types = ['.csv'])
                    btn2  = gr.Button(
                        'Run Comparison',
                        variant='primary')
                    gr.HTML("""
                    <div style="background:#F3F4F6;
                                border-left:4px solid #C62828;
                                padding:12px 14px;
                                border-radius:0 6px 6px 0;
                                margin-top:10px;
                                font-size:12px;
                                color:#37474F;
                                line-height:1.9;">
                        <b>Transformer</b><br>
                        F1: 0.9774 &nbsp;|&nbsp;
                        AUC: 0.9866<br>
                        Params: 2,412,545<br>
                        Inference: 1.32 ms<br><br>
                        <b>CNN-LSTM</b><br>
                        F1: 0.9759 &nbsp;|&nbsp;
                        AUC: 0.9849<br>
                        Params: 708,609<br>
                        Inference: 0.31 ms
                    </div>
                    """)
                with gr.Column(scale=2):
                    out2_report = gr.Textbox(
                        label     = 'Comparison report',
                        lines     = 28,
                        max_lines = 40)

            out2_fig = gr.Plot(
                label='Comparative analysis — '
                      'metrics, ROC curves, '
                      'efficiency, complexity')

            btn2.click(
                fn      = compare_models,
                inputs  = [file2],
                outputs = [out2_report, out2_fig])

        # ── Tab 3 ─────────────────────────────────────────
        with gr.Tab("Methodology"):
            gr.Markdown(
                "Complete documentation of dataset "
                "selection, preprocessing pipeline, "
                "model architectures, training "
                "configuration, and evaluation results."
            )
            gr.Textbox(
                value       = show_methodology(),
                label       = 'Project methodology',
                lines       = 45,
                max_lines   = 60,
                interactive = False)

    gr.HTML("""
    <div style="text-align:center; padding:14px;
                margin-top:16px;
                border-top:1px solid #E0E0E0;
                font-size:11px; color:#90A4AE;">
        Botnet Detection System &nbsp;|&nbsp;
        Transformer &amp; CNN-LSTM &nbsp;|&nbsp;
        CIC IoT Dataset 2023 &nbsp;|&nbsp;
        Biraj Pandey
    </div>
    """)

if __name__ == '__main__':
    app.launch(
        server_name = '127.0.0.1',
        server_port = APP_PORT,
        share       = False)