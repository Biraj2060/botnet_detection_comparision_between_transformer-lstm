# src/train.py
import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (f1_score, roc_auc_score,
                             confusion_matrix,
                             classification_report,
                             precision_score, recall_score,
                             roc_curve)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from config import (MODEL_DIR, MODEL_PATH, CNNLSTM_PATH,
                    EPOCHS, BATCH_SIZE, LEARNING_RATE,
                    PATIENCE, THRESHOLD, SEQUENCE_LEN,
                    D_MODEL, N_HEADS, N_LAYERS, DROPOUT)
from src.model    import BotnetTransformer
from src.cnn_lstm import BotnetCNNLSTM

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_sequences(X, y, seq_len=SEQUENCE_LEN):
    """
    Convert flat rows into overlapping sequences.
    Example with seq_len=3:
      rows [A,B,C,D,E] become:
      [A,B,C]->label(C), [B,C,D]->label(D), [C,D,E]->label(E)
    """
    Xs, ys = [], []
    for i in range(len(X) - seq_len + 1):
        Xs.append(X[i: i + seq_len])
        ys.append(y[i + seq_len - 1])
    return np.array(Xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def make_loader(X, y, shuffle=True):
    ds = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle)


def train_one_model(model, train_loader, val_loader,
                    save_path, model_name):
    print(f"\n  Training {model_name} on {device}...")
    model      = model.to(device)
    criterion  = nn.BCEWithLogitsLoss()
    optimiser  = torch.optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode='max', patience=3, factor=0.5)

    best_f1      = 0.0
    patience_ctr = 0
    history      = {'train_loss': [], 'val_loss': [],
                    'val_f1': [],    'val_auc': []}
    start        = time.time()

    for epoch in range(1, EPOCHS + 1):

        # ── Training phase ────────────────────────────────
        model.train()
        t_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            t_losses.append(loss.item())

        # ── Validation phase ──────────────────────────────
        model.eval()
        v_losses, probs_all, labels_all = [], [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb  = xb.to(device), yb.to(device)
                logits   = model(xb)
                v_losses.append(criterion(logits, yb).item())
                probs_all.extend(
                    torch.sigmoid(logits).cpu().numpy())
                labels_all.extend(yb.cpu().numpy())

        probs_all  = np.array(probs_all)
        labels_all = np.array(labels_all).astype(int)
        preds      = (probs_all > THRESHOLD).astype(int)
        val_f1     = f1_score(labels_all, preds, zero_division=0)
        val_auc    = roc_auc_score(labels_all, probs_all)
        t_loss     = np.mean(t_losses)
        v_loss     = np.mean(v_losses)

        history['train_loss'].append(t_loss)
        history['val_loss'].append(v_loss)
        history['val_f1'].append(val_f1)
        history['val_auc'].append(val_auc)

        print(f"  Epoch {epoch:3d}/{EPOCHS} | "
              f"train_loss={t_loss:.4f} | "
              f"val_loss={v_loss:.4f} | "
              f"val_F1={val_f1:.4f} | "
              f"val_AUC={val_auc:.4f}")

        scheduler.step(val_f1)

        if val_f1 > best_f1:
            best_f1      = val_f1
            patience_ctr = 0
            torch.save(model.state_dict(), save_path)
        else:
            patience_ctr += 1
            if patience_ctr >= PATIENCE:
                print(f"\n  Early stop at epoch {epoch} "
                      f"(best F1: {best_f1:.4f})")
                break

    train_time = time.time() - start
    print(f"  Training time : {train_time:.1f}s")
    print(f"  Best val F1   : {best_f1:.4f}")
    return history, train_time


def evaluate_model(model, test_loader, save_path, model_name):
    model.load_state_dict(
        torch.load(save_path, map_location=device))
    model = model.to(device)
    model.eval()

    probs_all, labels_all = [], []
    start = time.time()
    with torch.no_grad():
        for xb, yb in test_loader:
            probs = torch.sigmoid(
                model(xb.to(device))).cpu().numpy()
            probs_all.extend(probs)
            labels_all.extend(yb.numpy())

    inf_time   = (time.time() - start) / len(probs_all) * 1000
    probs_all  = np.array(probs_all)
    labels_all = np.array(labels_all).astype(int)
    preds      = (probs_all > THRESHOLD).astype(int)

    results = {
        'model'    : model_name,
        'accuracy' : float((preds == labels_all).mean()),
        'precision': float(precision_score(
                        labels_all, preds, zero_division=0)),
        'recall'   : float(recall_score(
                        labels_all, preds, zero_division=0)),
        'f1'       : float(f1_score(
                        labels_all, preds, zero_division=0)),
        'roc_auc'  : float(roc_auc_score(labels_all, probs_all)),
        'confusion' : confusion_matrix(labels_all, preds),
        'inf_ms'   : round(inf_time, 4),
        'params'   : sum(p.numel() for p in model.parameters()),
        'probs'    : probs_all,
        'labels'   : labels_all,
        'preds'    : preds,
    }

    print(f"\n{'='*55}")
    print(f"  TEST RESULTS — {model_name}")
    print(f"{'='*55}")
    print(classification_report(
        labels_all, preds,
        target_names=['Normal', 'Attack']))
    print(f"  ROC AUC   : {results['roc_auc']:.4f}")
    print(f"  Inference : {results['inf_ms']} ms/sample")
    print(f"  Params    : {results['params']:,}")
    return results


def save_charts(r1, r2, h1, h2):
    os.makedirs(MODEL_DIR, exist_ok=True)

    # ── Chart 1: Full comparison dashboard ───────────────
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle('Transformer vs CNN-LSTM — Full Comparison',
                 fontsize=14)
    colors = ['#378ADD', '#D85A30']
    names  = [r1['model'], r2['model']]

    # Metrics bar chart
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    ax = axes[0, 0]
    x  = np.arange(len(metrics))
    w  = 0.35
    ax.bar(x - w/2, [r1[m] for m in metrics],
           w, label=names[0], color=colors[0], alpha=0.85)
    ax.bar(x + w/2, [r2[m] for m in metrics],
           w, label=names[1], color=colors[1], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=15)
    ax.set_ylim(0.9, 1.02)
    ax.set_title('Metrics comparison')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    for i, (tv, cv) in enumerate(
            zip([r1[m] for m in metrics],
                [r2[m] for m in metrics])):
        ax.text(i - w/2, tv + 0.001, f'{tv:.3f}',
                ha='center', fontsize=7)
        ax.text(i + w/2, cv + 0.001, f'{cv:.3f}',
                ha='center', fontsize=7)

    # Val F1 curves
    ax = axes[0, 1]
    if h1['val_f1']:
        ax.plot(h1['val_f1'], color=colors[0],
                linewidth=2, label=names[0])
    if h2['val_f1']:
        ax.plot(h2['val_f1'], color=colors[1],
                linewidth=2, label=names[1])
    ax.set_title('Validation F1 over epochs')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('F1 Score')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Val Loss curves
    ax = axes[0, 2]
    if h1['train_loss']:
        ax.plot(h1['train_loss'], color=colors[0],
                label=f'{names[0]} train')
        ax.plot(h1['val_loss'],   color=colors[0],
                linestyle='--', label=f'{names[0]} val')
    if h2['train_loss']:
        ax.plot(h2['train_loss'], color=colors[1],
                label=f'{names[1]} train')
        ax.plot(h2['val_loss'],   color=colors[1],
                linestyle='--', label=f'{names[1]} val')
    ax.set_title('Loss over epochs')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # Confusion matrices
    for idx, (res, ax) in enumerate(
            [(r1, axes[1, 0]), (r2, axes[1, 1])]):
        cm = res['confusion']
        im = ax.imshow(cm, cmap='Blues')
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{cm[i,j]:,}',
                        ha='center', va='center',
                        fontsize=11, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Normal', 'Attack'])
        ax.set_yticklabels(['Normal', 'Attack'])
        ax.set_title(f'Confusion — {res["model"]}')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    # ROC curves
    ax = axes[1, 2]
    for res, color in [(r1, colors[0]), (r2, colors[1])]:
        fpr, tpr, _ = roc_curve(res['labels'], res['probs'])
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{res['model']} "
                      f"(AUC={res['roc_auc']:.4f})")
    ax.plot([0, 1], [0, 1], 'k--',
            linewidth=1, label='Random')
    ax.set_title('ROC Curve')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out1 = os.path.join(MODEL_DIR, 'comparison.png')
    plt.savefig(out1, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Chart saved : {out1}")

    # ── Chart 2: Individual training curves ──────────────
    for res, history, name, color in [
            (r1, h1, 'Transformer', colors[0]),
            (r2, h2, 'CNN-LSTM',    colors[1])]:

        if not history['val_f1']:
            continue

        fig, axes2 = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f'{name} — Training Curves', fontsize=13)

        axes2[0].plot(history['train_loss'],
                      color=color, label='Train loss')
        axes2[0].plot(history['val_loss'],
                      color=color, linestyle='--',
                      label='Val loss')
        axes2[0].set_title('Loss over epochs')
        axes2[0].set_xlabel('Epoch')
        axes2[0].set_ylabel('Loss')
        axes2[0].legend()
        axes2[0].grid(True, alpha=0.3)

        axes2[1].plot(history['val_f1'],
                      color='#1D9E75', linewidth=2,
                      label='Val F1')
        axes2[1].plot(history['val_auc'],
                      color='#7F77DD', linewidth=2,
                      label='Val AUC')
        axes2[1].set_title('F1 and AUC over epochs')
        axes2[1].set_xlabel('Epoch')
        axes2[1].set_ylim(0.95, 1.0)
        axes2[1].legend()
        axes2[1].grid(True, alpha=0.3)

        plt.tight_layout()
        out2 = os.path.join(
            MODEL_DIR,
            f'training_curves_{name.lower().replace("-","_")}.png')
        plt.savefig(out2, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Chart saved : {out2}")


def train(X_train, X_val, X_test,
          y_train, y_val, y_test, n_features):

    os.makedirs(MODEL_DIR, exist_ok=True)

    # Build sequences
    print("\n  Building sequences...")
    X_tr_s, y_tr_s = make_sequences(X_train, y_train)
    X_v_s,  y_v_s  = make_sequences(X_val,   y_val)
    X_te_s, y_te_s = make_sequences(X_test,  y_test)
    print(f"  Train : {X_tr_s.shape}")
    print(f"  Val   : {X_v_s.shape}")
    print(f"  Test  : {X_te_s.shape}")

    train_loader = make_loader(X_tr_s, y_tr_s, shuffle=True)
    val_loader   = make_loader(X_v_s,  y_v_s,  shuffle=False)
    test_loader  = make_loader(X_te_s, y_te_s, shuffle=False)

    # ── Train Transformer ─────────────────────────────────
    transformer = BotnetTransformer(
        input_dim = n_features,
        d_model   = D_MODEL,
        n_heads   = N_HEADS,
        n_layers  = N_LAYERS,
        dropout   = DROPOUT
    )
    h1, t1 = train_one_model(
        transformer, train_loader, val_loader,
        MODEL_PATH, 'Transformer')
    r1 = evaluate_model(
        transformer, test_loader, MODEL_PATH, 'Transformer')

    # ── Train CNN-LSTM ────────────────────────────────────
    cnn_lstm = BotnetCNNLSTM(
        input_dim = n_features,
        dropout   = DROPOUT)
    h2, t2 = train_one_model(
        cnn_lstm, train_loader, val_loader,
        CNNLSTM_PATH, 'CNN-LSTM')
    r2 = evaluate_model(
        cnn_lstm, test_loader, CNNLSTM_PATH, 'CNN-LSTM')

    # ── Save all charts ───────────────────────────────────
    save_charts(r1, r2, h1, h2)

    # ── Print final comparison table ──────────────────────
    print(f"\n{'='*55}")
    print(f"  FINAL COMPARISON")
    print(f"{'='*55}")
    print(f"  {'Metric':<12} {'Transformer':>14} {'CNN-LSTM':>12}")
    print(f"  {'-'*40}")
    for m in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        print(f"  {m:<12} {r1[m]:>14.4f} {r2[m]:>12.4f}")
    print(f"  {'inf_ms':<12} {r1['inf_ms']:>14} {r2['inf_ms']:>12}")
    print(f"  {'params':<12} {r1['params']:>14,} {r2['params']:>12,}")
    print(f"  {'time(s)':<12} {t1:>14.1f} {t2:>12.1f}")
    print(f"{'='*55}")

    return transformer, cnn_lstm