# evaluate_properly.py
import numpy as np
import torch
import pickle
from sklearn.metrics import (classification_report,
                             confusion_matrix,
                             f1_score, roc_auc_score)
from src.preprocess import load_data, clean_data, preprocess
from src.train      import make_sequences, make_loader
from src.model      import BotnetTransformer
from src.cnn_lstm   import BotnetCNNLSTM
from config         import (MODEL_PATH, CNNLSTM_PATH,
                            D_MODEL, N_HEADS, N_LAYERS,
                            DROPOUT, THRESHOLD)

device = torch.device('cpu')

print("Loading data...")
df = load_data()
df = clean_data(df)
_, _, X_test, _, _, y_test, _, n_features = preprocess(df)

X_te_s, y_te_s = make_sequences(X_test, y_test)
test_loader    = make_loader(X_te_s, y_te_s, shuffle=False)

def evaluate(model, loader, name):
    model.eval()
    probs_all, labels_all = [], []
    with torch.no_grad():
        for xb, yb in loader:
            probs = torch.sigmoid(
                model(xb)).cpu().numpy()
            probs_all.extend(probs)
            labels_all.extend(yb.numpy())

    probs_all  = np.array(probs_all)
    labels_all = np.array(labels_all).astype(int)
    preds      = (probs_all > THRESHOLD).astype(int)

    cm         = confusion_matrix(labels_all, preds)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n{'='*55}")
    print(f"  {name} — Detailed Evaluation")
    print(f"{'='*55}")
    print(f"\n  Test set distribution:")
    print(f"  Normal : {(labels_all==0).sum():,}")
    print(f"  Attack : {(labels_all==1).sum():,}")
    print(f"  Ratio  : "
          f"{(labels_all==1).sum()/(labels_all==0).sum():.1f}:1")

    print(f"\n  Classification report:")
    print(classification_report(
        labels_all, preds,
        target_names=['Normal', 'Attack']))

    print(f"  Confusion matrix breakdown:")
    print(f"  TN correct normal   : {tn:,}")
    print(f"  FP normal as attack : {fp:,}  "
          f"(false alarm rate: {fp/(tn+fp)*100:.1f}%)")
    print(f"  FN missed attack    : {fn:,}  "
          f"(miss rate: {fn/(fn+tp)*100:.1f}%)")
    print(f"  TP correct attack   : {tp:,}")
    print(f"\n  ROC AUC : "
          f"{roc_auc_score(labels_all, probs_all):.4f}")
    print(f"{'='*55}")

print("\nLoading Transformer...")
t_model = BotnetTransformer(
    n_features, D_MODEL, N_HEADS, N_LAYERS, DROPOUT)
t_model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device))
evaluate(t_model, test_loader, 'Transformer')

print("\nLoading CNN-LSTM...")
c_model = BotnetCNNLSTM(n_features, DROPOUT)
c_model.load_state_dict(
    torch.load(CNNLSTM_PATH, map_location=device))
evaluate(c_model, test_loader, 'CNN-LSTM')