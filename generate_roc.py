# generate_roc.py
import numpy as np
import torch
from sklearn.metrics import roc_curve
from src.preprocess import load_data, clean_data, preprocess
from src.train      import make_sequences, make_loader
from src.model      import BotnetTransformer
from src.cnn_lstm   import BotnetCNNLSTM
from config import (MODEL_PATH, CNNLSTM_PATH,
                    D_MODEL, N_HEADS, N_LAYERS, DROPOUT)

device = torch.device('cpu')

print('Loading data...')
df = load_data()
df = clean_data(df)
_, _, X_test, _, _, y_test, _, n_features = preprocess(df)

X_te_s, y_te_s = make_sequences(X_test, y_test)
test_loader    = make_loader(X_te_s, y_te_s, shuffle=False)

def get_roc(model, loader):
    model.eval()
    probs_all, labels_all = [], []
    with torch.no_grad():
        for xb, yb in loader:
            probs = torch.sigmoid(
                model(xb.to(device))).cpu().numpy()
            probs_all.extend(probs)
            labels_all.extend(yb.numpy())
    fpr, tpr, _ = roc_curve(
        np.array(labels_all).astype(int),
        np.array(probs_all))
    return fpr, tpr

print('Generating Transformer ROC...')
t_model = BotnetTransformer(
    n_features, D_MODEL, N_HEADS, N_LAYERS, DROPOUT)
t_model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device))
fpr_t, tpr_t = get_roc(t_model, test_loader)
np.save('model/roc_fpr_transformer.npy', fpr_t)
np.save('model/roc_tpr_transformer.npy', tpr_t)
print(f'  Transformer ROC saved — {len(fpr_t)} points')

print('Generating CNN-LSTM ROC...')
c_model = BotnetCNNLSTM(n_features, DROPOUT)
c_model.load_state_dict(
    torch.load(CNNLSTM_PATH, map_location=device))
fpr_c, tpr_c = get_roc(c_model, test_loader)
np.save('model/roc_fpr_cnnlstm.npy', fpr_c)
np.save('model/roc_tpr_cnnlstm.npy', tpr_c)
print(f'  CNN-LSTM ROC saved — {len(fpr_c)} points')

print('\nAll ROC files saved to model/ folder')
print('You can now run: python app.py')