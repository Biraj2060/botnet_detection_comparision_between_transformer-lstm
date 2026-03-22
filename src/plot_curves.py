# plot_curves.py  — run this now to save training curve charts
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

MODEL_DIR = 'model/'
os.makedirs(MODEL_DIR, exist_ok=True)

# Paste your actual epoch data here
transformer_f1   = [0.9732,0.9736,0.9735,0.9736,0.9743,0.9749,0.9748,
                    0.9751,0.9749,0.9752,0.9757,0.9753,0.9760,0.9762,
                    0.9763,0.9761,0.9764,0.9763,0.9766,0.9768,0.9745,
                    0.9765,0.9765,0.9757,0.9770,0.9760,0.9764,0.9764,
                    0.9766,0.9758,0.9764,0.9763]
transformer_loss = [0.0866,0.0566,0.0545,0.0531,0.0524,0.0515,0.0509,
                    0.0503,0.0499,0.0495,0.0491,0.0490,0.0488,0.0483,
                    0.0484,0.0483,0.0479,0.0477,0.0477,0.0474,0.0475,
                    0.0472,0.0471,0.0471,0.0466,0.0466,0.0465,0.0466,
                    0.0465,0.0463,0.0463,0.0462]
transformer_val  = [0.1019,0.0939,0.0947,0.0945,0.0910,0.0885,0.0879,
                    0.0873,0.0889,0.0880,0.0870,0.0864,0.0860,0.0863,
                    0.0845,0.0855,0.0835,0.0840,0.0840,0.0843,0.0859,
                    0.0837,0.0842,0.0859,0.0819,0.0840,0.0827,0.0824,
                    0.0822,0.0842,0.0827,0.0825]

cnnlstm_f1   = [0.9723,0.9728,0.9737,0.9738,0.9741,0.9747,0.9744,
                0.9751,0.9754,0.9755,0.9754,0.9760,0.9759,0.9757,
                0.9757,0.9758,0.9755,0.9755,0.9755]
cnnlstm_loss = [0.1191,0.0568,0.0541,0.0526,0.0514,0.0506,0.0498,
                0.0493,0.0490,0.0485,0.0481,0.0478,0.0475,0.0473,
                0.0467,0.0467,0.0461,0.0457,0.0456]
cnnlstm_val  = [0.1002,0.0968,0.0939,0.0923,0.0917,0.0901,0.0909,
                0.0896,0.0886,0.0887,0.0885,0.0882,0.0877,0.0872,
                0.0884,0.0881,0.0871,0.0873,0.0875]

# ── Chart 1: Training curves side by side ────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Training Curves — Transformer vs CNN-LSTM', fontsize=13)

axes[0].plot(transformer_loss, color='#378ADD',
             label='Transformer train')
axes[0].plot(transformer_val,  color='#378ADD',
             linestyle='--', label='Transformer val')
axes[0].plot(cnnlstm_loss, color='#D85A30',
             label='CNN-LSTM train')
axes[0].plot(cnnlstm_val,  color='#D85A30',
             linestyle='--', label='CNN-LSTM val')
axes[0].set_title('Loss over epochs')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

axes[1].plot(transformer_f1, color='#378ADD', linewidth=2,
             label='Transformer F1')
axes[1].plot(cnnlstm_f1,     color='#D85A30', linewidth=2,
             label='CNN-LSTM F1')
axes[1].set_title('Validation F1 over epochs')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('F1 Score')
axes[1].set_ylim(0.97, 0.98)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
out = os.path.join(MODEL_DIR, 'training_curves.png')
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out}")

# ── Chart 2: Final metrics bar chart ─────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
metrics  = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC']
t_vals   = [0.9587, 0.9745, 0.9802, 0.9774, 0.9866]
c_vals   = [0.9561, 0.9721, 0.9798, 0.9759, 0.9849]
x = range(len(metrics))
w = 0.35
ax.bar([i - w/2 for i in x], t_vals, w,
       label='Transformer', color='#378ADD', alpha=0.85)
ax.bar([i + w/2 for i in x], c_vals, w,
       label='CNN-LSTM', color='#D85A30', alpha=0.85)
ax.set_xticks(list(x))
ax.set_xticklabels(metrics)
ax.set_ylim(0.94, 1.0)
ax.set_title('Final Test Metrics Comparison')
ax.set_ylabel('Score')
ax.legend()
ax.grid(axis='y', alpha=0.3)
for i, (tv, cv) in enumerate(zip(t_vals, c_vals)):
    ax.text(i - w/2, tv + 0.001, f'{tv:.4f}',
            ha='center', fontsize=7)
    ax.text(i + w/2, cv + 0.001, f'{cv:.4f}',
            ha='center', fontsize=7)
plt.tight_layout()
out2 = os.path.join(MODEL_DIR, 'metrics_comparison.png')
plt.savefig(out2, dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved: {out2}")

print("\nAll charts saved to model/ folder!")
print("Files: training_curves.png, metrics_comparison.png")