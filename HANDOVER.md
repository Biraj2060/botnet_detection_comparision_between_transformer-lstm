# Botnet Detection System — Complete Project Handover
## For Codex / Future Development

---

## Project Overview

A deep learning system for botnet detection using CIC IoT Dataset 2023.
Compares two architectures: Transformer Encoder vs CNN-LSTM Hybrid.
Built by Biraj Pandey as academic project.

**GitHub:** https://github.com/Biraj2060/botnet_detection_comparision_between_transformer-lstm

---

## Final Results

| Metric | Transformer | CNN-LSTM |
|---|---|---|
| Accuracy | 0.9587 | 0.9561 |
| Precision | 0.9745 | 0.9721 |
| Recall | 0.9802 | 0.9798 |
| F1 Score | 0.9774 | 0.9759 |
| ROC AUC | 0.9866 | 0.9849 |
| PR AP | 0.9987 | 0.9985 |
| Inference | 1.3246 ms | 0.3073 ms |
| Parameters | 2,412,545 | 708,609 |
| Train epochs | 32 | 19 |
| False alarm rate | 25.5% | 28.0% |
| Miss rate | 2.0% | 2.0% |

---

## Project Structure
```
botnet_detection/
├── data/                         # CIC IoT 2023 CSV files
│   ├── BenignTraffic.pcap.csv
│   ├── Mirai-greeth_flood.pcap.csv
│   ├── Mirai-greip_flood.pcap.csv
│   ├── Mirai-udpplain.pcap.csv
│   ├── DDoS-ICMP_Flood.pcap.csv
│   ├── DDoS-SYN_Flood.pcap.csv
│   ├── DDoS-RSTFINFlood.pcap.csv
│   ├── DDoS-SlowLoris.pcap.csv
│   ├── DoS-SYN_Flood.pcap.csv
│   ├── DoS-UDP_Flood.pcap.csv
│   ├── Recon-PingSweep.pcap.csv
│   ├── Backdoor_Malware.pcap.csv
│   └── VulnerabilityScan.pcap.csv
│
├── src/
│   ├── __init__.py               # Empty — makes src a package
│   ├── preprocess.py             # Data loading, cleaning, SMOTE
│   ├── model.py                  # Transformer architecture
│   ├── cnn_lstm.py               # CNN-LSTM architecture
│   └── train.py                  # Training, evaluation, charts
│
├── model/                        # Saved after training
│   ├── best_transformer.pth      # Trained Transformer weights
│   ├── best_cnnlstm.pth          # Trained CNN-LSTM weights
│   ├── scaler.pkl                # Fitted StandardScaler
│   ├── roc_fpr_transformer.npy   # ROC curve data
│   ├── roc_tpr_transformer.npy
│   ├── roc_fpr_cnnlstm.npy
│   ├── roc_tpr_cnnlstm.npy
│   ├── pr_prec_transformer.npy   # PR curve data
│   ├── pr_rec_transformer.npy
│   ├── pr_prec_cnnlstm.npy
│   ├── pr_rec_cnnlstm.npy
│   ├── comparison.png
│   ├── training_curves.png
│   └── metrics_comparison.png
│
├── report_figures/               # All presentation figures
│   ├── figure1_dataset_overview.png
│   ├── figure2_preprocessing_pipeline.png
│   ├── figure3_architecture_comparison.png
│   ├── figure4_training_curves.png
│   ├── figure5_results_table_and_chart.png
│   ├── figure6_confusion_matrices.png
│   ├── figure7_roc_curves.png
│   ├── figure8_efficiency_comparison.png
│   ├── figure9_similarity_analysis.png
│   └── figure10_pr_curves.png
│
├── logs/                         # Training logs (auto-created)
│
├── config.py                     # All hyperparameters
├── main.py                       # Training entry point
├── app.py                        # Gradio web application
├── diagnose.py                   # Dataset diagnosis tool
├── generate_roc.py               # ROC curve generation
├── generate_pr_curve.py          # PR curve generation
├── generate_report_figures.py    # All 10 report figures
├── evaluate_properly.py          # Detailed evaluation
├── prove_selection.py            # File selection justification
├── check_similarity.py           # Same-category similarity
├── check_similarity2.py          # Cross-category comparison
└── HANDOVER.md                   # This file
```

---

## Step by Step — Everything Done

### Step 1 — Dataset Selection

**Dataset:** CIC IoT Dataset 2023
**Source:** https://www.unb.ca/cic/datasets/iotdataset-2023.html
**Total:** 33 CSV files, 7 attack categories

**Files selected (13 total):**
- 1 benign file: BenignTraffic.pcap.csv
- 3 Mirai botnet: greeth_flood, greip_flood, udpplain
- 4 DDoS: ICMP_Flood, SYN_Flood, RSTFINFlood, SlowLoris
- 2 DoS: SYN_Flood, UDP_Flood
- 1 Recon: PingSweep
- 2 Other: Backdoor_Malware, VulnerabilityScan

**Why these files:**
We ran prove_selection.py which showed:
- Same attack numbered variants: correlation = 0.9999 (near identical)
- Different attack categories: correlation = 0.15-0.32 (genuinely different)
This proves one file per category maximises diversity.

**Known limitation:**
- Within-attack class imbalance exists (DDoS has 113,426 rows vs Recon 2,262)
- Fix is coded in preprocess.py (per-category capping) but requires retraining
- Only 1 benign file exists in dataset — dataset limitation not methodology error

---

### Step 2 — Preprocessing Pipeline (src/preprocess.py)

**Order of operations — critical to get right:**
```
1. Load CSV files (MAX_ROWS_PER_FILE = 30,000 per file)
2. Assign binary labels (0=benign, 1=attack)
3. Shuffle combined dataset (prevents temporal leakage)
4. Remove NaN and infinite values (removed 11 rows)
5. Stratified train/val/test split (70/15/15)
   - stratified = maintains class ratio in each split
   - split BEFORE scaling (prevents leakage)
6. StandardScaler fitted on training set ONLY
   - applied to val and test but never fitted on them
7. SMOTE applied to training set ONLY
   - NEVER apply to val or test (causes data leakage)
   - k_neighbors=5
   - balances from 90.9% attack to 50/50
8. Sliding window sequences (length=10)
   - every 10 consecutive rows = 1 sequence
   - gives model temporal context
```

**Results after preprocessing:**
```
Raw rows loaded    : 328,906
After cleaning     : 328,895 (11 removed)
Train before SMOTE : 230,357 rows (90.9% attack)
Val                : 49,203 rows  (90.9% attack)
Test               : 49,335 rows  (90.9% attack)
Train after SMOTE  : 418,696 rows (50.0% attack)
Train sequences    : 418,687
Val sequences      : 49,194
Test sequences     : 49,326
```

---

### Step 3 — Model Architectures

**Transformer (src/model.py):**
```
Input: (batch, 10, 39)
Linear projection: 39 → 256 (d_model)
Positional encoding: sinusoidal
3x TransformerEncoderLayer:
  - 8 attention heads
  - FFN dimension: 1024
  - Dropout: 0.3
  - LayerNorm
Global average pooling over sequence
Classifier: Linear(256→128) → ReLU → Dropout(0.3) → Linear(128→1)
Output: raw logit (apply sigmoid for probability)
Parameters: 2,412,545
```

**CNN-LSTM (src/cnn_lstm.py):**
```
Input: (batch, 10, 39)
Permute to (batch, 39, 10) for Conv1D
Conv1D(39→64, kernel=3, padding=1) + BatchNorm + ReLU
Conv1D(64→128, kernel=3, padding=1) + BatchNorm + ReLU
MaxPool1D(kernel=2) → sequence length becomes 5
Dropout(0.3)
Permute back to (batch, 5, 128)
BiLSTM(128→128, 2 layers, bidirectional=True, dropout=0.3)
Take last time step → (batch, 256)
Classifier: Linear(256→64) → ReLU → Dropout(0.3) → Linear(64→1)
Output: raw logit
Parameters: 708,609
```

---

### Step 4 — Training Configuration (src/train.py)
```
Loss function  : BCEWithLogitsLoss
               (combines sigmoid + BCE, numerically stable)
Optimiser      : Adam (lr=1e-4, weight_decay=1e-4)
Scheduler      : ReduceLROnPlateau
               (mode=max, patience=3, factor=0.5)
               (monitors val F1, halves LR when stagnant)
Early stopping : patience=7 epochs
               (stops when val F1 no longer improves)
Batch size     : 512
Max epochs     : 50
Threshold      : 0.5 (confirmed optimal by PR analysis)
Gradient clip  : max_norm=1.0

Learning rate history (Transformer):
  Epoch 1-17  : LR = 0.0001
  Epoch 20    : LR = 0.00005 (halved — caused F1 spike)
  Epoch 29    : LR = 0.000025 (halved again)
  Epoch 32    : Early stop (patience exhausted)

Learning rate history (CNN-LSTM):
  Epoch 1-12  : LR = 0.0001
  Epoch 19    : Early stop
```

---

### Step 5 — Evaluation Scripts

**main.py** — full training pipeline, run to retrain everything

**evaluate_properly.py** — detailed test set evaluation:
```
Transformer:
  TN: 3,350  FP: 1,149  (false alarm rate: 25.5%)
  FN:   887  TP: 43,940 (miss rate: 2.0%)

CNN-LSTM:
  TN: 3,238  FP: 1,261  (false alarm rate: 28.0%)
  FN:   906  TP: 43,921 (miss rate: 2.0%)
```

**generate_roc.py** — generates ROC curve .npy files
Run after training, before app.py

**generate_pr_curve.py** — generates PR curve .npy files
Run after generate_roc.py

**generate_report_figures.py** — generates all 10 figures
Run after generate_pr_curve.py

---

### Step 6 — Web Application (app.py)

**3 tabs:**

Tab 1 — Traffic Analysis:
- Upload CSV (CIC IoT 2023 format, 39 features)
- Transformer model inference
- Shows: donut chart, risk gauge, summary table
- Shows: ROC curve from test set, metrics bar chart
- Risk levels: NORMAL (<10%), WARNING (10-30%), CRITICAL (>30%)

Tab 2 — Model Comparison:
- Upload CSV
- Runs both Transformer and CNN-LSTM
- Shows: metrics comparison, ROC curves, efficiency, parameters

Tab 3 — Methodology:
- Auto-loads full project documentation
- Dataset, preprocessing, architectures, results, limitations

**Run order:**
```
python main.py              # train (only once, ~11 hours)
python generate_roc.py      # ~15 minutes
python generate_pr_curve.py # ~15 minutes
python app.py               # launches at http://localhost:7861
```

---

### Step 7 — Known Issues and Limitations

**1. Within-attack class imbalance (identified, fix coded)**
DDoS: 113,426 rows vs Recon: 2,262 rows
Fix: use per-category capping in preprocess.py
Requires retraining after fix

**2. False alarm rate 25.5% on normal traffic**
Model trained on balanced (50/50) SMOTE data
Deployed on imbalanced (10:1) real-world data
Cause: only 1 benign file available in CIC IoT 2023
Fix: collect more diverse real benign traffic captures

**3. Single benign file**
CIC IoT 2023 only has BenignTraffic.pcap.csv
SMOTE synthetic samples bounded by this file's feature space
Fix: add benign files from other IoT network captures

**4. Laboratory dataset**
Traffic generated in controlled testbed
Real-world performance may differ
Fix: evaluate on real production network traffic

**5. Two attack categories excluded**
Brute Force and Spoofing not included
Justification: outside scope of botnet detection
Fix: include DictionaryBruteForce.pcap.csv and DNS_Spoofing.pcap.csv

---

### Step 8 — Future Enhancement Suggestions

**Priority 1 — Fix within-attack imbalance (no retraining needed for code)**
In config.py change to per-category capping
Already coded in preprocess.py load_data() function
Then retrain: python main.py

**Priority 2 — Add ablation study**
Train 3 variants to prove each design choice matters:
- Variant A: no SMOTE (proves SMOTE is necessary)
- Variant B: sequence length=1 (proves sequencing matters)
- Variant C: d_model=64 (proves model size matters)

**Priority 3 — Multi-class classification**
Currently binary (attack vs normal)
Extend to predict which attack type
Use softmax instead of sigmoid output
11 output classes (10 attacks + benign)

**Priority 4 — Add more models for comparison**
Random Forest (traditional ML baseline)
LSTM only (no CNN component)
GRU (alternative to LSTM)
This would make the comparison study stronger

**Priority 5 — Real-world validation**
Capture real IoT network traffic
Test trained models on live data
Measure performance degradation from lab to real world

**Priority 6 — Threshold optimization**
Current threshold: 0.5 (optimal for F1)
For security context: lower threshold (0.3) catches more attacks
but increases false alarms
For low-false-alarm context: higher threshold (0.7)
Add threshold slider to app.py

**Priority 7 — Explainability**
Add attention weight visualization for Transformer
Show which time steps in the sequence triggered detection
Use SHAP or LIME for feature importance

---

### Step 9 — How to Run Everything From Scratch
```bash
# 1. Install dependencies
pip install torch pandas numpy scikit-learn imbalanced-learn matplotlib gradio

# 2. Put CSV files in data/ folder

# 3. Diagnose your data
python diagnose.py

# 4. Train both models (~11 hours on CPU)
python main.py

# 5. Generate ROC curves (~15 minutes)
python generate_roc.py

# 6. Generate PR curves (~15 minutes)
python generate_pr_curve.py

# 7. Generate all report figures
python generate_report_figures.py

# 8. Run detailed evaluation
python evaluate_properly.py

# 9. Prove file selection is justified
python prove_selection.py

# 10. Launch web app
python app.py
# Open: http://localhost:7861
```

---

### Step 10 — Key Design Decisions and Justifications

**Why Transformer vs CNN-LSTM?**
Transformer uses self-attention — captures global dependencies
across all 10 time steps simultaneously.
CNN-LSTM uses local convolution then sequential processing.
Comparison shows Transformer marginally better on accuracy
but CNN-LSTM 4.3x faster — practical trade-off study.

**Why sequence length = 10?**
10 consecutive network flow records give temporal context.
Single row classification misses temporal attack patterns.
Mirai botnet generates sustained traffic — detectable in sequences.

**Why BCEWithLogitsLoss?**
Numerically more stable than BCELoss + separate sigmoid.
Combines log-sum-exp trick internally.
Industry standard for binary classification.

**Why SMOTE after splitting?**
If SMOTE applied before splitting, synthetic samples derived
from test data contaminate training — data leakage.
Always: split first, then SMOTE on training only.

**Why 70/15/15 split?**
70% gives model enough data to learn.
15% validation gives stable monitoring signal.
15% test gives reliable final evaluation.
Stratified ensures class ratio preserved in each split.

**Why threshold = 0.5?**
Confirmed optimal by PR curve analysis.
F1 score peaks at 0.5 for both models.
No threshold tuning required.

---

### Step 11 — Files NOT in GitHub (local only)

These files are too large for GitHub (.gitignore excludes them):
- data/*.csv (6GB total)
- model/*.pth (trained weights)
- model/scaler.pkl
- model/*.npy (ROC and PR curve data)
- logs/

Anyone cloning the repo must:
1. Download CIC IoT 2023 dataset manually
2. Put CSV files in data/ folder
3. Run python main.py to retrain
4. Run python generate_roc.py
5. Run python generate_pr_curve.py
6. Then run python app.py

---

### Step 12 — Hyperparameters (config.py)
```python
MAX_ROWS_PER_FILE = 30000    # rows loaded per CSV
SEQUENCE_LEN      = 10       # sliding window length
TEST_SIZE         = 0.15     # 15% for test
VAL_SIZE          = 0.15     # 15% for validation
RANDOM_STATE      = 42       # reproducibility seed

INPUT_DIM  = 38              # features (auto-detected)
D_MODEL    = 256             # Transformer embedding size
N_HEADS    = 8               # attention heads
N_LAYERS   = 3               # Transformer layers
DROPOUT    = 0.3             # dropout rate

EPOCHS        = 50           # maximum (early stop fires)
BATCH_SIZE    = 512          # sequences per batch
LEARNING_RATE = 1e-4         # initial learning rate
PATIENCE      = 7            # early stopping patience
THRESHOLD     = 0.5          # classification boundary

APP_HOST = '127.0.0.1'
APP_PORT = 7861
```

---

### Step 13 — Comparison with Related Work

**Alkahtani et al. 2021 (closest related paper):**
- Used CNN-LSTM on N-BaIoT dataset
- Achieved 88-91% accuracy
- No SMOTE, no validation set, fixed 20 epochs
- Multiclass (11 classes)

**This project improvements:**
- CIC IoT 2023 (more recent, broader dataset)
- Added Transformer as comparison model
- SMOTE for class balancing
- Proper 3-way split with validation set
- Early stopping prevents overfitting
- Binary classification (more practical)
- Achieved 95.87% accuracy (Transformer)
- Full evaluation: ROC, PR, confusion matrix

---

*End of handover document*
*Project completed March 2026*
*Author: Biraj Pandey*