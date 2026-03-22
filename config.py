# config.py
import os

# ── Paths ─────────────────────────────────────────────────
DATA_DIR      = 'data/'
MODEL_DIR     = 'model/'
LOG_DIR       = 'logs/'
MODEL_PATH    = os.path.join(MODEL_DIR, 'best_transformer.pth')
CNNLSTM_PATH  = os.path.join(MODEL_DIR, 'best_cnnlstm.pth')
SCALER_PATH   = os.path.join(MODEL_DIR, 'scaler.pkl')

# ── Dataset files ─────────────────────────────────────────
# We use ONE file per attack category on purpose.
# This ensures diversity and prevents one attack type
# from dominating the entire dataset.
BENIGN_FILES = [
    'BenignTraffic.pcap.csv',
]

ATTACK_FILES = [
    # Mirai botnet family — core of your project
    'Mirai-greeth_flood.pcap.csv',
    'Mirai-greip_flood.pcap.csv',
    'Mirai-udpplain.pcap.csv',
    # DDoS attacks
    'DDoS-ICMP_Flood.pcap.csv',
    'DDoS-SYN_Flood.pcap.csv',
    'DDoS-RSTFINFlood.pcap.csv',
    'DDoS-SlowLoris.pcap.csv',
    # DoS attacks
    'DoS-SYN_Flood.pcap.csv',
    'DoS-UDP_Flood.pcap.csv',
    # Reconnaissance
    'Recon-PingSweep.pcap.csv',
    # Other threats
    'Backdoor_Malware.pcap.csv',
    'VulnerabilityScan.pcap.csv',
]

# ── Preprocessing settings ────────────────────────────────
# 30,000 rows per file × 13 files = ~390,000 rows total
# This is enough data without being too slow to train
MAX_ROWS_PER_FILE = 30000
SEQUENCE_LEN      = 10      # 10 consecutive rows = 1 sequence
TEST_SIZE         = 0.15    # 15% for final testing
VAL_SIZE          = 0.15    # 15% for validation during training
RANDOM_STATE      = 42      # fixed seed so results are reproducible

# ── Model settings ────────────────────────────────────────
INPUT_DIM  = 46    # auto-updated during preprocessing
D_MODEL    = 256   # transformer internal size
N_HEADS    = 8     # number of attention heads
N_LAYERS   = 3     # number of transformer layers
DROPOUT    = 0.3   # dropout rate to prevent overfitting

# ── Training settings ─────────────────────────────────────
EPOCHS        = 50     # maximum training epochs
BATCH_SIZE    = 512   # rows processed at once
LEARNING_RATE = 1e-4   # how fast the model learns
PATIENCE      = 7      # stop if no improvement for 7 epochs
THRESHOLD     = 0.5    # above 0.5 = attack, below = normal

# ── Web app settings ──────────────────────────────────────
APP_HOST = '0.0.0.0'
APP_PORT = 7861