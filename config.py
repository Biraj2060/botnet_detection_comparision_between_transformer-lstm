import os


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "model")
LOG_DIR = os.path.join(BASE_DIR, "logs")
REPORT_DIR = os.path.join(BASE_DIR, "report_figures")

BENIGN_FILES = ["BenignTraffic.pcap.csv"]

ATTACK_CATEGORY_FILES = {
    "DDoS": [
        "DDoS-ACK_Fragmentation.pcap.csv",
        "DDoS-HTTP_Flood-.pcap.csv",
        "DDoS-ICMP_Flood.pcap.csv",
        "DDoS-ICMP_Fragmentation.pcap.csv",
        "DDoS-PSHACK_Flood.pcap.csv",
        "DDoS-RSTFINFlood.pcap.csv",
        "DDoS-SlowLoris.pcap.csv",
        "DDoS-SYN_Flood.pcap.csv",
        "DDoS-SynonymousIP_Flood.pcap.csv",
        "DDoS-TCP_Flood.pcap.csv",
        "DDoS-UDP_Flood.pcap.csv",
        "DDoS-UDP_Fragmentation.pcap.csv",
    ],
    "DoS": [
        "DoS-HTTP_Flood.pcap.csv",
        "DoS-SYN_Flood.pcap.csv",
        "DoS-TCP_Flood.pcap.csv",
        "DoS-UDP_Flood.pcap.csv",
    ],
    "Mirai Botnet": [
        "Mirai-greeth_flood.pcap.csv",
        "Mirai-greip_flood.pcap.csv",
        "Mirai-udpplain.pcap.csv",
    ],
    "Reconnaissance": [
        "Recon-HostDiscovery.pcap.csv",
        "Recon-OSScan.pcap.csv",
        "Recon-PingSweep.pcap.csv",
        "Recon-PortScan.pcap.csv",
    ],
    "Brute Force": [
        "DictionaryBruteForce.pcap.csv",
    ],
    "Spoofing": [
        "DNS_Spoofing.pcap.csv",
        "MITM-ArpSpoofing.pcap.csv",
    ],
    "Other": [
        "Backdoor_Malware.pcap.csv",
        "BrowserHijacking.pcap.csv",
        "CommandInjection.pcap.csv",
        "SqlInjection.pcap.csv",
        "Uploading_Attack.pcap.csv",
        "VulnerabilityScan.pcap.csv",
        "XSS.pcap.csv",
    ],
}

ATTACK_FILES = [
    file_name
    for files in ATTACK_CATEGORY_FILES.values()
    for file_name in files
]

CATEGORY_NAMES = ["Benign"] + list(ATTACK_CATEGORY_FILES.keys())

FILE_TO_CATEGORY = {BENIGN_FILES[0]: "Benign"}
for category_name, file_names in ATTACK_CATEGORY_FILES.items():
    for file_name in file_names:
        FILE_TO_CATEGORY[file_name] = category_name

CLASS_NAMES = ["Benign"] + [os.path.splitext(os.path.splitext(name)[0])[0] for name in ATTACK_FILES]
CLASS_FILES = BENIGN_FILES + ATTACK_FILES
FILE_TO_CLASS = dict(zip(CLASS_FILES, CLASS_NAMES))
CLASS_TO_INDEX = {class_name: index for index, class_name in enumerate(CLASS_NAMES)}
INDEX_TO_CLASS = {index: class_name for class_name, index in CLASS_TO_INDEX.items()}
FILE_TO_CLASS_INDEX = {
    file_name: CLASS_TO_INDEX[class_name]
    for file_name, class_name in FILE_TO_CLASS.items()
}
CLASS_TO_CATEGORY = {
    class_name: FILE_TO_CATEGORY[file_name]
    for file_name, class_name in FILE_TO_CLASS.items()
}

# Full-dataset multiclass training settings
MAX_ROWS_PER_FILE = None
SEQUENCE_LEN = 10
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15
RANDOM_STATE = 42

# Model settings
INPUT_DIM = None
D_MODEL = 256
N_HEADS = 8
N_LAYERS = 3
DROPOUT = 0.3

# Training settings
EPOCHS = 50
BATCH_SIZE = 256
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 7
GRAD_CLIP_NORM = 1.0
SVM_MAX_SAMPLES = 150000
TOP_K = 3
CONFIDENCE_THRESHOLD = 0.55

# Artifact paths
TRANSFORMER_PATH = os.path.join(MODEL_DIR, "best_transformer_multiclass.pth")
CNNLSTM_PATH = os.path.join(MODEL_DIR, "best_cnnlstm_multiclass.pth")
SVM_PATH = os.path.join(MODEL_DIR, "best_svm_multiclass.joblib")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_multiclass.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "dataset_metadata.json")
SPLIT_SUMMARY_PATH = os.path.join(MODEL_DIR, "split_summary.json")
CLASS_WEIGHT_PATH = os.path.join(MODEL_DIR, "class_weights.npy")
TRAINING_SUMMARY_PATH = os.path.join(MODEL_DIR, "training_summary.json")
RESULTS_PATH = os.path.join(MODEL_DIR, "results_summary.json")

TRANSFORMER_HISTORY_PATH = os.path.join(MODEL_DIR, "history_transformer.json")
CNNLSTM_HISTORY_PATH = os.path.join(MODEL_DIR, "history_cnn_lstm.json")
SVM_HISTORY_PATH = os.path.join(MODEL_DIR, "history_svm.json")

TRANSFORMER_PRED_PATH = os.path.join(MODEL_DIR, "predictions_transformer.npz")
CNNLSTM_PRED_PATH = os.path.join(MODEL_DIR, "predictions_cnn_lstm.npz")
SVM_PRED_PATH = os.path.join(MODEL_DIR, "predictions_svm.npz")

TRANSFORMER_METRICS_PATH = os.path.join(MODEL_DIR, "metrics_transformer.json")
CNNLSTM_METRICS_PATH = os.path.join(MODEL_DIR, "metrics_cnn_lstm.json")
SVM_METRICS_PATH = os.path.join(MODEL_DIR, "metrics_svm.json")

ROC_DATA_PATH = os.path.join(MODEL_DIR, "roc_data_multiclass.json")
PR_DATA_PATH = os.path.join(MODEL_DIR, "pr_data_multiclass.json")

APP_HOST = "127.0.0.1"
APP_PORT = 7861


def ensure_directories() -> None:
    for path in (MODEL_DIR, LOG_DIR, REPORT_DIR):
        os.makedirs(path, exist_ok=True)
