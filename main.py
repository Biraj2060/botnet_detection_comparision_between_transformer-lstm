import logging
import os
import pickle
import sys
from datetime import datetime

from config import LOG_DIR, SCALER_PATH, ensure_directories
from src.preprocess import build_sequence_splits
from src.train import train_all_models


def configure_logging() -> None:
    ensure_directories()
    log_name = f"training_multiclass_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(LOG_DIR, log_name)),
            logging.StreamHandler(sys.stdout),
        ],
    )


def main() -> None:
    configure_logging()
    print("\n" + "=" * 72)
    print("  MULTICLASS IOT BOTNET DETECTION TRAINING PIPELINE")
    print("  Models: Transformer, CNN-LSTM, SVM")
    print("=" * 72)

    print("\nSTEP 1: Build multiclass sequence splits")
    split_bundle = build_sequence_splits(os.path.join(os.path.dirname(__file__), "data"))

    with open(SCALER_PATH, "wb") as handle:
        pickle.dump(split_bundle["scaler"], handle)
    print(f"  Saved scaler to {SCALER_PATH}")

    print("\nSTEP 2: Train all models")
    results = train_all_models(
        split_bundle["train_parts"],
        split_bundle["val_parts"],
        split_bundle["test_parts"],
        feature_count=len(split_bundle["feature_names"]),
    )

    print("\nSTEP 3: Final summary")
    for model_name, metrics in results.items():
        print(
            f"  {model_name:<12} "
            f"acc={metrics['accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} "
            f"weighted_f1={metrics['weighted_f1']:.4f} "
            f"top3={metrics['top3_accuracy']:.4f}"
        )

    print("\nPipeline complete. Next run:")
    print("  python evaluate_properly.py")
    print("  python generate_roc.py")
    print("  python generate_pr_curve.py")
    print("  python generate_report_figures.py")
    print("  python app.py")


if __name__ == "__main__":
    main()
