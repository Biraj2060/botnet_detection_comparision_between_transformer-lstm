import joblib
import numpy as np


def load_svm(path: str):
    return joblib.load(path)


def predict_svm_scores(model, X_sequences: np.ndarray) -> np.ndarray:
    flat = X_sequences.reshape(len(X_sequences), -1)
    scores = model.decision_function(flat)
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    return scores
