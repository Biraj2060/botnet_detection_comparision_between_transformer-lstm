# Project Stack And What We Used

## Purpose

This project compares deep learning and classical machine learning approaches for IoT botnet detection and provides a UI for prediction and model comparison.

## Main Technologies Used

- Python
- Streamlit for the interactive UI
- PyTorch for deep learning models
- scikit-learn for preprocessing, metrics, and the SVM baseline
- pandas and numpy for data handling
- matplotlib for plots and report figures
- joblib and pickle for saving model artifacts

## Models Used

- Transformer
- CNN-LSTM
- SVM baseline

## Dataset And Data Handling

- CICIoT2023-style IoT traffic CSV files
- numeric feature cleaning
- scaling with `StandardScaler`
- sequence construction for sequence-based models
- train, validation, and test splitting

## What We Used For Training

- weighted cross-entropy for class imbalance handling
- dropout for regularization
- weight decay in the optimizer
- gradient clipping
- early stopping
- validation macro-F1 tracking

## What We Used For Evaluation

- accuracy
- macro precision
- macro recall
- macro F1
- weighted F1
- PR-AP
- top-3 accuracy
- inference time comparison
- confusion matrices

## What We Used In The UI

- CSV upload for prediction
- botnet or benign decision
- attack type display
- vulnerability guidance
- confidence level
- model comparison results
- efficiency comparison

## Project Structure

- `app.py`: Streamlit interface
- `main.py`: training pipeline entry point
- `src/`: preprocessing, model definitions, training, and SVM helpers
- `model/`: saved artifacts, metrics, and generated figures
- `report_figures/`: exported figures for reporting

## Notes

- The project combines model benchmarking and a user-facing interface.
- The UI is intended to make predictions easier to interpret by showing result type, attack category, confidence, and model comparison in a simpler format.
