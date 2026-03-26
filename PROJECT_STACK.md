# Project Stack And What Was Used For What

## Purpose

This project compares deep learning and classical machine learning approaches for IoT botnet detection and provides a UI for prediction and model comparison.

## Main Technologies Used

- `Python`
  Used as the main programming language for preprocessing, training, evaluation, plotting, and UI development.

- `Streamlit`
  Used to build the interactive dashboard where the user can upload CSV traffic data, run prediction, and compare model outputs.

- `PyTorch`
  Used to define, train, save, and load the Transformer and CNN-LSTM deep learning models.

- `scikit-learn`
  Used for:
  - `StandardScaler` for feature scaling
  - `LinearSVC` for the SVM baseline
  - evaluation metrics such as accuracy, precision, recall, F1, confusion matrix, PR-related scores, and top-k accuracy

- `pandas`
  Used to read CSV files, clean tabular traffic data, prepare uploaded samples, and create tables shown in the UI.

- `numpy`
  Used for numerical operations such as sequence preparation, softmax calculation, score processing, class weights, and array storage.

- `matplotlib`
  Used to generate report figures, comparison charts, training curves, confusion matrices, and performance summary plots.

- `joblib`
  Used to save and load the SVM model.

- `pickle`
  Used to save and load the fitted scaler used during preprocessing and prediction.

## Models Used

- `Transformer`
  Used as a sequence model to capture temporal relationships in IoT traffic features through self-attention.

- `CNN-LSTM`
  Used as a hybrid deep model where CNN layers extract local feature patterns and LSTM layers capture sequence behavior over time.

- `SVM baseline`
  Used as a classical machine learning baseline to compare deep learning performance against a simpler non-neural approach.

## Dataset And Data Handling

- `CICIoT2023-style IoT traffic CSV files`
  Used as the source data for benign traffic and multiple botnet attack classes.

- `Numeric feature cleaning`
  Used to remove non-numeric columns, replace infinite values, and drop invalid rows before training.

- `StandardScaler`
  Used to normalize feature values so the models train and predict on scaled inputs.

- `Sequence construction`
  Used to transform row-based traffic data into short sequences for the Transformer and CNN-LSTM models.

- `Train / validation / test split`
  Used to separate data for training, tuning, and final evaluation.

## What We Used For Training

- `Weighted cross-entropy`
  Used to reduce the effect of class imbalance during deep model training.

- `Dropout`
  Used to regularize the Transformer and CNN-LSTM models and reduce overfitting.

- `Weight decay`
  Used in the optimizer as an additional regularization method.

- `Gradient clipping`
  Used to stabilize training and avoid exploding gradients.

- `Early stopping`
  Used to stop training when validation macro-F1 no longer improves.

- `Validation macro-F1 tracking`
  Used to choose the best saved deep learning model checkpoint.

## What We Used For Evaluation

- `Accuracy`
  Used to measure overall prediction correctness.

- `Macro Precision`
  Used to evaluate average precision equally across all classes.

- `Macro Recall`
  Used to evaluate average recall equally across all classes.

- `Macro F1`
  Used as the main balanced score for class-sensitive performance comparison.

- `Weighted F1`
  Used to measure overall F1 while accounting for class frequency.

- `PR-AP`
  Used to assess precision-recall quality, especially useful under class imbalance.

- `Top-3 Accuracy`
  Used to check whether the correct label appears among the top three predicted classes.

- `Inference time comparison`
  Used to compare efficiency between models.

- `Confusion matrices`
  Used to inspect where models confuse benign traffic and different attacks.

## What We Used In The UI

- `CSV upload`
  Used to let the user submit traffic data for prediction.

- `Botnet or benign decision`
  Used to make the result easier to understand at a glance.

- `Attack type display`
  Used to show which attack class the model predicts when botnet behavior is detected.

- `Vulnerability guidance`
  Used to provide a short explanation of the likely security weakness related to the detected category.

- `Confidence level`
  Used to communicate how certain the selected model is about its prediction.

- `Risk level`
  Used to give a more practical severity-style interpretation of the prediction.

- `Model comparison results`
  Used to compare how Transformer, CNN-LSTM, and SVM behave on the same uploaded sample.

- `Efficiency comparison`
  Used to compare model speed and saved benchmark performance.

## Project Structure

- `app.py`
  Used for the Streamlit interface, prediction workflow, and model comparison view.

- `main.py`
  Used as the main training pipeline entry point.

- `src/`
  Used to store preprocessing logic, model definitions, training code, and SVM helpers.

- `model/`
  Used to store trained weights, metrics, histories, prediction outputs, scaler files, and generated figures.

- `report_figures/`
  Used to store exported figures prepared for reporting and presentation.

## Notes

- The project combines model benchmarking and a user-facing interface.
- The UI is intended to make predictions easier to interpret by showing result type, attack category, confidence, and model comparison in a simpler format.
