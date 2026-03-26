# Botnet Detection Comparison Between Transformer, CNN-LSTM, and SVM

## Overview

This project is an IoT botnet detection and model comparison system built to analyze network traffic data and classify whether the observed behavior is benign or malicious. It compares three different approaches:

- Transformer
- CNN-LSTM
- SVM

The goal of the project is to measure both detection performance and model efficiency, then present the results through an interactive interface where a user can upload traffic data and inspect predictions more easily.

## Project Goal

The project was designed to:

- detect botnet-related traffic from IoT network data
- compare deep learning and classical machine learning models
- evaluate model performance using multiple classification metrics
- provide a simple interface for prediction and model comparison
- help interpret predictions using confidence, attack category, vulnerability notes, and risk level

## Models Used

### Transformer

Used to capture sequential relationships in traffic features through self-attention.

### CNN-LSTM

Used to combine convolution-based feature extraction with temporal sequence learning.

### SVM

Used as a classical machine learning baseline for comparison against deep learning methods.

## Main Features

- multiclass botnet attack detection
- benign vs botnet decision support
- attack type prediction
- confidence level reporting
- vulnerability guidance
- risk level interpretation
- model comparison for performance and efficiency
- Streamlit-based interactive UI

## Workflow

1. Read and clean the traffic CSV files
2. Keep numeric features and remove invalid values
3. Scale features using a fitted scaler
4. Build short sequences for sequence-based models
5. Train Transformer, CNN-LSTM, and SVM models
6. Evaluate the models on the test set
7. Save metrics, predictions, histories, and figures
8. Use the Streamlit app to run prediction and compare models

## Evaluation Metrics

The project uses several metrics to compare models:

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- Weighted F1
- PR-AP
- Top-3 Accuracy
- Inference Time
- Confusion Matrix

## User Interface

The project includes a Streamlit interface where the user can:

- upload CSV traffic samples
- choose a prediction model
- see whether the system is botnet or benign
- inspect the predicted attack type
- view confidence and risk level
- compare outputs from all models on the same uploaded sample

## Project Structure

- `app.py`  
  Streamlit application for prediction and model comparison

- `main.py`  
  Main training pipeline entry point

- `config.py`  
  Configuration values for paths, model settings, and training setup

- `src/`  
  Source code for preprocessing, model definitions, training, and SVM support

- `model/`  
  Saved model artifacts, metrics, histories, predictions, and generated figures

- `report_figures/`  
  Report-ready figure outputs

## Technologies Used

- Python
- Streamlit
- PyTorch
- scikit-learn
- pandas
- numpy
- matplotlib
- joblib
- pickle

## Notes

- This project focuses on both performance and interpretability.
- The comparison between Transformer, CNN-LSTM, and SVM helps show the trade-off between detection quality and efficiency.
- The UI is designed to make model outputs easier to understand for practical use.
