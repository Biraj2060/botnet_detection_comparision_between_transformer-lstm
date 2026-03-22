# main.py
import os
import sys
import pickle
import logging
from datetime import datetime

os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level    = logging.INFO,
    format   = '%(asctime)s [%(levelname)s] %(message)s',
    handlers = [
        logging.FileHandler(
            f'logs/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

from src.preprocess import load_data, clean_data, preprocess
from src.train      import train
from config         import SCALER_PATH, MODEL_DIR


def main():
    print('\n' + '='*55)
    print('  BOTNET DETECTION — TRAINING PIPELINE')
    print('='*55)

    print('\nSTEP 1: Loading data')
    print('-'*40)
    df = load_data()

    print('\nSTEP 2: Cleaning data')
    print('-'*40)
    df = clean_data(df)

    print('\nSTEP 3: Preprocessing')
    print('-'*40)
    (X_train, X_val, X_test,
     y_train, y_val, y_test,
     scaler, n_features) = preprocess(df)

    os.makedirs(MODEL_DIR, exist_ok=True)
    with open(SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)
    print(f'  Scaler saved : {SCALER_PATH}')

    print('\nSTEP 4: Training both models')
    print('-'*40)
    train(X_train, X_val, X_test,
          y_train, y_val, y_test,
          n_features)

    print('\n' + '='*55)
    print('  DONE — run python app.py next')
    print('='*55 + '\n')


if __name__ == '__main__':
    main()