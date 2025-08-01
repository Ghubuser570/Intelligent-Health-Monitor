import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import pickle
import os
import random

MODEL_PATH = 'model.pkl'
NUM_NORMAL_SAMPLES = 1000
NUM_ANOMALY_SAMPLES = 50 

TRAINING_NORMAL_RANGES = {
    'temperature': {'min': 20.0, 'max': 25.0, 'std_dev': 1.0},
    'humidity': {'min': 40.0, 'max': 60.0, 'std_dev': 5.0},
    'pressure': {'min': 1000.0, 'max': 1015.0, 'std_dev': 2.0},
    'vibration': {'min': 0.5, 'max': 2.0, 'std_dev': 0.3}
}

TRAINING_ANOMALY_RANGES = {
    'temperature': {'min': 30.0, 'max': 40.0, 'std_dev': 3.0},
    'humidity': {'min': 80.0, 'max': 95.0, 'std_dev': 5.0},
    'pressure': {'min': 980.0, 'max': 990.0, 'std_dev': 3.0},
    'vibration': {'min': 5.0, 'max': 10.0, 'std_dev': 2.0}
}


def generate_synthetic_data(num_samples, ranges, is_anomaly=False):
    data = {}
    for sensor, props in ranges.items():
        midpoint = (props['min'] + props['max']) / 2
        values = [random.gauss(midpoint, props['std_dev']) for _ in range(num_samples)]
        data[sensor] = [max(props['min'], min(props['max'], v)) for v in values]
    return pd.DataFrame(data)

def train_model():
    print("Generating synthetic training data...")
    normal_data = generate_synthetic_data(NUM_NORMAL_SAMPLES, TRAINING_NORMAL_RANGES)
    
    features = ['temperature', 'humidity', 'pressure', 'vibration']
    X_train = normal_data[features]

    print(f"Training Isolation Forest model with {len(X_train)} normal samples...")
    model = IsolationForest(contamination=0.01, random_state=42)
    
    model.fit(X_train)
    print("Model training complete.")

    try:
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved successfully to {MODEL_PATH}")
    except Exception as e:
        print(f"Error saving model to {MODEL_PATH}: {e}")

if __name__ == "__main__":
    train_model()
