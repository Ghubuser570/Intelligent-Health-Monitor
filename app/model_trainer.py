# app/model_trainer.py
# This script is responsible for generating synthetic training data and
# training an Isolation Forest model for anomaly detection.
# The trained model is then saved to 'model.pkl' for use by the Flask application.

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import pickle
import os
import random

# --- Configuration ---
# Path where the trained model will be saved
MODEL_PATH = 'model.pkl'
# Number of normal data points to generate for training
NUM_NORMAL_SAMPLES = 1000
# Number of anomalous data points to generate (these help define the boundaries of 'normal')
NUM_ANOMALY_SAMPLES = 50 # A small number of known anomalies can sometimes help, but Isolation Forest is unsupervised

# --- Normal Data Ranges (for simulation of training data) ---
# These ranges should generally align with the NORMAL_RANGES in data_simulator.py
TRAINING_NORMAL_RANGES = {
    'temperature': {'min': 20.0, 'max': 25.0, 'std_dev': 1.0},
    'humidity': {'min': 40.0, 'max': 60.0, 'std_dev': 5.0},
    'pressure': {'min': 1000.0, 'max': 1015.0, 'std_dev': 2.0},
    'vibration': {'min': 0.5, 'max': 2.0, 'std_dev': 0.3}
}

# --- Anomaly Data Ranges (for generating *some* anomalies in training data, if desired) ---
# These ranges should generally align with the ANOMALY_RANGES in data_simulator.py
TRAINING_ANOMALY_RANGES = {
    'temperature': {'min': 30.0, 'max': 40.0, 'std_dev': 3.0},
    'humidity': {'min': 80.0, 'max': 95.0, 'std_dev': 5.0},
    'pressure': {'min': 980.0, 'max': 990.0, 'std_dev': 3.0},
    'vibration': {'min': 5.0, 'max': 10.0, 'std_dev': 2.0}
}


def generate_synthetic_data(num_samples, ranges, is_anomaly=False):
    """
    Generates synthetic sensor data based on specified ranges.
    """
    data = {}
    for sensor, props in ranges.items():
        midpoint = (props['min'] + props['max']) / 2
        # Use random.gauss for a more natural distribution around the midpoint
        values = [random.gauss(midpoint, props['std_dev']) for _ in range(num_samples)]
        # Clamp values to ensure they stay within min/max bounds
        data[sensor] = [max(props['min'], min(props['max'], v)) for v in values]
    return pd.DataFrame(data)

def train_model():
    """
    Generates synthetic normal data, trains an Isolation Forest model,
    and saves the trained model to MODEL_PATH.
    """
    print("Generating synthetic training data...")
    # Generate normal data
    normal_data = generate_synthetic_data(NUM_NORMAL_SAMPLES, TRAINING_NORMAL_RANGES)
    
    # Optionally, generate a small amount of anomalous data if you want the model
    # to "see" some examples of anomalies during training, though Isolation Forest
    # is designed to work well without explicit anomaly examples.
    # For Isolation Forest, it's often better to train primarily on normal data.
    # We will mostly train on normal data, as IsolationForest is an unsupervised algorithm
    # that works by isolating anomalies rather than classifying them based on examples.
    # However, a very small contamination can sometimes fine-tune its sensitivity.
    
    # Combine normal and a very small amount of anomaly data for training if desired
    # For Isolation Forest, it's common to train primarily on "normal" data.
    # The 'contamination' parameter helps it estimate the proportion of anomalies.
    
    # Features for the model
    features = ['temperature', 'humidity', 'pressure', 'vibration']
    X_train = normal_data[features]

    print(f"Training Isolation Forest model with {len(X_train)} normal samples...")
    # Initialize Isolation Forest model
    # contamination: The proportion of outliers in the data set.
    #                 It's an estimate for the model to use when fitting.
    # random_state: For reproducibility of results.
    model = IsolationForest(contamination=0.01, random_state=42) # Assuming 1% anomalies
    
    # Train the model
    model.fit(X_train)
    print("Model training complete.")

    # Save the trained model to a file
    try:
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved successfully to {MODEL_PATH}")
    except Exception as e:
        print(f"Error saving model to {MODEL_PATH}: {e}")

if __name__ == "__main__":
    train_model()
