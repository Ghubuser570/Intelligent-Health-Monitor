# app/data_simulator.py
# This script simulates sensor data (temperature, humidity, pressure, vibration)
# and sends it as JSON POST requests to the Flask application's /sensor_data endpoint.
# It can also introduce "anomalies" periodically to test the detection system.

import requests
import time
import random
import json

# --- Configuration ---
# The URL of your Flask application's sensor data endpoint.
# 'app' is the service name defined in docker-compose.yml, which resolves to the container's IP.
# 5000 is the port the Flask app is listening on.
FLASK_APP_URL = "http://app:5000/sensor_data"
# Interval between sending data points (in seconds)
SEND_INTERVAL_SECONDS = 1
# Probability of generating an anomaly (e.g., 0.1 means 10% chance)
ANOMALY_PROBABILITY = 0.15 # Increased slightly to demonstrate anomalies more often

# --- Normal Data Ranges (for simulation) ---
NORMAL_RANGES = {
    'temperature': {'min': 20.0, 'max': 25.0, 'std_dev': 1.0}, # Celsius
    'humidity': {'min': 40.0, 'max': 60.0, 'std_dev': 5.0},   # Percentage
    'pressure': {'min': 1000.0, 'max': 1015.0, 'std_dev': 2.0}, # hPa
    'vibration': {'min': 0.5, 'max': 2.0, 'std_dev': 0.3}     # Hz
}

# --- Anomaly Data Ranges (for simulation) ---
ANOMALY_RANGES = {
    'temperature': {'min': 30.0, 'max': 40.0, 'std_dev': 3.0}, # High temperature spike
    'humidity': {'min': 80.0, 'max': 95.0, 'std_dev': 5.0},   # Very high humidity
    'pressure': {'min': 980.0, 'max': 990.0, 'std_dev': 3.0}, # Sudden pressure drop
    'vibration': {'min': 5.0, 'max': 10.0, 'std_dev': 2.0}    # High vibration (e.g., failing machinery)
}

def generate_sensor_data(is_anomaly=False):
    """
    Generates a single dictionary of simulated sensor data.
    If is_anomaly is True, it generates data outside normal ranges.
    """
    data = {}
    ranges = ANOMALY_RANGES if is_anomaly else NORMAL_RANGES

    for sensor, props in ranges.items():
        # Generate a random value within the specified min/max range,
        # with some Gaussian noise around the midpoint.
        midpoint = (props['min'] + props['max']) / 2
        value = random.gauss(midpoint, props['std_dev'])
        # Clamp the value to ensure it stays within the min/max bounds
        data[sensor] = max(props['min'], min(props['max'], value))
    
    return data

def send_data():
    """
    Generates sensor data and sends it to the Flask application.
    Periodically generates anomalous data.
    """
    print(f"Starting data simulation. Sending data to {FLASK_APP_URL} every {SEND_INTERVAL_SECONDS} seconds...")
    print(f"Anomaly probability: {ANOMALY_PROBABILITY * 100}%")

    try:
        while True:
            # Decide whether to generate normal or anomalous data
            is_anomaly_event = random.random() < ANOMALY_PROBABILITY
            sensor_data = generate_sensor_data(is_anomaly=is_anomaly_event)

            # Convert data to JSON string
            json_data = json.dumps(sensor_data)

            try:
                # Send the POST request
                response = requests.post(FLASK_APP_URL, json=sensor_data)
                response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

                response_json = response.json()
                status = response_json.get("status", "unknown")
                message = response_json.get("message", "No message")
                detected_anomaly = response_json.get("is_anomaly", False)

                # Print status based on what was sent and what was detected
                if is_anomaly_event:
                    print(f"Sent ANOMALY: {json_data} -> App Status: {status}, Detected: {detected_anomaly}")
                else:
                    print(f"Sent NORMAL: {json_data} -> App Status: {status}, Detected: {detected_anomaly}")

            except requests.exceptions.ConnectionError as e:
                print(f"Connection Error: Could not connect to Flask app at {FLASK_APP_URL}. Is the app running? Error: {e}")
            except requests.exceptions.HTTPError as e:
                print(f"HTTP Error: {e.response.status_code} - {e.response.text}")
            except json.JSONDecodeError:
                print(f"JSON Decode Error: Could not parse response from {FLASK_APP_URL}")
            except Exception as e:
                print(f"An unexpected error occurred: {e}")

            time.sleep(SEND_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\nData simulation stopped by user.")
    except Exception as e:
        print(f"An error occurred in the main simulation loop: {e}")

if __name__ == "__main__":
    send_data()
