import requests
import time
import random
import json

FLASK_APP_URL = "http://localhost:5000/sensor_data"
SEND_INTERVAL_SECONDS = 1
ANOMALY_PROBABILITY = 0.15 

NORMAL_RANGES = {
    'temperature': {'min': 20.0, 'max': 25.0, 'std_dev': 1.0},
    'humidity': {'min': 40.0, 'max': 60.0, 'std_dev': 5.0},
    'pressure': {'min': 1000.0, 'max': 1015.0, 'std_dev': 2.0},
    'vibration': {'min': 0.5, 'max': 2.0, 'std_dev': 0.3}
}

ANOMALY_RANGES = {
    'temperature': {'min': 30.0, 'max': 40.0, 'std_dev': 3.0},
    'humidity': {'min': 80.0, 'max': 95.0, 'std_dev': 5.0},
    'pressure': {'min': 980.0, 'max': 990.0, 'std_dev': 3.0},
    'vibration': {'min': 5.0, 'max': 10.0, 'std_dev': 2.0}
}

def generate_sensor_data(is_anomaly=False):
    data = {}
    ranges = ANOMALY_RANGES if is_anomaly else NORMAL_RANGES

    for sensor, props in ranges.items():
        midpoint = (props['min'] + props['max']) / 2
        value = random.gauss(midpoint, props['std_dev'])
        data[sensor] = max(props['min'], min(props['max'], value))
    
    return data

def send_data():
    print(f"Starting data simulation. Sending data to {FLASK_APP_URL} every {SEND_INTERVAL_SECONDS} seconds...")
    print(f"Anomaly probability: {ANOMALY_PROBABILITY * 100}%")

    try:
        while True:
            is_anomaly_event = random.random() < ANOMALY_PROBABILITY
            sensor_data = generate_sensor_data(is_anomaly=is_anomaly_event)

            json_data = json.dumps(sensor_data)

            try:
                response = requests.post(FLASK_APP_URL, json=sensor_data)
                response.raise_for_status()

                response_json = response.json()
                status = response_json.get("status", "unknown")
                message = response_json.get("message", "No message")
                detected_anomaly = response_json.get("is_anomaly", False)

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
