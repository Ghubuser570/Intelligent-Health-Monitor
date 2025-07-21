# app/app.py
# This is the main Flask application for the Building Health Monitor.
# It serves as an API endpoint for receiving sensor data, performs anomaly detection
# using a pre-trained ML model, and exposes Prometheus metrics.
# It also provides a simple web interface to display the current status and anomalies.

from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import pickle
import os
import threading
import time
from prometheus_client import generate_latest, Counter, Gauge, Histogram, make_wsgi_app
from werkzeug.middleware.dispatcher import DispatcherMiddleware

# --- Flask Application Setup ---
app = Flask(__name__)

# --- Global Variables and Constants ---
# Path to the trained ML model. The model_trainer.py script will save it here.
MODEL_PATH = 'model.pkl'
# Path for the initial training data. This is a placeholder; in a real scenario,
# data would come from a database or stream.
INITIAL_TRAINING_DATA_PATH = 'initial_training_data.csv' # Placeholder, not used directly in this app.py for loading, but conceptually for model_trainer.py

# --- Prometheus Metrics Setup ---
# Counter for total data points received
DATA_POINTS_RECEIVED = Counter(
    'data_points_received_total',
    'Total number of data points received by the application'
)

# Counter for total anomalies detected
ANOMALIES_DETECTED = Counter(
    'anomalies_detected_total',
    'Total number of anomalies detected'
)

# Gauge for the current number of active anomalies
ACTIVE_ANOMALIES = Gauge(
    'active_anomalies',
    'Current number of active anomalies being reported'
)

# Histogram for request duration
REQUEST_DURATION_SECONDS = Histogram(
    'http_request_duration_seconds',
    'Duration of HTTP requests in seconds',
    ['method', 'endpoint']
)

# --- In-memory Data Storage ---
# Store the last N data points for display and anomaly detection
MAX_DATA_POINTS = 100
current_data = [] # Stores recent sensor data
anomalies = []    # Stores detected anomalies

# --- Machine Learning Model Loading ---
model = None # Global variable to hold the loaded ML model

def load_model():
    """
    Loads the pre-trained Isolation Forest model from disk.
    If the model file does not exist, it prints an error and returns None.
    """
    global model
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, 'rb') as f:
                model = pickle.load(f)
            print(f"ML model loaded successfully from {MODEL_PATH}")
        except Exception as e:
            print(f"Error loading ML model from {MODEL_PATH}: {e}")
            model = None
    else:
        print(f"ML model not found at {MODEL_PATH}. Please run model_trainer.py first.")
        model = None

# Load the model when the application starts
load_model()

# --- Anomaly Detection Logic ---
def detect_anomaly(data_point):
    """
    Performs anomaly detection on a single data point using the loaded ML model.
    The Isolation Forest model returns -1 for anomalies and 1 for normal points.
    """
    if model is None:
        print("ML model not loaded, skipping anomaly detection.")
        return False, "Model not loaded"

    # Convert the data point to a DataFrame suitable for the model
    # Ensure the order of features matches the training data
    df = pd.DataFrame([data_point], columns=['temperature', 'humidity', 'pressure', 'vibration'])
    
    try:
        # Predict the anomaly score (-1 for anomaly, 1 for normal)
        prediction = model.predict(df)
        is_anomaly = (prediction[0] == -1)
        return is_anomaly, "Detected" if is_anomaly else "Normal"
    except Exception as e:
        print(f"Error during anomaly prediction: {e}")
        return False, f"Prediction error: {e}"

# --- Flask Routes ---

@app.route('/')
def index():
    """
    Renders the main web page displaying current data and detected anomalies.
    """
    # Simple HTML template for the UI
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Building Health Monitor</title>
        <script src="https://cdn.tailwindcss.com"></script>
        <style>
            body {
                font-family: 'Inter', sans-serif;
                background-color: #f0f4f8;
                color: #334155;
            }
            .container {
                max-width: 1200px;
                margin: 2rem auto;
                padding: 1.5rem;
                background-color: #ffffff;
                border-radius: 0.75rem;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .anomaly-card {
                background-color: #fee2e2;
                border-left: 4px solid #ef4444;
                color: #dc2626;
            }
            .normal-card {
                background-color: #d1fae5;
                border-left: 4px solid #10b981;
                color: #059669;
            }
            .data-table th, .data-table td {
                padding: 0.75rem;
                border-bottom: 1px solid #e2e8f0;
                text-align: left;
            }
            .data-table th {
                background-color: #f8fafc;
                font-weight: 600;
                color: #475569;
            }
        </style>
    </head>
    <body class="p-4">
        <div class="container">
            <h1 class="text-3xl font-bold text-center mb-6 text-gray-800">Building Health Monitor</h1>

            <div class="grid grid-cols-1 md:grid-cols-2 gap-6 mb-8">
                <div class="bg-blue-50 p-6 rounded-lg shadow-md">
                    <h2 class="text-xl font-semibold mb-4 text-blue-700">Real-time Data Stream</h2>
                    <div id="data-stream" class="space-y-3 text-sm">
                        <!-- Data points will be inserted here by JavaScript -->
                        <p class="text-gray-500">Waiting for data...</p>
                    </div>
                </div>

                <div class="bg-red-50 p-6 rounded-lg shadow-md">
                    <h2 class="text-xl font-semibold mb-4 text-red-700">Detected Anomalies</h2>
                    <div id="anomaly-list" class="space-y-3">
                        <!-- Anomalies will be inserted here by JavaScript -->
                        <p class="text-gray-500">No anomalies detected yet.</p>
                    </div>
                </div>
            </div>

            <div class="bg-gray-50 p-6 rounded-lg shadow-md">
                <h2 class="text-xl font-semibold mb-4 text-gray-700">Recent Data History</h2>
                <div class="overflow-x-auto">
                    <table class="min-w-full bg-white rounded-lg data-table">
                        <thead>
                            <tr>
                                <th>Timestamp</th>
                                <th>Temperature</th>
                                <th>Humidity</th>
                                <th>Pressure</th>
                                <th>Vibration</th>
                                <th>Status</th>
                            </tr>
                        </thead>
                        <tbody id="data-history-table-body">
                            <!-- Data history will be inserted here by JavaScript -->
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <script>
            // JavaScript to fetch data and update the UI in real-time
            const dataStreamDiv = document.getElementById('data-stream');
            const anomalyListDiv = document.getElementById('anomaly-list');
            const dataHistoryTableBody = document.getElementById('data-history-table-body');

            async function fetchData() {
                try {
                    const response = await fetch('/data');
                    const result = await response.json();

                    // Update real-time data stream
                    dataStreamDiv.innerHTML = ''; // Clear previous data
                    if (result.recent_data.length > 0) {
                        const latest = result.recent_data[result.recent_data.length - 1];
                        const statusClass = latest.is_anomaly ? 'text-red-600 font-bold' : 'text-green-600';
                        dataStreamDiv.innerHTML = `
                            <p><strong>Timestamp:</strong> ${new Date(latest.timestamp).toLocaleTimeString()}</p>
                            <p><strong>Temperature:</strong> ${latest.temperature.toFixed(2)} °C</p>
                            <p><strong>Humidity:</strong> ${latest.humidity.toFixed(2)} %</p>
                            <p><strong>Pressure:</strong> ${latest.pressure.toFixed(2)} hPa</p>
                            <p><strong>Vibration:</strong> ${latest.vibration.toFixed(2)} Hz</p>
                            <p><strong>Status:</strong> <span class="${statusClass}">${latest.status}</span></p>
                        `;
                    } else {
                        dataStreamDiv.innerHTML = '<p class="text-gray-500">Waiting for data...</p>';
                    }

                    // Update anomalies list
                    anomalyListDiv.innerHTML = ''; // Clear previous anomalies
                    if (result.anomalies.length > 0) {
                        result.anomalies.forEach(anomaly => {
                            const anomalyCard = `
                                <div class="anomaly-card p-3 rounded-md shadow-sm text-sm">
                                    <p><strong>Anomaly Detected!</strong></p>
                                    <p>Timestamp: ${new Date(anomaly.timestamp).toLocaleTimeString()}</p>
                                    <p>Data: Temp=${anomaly.temperature.toFixed(2)}, Hum=${anomaly.humidity.toFixed(2)}, Pres=${anomaly.pressure.toFixed(2)}, Vib=${anomaly.vibration.toFixed(2)}</p>
                                </div>
                            `;
                            anomalyListDiv.innerHTML += anomalyCard;
                        });
                    } else {
                        anomalyListDiv.innerHTML = '<p class="text-gray-500">No anomalies detected yet.</p>';
                    }

                    // Update data history table
                    dataHistoryTableBody.innerHTML = ''; // Clear previous history
                    result.recent_data.slice().reverse().forEach(data => { // Reverse to show latest at top
                        const rowClass = data.is_anomaly ? 'anomaly-card' : 'normal-card';
                        const statusTextClass = data.is_anomaly ? 'text-red-600 font-bold' : 'text-green-600';
                        const row = `
                            <tr class="${rowClass.replace('-card', '')}">
                                <td>${new Date(data.timestamp).toLocaleTimeString()}</td>
                                <td>${data.temperature.toFixed(2)}</td>
                                <td>${data.humidity.toFixed(2)}</td>
                                <td>${data.pressure.toFixed(2)}</td>
                                <td>${data.vibration.toFixed(2)}</td>
                                <td><span class="${statusTextClass}">${data.status}</span></td>
                            </tr>
                        `;
                        dataHistoryTableBody.innerHTML += row;
                    });

                } catch (error) {
                    console.error('Error fetching data:', error);
                    dataStreamDiv.innerHTML = '<p class="text-red-500">Error loading data.</p>';
                    anomalyListDiv.innerHTML = '<p class="text-red-500">Error loading anomalies.</p>';
                }
            }

            // Fetch data every 2 seconds
            setInterval(fetchData, 2000);
            // Initial fetch
            fetchData();
        </script>
    </body>
    </html>
    """
    return render_template_string(html_template)

@app.route('/data', methods=['GET'])
@REQUEST_DURATION_SECONDS.labels(method='GET', endpoint='/data').time()
def get_data():
    """
    API endpoint to retrieve the current data and anomalies for the UI.
    """
    # Return a copy to avoid modification issues during iteration
    return jsonify({
        'recent_data': current_data,
        'anomalies': anomalies
    })

@app.route('/sensor_data', methods=['POST'])
@REQUEST_DURATION_SECONDS.labels(method='POST', endpoint='/sensor_data').time()
def receive_sensor_data():
    """
    API endpoint for receiving sensor data from the data simulator.
    Performs anomaly detection and updates in-memory storage.
    """
    DATA_POINTS_RECEIVED.inc() # Increment Prometheus counter

    data = request.json
    if not data:
        return jsonify({"status": "error", "message": "No JSON data received"}), 400

    # Ensure required fields are present
    required_fields = ['temperature', 'humidity', 'pressure', 'vibration']
    if not all(field in data for field in required_fields):
        return jsonify({"status": "error", "message": "Missing required sensor data fields"}), 400

    # Add timestamp to the data point
    data['timestamp'] = time.time() * 1000 # Milliseconds for JavaScript Date object

    # Perform anomaly detection
    is_anomaly, status_message = detect_anomaly(data)
    data['is_anomaly'] = is_anomaly
    data['status'] = status_message

    # Store the data point
    current_data.append(data)
    if len(current_data) > MAX_DATA_POINTS:
        current_data.pop(0) # Keep only the latest N data points

    # If anomaly, add to anomalies list and increment counter
    if is_anomaly:
        anomalies.append(data)
        ANOMALIES_DETECTED.inc()
        ACTIVE_ANOMALIES.set(len(anomalies)) # Update gauge for active anomalies
        print(f"ANOMALY DETECTED: {data}")
    else:
        # Decrement active anomalies if the latest point is normal and was previously an anomaly
        # (Simplified: In a real system, you'd have a more sophisticated way to clear active anomalies)
        if len(anomalies) > 0 and anomalies[-1]['timestamp'] == data['timestamp']: # Check if last anomaly was this point
             anomalies.pop() # Remove it if it was immediately followed by a normal point
        ACTIVE_ANOMALIES.set(len(anomalies)) # Update gauge

    return jsonify({"status": "success", "message": "Data received and processed", "is_anomaly": is_anomaly})

# --- Prometheus Metrics Endpoint ---
# Expose Prometheus metrics at /metrics
# This uses DispatcherMiddleware to serve Prometheus metrics alongside the Flask app.
# The `make_wsgi_app()` function creates a WSGI app that serves the metrics.
app.wsgi_app = DispatcherMiddleware(app.wsgi_app, {
    '/metrics': make_wsgi_app()
})

# --- Main Execution Block ---
if __name__ == '__main__':
    # When running directly (e.g., for local testing without Docker Compose),
    # ensure the model is loaded. In Docker, this is handled when the container starts.
    if model is None:
        load_model()
    # Run the Flask app in debug mode (not recommended for production)
    app.run(host='0.0.0.0', port=5000, debug=True)
