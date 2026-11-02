import time
import os
import joblib
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model

from prometheus_client import Gauge, start_http_server

# ==========================================
# Paths
# ==========================================

DATASET = "prediction/time_series_dataset.csv"

MODEL_PATH = "ml/models/lstm_model.keras"

SCALER_PATH = "ml/models/scaler.save"

SEQUENCE = 10

# ==========================================
# Load Model
# ==========================================

print("Loading LSTM Model...")

model = load_model(MODEL_PATH)

scaler = joblib.load(SCALER_PATH)

print("Model Loaded Successfully")

# ==========================================
# Prometheus Metrics
# ==========================================

predicted_cpu_metric = Gauge(
    "predicted_cpu_usage",
    "Predicted CPU Usage"
)

recommended_pods_metric = Gauge(
    "recommended_pods",
    "Recommended Pods"
)

# Expose metrics on port 9001
start_http_server(9001)

print("Prometheus Metrics Running on Port 9001")

# ==========================================
# Features
# ==========================================

features = [
    "CurrentUsers",
    "CPU",
    "Memory",
    "RequestsPerSecond",
    "ResponseTime"
]

# ==========================================
# Pod Recommendation
# ==========================================

def recommend_pods(cpu):

    if cpu <= 30:
        return 1
    elif cpu <= 50:
        return 2
    elif cpu <= 70:
        return 3
    elif cpu <= 90:
        return 4
    else:
        return 5

# ==========================================
# Live Prediction Loop
# ==========================================

while True:

    try:

        df = pd.read_csv(DATASET)

        df = df.ffill().bfill()

        data = df[features]

        scaled = scaler.transform(data)

        latest = scaled[-SEQUENCE:]

        latest = np.expand_dims(latest, axis=0)

        prediction = model.predict(latest, verbose=0)

        dummy = np.zeros((1, len(features)))

        dummy[:,1] = prediction[0]

        predicted_cpu = scaler.inverse_transform(dummy)[0][1]

        pods = recommend_pods(predicted_cpu)

        predicted_cpu_metric.set(predicted_cpu)

        recommended_pods_metric.set(pods)

        print("="*50)

        print(f"Predicted CPU : {predicted_cpu:.2f}%")

        print(f"Recommended Pods : {pods}")

        print("="*50)

    except Exception as e:

        print(e)

    time.sleep(2)