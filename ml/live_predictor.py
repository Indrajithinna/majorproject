import os
import time
import joblib
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model
from prometheus_client import Gauge, start_http_server


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

DATASET = os.path.join(
    PROJECT_ROOT,
    "prediction",
    "time_series_dataset.csv"
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "ml",
    "models",
    "lstm_cpu_model.keras"
)

SCALER_PATH = os.path.join(
    PROJECT_ROOT,
    "ml",
    "models",
    "cpu_scaler.pkl"
)


# ============================================================
# CONFIGURATION
# ============================================================

SEQUENCE_LENGTH = 10

PREDICTION_INTERVAL = 2

PROMETHEUS_PORT = 9001

# Dataset is considered stale after this many seconds.
# This prevents old high-load data from being predicted
# when the load generator/collector has stopped.
STALE_DATA_SECONDS = 15

# Idle detection
IDLE_USERS = 0
IDLE_RPS = 0.1

# Drift detection
DRIFT_ERROR_THRESHOLD = 15.0
DRIFT_WINDOW = 5


# ============================================================
# FEATURES
# ============================================================

FEATURES = [
    "CurrentUsers",
    "CPU",
    "Memory",
    "RequestsPerSecond",
    "ResponseTime"
]


# ============================================================
# KUBERNETES LABELS
# ============================================================

NAMESPACE = "default"

DEPLOYMENT = "mgrk-backend"


# ============================================================
# DISPLAY
# ============================================================

print("=" * 65)
print("       MGRK UNIVERSITY - LIVE LSTM PREDICTION")
print("=" * 65)

print("\nProject Root:")
print(PROJECT_ROOT)

print("\nDataset:")
print(DATASET)

print("\nLSTM Model:")
print(MODEL_PATH)

print("\nScaler:")
print(SCALER_PATH)


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

if not os.path.exists(DATASET):

    raise FileNotFoundError(
        f"\nDataset not found:\n{DATASET}"
    )


if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"\nLSTM model not found:\n{MODEL_PATH}\n\n"
        "Run train_lstm.py first."
    )


if not os.path.exists(SCALER_PATH):

    raise FileNotFoundError(
        f"\nScaler not found:\n{SCALER_PATH}\n\n"
        "Run train_lstm.py first."
    )


# ============================================================
# LOAD LSTM MODEL
# ============================================================

print("\nLoading LSTM model...")

model = load_model(
    MODEL_PATH
)

print("LSTM model loaded successfully.")


# ============================================================
# LOAD SCALER
# ============================================================

print("\nLoading scaler...")

scaler = joblib.load(
    SCALER_PATH
)

print("Scaler loaded successfully.")


# ============================================================
# PROMETHEUS METRICS
# ============================================================

predicted_cpu_metric = Gauge(
    "predicted_cpu_usage",
    "LSTM predicted CPU usage percentage",
    [
        "namespace",
        "deployment"
    ]
)


recommended_pods_metric = Gauge(
    "recommended_pods",
    "Recommended number of backend pods based on predicted CPU",
    [
        "namespace",
        "deployment"
    ]
)


prediction_error_metric = Gauge(
    "prediction_error",
    "Absolute error between predicted CPU and actual CPU",
    [
        "namespace",
        "deployment"
    ]
)


prediction_confidence_metric = Gauge(
    "prediction_confidence",
    "Heuristic confidence of the LSTM prediction",
    [
        "namespace",
        "deployment"
    ]
)


workload_drift_metric = Gauge(
    "workload_drift",
    "Indicates sustained workload prediction drift",
    [
        "namespace",
        "deployment"
    ]
)


prediction_active_metric = Gauge(
    "prediction_active",
    "Indicates whether LSTM prediction is currently active",
    [
        "namespace",
        "deployment"
    ]
)


live_data_freshness_metric = Gauge(
    "live_data_freshness",
    "Indicates whether the latest dataset observation is fresh",
    [
        "namespace",
        "deployment"
    ]
)


# ============================================================
# START PROMETHEUS SERVER
# ============================================================

start_http_server(
    PROMETHEUS_PORT
)

print(
    f"\nPrometheus metrics server running on port "
    f"{PROMETHEUS_PORT}"
)

print(
    f"Metrics URL: http://localhost:{PROMETHEUS_PORT}/metrics"
)


# ============================================================
# POD RECOMMENDATION
# ============================================================

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


# ============================================================
# DATA FRESHNESS
# ============================================================

def get_data_age(timestamp_value):

    """
    Returns the age of the latest dataset record in seconds.

    If timestamp cannot be interpreted, returns None.
    """

    try:

        timestamp = pd.to_datetime(
            timestamp_value,
            errors="coerce"
        )

        if pd.isna(timestamp):
            return None

        # Timezone-aware timestamp
        if timestamp.tzinfo is not None:

            now = pd.Timestamp.now(
                tz=timestamp.tz
            )

        # Timezone-naive timestamp
        else:

            now = pd.Timestamp.now()

        age = (
            now - timestamp
        ).total_seconds()

        # Protect against small clock differences
        return max(
            0,
            age
        )

    except Exception:

        return None


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    df = pd.read_csv(
        DATASET
    )

    if df.empty:

        raise ValueError(
            "Dataset is empty."
        )

    # --------------------------------------------------------
    # Check required columns
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in FEATURES
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing columns in dataset: "
            + str(missing_columns)
        )

    # --------------------------------------------------------
    # Convert required columns to numeric
    # --------------------------------------------------------

    for column in FEATURES:

        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Fill missing values
    # --------------------------------------------------------

    df[FEATURES] = (
        df[FEATURES]
        .ffill()
        .bfill()
    )

    return df


# ============================================================
# LOAD LATEST DATA
# ============================================================

def load_latest_data():

    df = load_dataset()

    # Need enough observations for LSTM
    if len(df) < SEQUENCE_LENGTH:

        raise ValueError(
            f"Need at least {SEQUENCE_LENGTH} "
            f"rows, but dataset contains {len(df)} rows."
        )

    latest_row = df.iloc[-1]

    # --------------------------------------------------------
    # Current live values
    # --------------------------------------------------------

    current_users = float(
        latest_row["CurrentUsers"]
    )

    actual_cpu = float(
        latest_row["CPU"]
    )

    current_rps = float(
        latest_row["RequestsPerSecond"]
    )

    # --------------------------------------------------------
    # Check timestamp freshness
    # --------------------------------------------------------

    data_age = None

    if "Timestamp" in df.columns:

        data_age = get_data_age(
            latest_row["Timestamp"]
        )

    if data_age is None:

        is_fresh = False

    else:

        is_fresh = (
            data_age <= STALE_DATA_SECONDS
        )

    return (
        df,
        current_users,
        actual_cpu,
        current_rps,
        data_age,
        is_fresh
    )


# ============================================================
# PREDICT CPU
# ============================================================

def predict_cpu(df):

    data = df[
        FEATURES
    ].copy()

    # Take only the latest required observations
    data = data.iloc[
        -SEQUENCE_LENGTH:
    ]

    data_values = data.values.astype(
        np.float32
    )

    # --------------------------------------------------------
    # Scale using training scaler
    # --------------------------------------------------------

    scaled = scaler.transform(
        data_values
    )

    # --------------------------------------------------------
    # Create LSTM sequence
    # --------------------------------------------------------

    latest_sequence = scaled[
        -SEQUENCE_LENGTH:
    ]

    latest_sequence = np.expand_dims(
        latest_sequence,
        axis=0
    )

    # --------------------------------------------------------
    # LSTM prediction
    # --------------------------------------------------------

    prediction = model.predict(
        latest_sequence,
        verbose=0
    )

    predicted_scaled_cpu = float(
        prediction[0][0]
    )

    # --------------------------------------------------------
    # Convert CPU back to original percentage
    # --------------------------------------------------------

    dummy = np.zeros(
        (
            1,
            len(FEATURES)
        )
    )

    # CPU = feature index 1
    dummy[0][1] = predicted_scaled_cpu

    restored = scaler.inverse_transform(
        dummy
    )

    predicted_cpu = float(
        restored[0][1]
    )

    # Keep CPU within realistic range
    predicted_cpu = float(
        np.clip(
            predicted_cpu,
            0,
            100
        )
    )

    return predicted_cpu


# ============================================================
# PREDICTION CONFIDENCE
# ============================================================

def calculate_confidence(error):

    """
    Heuristic confidence score.

    This is NOT a statistically calibrated probability.
    It is used only as a project-level indicator.
    """

    if error is None:

        return 0.50

    if error <= 5:

        return 0.95

    elif error <= 10:

        return 0.85

    elif error <= 15:

        return 0.70

    elif error <= 20:

        return 0.55

    else:

        return 0.35


# ============================================================
# GLOBAL PREDICTION STATE
# ============================================================

previous_prediction = None

previous_row_key = None

prediction_errors = []


# ============================================================
# UPDATE PROMETHEUS METRICS
# ============================================================

def set_metric(
    metric,
    value
):

    metric.labels(
        namespace=NAMESPACE,
        deployment=DEPLOYMENT
    ).set(
        value
    )


# ============================================================
# IDLE / NO LIVE DATA
# ============================================================

def handle_idle_state(
    reason,
    current_users=0,
    actual_cpu=0,
    current_rps=0,
    data_age=None
):

    global previous_prediction
    global prediction_errors

    # --------------------------------------------------------
    # Disable LSTM prediction
    # --------------------------------------------------------

    set_metric(
        prediction_active_metric,
        0
    )

    # --------------------------------------------------------
    # Predicted CPU = 0 while inactive
    #
    # Analytics can display "--" when prediction_active = 0.
    # This also allows HPA to safely scale toward 1 pod.
    # --------------------------------------------------------

    set_metric(
        predicted_cpu_metric,
        0
    )

    # --------------------------------------------------------
    # Minimum pod recommendation
    # --------------------------------------------------------

    set_metric(
        recommended_pods_metric,
        1
    )

    # --------------------------------------------------------
    # No prediction error while inactive
    # --------------------------------------------------------

    set_metric(
        prediction_error_metric,
        0
    )

    # --------------------------------------------------------
    # No active drift detection
    # --------------------------------------------------------

    set_metric(
        workload_drift_metric,
        0
    )

    # --------------------------------------------------------
    # Confidence = 0 while inactive
    # --------------------------------------------------------

    set_metric(
        prediction_confidence_metric,
        0
    )

    # --------------------------------------------------------
    # Freshness
    # --------------------------------------------------------

    if reason == "IDLE":

        set_metric(
            live_data_freshness_metric,
            1
        )

    else:

        set_metric(
            live_data_freshness_metric,
            0
        )

    # --------------------------------------------------------
    # Clear previous prediction.
    #
    # Important:
    # We must NOT compare an old high-load prediction against
    # a new idle observation.
    # --------------------------------------------------------

    previous_prediction = None

    prediction_errors = []

    print("\n" + "=" * 65)

    print(
        "System State        : "
        f"{reason}"
    )

    print(
        f"Current Users       : "
        f"{current_users:.0f}"
    )

    print(
        f"Actual CPU          : "
        f"{actual_cpu:.2f}%"
    )

    print(
        f"Requests/sec        : "
        f"{current_rps:.2f}"
    )

    if data_age is not None:

        print(
            f"Data Age            : "
            f"{data_age:.1f} seconds"
        )

    print(
        "Predicted CPU       : --"
    )

    print(
        "Recommended Pods    : 1"
    )

    print(
        "Prediction Active   : NO"
    )

    print(
        "Prediction Error    : --"
    )

    print(
        "Prediction Confidence : --"
    )

    print(
        "Workload Drift      : NO"
    )

    print(
        f"Reason              : {reason}"
    )

    print("=" * 65)


# ============================================================
# ACTIVE PREDICTION
# ============================================================

def process_active_data(
    df,
    current_users,
    actual_cpu,
    current_rps,
    data_age
):

    global previous_prediction
    global prediction_errors

    # --------------------------------------------------------
    # Compare current actual CPU with PREVIOUS prediction.
    #
    # The previous prediction was made from the previous
    # observation. Therefore the current observation gives us
    # the actual result against which we can measure it.
    # --------------------------------------------------------

    current_error = None

    if previous_prediction is not None:

        current_error = abs(
            actual_cpu -
            previous_prediction
        )

        prediction_errors.append(
            current_error
        )

        # Keep only recent errors
        if len(prediction_errors) > DRIFT_WINDOW:

            prediction_errors = (
                prediction_errors[
                    -DRIFT_WINDOW:
                ]
            )

    # --------------------------------------------------------
    # LSTM prediction
    # --------------------------------------------------------

    predicted_cpu = predict_cpu(
        df
    )

    # --------------------------------------------------------
    # Recommended pods
    # --------------------------------------------------------

    recommended_pods = recommend_pods(
        predicted_cpu
    )

    # --------------------------------------------------------
    # Confidence
    # --------------------------------------------------------

    confidence = calculate_confidence(
        current_error
    )

    # --------------------------------------------------------
    # Workload drift
    #
    # Drift occurs when the last N prediction errors are all
    # above the configured threshold.
    # --------------------------------------------------------

    drift = False

    if len(prediction_errors) >= DRIFT_WINDOW:

        recent_errors = (
            prediction_errors[
                -DRIFT_WINDOW:
            ]
        )

        drift = all(
            error >= DRIFT_ERROR_THRESHOLD
            for error in recent_errors
        )

    # --------------------------------------------------------
    # Prometheus metrics
    # --------------------------------------------------------

    set_metric(
        prediction_active_metric,
        1
    )

    set_metric(
        predicted_cpu_metric,
        predicted_cpu
    )

    set_metric(
        recommended_pods_metric,
        recommended_pods
    )

    if current_error is not None:

        set_metric(
            prediction_error_metric,
            current_error
        )

    else:

        set_metric(
            prediction_error_metric,
            0
        )

    set_metric(
        prediction_confidence_metric,
        confidence
    )

    set_metric(
        workload_drift_metric,
        1 if drift else 0
    )

    set_metric(
        live_data_freshness_metric,
        1
    )

    # --------------------------------------------------------
    # Store current prediction.
    #
    # It will be compared with the next actual CPU value.
    # --------------------------------------------------------

    previous_prediction = predicted_cpu

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print("\n" + "=" * 65)

    print(
        "System State        : ACTIVE"
    )

    print(
        f"Current Users       : "
        f"{current_users:.0f}"
    )

    print(
        f"Actual CPU          : "
        f"{actual_cpu:.2f}%"
    )

    print(
        f"Requests/sec        : "
        f"{current_rps:.2f}"
    )

    print(
        f"Data Age            : "
        f"{data_age:.1f} seconds"
    )

    print(
        f"Predicted CPU       : "
        f"{predicted_cpu:.2f}%"
    )

    print(
        f"Recommended Pods    : "
        f"{recommended_pods}"
    )

    if current_error is not None:

        print(
            f"Prediction Error    : "
            f"{current_error:.2f}%"
        )

    else:

        print(
            "Prediction Error    : --"
        )

    print(
        f"Prediction Confidence : "
        f"{confidence * 100:.1f}%"
    )

    print(
        "Workload Drift      : "
        f"{'YES' if drift else 'NO'}"
    )

    print(
        "Prediction Active   : YES"
    )

    print("=" * 65)


# ============================================================
# LIVE PREDICTION LOOP
# ============================================================

print("\nStarting live prediction...")

print(
    "Press CTRL+C to stop."
)

print(
    f"Stale data threshold : "
    f"{STALE_DATA_SECONDS} seconds"
)

print(
    f"Idle users           : "
    f"{IDLE_USERS}"
)

print(
    f"Idle RPS             : "
    f"{IDLE_RPS}"
)

print("=" * 65)


while True:

    try:

        # ----------------------------------------------------
        # Load current dataset
        # ----------------------------------------------------

        (
            df,
            current_users,
            actual_cpu,
            current_rps,
            data_age,
            is_fresh
        ) = load_latest_data()


        # ----------------------------------------------------
        # Create a unique key for latest observation.
        #
        # This prevents repeatedly predicting the exact same
        # dataset row every 2 seconds.
        # ----------------------------------------------------

        latest_timestamp = ""

        if "Timestamp" in df.columns:

            latest_timestamp = str(
                df.iloc[-1]["Timestamp"]
            )

        current_row_key = (
            f"{len(df)}|{latest_timestamp}"
        )


        # ----------------------------------------------------
        # If the dataset has not changed, wait.
        #
        # This means prediction happens once for each new
        # collected observation instead of predicting the same
        # data repeatedly.
        # ----------------------------------------------------

        if current_row_key == previous_row_key:

            time.sleep(
                PREDICTION_INTERVAL
            )

            continue


        previous_row_key = current_row_key


        # ----------------------------------------------------
        # CHECK 1: STALE DATA
        # ----------------------------------------------------

        if not is_fresh:

            handle_idle_state(
                reason="NO LIVE DATA",
                current_users=current_users,
                actual_cpu=actual_cpu,
                current_rps=current_rps,
                data_age=data_age
            )

            time.sleep(
                PREDICTION_INTERVAL
            )

            continue


        # ----------------------------------------------------
        # CHECK 2: IDLE SYSTEM
        # ----------------------------------------------------

        if (
            current_users <= IDLE_USERS
            and
            current_rps <= IDLE_RPS
        ):

            handle_idle_state(
                reason="IDLE",
                current_users=current_users,
                actual_cpu=actual_cpu,
                current_rps=current_rps,
                data_age=data_age
            )

            time.sleep(
                PREDICTION_INTERVAL
            )

            continue


        # ----------------------------------------------------
        # ACTIVE SYSTEM
        # ----------------------------------------------------

        if len(df) < SEQUENCE_LENGTH:

            handle_idle_state(
                reason="WAITING FOR DATA",
                current_users=current_users,
                actual_cpu=actual_cpu,
                current_rps=current_rps,
                data_age=data_age
            )

            time.sleep(
                PREDICTION_INTERVAL
            )

            continue


        # ----------------------------------------------------
        # RUN ACTIVE LSTM PREDICTION
        # ----------------------------------------------------

        process_active_data(
            df=df,
            current_users=current_users,
            actual_cpu=actual_cpu,
            current_rps=current_rps,
            data_age=data_age
        )


    except KeyboardInterrupt:

        print(
            "\n\nLive predictor stopped."
        )

        break


    except Exception as error:

        print(
            "\nPrediction Error:"
        )

        print(
            error
        )


    # --------------------------------------------------------
    # Wait for next dataset update
    # --------------------------------------------------------

    time.sleep(
        PREDICTION_INTERVAL
    )