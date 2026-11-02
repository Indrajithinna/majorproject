import os
import time
import subprocess
import requests
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from prometheus_client import Gauge, start_http_server


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "prediction",
    "time_series_dataset.csv"
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "ml",
    "models",
    "lstm_cpu_model.keras"
)

SCALER_PATH = os.path.join(
    BASE_DIR,
    "ml",
    "models",
    "cpu_scaler.pkl"
)

WORKLOAD_MODEL_PATH = os.path.join(
    BASE_DIR,
    "ml",
    "models",
    "workload_classifier.pkl"
)

WORKLOAD_SCALER_PATH = os.path.join(
    BASE_DIR,
    "ml",
    "models",
    "workload_scaler.pkl"
)


# ============================================================
# CONFIGURATION
# ============================================================

PROMETHEUS_URL = "http://localhost:9090/api/v1/query"

PREDICTION_INTERVAL = 5

SEQUENCE_LENGTH = 10

METRICS_PORT = 9002

NAMESPACE = "default"

DEPLOYMENT = "mgrk-backend"

MIN_PODS = 1

MAX_PODS = 5

SCALING_COOLDOWN = 20


# ============================================================
# LSTM FEATURES
# ============================================================

LSTM_FEATURES = [
    "CurrentUsers",
    "CPU",
    "Memory",
    "RequestsPerSecond",
    "ResponseTime"
]


# ============================================================
# WORKLOAD CLASSIFIER FEATURES
# ============================================================

WORKLOAD_FEATURES = [

    "CurrentUsers",
    "CPU",
    "Memory",
    "RequestsPerSecond",
    "ResponseTime",

    "UsersChange",
    "CPUChange",
    "RPSChange",
    "MemoryChange",
    "ResponseTimeChange",

    "UserGrowthRate",
    "RPSGrowthRate",
    "CPUGrowthRate",

    "UsersIntensity",
    "RPSIntensity",
    "ResponseIntensity",
    "CPUIntensity",

    "WorkloadScore"

]


# ============================================================
# PROMETHEUS METRICS
# ============================================================

actual_cpu = Gauge(
    "workload_actual_cpu",
    "Actual CPU from custom_cpu_usage",
    ["namespace", "deployment"]
)


predicted_cpu = Gauge(
    "workload_predicted_cpu",
    "CPU predicted by LSTM",
    ["namespace", "deployment"]
)


current_pods = Gauge(
    "workload_current_pods",
    "Actual current backend pods",
    ["namespace", "deployment"]
)


recommended_pods = Gauge(
    "workload_recommended_pods",
    "AI recommended pods from predicted CPU",
    ["namespace", "deployment"]
)


workload_state_metric = Gauge(
    "workload_state",
    "Separate workload classifier state",
    ["namespace", "deployment", "state"]
)


scaling_score = Gauge(
    "workload_scaling_score",
    "AI scaling score",
    ["namespace", "deployment"]
)


prediction_active = Gauge(
    "workload_prediction_active",
    "LSTM prediction active",
    ["namespace", "deployment"]
)


data_age_metric = Gauge(
    "workload_data_age_seconds",
    "Age of dataset data",
    ["namespace", "deployment"]
)


# ============================================================
# LOAD MODELS
# ============================================================

print()
print("=" * 70)
print("          MGRK UNIVERSITY AI PREDICTOR")
print("=" * 70)
print()

print("Loading LSTM model...")

lstm_model = tf.keras.models.load_model(
    MODEL_PATH
)

lstm_scaler = joblib.load(
    SCALER_PATH
)

print("LSTM loaded successfully.")


print()
print("Loading workload classifier...")

workload_model = joblib.load(
    WORKLOAD_MODEL_PATH
)

workload_scaler = joblib.load(
    WORKLOAD_SCALER_PATH
)

print("Workload classifier loaded successfully.")


# ============================================================
# VERIFY WORKLOAD FEATURES
# ============================================================

expected_features = len(
    WORKLOAD_FEATURES
)

actual_features = getattr(
    workload_scaler,
    "n_features_in_",
    None
)

print()
print(
    f"Expected workload features : "
    f"{expected_features}"
)

print(
    f"Scaler workload features   : "
    f"{actual_features}"
)

if (
    actual_features is not None
    and
    actual_features != expected_features
):

    raise RuntimeError(
        f"Feature mismatch. "
        f"Scaler expects {actual_features}, "
        f"predictor provides {expected_features}."
    )

print("Feature configuration verified.")


# ============================================================
# PROMETHEUS QUERY
# ============================================================

def query_prometheus(query):

    try:

        response = requests.get(
            PROMETHEUS_URL,
            params={
                "query": query
            },
            timeout=5
        )


        if response.status_code != 200:

            print(
                f"Prometheus HTTP error: "
                f"{response.status_code}"
            )

            return None


        data = response.json()


        if data.get("status") != "success":

            return None


        results = (
            data
            .get("data", {})
            .get("result", [])
        )


        if not results:

            return None


        return float(
            results[0]["value"][1]
        )


    except Exception as error:

        print(
            f"Prometheus query error: {error}"
        )

        return None


# ============================================================
# REAL ACTUAL CPU
#
# IMPORTANT:
# This MUST match Grafana.
#
# Grafana uses:
# custom_cpu_usage
# ============================================================

def get_actual_cpu():

    value = query_prometheus(
        "custom_cpu_usage"
    )


    if value is None:

        return None


    return float(value)


# ============================================================
# REAL CURRENT PODS
#
# This comes from Prometheus current_pods.
# ============================================================

def get_current_pods():

    # First try the exact metric.
    value = query_prometheus(
        "current_pods"
    )


    if value is not None:

        return int(
            round(value)
        )


    # Fallback to Kubernetes if the metric
    # is temporarily unavailable.

    try:

        result = subprocess.run(
            [
                "kubectl",
                "get",
                "deployment",
                DEPLOYMENT,
                "-n",
                NAMESPACE,
                "-o",
                "jsonpath={.status.availableReplicas}"
            ],
            capture_output=True,
            text=True,
            timeout=5
        )


        if result.returncode == 0:

            output = result.stdout.strip()

            if output:

                return int(output)


    except Exception:
        pass


    return 0


# ============================================================
# LIVE MEMORY
# ============================================================

def get_actual_memory():

    value = query_prometheus(
        "custom_memory_usage"
    )


    if value is None:

        return 0.0


    return float(value)


# ============================================================
# LIVE REQUESTS PER SECOND
# ============================================================

def get_actual_rps():

    value = query_prometheus(
        'sum(rate(http_requests_total{job="backend"}[1m]))'
    )


    if value is None:

        value = query_prometheus(
            "sum(rate(http_requests_total[1m]))"
        )


    if value is None:

        return 0.0


    return max(
        float(value),
        0.0
    )


# ============================================================
# LIVE RESPONSE TIME
# ============================================================

def get_actual_response_time():

    query = """
    (
        sum(
            rate(
                http_request_duration_seconds_sum{job="backend"}[1m]
            )
        )
        /
        sum(
            rate(
                http_request_duration_seconds_count{job="backend"}[1m]
            )
        )
    )
    """

    value = query_prometheus(
        query
    )


    if value is None:

        return 0.0


    return max(
        float(value),
        0.0
    )


# ============================================================
# DATASET
# ============================================================

def load_dataset():

    if not os.path.exists(
        DATASET_PATH
    ):

        return None


    try:

        df = pd.read_csv(
            DATASET_PATH
        )


        if len(df) == 0:

            return None


        df["Timestamp"] = pd.to_datetime(
            df["Timestamp"],
            format="mixed",
            errors="coerce"
        )


        df = df.dropna(
            subset=["Timestamp"]
        )


        for column in LSTM_FEATURES:

            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )


        df = df.dropna(
            subset=LSTM_FEATURES
        )


        df = df.sort_values(
            "Timestamp"
        ).reset_index(
            drop=True
        )


        return df


    except Exception as error:

        print(
            f"Dataset error: {error}"
        )

        return None


# ============================================================
# UPDATE LATEST DATASET ROW WITH REAL LIVE VALUES
#
# This makes sure the newest prediction input uses the
# same custom_cpu_usage value that Grafana shows.
# ============================================================

def update_latest_row_with_live_values(
    df,
    actual_cpu_value,
    memory_value,
    rps_value,
    response_time_value
):

    df = df.copy()


    if len(df) == 0:

        return df


    latest_index = df.index[-1]


    df.loc[
        latest_index,
        "CPU"
    ] = actual_cpu_value


    df.loc[
        latest_index,
        "Memory"
    ] = memory_value


    df.loc[
        latest_index,
        "RequestsPerSecond"
    ] = rps_value


    df.loc[
        latest_index,
        "ResponseTime"
    ] = response_time_value


    return df


# ============================================================
# WORKLOAD FEATURE ENGINEERING
# ============================================================

def create_workload_features(df):

    df = df.copy()


    # --------------------------------------------------------
    # Changes
    # --------------------------------------------------------

    df["UsersChange"] = (
        df["CurrentUsers"]
        .diff()
        .fillna(0)
    )


    df["CPUChange"] = (
        df["CPU"]
        .diff()
        .fillna(0)
    )


    df["RPSChange"] = (
        df["RequestsPerSecond"]
        .diff()
        .fillna(0)
    )


    df["MemoryChange"] = (
        df["Memory"]
        .diff()
        .fillna(0)
    )


    df["ResponseTimeChange"] = (
        df["ResponseTime"]
        .diff()
        .fillna(0)
    )


    # --------------------------------------------------------
    # Growth
    # --------------------------------------------------------

    df["UserGrowthRate"] = (

        df["CurrentUsers"]
        .pct_change()

        .replace(
            [np.inf, -np.inf],
            0
        )

        .fillna(0)

    )


    df["RPSGrowthRate"] = (

        df["RequestsPerSecond"]
        .pct_change()

        .replace(
            [np.inf, -np.inf],
            0
        )

        .fillna(0)

    )


    df["CPUGrowthRate"] = (

        df["CPU"]
        .pct_change()

        .replace(
            [np.inf, -np.inf],
            0
        )

        .fillna(0)

    )


    # --------------------------------------------------------
    # Intensity
    # --------------------------------------------------------

    df["UsersIntensity"] = np.clip(
        df["CurrentUsers"] / 500.0,
        0,
        1
    )


    df["RPSIntensity"] = np.clip(
        df["RequestsPerSecond"] / 100.0,
        0,
        1
    )


    df["ResponseIntensity"] = np.clip(
        df["ResponseTime"] / 5.0,
        0,
        1
    )


    df["CPUIntensity"] = np.clip(
        df["CPU"] / 100.0,
        0,
        1
    )


    # --------------------------------------------------------
    # Positive growth
    # --------------------------------------------------------

    positive_growth = (

        0.40
        *
        np.clip(
            df["UserGrowthRate"],
            0,
            1
        )

        +

        0.40
        *
        np.clip(
            df["RPSGrowthRate"],
            0,
            1
        )

        +

        0.20
        *
        np.clip(
            df["CPUGrowthRate"],
            0,
            1
        )

    )


    # --------------------------------------------------------
    # Workload Score
    # --------------------------------------------------------

    df["WorkloadScore"] = (

        0.35
        *
        df["UsersIntensity"]

        +

        0.35
        *
        df["RPSIntensity"]

        +

        0.10
        *
        df["ResponseIntensity"]

        +

        0.10
        *
        df["CPUIntensity"]

        +

        0.10
        *
        positive_growth

    )


    return df


# ============================================================
# WORKLOAD CLASSIFICATION
#
# IMPORTANT:
# Workload is NOT obtained from predicted CPU.
# It is a separate classifier.
# ============================================================

def classify_workload(row):

    values = [

        row["CurrentUsers"],

        row["CPU"],

        row["Memory"],

        row["RequestsPerSecond"],

        row["ResponseTime"],


        row["UsersChange"],

        row["CPUChange"],

        row["RPSChange"],

        row["MemoryChange"],

        row["ResponseTimeChange"],


        row["UserGrowthRate"],

        row["RPSGrowthRate"],

        row["CPUGrowthRate"],


        row["UsersIntensity"],

        row["RPSIntensity"],

        row["ResponseIntensity"],

        row["CPUIntensity"],


        row["WorkloadScore"]

    ]


    input_df = pd.DataFrame(
        [values],
        columns=WORKLOAD_FEATURES
    )


    scaled = workload_scaler.transform(
        input_df
    )


    result = workload_model.predict(
        scaled
    )


    return str(
        result[0]
    )


# ============================================================
# AI RECOMMENDED PODS
#
# ONLY predicted CPU controls this.
# ============================================================

def get_recommended_pods(
    predicted_cpu_value
):

    predicted_cpu_value = float(
        predicted_cpu_value
    )


    if predicted_cpu_value < 50:

        pods = 1

    elif predicted_cpu_value < 65:

        pods = 2

    elif predicted_cpu_value < 75:

        pods = 3

    elif predicted_cpu_value < 85:

        pods = 4

    else:

        pods = 5


    return int(
        np.clip(
            pods,
            MIN_PODS,
            MAX_PODS
        )
    )


# ============================================================
# AI SCALING SCORE
# ============================================================

def calculate_scaling_score(
    predicted_cpu_value
):

    return float(
        np.clip(
            predicted_cpu_value / 100.0,
            0,
            1
        )
    )


# ============================================================
# KUBERNETES SCALING
# ============================================================

def scale_kubernetes(
    target_pods,
    current_pod_count
):

    target_pods = int(
        np.clip(
            target_pods,
            MIN_PODS,
            MAX_PODS
        )
    )


    if target_pods == current_pod_count:

        return True


    print()
    print("=" * 70)
    print("AI KUBERNETES SCALING")
    print("=" * 70)


    if target_pods > current_pod_count:

        print(
            f"SCALE UP: "
            f"{current_pod_count} -> "
            f"{target_pods}"
        )

    else:

        print(
            f"SCALE DOWN: "
            f"{current_pod_count} -> "
            f"{target_pods}"
        )


    try:

        result = subprocess.run(
            [
                "kubectl",
                "scale",
                "deployment",
                DEPLOYMENT,
                "-n",
                NAMESPACE,
                f"--replicas={target_pods}"
            ],
            capture_output=True,
            text=True,
            timeout=15
        )


        if result.returncode != 0:

            print(
                "Scaling failed:"
            )

            print(
                result.stderr.strip()
            )

            return False


        print(
            "Kubernetes scaling command "
            "executed successfully."
        )

        print("=" * 70)

        return True


    except Exception as error:

        print(
            f"Scaling error: {error}"
        )

        return False


# ============================================================
# RESET
# ============================================================

def reset_metrics():

    actual_cpu.labels(
        namespace=NAMESPACE,
        deployment=DEPLOYMENT
    ).set(0)


    predicted_cpu.labels(
        namespace=NAMESPACE,
        deployment=DEPLOYMENT
    ).set(0)


    current_pods.labels(
        namespace=NAMESPACE,
        deployment=DEPLOYMENT
    ).set(
        get_current_pods()
    )


    recommended_pods.labels(
        namespace=NAMESPACE,
        deployment=DEPLOYMENT
    ).set(1)


    scaling_score.labels(
        namespace=NAMESPACE,
        deployment=DEPLOYMENT
    ).set(0)


    prediction_active.labels(
        namespace=NAMESPACE,
        deployment=DEPLOYMENT
    ).set(0)


    data_age_metric.labels(
        namespace=NAMESPACE,
        deployment=DEPLOYMENT
    ).set(0)


    for state in [
        "LOW",
        "MODERATE",
        "HIGH",
        "VERY_HIGH"
    ]:

        workload_state_metric.labels(
            namespace=NAMESPACE,
            deployment=DEPLOYMENT,
            state=state
        ).set(0)


# ============================================================
# START METRICS SERVER
# ============================================================

start_http_server(
    METRICS_PORT
)


print()
print(
    f"Predictor metrics running on "
    f"http://127.0.0.1:{METRICS_PORT}/metrics"
)
print()


# ============================================================
# MAIN LOOP
# ============================================================

last_scaling_time = 0


while True:

    try:

        # ====================================================
        # REAL PROMETHEUS VALUES
        # ====================================================

        actual_cpu_value = (
            get_actual_cpu()
        )


        if actual_cpu_value is None:

            print(
                "custom_cpu_usage unavailable."
            )

            time.sleep(
                PREDICTION_INTERVAL
            )

            continue


        memory_value = (
            get_actual_memory()
        )


        rps_value = (
            get_actual_rps()
        )


        response_time_value = (
            get_actual_response_time()
        )


        real_current_pods = (
            get_current_pods()
        )


        # ====================================================
        # LOAD DATASET
        # ====================================================

        df = load_dataset()


        if df is None:

            print(
                "Dataset unavailable."
            )

            reset_metrics()

            time.sleep(
                PREDICTION_INTERVAL
            )

            continue


        if len(df) < SEQUENCE_LENGTH:

            print(
                f"Waiting for dataset: "
                f"{len(df)}/{SEQUENCE_LENGTH}"
            )

            time.sleep(
                PREDICTION_INTERVAL
            )

            continue


        # ====================================================
        # USE REAL LIVE CPU IN LATEST PREDICTION INPUT
        # ====================================================

        df = (
            update_latest_row_with_live_values(
                df,
                actual_cpu_value,
                memory_value,
                rps_value,
                response_time_value
            )
        )


        # ====================================================
        # DATA AGE
        # ====================================================

        latest_timestamp = (
            df.iloc[-1]["Timestamp"]
        )


        data_age = (

            pd.Timestamp.now()
            -
            latest_timestamp

        ).total_seconds()


        data_age = max(
            data_age,
            0
        )


        data_age_metric.labels(
            namespace=NAMESPACE,
            deployment=DEPLOYMENT
        ).set(
            data_age
        )


        # ====================================================
        # REAL METRICS → PROMETHEUS
        # ====================================================

        actual_cpu.labels(
            namespace=NAMESPACE,
            deployment=DEPLOYMENT
        ).set(
            actual_cpu_value
        )


        current_pods.labels(
            namespace=NAMESPACE,
            deployment=DEPLOYMENT
        ).set(
            real_current_pods
        )


        # ====================================================
        # WORKLOAD FEATURES
        # ====================================================

        df = create_workload_features(
            df
        )


        # ====================================================
        # SEPARATE WORKLOAD CLASSIFIER
        # ====================================================

        workload = classify_workload(
            df.iloc[-1]
        )


        # ====================================================
        # LSTM INPUT
        # ====================================================

        sequence = (

            df[
                LSTM_FEATURES
            ]

            .tail(
                SEQUENCE_LENGTH
            )

            .values

        )


        scaled_sequence = (
            lstm_scaler.transform(
                sequence
            )
        )


        input_data = np.expand_dims(
            scaled_sequence,
            axis=0
        )


        # ====================================================
        # LSTM PREDICTION
        # ====================================================

        prediction_scaled = (

            lstm_model.predict(
                input_data,
                verbose=0
            )[0][0]

        )


        # ====================================================
        # INVERSE CPU
        # ====================================================

        dummy = np.zeros(
            (
                1,
                len(LSTM_FEATURES)
            )
        )


        dummy[0, 1] = (
            prediction_scaled
        )


        predicted_cpu_value = (

            lstm_scaler
            .inverse_transform(
                dummy
            )[0][1]

        )


        predicted_cpu_value = float(
            np.clip(
                predicted_cpu_value,
                0,
                100
            )
        )


        # ====================================================
        # AI RECOMMENDED PODS
        #
        # ONLY PREDICTED CPU
        # ====================================================

        ai_pods = (
            get_recommended_pods(
                predicted_cpu_value
            )
        )


        # ====================================================
        # SCALING SCORE
        # ====================================================

        score = (
            calculate_scaling_score(
                predicted_cpu_value
            )
        )


        # ====================================================
        # PUBLISH LSTM PREDICTION
        # ====================================================

        predicted_cpu.labels(
            namespace=NAMESPACE,
            deployment=DEPLOYMENT
        ).set(
            predicted_cpu_value
        )


        recommended_pods.labels(
            namespace=NAMESPACE,
            deployment=DEPLOYMENT
        ).set(
            ai_pods
        )


        scaling_score.labels(
            namespace=NAMESPACE,
            deployment=DEPLOYMENT
        ).set(
            score
        )


        prediction_active.labels(
            namespace=NAMESPACE,
            deployment=DEPLOYMENT
        ).set(1)


        # ====================================================
        # PUBLISH WORKLOAD SEPARATELY
        # ====================================================

        for state in [
            "LOW",
            "MODERATE",
            "HIGH",
            "VERY_HIGH"
        ]:

            workload_state_metric.labels(
                namespace=NAMESPACE,
                deployment=DEPLOYMENT,
                state=state
            ).set(
                1
                if workload == state
                else 0
            )


        # ====================================================
        # AUTOMATIC AI SCALING
        # ====================================================

        now = time.time()


        cooldown_finished = (

            now
            -
            last_scaling_time

        ) >= SCALING_COOLDOWN


        if (

            ai_pods != real_current_pods

            and

            cooldown_finished

        ):

            success = scale_kubernetes(
                ai_pods,
                real_current_pods
            )


            if success:

                last_scaling_time = now


        # ====================================================
        # DISPLAY
        # ====================================================

        print()
        print("=" * 70)
        print("             MGRK AI LIVE ANALYSIS")
        print("=" * 70)

        print()
        print("REAL SYSTEM")
        print("-" * 40)

        print(
            f"Actual CPU          : "
            f"{actual_cpu_value:.2f}%"
        )

        print(
            f"Current Pods        : "
            f"{real_current_pods}"
        )

        print()
        print("LSTM PREDICTION")
        print("-" * 40)

        print(
            f"Predicted CPU       : "
            f"{predicted_cpu_value:.2f}%"
        )

        print()
        print("AI SCALING")
        print("-" * 40)

        print(
            f"AI Recommended Pods : "
            f"{ai_pods}"
        )

        print(
            f"Scaling Score       : "
            f"{score:.3f}"
        )

        print()
        print("WORKLOAD ANALYSIS")
        print("-" * 40)

        print(
            f"Workload            : "
            f"{workload}"
        )

        print()
        print("LIVE INPUTS")
        print("-" * 40)

        print(
            f"Memory              : "
            f"{memory_value:.2f}%"
        )

        print(
            f"Requests/sec        : "
            f"{rps_value:.2f}"
        )

        print(
            f"Response Time       : "
            f"{response_time_value:.4f}s"
        )

        print("=" * 70)


        time.sleep(
            PREDICTION_INTERVAL
        )


    except KeyboardInterrupt:

        print()
        print(
            "Predictor stopped."
        )

        break


    except Exception as error:

        print()
        print(
            f"PREDICTOR ERROR: {error}"
        )

        time.sleep(
            PREDICTION_INTERVAL
        )