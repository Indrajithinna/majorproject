import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

DATASET_PATH = os.path.join(
    PROJECT_ROOT,
    "prediction",
    "time_series_dataset.csv"
)

MODEL_DIR = os.path.join(
    PROJECT_ROOT,
    "ml",
    "models"
)

GRAPH_DIR = os.path.join(
    PROJECT_ROOT,
    "ml",
    "graphs"
)


# ============================================================
# V4 OUTPUT FILES
# ============================================================

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "lstm_cpu_model_v4.keras"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "cpu_scaler_v4.pkl"
)

PREDICTION_FILE = os.path.join(
    MODEL_DIR,
    "cpu_predictions_v4.csv"
)

PREDICTION_GRAPH = os.path.join(
    GRAPH_DIR,
    "lstm_prediction_v4.png"
)

LOSS_GRAPH = os.path.join(
    GRAPH_DIR,
    "lstm_training_loss_v4.png"
)


# ============================================================
# CONFIGURATION
# ============================================================

SEQUENCE_LENGTH = 10

TRAIN_RATIO = 0.80

EPOCHS = 150

BATCH_SIZE = 16

PATIENCE = 20


# ============================================================
# BASE FEATURES
# ============================================================

BASE_FEATURES = [
    "CurrentUsers",
    "CPU",
    "Memory",
    "RequestsPerSecond",
    "ResponseTime"
]


# ============================================================
# V4 FEATURES
# ============================================================

FEATURES = [
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

    "WorkloadScore"
]

TARGET = "CPU"


# ============================================================
# DIRECTORIES
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    GRAPH_DIR,
    exist_ok=True
)


# ============================================================
# HEADER
# ============================================================

print("=" * 80)
print("          MGRK UNIVERSITY - LSTM MODEL V4 TRAINING")
print("=" * 80)

print("\nDataset:")
print(DATASET_PATH)

print("\nV4 Model:")
print(MODEL_PATH)

print("\nV4 Scaler:")
print(SCALER_PATH)


# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.exists(DATASET_PATH):

    raise FileNotFoundError(
        f"\nDataset not found:\n{DATASET_PATH}"
    )


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(
    DATASET_PATH
)

print(
    f"Original dataset rows: {len(df)}"
)


# ============================================================
# CHECK COLUMNS
# ============================================================

required_columns = [
    "Timestamp"
] + BASE_FEATURES

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        "Missing columns:\n"
        + str(missing_columns)
    )


# ============================================================
# TIMESTAMP
# ============================================================

print("\nParsing timestamps...")

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    format="mixed",
    errors="coerce"
)


# ============================================================
# SORT
# ============================================================

df = df.sort_values(
    "Timestamp"
)


# ============================================================
# NUMERIC CONVERSION
# ============================================================

for column in BASE_FEATURES:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# CLEAN
# ============================================================

before_cleaning = len(df)

df = df.dropna(
    subset=required_columns
)

after_cleaning = len(df)

print(
    f"Rows before cleaning : {before_cleaning}"
)

print(
    f"Rows after cleaning  : {after_cleaning}"
)

df = df.reset_index(
    drop=True
)


# ============================================================
# DATA RANGE
# ============================================================

print("\nDataset time range:")

print(
    "Start:",
    df["Timestamp"].iloc[0]
)

print(
    "End  :",
    df["Timestamp"].iloc[-1]
)


# ============================================================
# CREATE TREND FEATURES
# ============================================================

print("\nCreating V4 workload features...")


# ------------------------------------------------------------
# Basic changes
# ------------------------------------------------------------

df["UsersChange"] = (
    df["CurrentUsers"].diff()
)

df["CPUChange"] = (
    df["CPU"].diff()
)

df["RPSChange"] = (
    df["RequestsPerSecond"].diff()
)

df["MemoryChange"] = (
    df["Memory"].diff()
)

df["ResponseTimeChange"] = (
    df["ResponseTime"].diff()
)


# ============================================================
# GROWTH RATES
# ============================================================

# Small epsilon prevents division by zero.

EPSILON = 0.001


df["UserGrowthRate"] = (
    df["UsersChange"] /
    (df["CurrentUsers"].shift(1).abs() + EPSILON)
)

df["RPSGrowthRate"] = (
    df["RPSChange"] /
    (df["RequestsPerSecond"].shift(1).abs() + EPSILON)
)

df["CPUGrowthRate"] = (
    df["CPUChange"] /
    (df["CPU"].shift(1).abs() + EPSILON)
)


# ============================================================
# WORKLOAD SCORE
# ============================================================

# Normalize workload components into a single
# workload-intensity signal.

users_norm = (
    df["CurrentUsers"] /
    max(df["CurrentUsers"].max(), 1)
)

rps_norm = (
    df["RequestsPerSecond"] /
    max(df["RequestsPerSecond"].max(), 1)
)

response_norm = (
    df["ResponseTime"] /
    max(df["ResponseTime"].max(), 0.001)
)

memory_norm = (
    df["Memory"] /
    max(df["Memory"].max(), 0.001)
)

df["WorkloadScore"] = (
    0.40 * users_norm +
    0.40 * rps_norm +
    0.10 * response_norm +
    0.10 * memory_norm
)


# ============================================================
# CLEAN ENGINEERED FEATURES
# ============================================================

TREND_FEATURES = [
    "UsersChange",
    "CPUChange",
    "RPSChange",
    "MemoryChange",
    "ResponseTimeChange",
    "UserGrowthRate",
    "RPSGrowthRate",
    "CPUGrowthRate",
    "WorkloadScore"
]

df[TREND_FEATURES] = (
    df[TREND_FEATURES]
    .replace(
        [np.inf, -np.inf],
        np.nan
    )
    .fillna(0)
)


# ============================================================
# FEATURE CHECK
# ============================================================

print("\nV4 feature check:")

print(
    df[FEATURES].isna().sum()
)

print("\nFeatures used:")

for feature in FEATURES:

    print(
        " -",
        feature
    )

print(
    "\nTotal features:",
    len(FEATURES)
)

print(
    "\nTarget:",
    TARGET
)


# ============================================================
# CPU DISTRIBUTION
# ============================================================

print("\nCPU distribution:")

print(
    df["CPU"].describe()
)


# ============================================================
# PREPARE DATA
# ============================================================

data = df[
    FEATURES
].values.astype(
    np.float32
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

split_index = int(
    len(data) * TRAIN_RATIO
)

train_data = data[
    :split_index
]

test_data = data[
    split_index:
]


print("\n" + "=" * 80)
print("DATASET SPLIT")
print("=" * 80)

print(
    f"Total rows   : {len(data)}"
)

print(
    f"Training rows: {len(train_data)}"
)

print(
    f"Testing rows : {len(test_data)}"
)


# ============================================================
# SCALER
# ============================================================

print("\nCreating V4 scaler...")

scaler = MinMaxScaler()

train_scaled = scaler.fit_transform(
    train_data
)

test_scaled = scaler.transform(
    test_data
)


# ============================================================
# SAVE SCALER
# ============================================================

joblib.dump(
    scaler,
    SCALER_PATH
)

print(
    "\nV4 scaler saved:"
)

print(
    SCALER_PATH
)


# ============================================================
# SEQUENCE CREATION
# ============================================================

def create_sequences(
    data,
    sequence_length
):

    X = []
    y = []

    for i in range(
        sequence_length,
        len(data)
    ):

        X.append(
            data[
                i - sequence_length:i
            ]
        )

        # CPU is feature index 1.

        y.append(
            data[i, 1]
        )

    return (
        np.array(X),
        np.array(y)
    )


# ============================================================
# TRAIN SEQUENCES
# ============================================================

X_train, y_train = create_sequences(
    train_scaled,
    SEQUENCE_LENGTH
)


# ============================================================
# TEST SEQUENCES
# ============================================================

combined_data = np.concatenate(
    [
        train_data[
            -SEQUENCE_LENGTH:
        ],
        test_data
    ],
    axis=0
)

combined_scaled = scaler.transform(
    combined_data
)

X_test, y_test = create_sequences(
    combined_scaled,
    SEQUENCE_LENGTH
)


# ============================================================
# SEQUENCE INFORMATION
# ============================================================

print("\n" + "=" * 80)
print("SEQUENCE INFORMATION")
print("=" * 80)

print(
    "X_train:",
    X_train.shape
)

print(
    "y_train:",
    y_train.shape
)

print(
    "X_test :",
    X_test.shape
)

print(
    "y_test :",
    y_test.shape
)


# ============================================================
# BUILD V4 MODEL
# ============================================================

print("\nBuilding V4 model...")

model = Sequential(
    [

        Input(
            shape=(
                SEQUENCE_LENGTH,
                len(FEATURES)
            )
        ),

        LSTM(
            96,
            return_sequences=True
        ),

        Dropout(
            0.20
        ),

        LSTM(
            48,
            return_sequences=False
        ),

        Dropout(
            0.20
        ),

        Dense(
            32,
            activation="relu"
        ),

        Dense(
            16,
            activation="relu"
        ),

        Dense(
            1
        )
    ]
)


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer="adam",
    loss="mean_squared_error"
)

print("\nV4 Model summary:\n")

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

early_stopping = EarlyStopping(
    monitor="val_loss",
    patience=PATIENCE,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=6,
    min_lr=0.00001,
    verbose=1
)


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 80)
print("STARTING MODEL V4 TRAINING")
print("=" * 80)

history = model.fit(

    X_train,

    y_train,

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    validation_split=0.20,

    shuffle=False,

    callbacks=[
        early_stopping,
        reduce_lr
    ],

    verbose=1
)


# ============================================================
# SAVE MODEL
# ============================================================

model.save(
    MODEL_PATH
)

print("\n" + "=" * 80)
print("MODEL V4 SAVED")
print("=" * 80)

print(
    MODEL_PATH
)


# ============================================================
# PREDICT
# ============================================================

print("\nGenerating V4 predictions...")

predicted_scaled = model.predict(
    X_test,
    verbose=0
)


# ============================================================
# INVERSE CPU
# ============================================================

dummy_pred = np.zeros(
    (
        len(predicted_scaled),
        len(FEATURES)
    )
)

dummy_actual = np.zeros(
    (
        len(y_test),
        len(FEATURES)
    )
)

dummy_pred[:, 1] = (
    predicted_scaled[:, 0]
)

dummy_actual[:, 1] = (
    y_test
)

predicted_original = (
    scaler.inverse_transform(
        dummy_pred
    )[:, 1]
)

actual_original = (
    scaler.inverse_transform(
        dummy_actual
    )[:, 1]
)


# ============================================================
# CLIP
# ============================================================

predicted_original = np.clip(
    predicted_original,
    0,
    100
)


# ============================================================
# ERROR
# ============================================================

absolute_errors = np.abs(
    actual_original -
    predicted_original
)


# ============================================================
# METRICS
# ============================================================

mae = mean_absolute_error(
    actual_original,
    predicted_original
)

rmse = np.sqrt(
    mean_squared_error(
        actual_original,
        predicted_original
    )
)

max_error = np.max(
    absolute_errors
)

within_10 = (
    np.mean(
        absolute_errors <= 10
    ) * 100
)

within_20 = (
    np.mean(
        absolute_errors <= 20
    ) * 100
)


# ============================================================
# PERFORMANCE
# ============================================================

print("\n" + "=" * 80)
print("                    MODEL V4 PERFORMANCE")
print("=" * 80)

print(
    f"MAE                    : {mae:.2f}%"
)

print(
    f"RMSE                   : {rmse:.2f}%"
)

print(
    f"Maximum Error          : {max_error:.2f}%"
)

print(
    f"Predictions <= 10% err : {within_10:.2f}%"
)

print(
    f"Predictions <= 20% err : {within_20:.2f}%"
)

print("=" * 80)


# ============================================================
# LOW / MEDIUM / HIGH CPU ANALYSIS
# ============================================================

print("\nCPU RANGE PERFORMANCE:")


low_mask = actual_original < 40

medium_mask = (
    (actual_original >= 40) &
    (actual_original < 70)
)

high_mask = actual_original >= 70


def range_mae(mask):

    if np.sum(mask) == 0:
        return 0.0

    return mean_absolute_error(
        actual_original[mask],
        predicted_original[mask]
    )


print(
    f"Low CPU (<40%) MAE     : "
    f"{range_mae(low_mask):.2f}%"
)

print(
    f"Medium CPU (40-70%) MAE: "
    f"{range_mae(medium_mask):.2f}%"
)

print(
    f"High CPU (>=70%) MAE   : "
    f"{range_mae(high_mask):.2f}%"
)


# ============================================================
# HIGH-LOAD DETECTION
# ============================================================

high_actual = actual_original >= 70

high_predicted = predicted_original >= 70

if np.sum(high_actual) > 0:

    high_detection = (
        np.sum(
            high_actual &
            high_predicted
        )
        /
        np.sum(high_actual)
    ) * 100

else:

    high_detection = 0


print(
    f"\nHigh CPU detection rate: "
    f"{high_detection:.2f}%"
)


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results = pd.DataFrame(
    {
        "Actual_CPU":
            actual_original,

        "Predicted_CPU":
            predicted_original,

        "Absolute_Error":
            absolute_errors
    }
)


# ============================================================
# RECENT PREDICTIONS
# ============================================================

print("\nRecent V4 predictions:")

print(
    results.tail(30).to_string(
        index=False
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

results.to_csv(
    PREDICTION_FILE,
    index=False
)

print(
    "\nV4 predictions saved:"
)

print(
    PREDICTION_FILE
)


# ============================================================
# PREDICTION GRAPH
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    actual_original,
    label="Actual CPU"
)

plt.plot(
    predicted_original,
    label="V4 Predicted CPU"
)

plt.title(
    "LSTM Model V4 - Actual vs Predicted CPU"
)

plt.xlabel(
    "Time Sequence"
)

plt.ylabel(
    "CPU Usage (%)"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(
    PREDICTION_GRAPH,
    dpi=150
)

plt.close()

print(
    "\nV4 prediction graph saved:"
)

print(
    PREDICTION_GRAPH
)


# ============================================================
# LOSS GRAPH
# ============================================================

plt.figure(
    figsize=(12, 6)
)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "LSTM Model V4 Training and Validation Loss"
)

plt.xlabel(
    "Epoch"
)

plt.ylabel(
    "Loss"
)

plt.legend()

plt.grid(
    True
)

plt.tight_layout()

plt.savefig(
    LOSS_GRAPH,
    dpi=150
)

plt.close()

print(
    "\nV4 training graph saved:"
)

print(
    LOSS_GRAPH
)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 80)
print("                 MODEL V4 TRAINING COMPLETE")
print("=" * 80)

print("\nModel:")
print(
    MODEL_PATH
)

print("\nScaler:")
print(
    SCALER_PATH
)

print("\nPredictions:")
print(
    PREDICTION_FILE
)

print("\nGraphs:")
print(
    PREDICTION_GRAPH
)

print(
    LOSS_GRAPH
)

print("\nIMPORTANT:")
print(
    "V1, V2 and V3 have NOT been modified."
)

print(
    "V4 must be compared against previous models "
    "before production deployment."
)

print("=" * 80)