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
# V3 OUTPUT FILES
# ============================================================

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "lstm_cpu_model_v3.keras"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "cpu_scaler_v3.pkl"
)

PREDICTION_FILE = os.path.join(
    MODEL_DIR,
    "cpu_predictions_v3.csv"
)

PREDICTION_GRAPH = os.path.join(
    GRAPH_DIR,
    "lstm_prediction_v3.png"
)

LOSS_GRAPH = os.path.join(
    GRAPH_DIR,
    "lstm_training_loss_v3.png"
)


# ============================================================
# CONFIGURATION
# ============================================================

SEQUENCE_LENGTH = 10

TRAIN_RATIO = 0.80

EPOCHS = 120

BATCH_SIZE = 16

PATIENCE = 15


# ============================================================
# ORIGINAL FEATURES
# ============================================================

BASE_FEATURES = [
    "CurrentUsers",
    "CPU",
    "Memory",
    "RequestsPerSecond",
    "ResponseTime"
]


# ============================================================
# V3 FEATURES
# ============================================================

# V3 adds workload trend information.
#
# CPU itself is still included because CPU history
# is important for time-series prediction.

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
    "ResponseTimeChange"
]

TARGET = "CPU"


# ============================================================
# CREATE DIRECTORIES
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

print("=" * 75)
print("          MGRK UNIVERSITY - LSTM MODEL V3 TRAINING")
print("=" * 75)

print("\nDataset:")
print(DATASET_PATH)

print("\nV3 Model:")
print(MODEL_PATH)

print("\nV3 Scaler:")
print(SCALER_PATH)


# ============================================================
# CHECK DATASET
# ============================================================

if not os.path.exists(DATASET_PATH):

    raise FileNotFoundError(
        f"\nDataset not found:\n{DATASET_PATH}"
    )


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(
    DATASET_PATH
)

print(
    f"Original dataset rows: {len(df)}"
)


# ============================================================
# CHECK REQUIRED BASE COLUMNS
# ============================================================

required_base_columns = [
    "Timestamp"
] + BASE_FEATURES

missing_columns = [
    column
    for column in required_base_columns
    if column not in df.columns
]

if missing_columns:

    raise ValueError(
        "Missing columns:\n"
        + str(missing_columns)
    )


# ============================================================
# TIMESTAMP CLEANING
# ============================================================

print("\nCleaning timestamps...")

# IMPORTANT:
# Dataset contains mixed timestamp formats.
#
# Example:
# 2026-06-30 11:47:54.888635
# 2026-10-06 19:51:14
#
# format="mixed" handles both formats.

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    format="mixed",
    errors="coerce"
)


# ============================================================
# SORT CHRONOLOGICALLY
# ============================================================

df = df.sort_values(
    "Timestamp"
)


# ============================================================
# NUMERIC CONVERSION
# ============================================================

print("\nConverting numerical columns...")

for column in BASE_FEATURES:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ============================================================
# REMOVE INVALID BASE DATA
# ============================================================

before_cleaning = len(df)

df = df.dropna(
    subset=required_base_columns
)

after_cleaning = len(df)

print(
    f"Rows before cleaning : {before_cleaning}"
)

print(
    f"Rows after cleaning  : {after_cleaning}"
)


# ============================================================
# RESET INDEX
# ============================================================

df = df.reset_index(
    drop=True
)


# ============================================================
# DATASET INFORMATION
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

print("\nCPU statistics:")

print(
    df["CPU"].describe()
)


# ============================================================
# CREATE WORKLOAD TREND FEATURES
# ============================================================

print("\nCreating V3 workload features...")


# ------------------------------------------------------------
# User change
# ------------------------------------------------------------

df["UsersChange"] = (
    df["CurrentUsers"].diff()
)


# ------------------------------------------------------------
# CPU change
# ------------------------------------------------------------

df["CPUChange"] = (
    df["CPU"].diff()
)


# ------------------------------------------------------------
# Requests-per-second change
# ------------------------------------------------------------

df["RPSChange"] = (
    df["RequestsPerSecond"].diff()
)


# ------------------------------------------------------------
# Memory change
# ------------------------------------------------------------

df["MemoryChange"] = (
    df["Memory"].diff()
)


# ------------------------------------------------------------
# Response-time change
# ------------------------------------------------------------

df["ResponseTimeChange"] = (
    df["ResponseTime"].diff()
)


# ------------------------------------------------------------
# First row has no previous observation.
# Fill trend features with zero.
# ------------------------------------------------------------

TREND_FEATURES = [
    "UsersChange",
    "CPUChange",
    "RPSChange",
    "MemoryChange",
    "ResponseTimeChange"
]

df[TREND_FEATURES] = (
    df[TREND_FEATURES]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
)


# ============================================================
# FINAL DATA CHECK
# ============================================================

print("\nV3 feature check:")

print(
    df[FEATURES].isna().sum()
)

print("\nFeatures used by V3:")

for feature in FEATURES:

    print(
        " -",
        feature
    )

print(
    "\nTarget:",
    TARGET
)


# ============================================================
# CHECK DATA SIZE
# ============================================================

minimum_rows = (
    SEQUENCE_LENGTH + 100
)

if len(df) < minimum_rows:

    raise ValueError(
        f"Not enough data.\n"
        f"Need at least {minimum_rows} rows."
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
# CHRONOLOGICAL TRAIN / TEST SPLIT
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


print("\n" + "=" * 75)
print("DATASET SPLIT")
print("=" * 75)

print(
    f"Total rows   : {len(data)}"
)

print(
    f"Training rows: {len(train_data)}"
)

print(
    f"Testing rows : {len(test_data)}"
)

print(
    "\nNewest 20% is reserved for testing."
)


# ============================================================
# NORMALIZATION
# ============================================================

print("\nCreating V3 MinMaxScaler...")

scaler = MinMaxScaler()

# Fit ONLY on training data.

train_scaled = scaler.fit_transform(
    train_data
)

# Transform test data using the
# training scaler.

test_scaled = scaler.transform(
    test_data
)


# ============================================================
# SAVE V3 SCALER
# ============================================================

joblib.dump(
    scaler,
    SCALER_PATH
)

print(
    "\nV3 scaler saved:"
)

print(
    SCALER_PATH
)


# ============================================================
# CREATE SEQUENCES
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

        # CPU is feature index 1
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

print("\n" + "=" * 75)
print("SEQUENCE INFORMATION")
print("=" * 75)

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
# BUILD V3 MODEL
# ============================================================

print("\nBuilding LSTM Model V3...")

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
            return_sequences=True
        ),

        Dropout(
            0.20
        ),

        LSTM(
            24,
            return_sequences=False
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

print("\nV3 Model summary:\n")

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

    patience=5,

    min_lr=0.00001,

    verbose=1

)


# ============================================================
# TRAIN V3
# ============================================================

print("\n" + "=" * 75)
print("STARTING MODEL V3 TRAINING")
print("=" * 75)

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
# SAVE V3 MODEL
# ============================================================

model.save(
    MODEL_PATH
)

print("\n" + "=" * 75)

print(
    "MODEL V3 SAVED"
)

print("=" * 75)

print(
    MODEL_PATH
)


# ============================================================
# PREDICTION
# ============================================================

print("\nGenerating V3 predictions...")

predicted_scaled = model.predict(
    X_test,
    verbose=0
)


# ============================================================
# INVERSE TRANSFORM CPU
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

# CPU = index 1

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
# CLIP CPU
# ============================================================

predicted_original = np.clip(
    predicted_original,
    0,
    100
)


# ============================================================
# EVALUATION
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

absolute_errors = np.abs(
    actual_original -
    predicted_original
)

mean_error = np.mean(
    absolute_errors
)

max_error = np.max(
    absolute_errors
)

within_10_percent = (
    np.mean(
        absolute_errors <= 10
    ) * 100
)

within_20_percent = (
    np.mean(
        absolute_errors <= 20
    ) * 100
)


# ============================================================
# V3 PERFORMANCE
# ============================================================

print("\n" + "=" * 75)
print("                 MODEL V3 PERFORMANCE")
print("=" * 75)

print(
    f"MAE                    : {mae:.2f}%"
)

print(
    f"RMSE                   : {rmse:.2f}%"
)

print(
    f"Mean Absolute Error    : {mean_error:.2f}%"
)

print(
    f"Maximum Error          : {max_error:.2f}%"
)

print(
    f"Predictions <= 10% err : "
    f"{within_10_percent:.2f}%"
)

print(
    f"Predictions <= 20% err : "
    f"{within_20_percent:.2f}%"
)

print("=" * 75)


# ============================================================
# SAMPLE PREDICTIONS
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

print("\nRecent V3 predictions:")

print(
    results.tail(30).to_string(
        index=False
    )
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

results.to_csv(
    PREDICTION_FILE,
    index=False
)

print(
    "\nV3 predictions saved:"
)

print(
    PREDICTION_FILE
)


# ============================================================
# ACTUAL VS PREDICTED GRAPH
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
    label="V3 Predicted CPU"
)

plt.title(
    "LSTM Model V3 - Actual vs Predicted CPU"
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
    "\nV3 prediction graph saved:"
)

print(
    PREDICTION_GRAPH
)


# ============================================================
# TRAINING LOSS GRAPH
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
    "LSTM Model V3 Training and Validation Loss"
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
    "\nV3 training graph saved:"
)

print(
    LOSS_GRAPH
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("                MODEL V3 TRAINING COMPLETE")
print("=" * 75)

print("\nModel V3:")
print(
    MODEL_PATH
)

print("\nScaler V3:")
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
    "V1 and V2 have NOT been modified."
)

print(
    "V3 must be compared against V1 and V2 "
    "before becoming the production model."
)

print("=" * 75)