import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


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
# V2 OUTPUT FILES
# ============================================================

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "lstm_cpu_model_v2.keras"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "cpu_scaler_v2.pkl"
)

PREDICTION_FILE = os.path.join(
    MODEL_DIR,
    "cpu_predictions_v2.csv"
)

PREDICTION_GRAPH = os.path.join(
    GRAPH_DIR,
    "lstm_prediction_v2.png"
)

LOSS_GRAPH = os.path.join(
    GRAPH_DIR,
    "lstm_training_loss_v2.png"
)


# ============================================================
# CONFIGURATION
# ============================================================

SEQUENCE_LENGTH = 10
TRAIN_RATIO = 0.80
EPOCHS = 100
BATCH_SIZE = 16
PATIENCE = 10


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

print("=" * 70)
print("          MGRK UNIVERSITY - LSTM MODEL V2 TRAINING")
print("=" * 70)

print("\nDataset:")
print(DATASET_PATH)

print("\nV2 Model:")
print(MODEL_PATH)

print("\nV2 Scaler:")
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
# CHECK COLUMNS
# ============================================================

required_columns = [
    "Timestamp"
] + FEATURES

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
# DATA CLEANING
# ============================================================

print("\nCleaning dataset...")

# ------------------------------------------------------------
# IMPORTANT:
# The dataset contains mixed timestamp formats.
#
# Example:
# 2026-06-30 11:47:54.888635
# 2026-10-06 19:51:14
#
# format="mixed" allows pandas to correctly parse both.
# ------------------------------------------------------------

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    format="mixed",
    errors="coerce"
)

# Sort chronologically
df = df.sort_values(
    "Timestamp"
)

# Keep required columns
df = df[
    required_columns
]

# Convert numerical columns
for column in FEATURES:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )


# ------------------------------------------------------------
# REMOVE INVALID ROWS
# ------------------------------------------------------------

before_cleaning = len(df)

print(
    f"\nRows before cleaning : {before_cleaning}"
)

# Show missing values before removal
print("\nMissing values before cleaning:")

print(
    df.isna().sum()
)

df = df.dropna()

after_cleaning = len(df)

print(
    f"\nRows after cleaning  : {after_cleaning}"
)

# Reset index
df = df.reset_index(
    drop=True
)

print(
    f"Rows after cleaning: {len(df)}"
)


# ============================================================
# CHECK DATA SIZE
# ============================================================

minimum_rows = (
    SEQUENCE_LENGTH + 50
)

if len(df) < minimum_rows:

    raise ValueError(
        f"Not enough data.\n"
        f"Need at least {minimum_rows} rows."
    )


# ============================================================
# DISPLAY DATASET INFORMATION
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
# PREPARE DATA
# ============================================================

data = df[
    FEATURES
].values.astype(
    np.float32
)

print("\nFeatures used:")

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


print("\n" + "=" * 70)
print("DATASET SPLIT")
print("=" * 70)

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
    "\nThe newest 20% of the time series "
    "is reserved for testing."
)


# ============================================================
# NORMALIZATION
# ============================================================

print("\nCreating MinMaxScaler...")

scaler = MinMaxScaler()

# Fit scaler ONLY on training data
train_scaled = scaler.fit_transform(
    train_data
)

# Transform test using same scaler
test_scaled = scaler.transform(
    test_data
)


# ============================================================
# SAVE V2 SCALER
# ============================================================

joblib.dump(
    scaler,
    SCALER_PATH
)

print(
    "\nV2 scaler saved:"
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

# Include the final 10 training observations
# so the first test sequence has historical context.

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

print("\n" + "=" * 70)
print("SEQUENCE INFORMATION")
print("=" * 70)

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
# BUILD V2 MODEL
# ============================================================

print("\nBuilding LSTM Model V2...")

model = Sequential(
    [

        LSTM(
            64,
            return_sequences=True,
            input_shape=(
                SEQUENCE_LENGTH,
                len(FEATURES)
            )
        ),

        Dropout(
            0.2
        ),

        LSTM(
            32,
            return_sequences=False
        ),

        Dropout(
            0.2
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

print("\nModel V2 summary:\n")

model.summary()


# ============================================================
# EARLY STOPPING
# ============================================================

early_stopping = EarlyStopping(

    monitor="val_loss",

    patience=PATIENCE,

    restore_best_weights=True

)


# ============================================================
# TRAIN V2
# ============================================================

print("\n" + "=" * 70)
print("STARTING MODEL V2 TRAINING")
print("=" * 70)

history = model.fit(

    X_train,

    y_train,

    epochs=EPOCHS,

    batch_size=BATCH_SIZE,

    validation_split=0.20,

    shuffle=False,

    callbacks=[
        early_stopping
    ],

    verbose=1

)


# ============================================================
# SAVE V2 MODEL
# ============================================================

model.save(
    MODEL_PATH
)

print("\n" + "=" * 70)

print(
    "MODEL V2 SAVED"
)

print("=" * 70)

print(
    MODEL_PATH
)


# ============================================================
# PREDICTION ON UNSEEN TEST DATA
# ============================================================

print("\nGenerating V2 predictions...")

predicted_scaled = model.predict(
    X_test,
    verbose=0
)


# ============================================================
# CONVERT BACK TO CPU %
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

# CPU = feature index 1

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


# Keep predictions within realistic CPU range

predicted_original = np.clip(
    predicted_original,
    0,
    100
)


# ============================================================
# MODEL EVALUATION
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


# ============================================================
# ADDITIONAL ERROR INFORMATION
# ============================================================

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
# PRINT PERFORMANCE
# ============================================================

print("\n" + "=" * 70)
print("             MODEL V2 PERFORMANCE")
print("=" * 70)

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

print("=" * 70)


# ============================================================
# SHOW SAMPLE PREDICTIONS
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

print("\nRecent V2 predictions:")

print(
    results.tail(20).to_string(
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
    "\nV2 predictions saved:"
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
    label="V2 Predicted CPU"
)

plt.title(
    "LSTM Model V2 - Actual vs Predicted CPU"
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
    "\nV2 prediction graph saved:"
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
    "LSTM Model V2 Training and Validation Loss"
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
    "\nV2 training graph saved:"
)

print(
    LOSS_GRAPH
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("              MODEL V2 TRAINING COMPLETE")
print("=" * 70)

print("\nModel V2:")
print(
    MODEL_PATH
)

print("\nScaler V2:")
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
    "Model V1 has NOT been modified."
)

print(
    "Model V2 must be compared with Model V1 "
    "before replacing the production model."
)

print("=" * 70)