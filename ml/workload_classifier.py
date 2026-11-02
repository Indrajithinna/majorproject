import os
import joblib
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score
)


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

MODEL_DIR = os.path.join(
    BASE_DIR,
    "ml",
    "models"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "ml",
    "outputs"
)

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "workload_classifier.pkl"
)

SCALER_PATH = os.path.join(
    MODEL_DIR,
    "workload_scaler.pkl"
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "workload_classified_dataset.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("\n==============================================")
print("   WORKLOAD CLASSIFIER - V3")
print("==============================================\n")

df = pd.read_csv(
    DATASET_PATH
)

print(
    f"Total rows: {len(df)}"
)


# ============================================================
# CLEAN DATA
# ============================================================

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    format="mixed",
    errors="coerce"
)

numeric_columns = [
    "CurrentUsers",
    "CPU",
    "Memory",
    "RequestsPerSecond",
    "ResponseTime"
]

for column in numeric_columns:

    df[column] = pd.to_numeric(
        df[column],
        errors="coerce"
    )

df = df.dropna(
    subset=[
        "Timestamp",
        *numeric_columns
    ]
).copy()

df = df.sort_values(
    "Timestamp"
).reset_index(drop=True)


# ============================================================
# REMOVE OBVIOUS BAD RPS VALUES
# ============================================================

print("\nChecking RPS values...")

rps_q99 = df[
    "RequestsPerSecond"
].quantile(0.99)

print(
    f"RPS 99th percentile: {rps_q99:.2f}"
)

# Anything above 5x the 99th percentile
# is considered an abnormal measurement.

rps_limit = max(
    rps_q99 * 5,
    500
)

bad_rps = (
    df["RequestsPerSecond"]
    > rps_limit
)

print(
    f"RPS limit: {rps_limit:.2f}"
)

print(
    f"Abnormal RPS rows: "
    f"{bad_rps.sum()}"
)

df.loc[
    bad_rps,
    "RequestsPerSecond"
] = np.nan

df["RequestsPerSecond"] = (
    df["RequestsPerSecond"]
    .interpolate()
    .bfill()
    .ffill()
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

print(
    "\nCreating workload features..."
)

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


# ============================================================
# GROWTH FEATURES
# ============================================================

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


# ============================================================
# NORMALIZED WORKLOAD SIGNALS
# ============================================================

users_max = max(
    df["CurrentUsers"].quantile(0.95),
    1
)

rps_max = max(
    df["RequestsPerSecond"].quantile(0.95),
    1
)

response_max = max(
    df["ResponseTime"].quantile(0.95),
    1
)


df["UsersIntensity"] = (
    df["CurrentUsers"] /
    users_max
)

df["RPSIntensity"] = (
    df["RequestsPerSecond"] /
    rps_max
)

df["ResponseIntensity"] = (
    df["ResponseTime"] /
    response_max
)

df["CPUIntensity"] = (
    df["CPU"] / 100.0
)


# Limit values to sensible range.

df["UsersIntensity"] = (
    df["UsersIntensity"]
    .clip(0, 1)
)

df["RPSIntensity"] = (
    df["RPSIntensity"]
    .clip(0, 1)
)

df["ResponseIntensity"] = (
    df["ResponseIntensity"]
    .clip(0, 1)
)

df["CPUIntensity"] = (
    df["CPUIntensity"]
    .clip(0, 1)
)


# ============================================================
# WORKLOAD SCORE
# ============================================================

# Workload is now primarily determined by:
#
# Users       35%
# RPS         35%
# Response    10%
# CPU         10%
# Growth      10%

growth_signal = (
    np.clip(
        df["UserGrowthRate"],
        0,
        1
    )
    +
    np.clip(
        df["RPSGrowthRate"],
        0,
        1
    )
) / 2


df["WorkloadScore"] = (
    0.35 * df["UsersIntensity"]
    +
    0.35 * df["RPSIntensity"]
    +
    0.10 * df["ResponseIntensity"]
    +
    0.10 * df["CPUIntensity"]
    +
    0.10 * growth_signal
)


df["WorkloadScore"] = (
    df["WorkloadScore"]
    .clip(0, 1)
)


# ============================================================
# DATA-DRIVEN WORKLOAD BOUNDARIES
# ============================================================

q25 = df[
    "WorkloadScore"
].quantile(0.25)

q50 = df[
    "WorkloadScore"
].quantile(0.50)

q75 = df[
    "WorkloadScore"
].quantile(0.75)


print("\n==============================================")
print("WORKLOAD SCORE BOUNDARIES")
print("==============================================")

print(
    f"LOW / MODERATE  : {q25:.3f}"
)

print(
    f"MODERATE / HIGH : {q50:.3f}"
)

print(
    f"HIGH / VERY HIGH: {q75:.3f}"
)


# ============================================================
# CLASSIFICATION
# ============================================================

def classify_workload(score):

    if score <= q25:
        return "LOW"

    elif score <= q50:
        return "MODERATE"

    elif score <= q75:
        return "HIGH"

    return "VERY_HIGH"


df["WorkloadState"] = (
    df["WorkloadScore"]
    .apply(
        classify_workload
    )
)


# ============================================================
# FEATURES FOR MODEL
# ============================================================

features = [

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


X = df[features].copy()

y = df[
    "WorkloadState"
].copy()


# ============================================================
# CLEAN
# ============================================================

X = X.replace(
    [np.inf, -np.inf],
    np.nan
)

valid = X.notna().all(
    axis=1
)

X = X.loc[
    valid
]

y = y.loc[
    valid
]


# ============================================================
# DISTRIBUTION
# ============================================================

print("\n==============================================")
print("WORKLOAD DISTRIBUTION")
print("==============================================")

distribution = (
    y.value_counts()
    .reindex(
        [
            "LOW",
            "MODERATE",
            "HIGH",
            "VERY_HIGH"
        ],
        fill_value=0
    )
)

print(
    distribution
)


# ============================================================
# TRAIN / TEST
# ============================================================

X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )
)


print(
    f"\nTraining samples : "
    f"{len(X_train)}"
)

print(
    f"Testing samples  : "
    f"{len(X_test)}"
)


# ============================================================
# SCALER
# ============================================================

scaler = StandardScaler()

X_train_scaled = (
    scaler.fit_transform(
        X_train
    )
)

X_test_scaled = (
    scaler.transform(
        X_test
    )
)


# ============================================================
# RANDOM FOREST
# ============================================================

print(
    "\nTraining Random Forest..."
)

model = RandomForestClassifier(
    n_estimators=250,
    max_depth=12,
    min_samples_split=5,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train_scaled,
    y_train
)


# ============================================================
# EVALUATION
# ============================================================

y_pred = model.predict(
    X_test_scaled
)

accuracy = accuracy_score(
    y_test,
    y_pred
)


print("\n==============================================")
print("MODEL RESULTS")
print("==============================================")

print(
    f"\nAccuracy: "
    f"{accuracy * 100:.2f}%"
)


labels = [
    "LOW",
    "MODERATE",
    "HIGH",
    "VERY_HIGH"
]


print(
    "\nClassification Report:"
)

print(
    classification_report(
        y_test,
        y_pred,
        labels=labels,
        zero_division=0
    )
)


print(
    "\nConfusion Matrix:"
)

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=labels
)

print(
    pd.DataFrame(
        cm,
        index=labels,
        columns=labels
    )
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

print("\n==============================================")
print("FEATURE IMPORTANCE")
print("==============================================")


importance = pd.DataFrame({

    "Feature": features,

    "Importance":
        model.feature_importances_

})

importance = (
    importance
    .sort_values(
        "Importance",
        ascending=False
    )
)


print(
    importance.to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

joblib.dump(
    model,
    MODEL_PATH
)

joblib.dump(
    scaler,
    SCALER_PATH
)

df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# FINAL
# ============================================================

print("\n==============================================")
print("TRAINING COMPLETE")
print("==============================================")

print(
    f"\nModel saved:\n{MODEL_PATH}"
)

print(
    f"\nScaler saved:\n{SCALER_PATH}"
)

print(
    f"\nDataset saved:\n{OUTPUT_PATH}"
)

print(
    "\nFinal workload distribution:"
)

print(
    df["WorkloadState"]
    .value_counts()
    .reindex(
        [
            "LOW",
            "MODERATE",
            "HIGH",
            "VERY_HIGH"
        ],
        fill_value=0
    )
)

print("\n==============================================")
print("DONE")
print("==============================================")