import csv
import os
from datetime import datetime

import load_test.config as config


# ============================================================
# TRAFFIC LOG CONFIGURATION
# ============================================================

LOG_FILE = "load_test/logs/traffic_log.csv"

LOG_HEADER = [
    "Timestamp",
    "CurrentUsers",
    "LoadLevel",
    "MostRequestedAPI"
]


# ============================================================
# INITIALIZE LOG FILE
# ============================================================

def initialize_log():

    os.makedirs(
        os.path.dirname(LOG_FILE),
        exist_ok=True
    )

    # Create a new file if it does not exist
    if not os.path.exists(LOG_FILE):

        with open(
            LOG_FILE,
            "w",
            newline="",
            encoding="utf-8"
        ) as f:

            writer = csv.writer(f)

            writer.writerow(LOG_HEADER)


# ============================================================
# SAVE TRAFFIC REPORT
# ============================================================

def save_report(users, level):

    initialize_log()

    with open(
        LOG_FILE,
        "a",
        newline="",
        encoding="utf-8"
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            users,
            level,
            config.MOST_REQUESTED_API
        ])