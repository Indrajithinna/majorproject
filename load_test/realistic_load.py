import threading
import random
import time
import csv
import os
import requests
import psutil

import load_test.config as config

from load_test.config import (
    INITIAL_USERS,
    MAX_USERS,
    USER_INCREMENT,
    REQUEST_INTERVAL
)

from load_test.traffic_schedule import traffic_levels
from load_test.user_simulator import simulate_student
from load_test.report_generator import save_report


# ============================================================
# DATASET CONFIGURATION
# ============================================================

DATASET_PATH = os.path.join(
    "prediction",
    "time_series_dataset.csv"
)

DATASET_INTERVAL = 5


# ============================================================
# DATASET INITIALIZATION
# ============================================================

def initialize_dataset():

    os.makedirs(
        "prediction",
        exist_ok=True
    )

    if not os.path.exists(DATASET_PATH):

        with open(
            DATASET_PATH,
            "w",
            newline=""
        ) as file:

            writer = csv.writer(file)

            writer.writerow([
                "Timestamp",
                "CurrentUsers",
                "CPU",
                "Memory",
                "RequestsPerSecond",
                "ResponseTime",
                "LoadLevel",
                "MostRequestedAPI"
            ])


# ============================================================
# DATASET COLLECTOR
# ============================================================

def collect_metrics():

    previous_requests = 0
    previous_time = time.time()

    while True:

        try:

            current_time = time.time()

            # --------------------------------------------
            # CPU
            # --------------------------------------------

            cpu = psutil.cpu_percent(
                interval=1
            )

            # --------------------------------------------
            # Memory
            # --------------------------------------------

            memory = psutil.virtual_memory().percent

            # --------------------------------------------
            # Requests
            # --------------------------------------------

            total_requests = sum(
                config.API_COUNTER.values()
            )

            elapsed = current_time - previous_time

            if elapsed > 0:

                requests_per_second = (
                    total_requests - previous_requests
                ) / elapsed

            else:

                requests_per_second = 0


            # --------------------------------------------
            # Response time
            # --------------------------------------------

            # Approximate response time based on
            # current request rate.
            if requests_per_second > 0:

                response_time = (
                    1 / requests_per_second
                )

            else:

                response_time = 0


            # --------------------------------------------
            # Save row
            # --------------------------------------------

            with open(
                DATASET_PATH,
                "a",
                newline=""
            ) as file:

                writer = csv.writer(file)

                writer.writerow([
                    time.strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    config.CURRENT_USERS,
                    round(cpu, 2),
                    round(memory, 2),
                    round(
                        requests_per_second,
                        4
                    ),
                    round(
                        response_time,
                        6
                    ),
                    config.CURRENT_LOAD_LEVEL,
                    config.MOST_REQUESTED_API
                ])


            print(
                f"[DATASET] "
                f"Users={config.CURRENT_USERS} | "
                f"CPU={cpu:.2f}% | "
                f"Memory={memory:.2f}% | "
                f"RPS={requests_per_second:.2f}"
            )


            previous_requests = total_requests
            previous_time = current_time


        except Exception as error:

            print(
                f"[DATASET ERROR] {error}"
            )


        time.sleep(
            DATASET_INTERVAL
        )


# ============================================================
# INITIALIZE DATASET
# ============================================================

initialize_dataset()


# ============================================================
# START DATASET COLLECTOR
# ============================================================

collector_thread = threading.Thread(
    target=collect_metrics,
    daemon=True
)

collector_thread.start()


# ============================================================
# INITIAL VALUES
# ============================================================

current_users = INITIAL_USERS

direction = 1

student_id = 1

active_threads = []


print("=" * 60)
print("      MGRK UNIVERSITY REALISTIC LOAD SIMULATION")
print("=" * 60)

print(
    f"Dataset collector running every "
    f"{DATASET_INTERVAL} seconds"
)


# ============================================================
# MAIN LOAD SIMULATION
# ============================================================

while True:

    # ----------------------------------------
    # Determine Current Load Level
    # ----------------------------------------

    level = "Low"

    for users, load in traffic_levels:

        if current_users <= users:

            level = load

            break


    # ----------------------------------------
    # Update Runtime Variables
    # ----------------------------------------

    config.CURRENT_USERS = current_users

    config.CURRENT_LOAD_LEVEL = level


    # ----------------------------------------
    # Display Current Status
    # ----------------------------------------

    print(
        "\n==================================================="
    )

    print(
        f"Current Active Users : {current_users}"
    )

    print(
        f"Traffic Level        : {level}"
    )

    print(
        f"Most Requested API   : "
        f"{config.MOST_REQUESTED_API}"
    )

    print(
        "==================================================="
    )


    # ----------------------------------------
    # Save Report
    # ----------------------------------------

    save_report(
        current_users,
        level
    )


    # ----------------------------------------
    # Create Student Threads
    # ----------------------------------------

    while len(active_threads) < current_users:

        thread = threading.Thread(
            target=simulate_student,
            args=(student_id,),
            daemon=True
        )

        thread.start()

        active_threads.append(
            thread
        )

        student_id += 1

        time.sleep(0.01)


    # ----------------------------------------
    # Remove Finished Threads
    # ----------------------------------------

    active_threads = [

        thread

        for thread in active_threads

        if thread.is_alive()

    ]


    # ----------------------------------------
    # Wait
    # ----------------------------------------

    time.sleep(
        REQUEST_INTERVAL
    )


    # ----------------------------------------
    # Increase / Decrease Users
    # ----------------------------------------

    current_users += (
        USER_INCREMENT * direction
    )


    if current_users >= MAX_USERS:

        current_users = MAX_USERS

        direction = -1


    elif current_users <= INITIAL_USERS:

        current_users = INITIAL_USERS

        direction = 1