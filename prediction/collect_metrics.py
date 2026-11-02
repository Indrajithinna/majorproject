import requests
import pandas as pd
import time
import os

from datetime import datetime


# ============================================================
# CONFIGURATION
# ============================================================

PROMETHEUS = "http://127.0.0.1:9091/api/v1/query"

CSV = "prediction/time_series_dataset.csv"

TRAFFIC_LOG = "load_test/logs/traffic_log.csv"

COLLECTION_INTERVAL = 5


# ============================================================
# PROMETHEUS QUERY
# ============================================================

def query(metric):

    try:

        response = requests.get(
            PROMETHEUS,
            params={"query": metric},
            timeout=5
        )

        response.raise_for_status()

        data = response.json()

        result = data.get(
            "data",
            {}
        ).get(
            "result",
            []
        )

        if result:

            value = float(
                result[0]["value"][1]
            )

            return value

        print(
            f"Warning: No Prometheus data for: {metric}"
        )

        return None

    except Exception as e:

        print(
            f"Prometheus query error: {e}"
        )

        return None


# ============================================================
# GET CURRENT TRAFFIC INFORMATION
# ============================================================

def get_latest_traffic():

    if not os.path.exists(TRAFFIC_LOG):

        return 0, "Low", "None"

    try:

        # ----------------------------------------------------
        # Read traffic log line-by-line.
        #
        # Historical rows may contain 3 fields while newer
        # rows contain 4 fields.
        # ----------------------------------------------------

        with open(
            TRAFFIC_LOG,
            "r",
            encoding="utf-8"
        ) as f:

            lines = [
                line.strip()
                for line in f
                if line.strip()
            ]

        if not lines:

            return 0, "Low", "None"

        # ----------------------------------------------------
        # Get latest traffic record
        # ----------------------------------------------------

        last = lines[-1]

        parts = [
            part.strip()
            for part in last.split(",")
        ]

        # ----------------------------------------------------
        # Required fields:
        #
        # 0 = Timestamp
        # 1 = CurrentUsers
        # 2 = LoadLevel
        # 3 = MostRequestedAPI
        # ----------------------------------------------------

        if len(parts) < 3:

            return 0, "Low", "None"

        # ----------------------------------------------------
        # Current Users
        # ----------------------------------------------------

        try:

            users = int(
                float(
                    parts[1]
                )
            )

        except Exception:

            users = 0

        # ----------------------------------------------------
        # Load Level
        # ----------------------------------------------------

        load = parts[2]

        if not load:

            load = "Low"

        # ----------------------------------------------------
        # Most Requested API
        # ----------------------------------------------------

        if len(parts) >= 4:

            api = parts[3]

        else:

            api = "None"

        if not api:

            api = "None"

        return users, load, api

    except Exception as e:

        print(
            f"Traffic log error: {e}"
        )

        return 0, "Low", "None"


# ============================================================
# GET CPU
# ============================================================

def get_cpu():

    value = query(
        "custom_cpu_usage"
    )

    if value is None:

        return None

    return value


# ============================================================
# GET MEMORY
# ============================================================

def get_memory():

    value = query(
        "custom_memory_usage"
    )

    if value is None:

        return None

    return value


# ============================================================
# GET REQUEST RATE
# ============================================================

def get_requests_per_second():

    value = query(
        'sum(rate(http_requests_total{job="backend"}[1m]))'
    )

    # --------------------------------------------------------
    # Prometheus returned no data
    # --------------------------------------------------------

    if value is None:

        return None

    # --------------------------------------------------------
    # Reject negative values
    # --------------------------------------------------------

    if value < 0:

        return 0.0

    # --------------------------------------------------------
    # Reject impossible request rates
    # --------------------------------------------------------

    if value > 1000:

        print(
            f"Warning: Invalid backend request rate detected: "
            f"{value:.2f}"
        )

        return None

    return round(
        value,
        4
    )


# ============================================================
# GET AVERAGE RESPONSE TIME
# ============================================================

def get_response_time():

    query_string = """
    (
        sum(rate(http_request_duration_seconds_sum{job="backend"}[1m]))
        /
        sum(rate(http_request_duration_seconds_count{job="backend"}[1m]))
    )
    """

    value = query(
        query_string
    )

    # --------------------------------------------------------
    # Prometheus returned no data
    # --------------------------------------------------------

    if value is None:

        return None

    # --------------------------------------------------------
    # Reject negative values
    # --------------------------------------------------------

    if value < 0:

        return 0.0

    # --------------------------------------------------------
    # Reject impossible response times
    # --------------------------------------------------------

    if value > 60:

        print(
            f"Warning: Invalid response time detected: "
            f"{value:.2f}s"
        )

        return None

    return value


# ============================================================
# SAVE DATASET
# ============================================================

def save_row(
    cpu,
    memory,
    requests_sec,
    response_time,
    users,
    load,
    api
):

    row = {

        "Timestamp":
            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            ),

        "CurrentUsers":
            users,

        "CPU":
            round(cpu, 2),

        "Memory":
            round(memory, 2),

        "RequestsPerSecond":
            round(requests_sec, 4),

        "ResponseTime":
            round(response_time, 4),

        "LoadLevel":
            load,

        "MostRequestedAPI":
            api

    }

    new_row = pd.DataFrame(
        [row]
    )

    # --------------------------------------------------------
    # Create directory if necessary
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(CSV),
        exist_ok=True
    )

    # --------------------------------------------------------
    # Append to existing dataset
    # --------------------------------------------------------

    if os.path.exists(CSV):

        try:

            old = pd.read_csv(
                CSV
            )

            new_row = pd.concat(
                [
                    old,
                    new_row
                ],
                ignore_index=True
            )

        except Exception as e:

            print(
                f"Dataset read error: {e}"
            )

            # Do not overwrite the existing dataset if it
            # cannot be read correctly.
            return row

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    new_row.to_csv(
        CSV,
        index=False
    )

    return row


# ============================================================
# MAIN MONITORING LOOP
# ============================================================

print("=" * 70)
print("MGRK UNIVERSITY")
print("CLOUD MONITORING DATA COLLECTOR")
print("=" * 70)

print(
    f"Prometheus : {PROMETHEUS}"
)

print(
    f"Dataset    : {CSV}"
)

print(
    f"Interval   : {COLLECTION_INTERVAL} seconds"
)

print("=" * 70)


while True:

    try:

        # ----------------------------------------------------
        # Collect metrics
        # ----------------------------------------------------

        cpu = get_cpu()

        memory = get_memory()

        requests_sec = (
            get_requests_per_second()
        )

        response_time = (
            get_response_time()
        )

        # ----------------------------------------------------
        # If any critical Prometheus metric is unavailable,
        # DO NOT write a fake zero into the ML dataset.
        # ----------------------------------------------------

        if (
            cpu is None
            or memory is None
            or requests_sec is None
            or response_time is None
        ):

            print()
            print("-" * 70)
            print(
                "Skipping dataset row because "
                "Prometheus metrics are unavailable."
            )
            print("-" * 70)

            time.sleep(
                COLLECTION_INTERVAL
            )

            continue

        # ----------------------------------------------------
        # Get traffic information
        # ----------------------------------------------------

        users, load, api = (
            get_latest_traffic()
        )

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        row = save_row(

            cpu=cpu,

            memory=memory,

            requests_sec=requests_sec,

            response_time=response_time,

            users=users,

            load=load,

            api=api

        )

        # ----------------------------------------------------
        # Display
        # ----------------------------------------------------

        print()
        print("-" * 70)

        print(
            "Timestamp          :",
            row["Timestamp"]
        )

        print(
            "Current Users      :",
            row["CurrentUsers"]
        )

        print(
            "CPU                :",
            f"{row['CPU']:.2f}%"
        )

        print(
            "Memory             :",
            f"{row['Memory']:.2f}%"
        )

        print(
            "Requests/sec       :",
            f"{row['RequestsPerSecond']:.4f}"
        )

        print(
            "Response Time      :",
            f"{row['ResponseTime']:.4f}s"
        )

        print(
            "Load Level         :",
            row["LoadLevel"]
        )

        print(
            "Most Requested API :",
            row["MostRequestedAPI"]
        )

        print("-" * 70)

        time.sleep(
            COLLECTION_INTERVAL
        )

    except KeyboardInterrupt:

        print()

        print(
            "Monitoring stopped."
        )

        break

    except Exception as e:

        print(
            f"Collector error: {e}"
        )

        time.sleep(
            COLLECTION_INTERVAL
        )