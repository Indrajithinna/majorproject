from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import psycopg2
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Gauge
import psutil
import threading
import time
import os
import random

from .workload import (
    light_workload,
    medium_workload,
    heavy_workload
)

app = FastAPI()

# ==========================================
# Runtime Request Tracking
# ==========================================

ACTIVE_REQUESTS = 0
LOCK = threading.Lock()

# ==========================================
# Prometheus Metrics
# ==========================================

cpu_metric = Gauge(
    "custom_cpu_usage",
    "CPU Usage Percentage"
)

memory_metric = Gauge(
    "custom_memory_usage",
    "Memory Usage Percentage"
)

pod_metric = Gauge(
    "current_pods",
    "Current Running Pods"
)

active_requests_metric = Gauge(
    "active_requests",
    "Current Active Requests"
)

recommended_pods_metric = Gauge(
    "recommended_pods",
    "Recommended Pods"
)

# ==========================================
# Background Monitoring
# ==========================================

def monitor_system():

    while True:

        cpu = psutil.cpu_percent(interval=1)

        memory = psutil.virtual_memory().percent

        cpu_metric.set(cpu)

        memory_metric.set(memory)

        # -----------------------------
        # Simulated Pod Scaling
        # -----------------------------

        if cpu < 30:

            pods = 1

        elif cpu < 50:

            pods = 2

        elif cpu < 70:

            pods = 3

        elif cpu < 90:

            pods = 4

        else:

            pods = 5

        pod_metric.set(pods)

        # -----------------------------
        # Save Training Dataset
        # -----------------------------

        try:

          with open("dataset/metrics.csv", "a") as f:

            f.write(
            f"{time.time()},"
            f"{cpu},"
            f"{memory},"
            f"{ACTIVE_REQUESTS},"
            f"{pods}\n"
        )

        except Exception as e:

         print("Dataset error:", e)
    time.sleep(2)

# ==========================================
# CORS
# ==========================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# Middleware
# ==========================================
@app.middleware("http")
async def track_requests(request, call_next):

    global ACTIVE_REQUESTS

    ACTIVE_REQUESTS += 1

    active_requests_metric.set(ACTIVE_REQUESTS)

    try:
        response = await call_next(request)
        return response

    finally:
        ACTIVE_REQUESTS -= 1
        active_requests_metric.set(ACTIVE_REQUESTS)



# ==========================================
# Overload Simulation
# ==========================================

def simulate_overload():

    global ACTIVE_REQUESTS

    if ACTIVE_REQUESTS < 20:

        recommended_pods_metric.set(1)

    elif ACTIVE_REQUESTS < 50:

        recommended_pods_metric.set(2)

        time.sleep(random.uniform(0.5, 1.5))

    elif ACTIVE_REQUESTS < 100:

        recommended_pods_metric.set(3)

        time.sleep(random.uniform(2, 4))

    elif ACTIVE_REQUESTS < 150:

        recommended_pods_metric.set(4)

        time.sleep(random.uniform(5, 8))

    else:

        recommended_pods_metric.set(5)

        raise HTTPException(
            status_code=503,
            detail="Server Busy. Please Try Again."
        )

# ==========================================
# PostgreSQL
# ==========================================

DB_HOST = os.getenv("DB_HOST", "localhost")

print("DB_HOST =", DB_HOST)

conn = psycopg2.connect(
    host=DB_HOST,
    port=5432,
    database="university",
    user="postgres",
    password="nithya"
)

# ==========================================
# Home
# ==========================================

@app.get("/")
def home():

    simulate_overload()

    return {

        "message": "MGRK University API Running",

        "db_host": DB_HOST,

        "active_requests": ACTIVE_REQUESTS

    }

# ==========================================
# Branch Rankings
# ==========================================

@app.get("/branch-ranks")
def branch_ranks():

    simulate_overload()

    medium_workload()

    cur = conn.cursor()

    cur.execute("""

        SELECT students.name,
               students.branch,
               AVG(marks.marks)

        FROM students

        JOIN marks
        ON students.id = marks.student_id

        GROUP BY students.id

        ORDER BY AVG(marks.marks) DESC

        LIMIT 10

    """)

    data = cur.fetchall()

    cur.close()

    return {

        "branch_ranks": data,

        "active_requests": ACTIVE_REQUESTS

    }

# ==========================================
# College Rankings
# ==========================================

@app.get("/college-ranks")
def college_ranks():

    simulate_overload()

    heavy_workload()

    cur = conn.cursor()

    cur.execute("""

        SELECT students.college,
               AVG(marks.marks)

        FROM students

        JOIN marks
        ON students.id = marks.student_id

        GROUP BY students.college

        ORDER BY AVG(marks.marks) DESC

    """)

    data = cur.fetchall()

    cur.close()

    return {

        "college_ranks": data,

        "active_requests": ACTIVE_REQUESTS

    }

# ==========================================
# Student Result
# ==========================================

@app.get("/result/{usn}")
def result(usn: str):

    simulate_overload()

    light_workload()

    cur = conn.cursor()

    cur.execute("""

        SELECT students.name,
               students.usn,
               students.branch,
               students.college,
               marks.subject,
               marks.marks

        FROM students

        JOIN marks
        ON students.id = marks.student_id

        WHERE students.usn=%s

    """, (usn,))

    data = cur.fetchall()

    cur.close()

    return {

        "result": data,

        "active_requests": ACTIVE_REQUESTS

    }

# ==========================================
# Prometheus Metrics
# ==========================================

Instrumentator().instrument(app).expose(app)