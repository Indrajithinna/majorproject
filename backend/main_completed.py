from __future__ import annotations

import csv
import os
import random
import threading
import time
from pathlib import Path

import psutil
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Gauge
from prometheus_fastapi_instrumentator import Instrumentator

try:
    import psycopg2
except ImportError:
    psycopg2 = None

try:
    from workload import heavy_workload, light_workload, medium_workload
except ImportError:
    from workload import heavy_workload, light_workload, medium_workload

app = FastAPI(title="MGRK University API", version="2.0.0")

cpu_metric = Gauge("custom_cpu_usage", "CPU Usage Percentage")
memory_metric = Gauge("custom_memory_usage", "Memory Usage Percentage")
pod_metric = Gauge("current_pods", "Current Running Pods")

BRANCHES = ["CSE", "ISE", "ECE", "MECH"]
COLLEGES = ["MGRK College", "BIT", "RVCE", "BMSCE", "MSRIT"]
SUBJECTS = ["Math", "DBMS", "OS", "CN", "AI"]
FIRST_NAMES = [
    "Aarav", "Aditi", "Ananya", "Arjun", "Bhavana", "Deepak", "Diya", "Ishaan",
    "Kavya", "Kiran", "Meera", "Nikhil", "Priya", "Rahul", "Rohan", "Saanvi",
    "Sneha", "Tanvi", "Vikram", "Yash",
]
LAST_NAMES = [
    "Gowda", "Kumar", "Rao", "Shetty", "Reddy", "Patil", "Sharma", "Nair",
    "Prasad", "Iyer", "Naidu", "Hegde", "Murthy", "Joshi", "Kulkarni",
]


def monitor_system() -> None:
    while True:
        cpu = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory().percent
        cpu_metric.set(cpu)
        memory_metric.set(memory)

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
        time.sleep(2)


@app.on_event("startup")
def startup_event() -> None:
    threading.Thread(target=monitor_system, daemon=True).start()


app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db_connection():
    if psycopg2 is None or os.getenv("USE_POSTGRES", "false").lower() != "true":
        return None

    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5432"),
        database=os.getenv("POSTGRES_DB", "university"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
    )


def build_sample_records() -> list[dict]:
    random.seed(2026)
    students = []

    for year in ["23", "24", "25", "26"]:
        for index in range(1, 126):
            usn = f"1MG{year}IS{index:03}"
            marks = [
                {"subject": subject, "marks": random.randint(35, 100)}
                for subject in SUBJECTS
            ]
            average = round(sum(item["marks"] for item in marks) / len(marks), 2)
            students.append(
                {
                    "usn": usn,
                    "name": f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}",
                    "branch": random.choice(BRANCHES),
                    "college": random.choice(COLLEGES),
                    "marks": marks,
                    "average": average,
                }
            )

    return students


SAMPLE_RECORDS = build_sample_records()


def read_prediction_summary() -> dict:
    csv_path = Path(__file__).resolve().parents[1] / "prediction" / "time_series_dataset.csv"
    if not csv_path.exists():
        return {
            "samples": 0,
            "average_cpu": 0,
            "average_memory": 0,
            "average_response_time": 0,
            "peak_users": 0,
        }

    with csv_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    if not rows:
        return {
            "samples": 0,
            "average_cpu": 0,
            "average_memory": 0,
            "average_response_time": 0,
            "peak_users": 0,
        }

    def average(column: str) -> float:
        values = [float(row[column]) for row in rows if row.get(column)]
        return round(sum(values) / len(values), 2) if values else 0

    return {
        "samples": len(rows),
        "average_cpu": average("CPU"),
        "average_memory": average("Memory"),
        "average_response_time": average("ResponseTime"),
        "peak_users": max(int(float(row.get("CurrentUsers") or 0)) for row in rows),
    }


def as_legacy_result(student: dict) -> list[list]:
    return [
        [
            student["name"],
            student["usn"],
            student["branch"],
            student["college"],
            item["subject"],
            item["marks"],
        ]
        for item in student["marks"]
    ]


def db_result_to_student(rowset) -> dict:
    if not rowset:
        raise HTTPException(status_code=404, detail="No result found for this USN")

    first = rowset[0]
    marks = [{"subject": row[4], "marks": row[5]} for row in rowset]
    average = round(sum(item["marks"] for item in marks) / len(marks), 2)
    return {
        "name": first[0],
        "usn": first[1],
        "branch": first[2],
        "college": first[3],
        "marks": marks,
        "average": average,
    }


def find_student(usn: str) -> dict:
    normalized_usn = usn.strip().upper()
    student = next((item for item in SAMPLE_RECORDS if item["usn"] == normalized_usn), None)
    if not student:
        raise HTTPException(status_code=404, detail="No result found for this USN")
    return student


@app.get("/")
def home() -> dict:
    return {
        "message": "MGRK University API Running",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }


@app.get("/api/health")
def health() -> dict:
    database = "sample-data"
    conn = None

    try:
        conn = get_db_connection()
        if conn:
            database = "postgres"
    except Exception:
        database = "sample-data"
    finally:
        if conn:
            conn.close()

    return {
        "status": "ok",
        "database": database,
        "students": len(SAMPLE_RECORDS),
    }


@app.get("/api/result/{usn}")
def api_result(usn: str) -> dict:
    light_workload()
    conn = get_db_connection()

    if conn:
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT students.name, students.usn, students.branch, students.college,
                       marks.subject, marks.marks
                FROM students
                JOIN marks ON students.id = marks.student_id
                WHERE students.usn=%s
                ORDER BY marks.subject
                """,
                (usn.strip().upper(),),
            )
            student = db_result_to_student(cur.fetchall())
            cur.close()
            return {"result": student}
        finally:
            conn.close()

    return {"result": find_student(usn)}


@app.get("/api/branch-ranks")
def api_branch_ranks() -> dict:
    medium_workload()
    conn = get_db_connection()

    if conn:
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT students.name, students.usn, students.branch, students.college,
                       AVG(marks.marks) AS average
                FROM students
                JOIN marks ON students.id = marks.student_id
                GROUP BY students.id
                ORDER BY average DESC
                LIMIT 10
                """
            )
            data = [
                {
                    "name": row[0],
                    "usn": row[1],
                    "branch": row[2],
                    "college": row[3],
                    "average": round(float(row[4]), 2),
                }
                for row in cur.fetchall()
            ]
            cur.close()
            return {"branch_ranks": data}
        finally:
            conn.close()

    data = sorted(SAMPLE_RECORDS, key=lambda item: item["average"], reverse=True)[:10]
    return {
        "branch_ranks": [
            {
                "name": item["name"],
                "usn": item["usn"],
                "branch": item["branch"],
                "college": item["college"],
                "average": item["average"],
            }
            for item in data
        ]
    }


@app.get("/api/college-ranks")
def api_college_ranks() -> dict:
    heavy_workload()
    conn = get_db_connection()

    if conn:
        try:
            cur = conn.cursor()
            cur.execute(
                """
                SELECT students.college, AVG(marks.marks) AS average
                FROM students
                JOIN marks ON students.id = marks.student_id
                GROUP BY students.college
                ORDER BY average DESC
                """
            )
            data = [
                {"college": row[0], "average": round(float(row[1]), 2)}
                for row in cur.fetchall()
            ]
            cur.close()
            return {"college_ranks": data}
        finally:
            conn.close()

    totals = {}
    for student in SAMPLE_RECORDS:
        totals.setdefault(student["college"], []).append(student["average"])

    data = [
        {"college": college, "average": round(sum(scores) / len(scores), 2)}
        for college, scores in totals.items()
    ]
    data.sort(key=lambda item: item["average"], reverse=True)
    return {"college_ranks": data}


@app.get("/api/analytics")
def api_analytics() -> dict:
    return {
        "students": len(SAMPLE_RECORDS),
        "marks_records": len(SAMPLE_RECORDS) * len(SUBJECTS),
        "branches": len(BRANCHES),
        "colleges": len(COLLEGES),
        "autoscaling": "Enabled",
        "prediction": read_prediction_summary(),
    }


@app.get("/result/{usn}")
def legacy_result(usn: str) -> dict:
    response = api_result(usn)
    return {"result": as_legacy_result(response["result"])}


@app.get("/branch-ranks")
def legacy_branch_ranks() -> dict:
    response = api_branch_ranks()
    return {
        "branch_ranks": [
            [item["name"], item["branch"], item["average"]]
            for item in response["branch_ranks"]
        ]
    }


@app.get("/college-ranks")
def legacy_college_ranks() -> dict:
    response = api_college_ranks()
    return {
        "college_ranks": [
            [item["college"], item["average"]]
            for item in response["college_ranks"]
        ]
    }


Instrumentator().instrument(app).expose(app)
