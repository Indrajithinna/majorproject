import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle
)
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    Image,
    Table,
    TableStyle
)


# ============================================================
# CONFIGURATION
# ============================================================

DATASET = "prediction/time_series_dataset.csv"

REPORT_FOLDER = "prediction/reports"

GRAPH_FOLDER = "prediction/graphs"

PDF_FILE = os.path.join(
    REPORT_FOLDER,
    "load_report.pdf"
)


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(
    REPORT_FOLDER,
    exist_ok=True
)

os.makedirs(
    GRAPH_FOLDER,
    exist_ok=True
)


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 70)
print("MGRK UNIVERSITY")
print("CLOUD MONITORING REPORT GENERATOR")
print("=" * 70)

print()
print("Loading dataset...")

if not os.path.exists(DATASET):

    raise FileNotFoundError(
        f"Dataset not found: {DATASET}"
    )


df = pd.read_csv(
    DATASET
)

print(
    "Total historical rows:",
    len(df)
)


# ============================================================
# TIMESTAMP CLEANING
# ============================================================

df["Timestamp"] = pd.to_datetime(
    df["Timestamp"],
    format="mixed",
    errors="coerce"
)


df = df.dropna(
    subset=["Timestamp"]
)


# ============================================================
# SORT CHRONOLOGICALLY
# ============================================================

df = df.sort_values(
    "Timestamp"
).reset_index(
    drop=True
)


# ============================================================
# NUMERIC CONVERSION
# ============================================================

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


# ============================================================
# REMOVE INVALID CPU / MEMORY VALUES
# ============================================================

df = df[
    (df["CPU"] >= 0) &
    (df["CPU"] <= 100)
]


df = df[
    (df["Memory"] >= 0) &
    (df["Memory"] <= 100)
]


# ============================================================
# REMOVE INVALID REQUEST RATE
# ============================================================

df.loc[
    (df["RequestsPerSecond"] < 0) |
    (df["RequestsPerSecond"] > 1000),
    "RequestsPerSecond"
] = float("nan")


# ============================================================
# REMOVE INVALID RESPONSE TIME
# ============================================================

df.loc[
    (df["ResponseTime"] < 0) |
    (df["ResponseTime"] > 60),
    "ResponseTime"
] = float("nan")


# ============================================================
# SELECT LATEST MONITORING DAY
# ============================================================

latest_date = (
    df["Timestamp"]
    .dt.date
    .max()
)


report_df = df[
    df["Timestamp"].dt.date == latest_date
].copy()


# ============================================================
# CHECK LATEST DAY
# ============================================================

if report_df.empty:

    raise ValueError(
        "No monitoring data available for the latest day."
    )


# ============================================================
# FILL SMALL DATA GAPS
# ============================================================

report_df[
    "RequestsPerSecond"
] = (
    report_df[
        "RequestsPerSecond"
    ]
    .interpolate()
    .ffill()
    .bfill()
)


report_df[
    "ResponseTime"
] = (
    report_df[
        "ResponseTime"
    ]
    .interpolate()
    .ffill()
    .bfill()
)


# ============================================================
# SORT AGAIN
# ============================================================

report_df = report_df.sort_values(
    "Timestamp"
).reset_index(
    drop=True
)


# ============================================================
# REPORT INFORMATION
# ============================================================

print()
print("=" * 70)
print("REPORT DATA")
print("=" * 70)

print(
    "Report Date        :",
    latest_date
)

print(
    "Rows Used          :",
    len(report_df)
)

print(
    "First Timestamp    :",
    report_df["Timestamp"].min()
)

print(
    "Last Timestamp     :",
    report_df["Timestamp"].max()
)

print("=" * 70)


# ============================================================
# STATISTICS
# ============================================================

highest_cpu = (
    report_df["CPU"].max()
)

average_cpu = (
    report_df["CPU"].mean()
)

highest_memory = (
    report_df["Memory"].max()
)

average_memory = (
    report_df["Memory"].mean()
)

highest_requests = (
    report_df["RequestsPerSecond"].max()
)

average_requests = (
    report_df["RequestsPerSecond"].mean()
)

highest_response = (
    report_df["ResponseTime"].max()
)

average_response = (
    report_df["ResponseTime"].mean()
)

peak_users = (
    report_df["CurrentUsers"].max()
)


# ============================================================
# PEAK CPU ROW
# ============================================================

peak_row = report_df.loc[
    report_df["CPU"].idxmax()
]


# ============================================================
# MONITORING DURATION
# ============================================================

monitoring_duration = (

    report_df["Timestamp"].max()

    -

    report_df["Timestamp"].min()

)


# ============================================================
# GRAPH FUNCTION
# ============================================================

def create_graph(
    column,
    filename,
    title,
    ylabel,
    y_min=None,
    y_max=None
):

    path = os.path.join(
        GRAPH_FOLDER,
        filename
    )

    plt.figure(
        figsize=(11, 4.5)
    )


    plt.plot(

        report_df["Timestamp"],

        report_df[column],

        linewidth=2,

        marker="o",

        markersize=3

    )


    plt.title(
        title,
        fontsize=13,
        fontweight="bold"
    )

    plt.xlabel(
        "Time"
    )

    plt.ylabel(
        ylabel
    )


    # --------------------------------------------------------
    # Better time axis
    # --------------------------------------------------------

    ax = plt.gca()

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter(
            "%H:%M"
        )
    )


    ax.xaxis.set_major_locator(
        mdates.AutoDateLocator(
            minticks=5,
            maxticks=10
        )
    )


    plt.xticks(
        rotation=30
    )


    # --------------------------------------------------------
    # Optional Y-axis limits
    # --------------------------------------------------------

    if y_min is not None:

        plt.ylim(
            bottom=y_min
        )


    if y_max is not None:

        plt.ylim(
            top=y_max
        )


    plt.grid(
        True,
        alpha=0.3
    )


    plt.tight_layout()


    # --------------------------------------------------------
    # Always overwrite old graph
    # --------------------------------------------------------

    plt.savefig(

        path,

        dpi=160,

        bbox_inches="tight"

    )


    plt.close()


    print(
        "Graph updated:",
        path
    )


# ============================================================
# GENERATE GRAPHS
# ============================================================

create_graph(

    "CPU",

    "cpu_usage.png",

    "CPU Usage During Latest Monitoring Session",

    "CPU Usage (%)",

    0,

    100

)


create_graph(

    "Memory",

    "memory_usage.png",

    "Memory Usage During Latest Monitoring Session",

    "Memory Usage (%)",

    0,

    100

)


create_graph(

    "RequestsPerSecond",

    "requests.png",

    "Requests per Second During Latest Monitoring Session",

    "Requests / Second",

    0,

    None

)


create_graph(

    "ResponseTime",

    "response_time.png",

    "Average Response Time During Latest Monitoring Session",

    "Response Time (seconds)",

    0,

    None

)


print()
print(
    "All graphs regenerated successfully."
)


# ============================================================
# API USAGE
# ============================================================

api_counts = (

    report_df[
        "MostRequestedAPI"
    ]

    .fillna("Unknown")

    .astype(str)

    .value_counts()

    .reset_index()

)


api_counts.columns = [

    "API",

    "Samples"

]


total_api_samples = (
    api_counts["Samples"].sum()
)


# ============================================================
# PDF STYLES
# ============================================================

styles = getSampleStyleSheet()


title_style = ParagraphStyle(

    "CustomTitle",

    parent=styles["Title"],

    alignment=TA_CENTER,

    fontSize=20,

    leading=24

)


heading_style = ParagraphStyle(

    "CustomHeading",

    parent=styles["Heading1"],

    fontSize=17,

    leading=21

)


normal_style = ParagraphStyle(

    "CustomNormal",

    parent=styles["BodyText"],

    fontSize=10,

    leading=14

)


story = []


# ============================================================
# PAGE 1 - COVER
# ============================================================

story.append(

    Paragraph(

        "MGRK UNIVERSITY",

        title_style

    )

)


story.append(
    Spacer(1, 20)
)


story.append(

    Paragraph(

        "Cloud Monitoring & Load Analysis Report",

        heading_style

    )

)


story.append(
    Spacer(1, 20)
)


story.append(

    Paragraph(

        """
        <b>Project Title</b><br/><br/>

        Proactive ML-Based Load Balancing for Cloud
        Microservices Using LSTM, Reinforcement
        Learning and Kubernetes.
        """,

        normal_style

    )

)


story.append(
    Spacer(1, 15)
)


story.append(

    Paragraph(

        """
        <b>About MGRK University</b><br/><br/>

        MGRK University provides a modern,
        technology-driven academic environment
        that supports digital education and
        cloud-based student services.

        The Student Result Management System
        allows students to securely access
        their academic results, branch rankings,
        and analytics through a centralized
        cloud platform.

        This report presents the monitoring
        and load analysis of the system during
        the latest load-monitoring session.
        """,

        normal_style

    )

)


story.append(
    Spacer(1, 20)
)


info = [

    [
        "Project",
        "MGRK University Student Result Portal"
    ],

    [
        "Report Date",
        str(latest_date)
    ],

    [
        "Generated Date",
        datetime.now().strftime(
            "%d-%m-%Y"
        )
    ],

    [
        "Generated Time",
        datetime.now().strftime(
            "%H:%M:%S"
        )
    ],

    [
        "Monitoring Duration",
        str(monitoring_duration)
    ],

    [
        "Data Points",
        str(len(report_df))
    ],

    [
        "Generated By",
        "Cloud Monitoring System"
    ],

    [
        "Version",
        "2.0"
    ]

]


table = Table(

    info,

    colWidths=[
        170,
        260
    ]

)


table.setStyle(

    TableStyle([

        (
            "GRID",
            (0, 0),
            (-1, -1),
            1,
            colors.black
        ),

        (
            "BACKGROUND",
            (0, 0),
            (0, -1),
            colors.lightgrey
        ),

        (
            "BOTTOMPADDING",
            (0, 0),
            (-1, -1),
            8
        ),

        (
            "TOPPADDING",
            (0, 0),
            (-1, -1),
            8
        )

    ])

)


story.append(
    table
)


story.append(
    Spacer(1, 20)
)


story.append(

    Paragraph(

        "© 2026 MGRK University | Cloud Monitoring System",

        styles["Italic"]

    )

)


story.append(
    PageBreak()
)


# ============================================================
# PAGE 2 - SYSTEM SUMMARY
# ============================================================

story.append(

    Paragraph(
        "System Summary",
        heading_style
    )

)


story.append(
    Spacer(1, 15)
)


summary = [

    [
        "Highest CPU",
        f"{highest_cpu:.2f}%"
    ],

    [
        "Average CPU",
        f"{average_cpu:.2f}%"
    ],

    [
        "Highest Memory",
        f"{highest_memory:.2f}%"
    ],

    [
        "Average Memory",
        f"{average_memory:.2f}%"
    ],

    [
        "Highest Requests/sec",
        f"{highest_requests:.2f}"
    ],

    [
        "Average Requests/sec",
        f"{average_requests:.2f}"
    ],

    [
        "Highest Response Time",
        f"{highest_response:.3f} sec"
    ],

    [
        "Average Response Time",
        f"{average_response:.3f} sec"
    ],

    [
        "Peak Users",
        str(peak_users)
    ]

]


table = Table(

    summary,

    colWidths=[
        220,
        180
    ]

)


table.setStyle(

    TableStyle([

        (
            "GRID",
            (0, 0),
            (-1, -1),
            1,
            colors.black
        ),

        (
            "BACKGROUND",
            (0, 0),
            (0, -1),
            colors.lightgrey
        ),

        (
            "BOTTOMPADDING",
            (0, 0),
            (-1, -1),
            8
        ),

        (
            "TOPPADDING",
            (0, 0),
            (-1, -1),
            8
        )

    ])

)


story.append(
    table
)


story.append(
    PageBreak()
)


# ============================================================
# PAGE 3 - PEAK LOAD
# ============================================================

story.append(

    Paragraph(

        "Peak Load Analysis",

        heading_style

    )

)


story.append(
    Spacer(1, 15)
)


peak_load = [

    [
        "Peak Time",
        str(
            peak_row["Timestamp"]
        )
    ],

    [
        "CPU",
        f"{peak_row['CPU']:.2f}%"
    ],

    [
        "Memory",
        f"{peak_row['Memory']:.2f}%"
    ],

    [
        "Requests/sec",
        f"{peak_row['RequestsPerSecond']:.2f}"
    ],

    [
        "Response Time",
        f"{peak_row['ResponseTime']:.3f} sec"
    ],

    [
        "Current Users",
        str(
            int(
                peak_row["CurrentUsers"]
            )
        )
    ],

    [
        "Load Level",
        str(
            peak_row["LoadLevel"]
        )
    ],

    [
        "Most Requested API",
        str(
            peak_row["MostRequestedAPI"]
        )
    ]

]


table = Table(

    peak_load,

    colWidths=[
        220,
        180
    ]

)


table.setStyle(

    TableStyle([

        (
            "GRID",
            (0, 0),
            (-1, -1),
            1,
            colors.black
        ),

        (
            "BACKGROUND",
            (0, 0),
            (0, -1),
            colors.lightgrey
        ),

        (
            "BOTTOMPADDING",
            (0, 0),
            (-1, -1),
            8
        ),

        (
            "TOPPADDING",
            (0, 0),
            (-1, -1),
            8
        )

    ])

)


story.append(
    table
)


story.append(
    PageBreak()
)


# ============================================================
# PAGE 4 - API USAGE
# ============================================================

story.append(

    Paragraph(

        "API Usage Analysis",

        heading_style

    )

)


story.append(
    Spacer(1, 15)
)


api_table_data = [

    [
        "API",
        "Samples",
        "Percentage"
    ]

]


for _, row in api_counts.iterrows():

    percentage = (

        row["Samples"]

        /

        total_api_samples

    ) * 100


    api_table_data.append(

        [

            row["API"],

            int(
                row["Samples"]
            ),

            f"{percentage:.2f}%"

        ]

    )


table = Table(

    api_table_data,

    colWidths=[
        180,
        120,
        120
    ]

)


table.setStyle(

    TableStyle([

        (
            "GRID",
            (0, 0),
            (-1, -1),
            1,
            colors.black
        ),

        (
            "BACKGROUND",
            (0, 0),
            (-1, 0),
            colors.grey
        ),

        (
            "TEXTCOLOR",
            (0, 0),
            (-1, 0),
            colors.white
        ),

        (
            "BACKGROUND",
            (0, 1),
            (0, -1),
            colors.beige
        ),

        (
            "BOTTOMPADDING",
            (0, 0),
            (-1, -1),
            8
        )

    ])

)


story.append(
    table
)


story.append(
    Spacer(1, 25)
)


story.append(

    Paragraph(

        """
        API usage percentages represent the distribution
        of monitored traffic samples by the API recorded
        as the most requested endpoint at each collection
        interval.
        """,

        normal_style

    )

)


story.append(
    PageBreak()
)


# ============================================================
# PAGE 5 - CPU AND MEMORY GRAPHS
# ============================================================

story.append(

    Paragraph(

        "Performance Graphs",

        heading_style

    )

)


story.append(
    Spacer(1, 15)
)


story.append(

    Image(

        os.path.join(
            GRAPH_FOLDER,
            "cpu_usage.png"
        ),

        width=500,

        height=205

    )

)


story.append(
    Spacer(1, 10)
)


story.append(

    Image(

        os.path.join(
            GRAPH_FOLDER,
            "memory_usage.png"
        ),

        width=500,

        height=205

    )

)


story.append(
    PageBreak()
)


# ============================================================
# PAGE 6 - TRAFFIC GRAPHS
# ============================================================

story.append(

    Paragraph(

        "Traffic Analysis",

        heading_style

    )

)


story.append(
    Spacer(1, 15)
)


story.append(

    Image(

        os.path.join(
            GRAPH_FOLDER,
            "requests.png"
        ),

        width=500,

        height=205

    )

)


story.append(
    Spacer(1, 10)
)


story.append(

    Image(

        os.path.join(
            GRAPH_FOLDER,
            "response_time.png"
        ),

        width=500,

        height=205

    )

)


story.append(
    PageBreak()
)


# ============================================================
# PAGE 7 - OVERALL ANALYSIS
# ============================================================

story.append(

    Paragraph(

        "Overall Analysis & Recommendations",

        heading_style

    )

)


story.append(
    Spacer(1, 15)
)


recommendation = f"""

<b>Monitoring Period</b><br/>

{latest_date}<br/><br/>


<b>Overall Analysis</b><br/><br/>

Highest CPU Usage :
<b>{highest_cpu:.2f}%</b>

<br/><br/>

Average CPU Usage :
<b>{average_cpu:.2f}%</b>

<br/><br/>

Highest Memory Usage :
<b>{highest_memory:.2f}%</b>

<br/><br/>

Average Memory Usage :
<b>{average_memory:.2f}%</b>

<br/><br/>

Peak Users :
<b>{peak_users}</b>

<br/><br/>

Highest Requests/sec :
<b>{highest_requests:.2f}</b>

<br/><br/>

Average Requests/sec :
<b>{average_requests:.2f}</b>

<br/><br/>

Highest Response Time :
<b>{highest_response:.3f} seconds</b>

<br/><br/>

Average Response Time :
<b>{average_response:.3f} seconds</b>

<br/><br/>


<b>Recommendations</b>

<br/><br/>

• Monitor CPU and Memory continuously.

<br/><br/>

• Predict traffic before peak load using LSTM.

<br/><br/>

• Scale Kubernetes Pods automatically during
High and Critical load.

<br/><br/>

• Use the predicted CPU metric with Kubernetes
autoscaling to support proactive scaling.

<br/><br/>

• Continue collecting time-series data to improve
prediction accuracy.

<br/><br/>

• Compare predicted CPU with actual CPU during
future monitoring sessions.

"""


story.append(

    Paragraph(

        recommendation,

        normal_style

    )

)


story.append(
    Spacer(1, 30)
)


story.append(

    Paragraph(

        "<b>End of Report</b>",

        title_style

    )

)


# ============================================================
# BUILD PDF
# ============================================================

print()
print("Generating PDF...")


doc = SimpleDocTemplate(

    PDF_FILE,

    pagesize=A4,

    rightMargin=40,

    leftMargin=40,

    topMargin=40,

    bottomMargin=40

)


doc.build(
    story
)


# ============================================================
# FINAL MESSAGE
# ============================================================

print()
print("=" * 70)
print("REPORT GENERATED SUCCESSFULLY")
print("=" * 70)

print(
    "PDF:",
    PDF_FILE
)

print(
    "Graphs:",
    GRAPH_FOLDER
)

print(
    "Report date:",
    latest_date
)

print(
    "Data points:",
    len(report_df)
)

print("=" * 70)