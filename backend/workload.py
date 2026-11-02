import math

# -----------------------------
# Light Workload
# -----------------------------
def light_workload():

    total = 0

    for _ in range(2):
        for i in range(300000):
            total += math.sqrt(i)

    return total


# -----------------------------
# Medium Workload
# -----------------------------
def medium_workload():

    total = 0

    for _ in range(8):
        for i in range(1000000):
            total += math.sqrt(i)

    return total


# -----------------------------
# Heavy Workload
# -----------------------------
def heavy_workload():

    total = 0

    for _ in range(20):
        for i in range(3000000):
            total += math.sqrt(i)

    return total