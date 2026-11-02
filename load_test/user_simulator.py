import random
import time
import requests
import load_test.config as config

from load_test.config import BASE_URL, USNS


# ============================================================
# REQUEST CONFIGURATION
# ============================================================

REQUEST_TIMEOUT = 30

# Keep the load-generation model the same,
# but prevent every user from hammering the API continuously.
MIN_REQUEST_DELAY = 0.5
MAX_REQUEST_DELAY = 1.0


# ============================================================
# SIMULATE ONE STUDENT
# ============================================================

def simulate_student(student_id):

    while True:

        try:

            # ------------------------------------------------
            # Select API according to existing workload model
            # ------------------------------------------------

            action = random.choices(
                population=[
                    "result",
                    "branch",
                    "college"
                ],
                weights=[
                    70,
                    20,
                    10
                ],
                k=1
            )[0]


            # =================================================
            # RESULT API
            # =================================================

            if action == "result":

                usn = random.choice(USNS)

                config.API_COUNTER["/result"] += 1
                config.MOST_REQUESTED_API = "/result"

                response = requests.get(
                    f"{BASE_URL}/result/{usn}",
                    timeout=REQUEST_TIMEOUT
                )

                response.raise_for_status()


            # =================================================
            # BRANCH RANK API
            # =================================================

            elif action == "branch":

                config.API_COUNTER["/branch-ranks"] += 1
                config.MOST_REQUESTED_API = "/branch-ranks"

                response = requests.get(
                    f"{BASE_URL}/branch-ranks",
                    timeout=REQUEST_TIMEOUT
                )

                response.raise_for_status()


            # =================================================
            # COLLEGE RANK API
            # =================================================

            else:

                config.API_COUNTER["/college-ranks"] += 1
                config.MOST_REQUESTED_API = "/college-ranks"

                response = requests.get(
                    f"{BASE_URL}/college-ranks",
                    timeout=REQUEST_TIMEOUT
                )

                response.raise_for_status()


            # ------------------------------------------------
            # REALISTIC THINKING TIME
            # ------------------------------------------------

            time.sleep(
                random.uniform(
                    MIN_REQUEST_DELAY,
                    MAX_REQUEST_DELAY
                )
            )


        # ====================================================
        # TIMEOUT
        # ====================================================

        except requests.exceptions.Timeout:

            print(
                f"[TIMEOUT] Student {student_id}: "
                f"request exceeded {REQUEST_TIMEOUT}s"
            )

            # Don't immediately retry.
            # Give Kubernetes time to respond/scale.
            time.sleep(2)


        # ====================================================
        # CONNECTION ERROR
        # ====================================================

        except requests.exceptions.ConnectionError as e:

            print(
                f"[CONNECTION ERROR] "
                f"Student {student_id}: {e}"
            )

            time.sleep(2)


        # ====================================================
        # HTTP ERROR
        # ====================================================

        except requests.exceptions.HTTPError as e:

            print(
                f"[HTTP ERROR] "
                f"Student {student_id}: {e}"
            )

            time.sleep(1)


        # ====================================================
        # OTHER ERROR
        # ====================================================

        except Exception as e:

            print(
                f"[REQUEST ERROR] "
                f"Student {student_id}: {e}"
            )

            time.sleep(1)