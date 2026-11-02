from faker import Faker
import psycopg2
import random

fake = Faker()

conn = psycopg2.connect(
    host="localhost",
    database="university",
    user="postgres",
    password="nithya"
)

cur = conn.cursor()

branches = [
    "CSE",
    "ISE",
    "ECE",
    "MECH"
]

colleges = [
    "BIT",
    "RVCE",
    "BMSCE",
    "MSRIT"
]

subjects = [
    "Math",
    "DBMS",
    "OS",
    "CN",
    "AI"
]

years = ["23","24","25","26"]

for year in years:

    for i in range(1,501):

        usn = f"1MG{year}IS{i:03}"

        cur.execute("""

            INSERT INTO students(
                usn,
                name,
                branch,
                college
            )

            VALUES(%s,%s,%s,%s)

            RETURNING id

        """,(

            usn,
            fake.name(),
            random.choice(branches),
            random.choice(colleges)

        ))

        student_id = cur.fetchone()[0]

        for sub in subjects:

            cur.execute("""

                INSERT INTO marks(
                    student_id,
                    subject,
                    marks
                )

                VALUES(%s,%s,%s)

            """,(

                student_id,
                sub,
                random.randint(35,100)

            ))

conn.commit()

print("Dataset Created Successfully")