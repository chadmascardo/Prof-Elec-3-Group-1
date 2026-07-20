"""Create and populate the application's local SQLite database."""

import csv
import sqlite3
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DATABASE_PATH = DATA_DIR / "student_records.db"
SOURCE_DIR = ROOT / "templates" / "data"


def read_csv(name: str) -> list[dict[str, str]]:
    with (SOURCE_DIR / name).open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def create_database() -> None:
    DATA_DIR.mkdir(exist_ok=True)

    with sqlite3.connect(DATABASE_PATH) as connection:
        connection.execute("PRAGMA foreign_keys = ON")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS students (
                studentID TEXT PRIMARY KEY,
                lastname TEXT NOT NULL,
                first_name TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS course_schedule (
                courseCode TEXT PRIMARY KEY,
                days TEXT NOT NULL,
                time TEXT NOT NULL,
                room TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS course_student_enrol (
                studentID TEXT NOT NULL,
                courseCode TEXT NOT NULL,
                PRIMARY KEY (studentID, courseCode),
                FOREIGN KEY (studentID) REFERENCES students(studentID),
                FOREIGN KEY (courseCode) REFERENCES course_schedule(courseCode)
            );
            """
        )

        connection.executemany(
            "INSERT OR IGNORE INTO students (studentID, lastname, first_name) "
            "VALUES (:studentID, :lastname, :first_name)",
            [
                {
                    "studentID": row["studentID"],
                    "lastname": row["lastname"],
                    "first_name": row["first name"],
                }
                for row in read_csv("students.csv")
            ],
        )
        connection.executemany(
            "INSERT OR IGNORE INTO course_schedule (courseCode, days, time, room) "
            "VALUES (:courseCode, :days, :time, :room)",
            read_csv("schedule.csv"),
        )
        connection.executemany(
            "INSERT OR IGNORE INTO course_student_enrol (studentID, courseCode) "
            "VALUES (:studentID, :courseCode)",
            read_csv("enrollment.csv"),
        )

    counts = {
        table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in ("students", "course_schedule", "course_student_enrol")
    }
    print(f"Database ready: {DATABASE_PATH}")
    print(f"Records: {counts}")


if __name__ == "__main__":
    create_database()
