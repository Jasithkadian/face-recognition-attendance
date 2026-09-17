"""
app/database.py
Web-adapted database layer for the Face Recognition Attendance System.

Key changes from the original database.py:
- Uses DATA_DIR environment variable for Render persistent disk support
- Uses pickle for face encodings (matching original schema)
- Returns dicts for easy JSON serialization
- WAL journal mode for better concurrency under web requests
"""

import sqlite3
import pickle
import datetime
import os
from pathlib import Path

# Use DATA_DIR env var for persistent disk mount on Render, fallback to ./data
DATA_DIR = os.environ.get("DATA_DIR", str(Path(__file__).parent.parent / "data"))
DB_PATH = os.path.join(DATA_DIR, "office.db")


def get_connection():
    """Get a database connection, creating the data directory if needed."""
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    """Initialize the database tables if they don't exist."""
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            encoding BLOB NOT NULL,
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS presence_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id INTEGER NOT NULL,
            seen_at TEXT NOT NULL,
            FOREIGN KEY (employee_id) REFERENCES employees (id)
        )
    """)

    conn.commit()
    conn.close()


def add_employee(name: str, encoding) -> int:
    """Store a new employee with their face encoding."""
    conn = get_connection()
    cur = conn.cursor()
    blob = pickle.dumps(encoding)
    cur.execute(
        "INSERT INTO employees (name, encoding, created_at) VALUES (?, ?, ?)",
        (name, blob, datetime.datetime.now().isoformat()),
    )
    conn.commit()
    employee_id = cur.lastrowid
    conn.close()
    return employee_id


def delete_employee(employee_id: int):
    """Remove an employee and all their presence records."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM employees WHERE id = ?", (employee_id,))
    cur.execute("DELETE FROM presence_log WHERE employee_id = ?", (employee_id,))
    conn.commit()
    conn.close()


def get_all_employees():
    """Returns list of dicts: id, name, encoding (decoded numpy array)."""
    conn = get_connection()
    cur = conn.cursor()
    rows = cur.execute("SELECT id, name, encoding FROM employees").fetchall()
    conn.close()
    return [
        {"id": r["id"], "name": r["name"], "encoding": pickle.loads(r["encoding"])}
        for r in rows
    ]


def get_all_people():
    """Get all enrolled people (without encodings, for the API)."""
    conn = get_connection()
    cur = conn.cursor()
    rows = cur.execute("SELECT id, name FROM employees ORDER BY name").fetchall()
    conn.close()
    return [{"id": r["id"], "name": r["name"]} for r in rows]


def log_presence(employee_id: int, timestamp: datetime.datetime = None):
    """Record that an employee's face was seen right now."""
    conn = get_connection()
    cur = conn.cursor()
    ts = (timestamp or datetime.datetime.now()).isoformat()
    cur.execute(
        "INSERT INTO presence_log (employee_id, seen_at) VALUES (?, ?)",
        (employee_id, ts),
    )
    conn.commit()
    conn.close()


def get_recently_present(window_seconds: int = 15):
    """
    Employees whose face was detected within the last `window_seconds`.
    Used to populate the "Currently Present" panel.
    """
    conn = get_connection()
    cur = conn.cursor()
    cutoff = (
        datetime.datetime.now() - datetime.timedelta(seconds=window_seconds)
    ).isoformat()
    rows = cur.execute(
        """
        SELECT e.name AS name, MAX(p.seen_at) AS last_seen
        FROM presence_log p
        JOIN employees e ON e.id = p.employee_id
        WHERE p.seen_at >= ?
        GROUP BY e.id
        ORDER BY last_seen DESC
        """,
        (cutoff,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_today_summary():
    """
    For each employee seen today, return first-seen and last-seen times.
    """
    conn = get_connection()
    cur = conn.cursor()
    today = datetime.date.today().isoformat()
    rows = cur.execute(
        """
        SELECT e.name AS name,
               MIN(p.seen_at) AS first_seen,
               MAX(p.seen_at) AS last_seen,
               COUNT(p.id) AS detections
        FROM presence_log p
        JOIN employees e ON e.id = p.employee_id
        WHERE p.seen_at LIKE ?
        GROUP BY e.id
        ORDER BY first_seen ASC
        """,
        (f"{today}%",),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_attendance_by_date(date_filter: str):
    """Get attendance/presence records for a specific date."""
    conn = get_connection()
    cur = conn.cursor()
    rows = cur.execute(
        """
        SELECT e.name AS name, p.seen_at AS time
        FROM presence_log p
        JOIN employees e ON e.id = p.employee_id
        WHERE p.seen_at LIKE ?
        ORDER BY p.seen_at DESC
        """,
        (f"{date_filter}%",),
    ).fetchall()
    conn.close()
    return [{"name": r["name"], "time": r["time"]} for r in rows]
