"""
database.py
Handles all persistence: employee records (with face encodings) and
a rolling presence/attendance log, stored in a local SQLite file.
"""

import sqlite3
import pickle
import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent / "data" / "office.db"
DB_PATH.parent.mkdir(exist_ok=True)


def get_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
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
    """Store a new employee with their averaged 128-d face encoding."""
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
    """Get all enrolled people (without encodings)."""
    conn = get_connection()
    cur = conn.cursor()
    rows = cur.execute("SELECT id, name, created_at FROM employees ORDER BY name").fetchall()
    conn.close()
    return [dict(r) for r in rows]



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


def get_today_summary():
    """
    For each employee seen today, return first-seen and last-seen times.
    Used for the dashboard's daily log table.
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


def get_recently_present(window_seconds: int = 8):
    """
    Employees whose face was detected within the last `window_seconds`.
    Used to populate the "Currently in view" live list.
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
