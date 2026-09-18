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
import numpy as np

from recognizer import distance_to_confidence


# Use DATA_DIR env var for persistent disk mount on Render, fallback to ./data
DATA_DIR = os.environ.get("DATA_DIR", str(Path(__file__).parent.parent / "data"))
DB_PATH = os.path.join(DATA_DIR, "office.db")
 
 
def get_connection():
    """Get a database connection, creating the data directory if needed."""
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=20.0, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 20000")
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def init_db():
    """Initialize the database tables if they don't exist and run schema migrations."""
    conn = get_connection()
    try:
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
                distance REAL,
                FOREIGN KEY (employee_id) REFERENCES employees (id)
            )
        """)
        conn.commit()

        # DB Schema Migration: Ensure 'distance' column exists in presence_log table
        cur.execute("PRAGMA table_info(presence_log)")
        columns = [row["name"] for row in cur.fetchall()]
        if "distance" not in columns:
            cur.execute("ALTER TABLE presence_log ADD COLUMN distance REAL")
            conn.commit()
    finally:
        conn.close()


def add_employee(name: str, encoding) -> int:
    """Store a new employee with their averaged face encoding."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        # Ensure encoding is a float64 numpy array normalized to unit length
        enc_arr = np.array(encoding, dtype=np.float64)
        norm = np.linalg.norm(enc_arr)
        if norm > 0:
            enc_arr = enc_arr / norm
        blob = pickle.dumps(enc_arr)
        cur.execute(
            "INSERT INTO employees (name, encoding, created_at) VALUES (?, ?, ?)",
            (name, blob, datetime.datetime.now().isoformat()),
        )
        conn.commit()
        employee_id = cur.lastrowid
        return employee_id
    finally:
        conn.close()


def delete_employee(employee_id: int):
    """Remove an employee and all their presence records."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM employees WHERE id = ?", (employee_id,))
        cur.execute("DELETE FROM presence_log WHERE employee_id = ?", (employee_id,))
        conn.commit()
    finally:
        conn.close()


def get_all_employees():
    """Returns list of dicts: id, name, encoding (decoded numpy array, L2 normalized)."""
    conn = get_connection()
    cur = conn.cursor()
    rows = cur.execute("SELECT id, name, encoding FROM employees").fetchall()
    conn.close()
    result = []
    for r in rows:
        try:
            enc = pickle.loads(r["encoding"])
            if isinstance(enc, np.ndarray) and enc.size == 128:
                enc = enc.flatten().astype(np.float64)
                norm = np.linalg.norm(enc)
                if norm > 0:
                    enc = enc / norm
                result.append({"id": r["id"], "name": r["name"], "encoding": enc})
        except Exception as e:
            print(f"Error loading employee encoding id={r['id']}: {e}")
    return result


def get_all_people():
    """Get all enrolled people (without encodings, for the API)."""
    conn = get_connection()
    cur = conn.cursor()
    rows = cur.execute("SELECT id, name, created_at FROM employees ORDER BY name").fetchall()
    conn.close()
    return [dict(r) for r in rows]


def log_presence(employee_id: int, distance: float = None, min_interval_seconds: int = 30) -> bool:
    """
    Log presence for an employee.
    If seen within min_interval_seconds (default 30s), updates seen_at on the existing record
    to keep active presence fresh while avoiding duplicate activity log spam.
    Otherwise inserts a new record.
    """
    conn = get_connection()
    cur = conn.cursor()

    now = datetime.datetime.now(datetime.timezone.utc)
    now_str = now.isoformat()

    # Check last logged entry for this employee
    cur.execute(
        "SELECT id, seen_at FROM presence_log WHERE employee_id = ? ORDER BY id DESC LIMIT 1",
        (employee_id,)
    )
    last_row = cur.fetchone()

    if last_row:
        try:
            last_str = last_row["seen_at"]
            if not last_str.endswith("Z") and "+" not in last_str and "-" not in last_str[10:]:
                last_str += "+00:00"
            last_time = datetime.datetime.fromisoformat(last_str)
            if last_time.tzinfo is None:
                last_time = last_time.replace(tzinfo=datetime.timezone.utc)

            if (now - last_time).total_seconds() < min_interval_seconds:
                # Update timestamp on latest row to keep presence active without creating duplicate log clutter
                cur.execute(
                    "UPDATE presence_log SET seen_at = ?, distance = ? WHERE id = ?",
                    (now_str, distance, last_row["id"])
                )
                conn.commit()
                conn.close()
                return True
        except Exception as e:
            print(f"Error checking last presence: {e}")

    cur.execute(
        "INSERT INTO presence_log (employee_id, seen_at, distance) VALUES (?, ?, ?)",
        (employee_id, now_str, distance),
    )
    conn.commit()
    conn.close()
    return True


def get_recently_present(window_seconds: int = 20):
    """
    Employees whose face was detected within the last `window_seconds`.
    Used to populate the "Currently Present" / "Active Attendees" panel.
    Safely calculates elapsed time in Python to avoid SQLite string timezone issues.
    """
    conn = get_connection()
    cur = conn.cursor()
    now_utc = datetime.datetime.now(datetime.timezone.utc)

    rows = cur.execute(
        """
        SELECT e.name AS name, MAX(p.seen_at) AS last_seen, p.distance
        FROM presence_log p
        JOIN employees e ON e.id = p.employee_id
        GROUP BY e.id
        ORDER BY last_seen DESC
        """
    ).fetchall()
    conn.close()

    result = []
    for r in rows:
        last_str = r["last_seen"]
        if not last_str:
            continue
        try:
            clean_str = last_str
            if not clean_str.endswith("Z") and "+" not in clean_str and "-" not in clean_str[10:]:
                clean_str += "+00:00"
            dt = datetime.datetime.fromisoformat(clean_str)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=datetime.timezone.utc)

            elapsed = (now_utc - dt).total_seconds()
            if elapsed <= window_seconds:
                result.append({
                    "name": r["name"],
                    "last_seen": r["last_seen"],
                    "distance": r["distance"],
                })
        except Exception as e:
            print(f"Error parsing last_seen timestamp '{last_str}': {e}")

    return result


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


def get_activity_log(limit: int = 100, name_filter: str = None, date_filter: str = None):
    """
    Get detailed timestamped activity log records with face match distance and confidence scores using canonical formula.
    """
    conn = get_connection()
    cur = conn.cursor()

    query = """
        SELECT p.id, e.name AS name, p.seen_at, p.distance
        FROM presence_log p
        JOIN employees e ON e.id = p.employee_id
        WHERE 1=1
    """
    params = []

    if name_filter:
        query += " AND LOWER(e.name) LIKE ?"
        params.append(f"%{name_filter.lower()}%")

    if date_filter:
        query += " AND p.seen_at LIKE ?"
        params.append(f"{date_filter}%")

    query += " ORDER BY p.id DESC LIMIT ?"
    params.append(limit)

    rows = cur.execute(query, params).fetchall()
    conn.close()

    result = []
    for r in rows:
        dist = r["distance"]
        conf = distance_to_confidence(dist) if dist is not None else None
        result.append({
            "id": r["id"],
            "name": r["name"],
            "seen_at": r["seen_at"],
            "distance": round(dist, 4) if dist is not None else None,
            "confidence": conf,
        })
    return result


