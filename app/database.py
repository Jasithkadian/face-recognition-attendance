"""
app/database.py
PostgreSQL / Supabase database layer for the Face Recognition Attendance System.

Migrated from SQLite to Supabase Postgres to survive ephemeral container restarts on Render.
Key features:
- Uses ThreadedConnectionPool for safe, thread-pooled concurrent connections
- Uses context managers (with get_connection() as conn:) so pool.putconn() is guaranteed
  to run on completion or error, with automatic rollback on unhandled exceptions
- Stores 512-d normalized face encodings as bytea (using identical pickle serialization)
- Loads SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, and DATABASE_URL via python-dotenv
- Preserves exact return dictionary signatures for zero disruption to recognizer and main.py
"""

import os
import pickle
import datetime
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
from dotenv import load_dotenv

import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import RealDictCursor

from recognizer import distance_to_confidence

# Load environment variables from .env file
BASE_DIR = Path(__file__).parent.parent
load_dotenv(BASE_DIR / ".env")

# Supabase / Postgres connection configuration
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
DATABASE_URL = os.environ.get("DATABASE_URL")

# Global connection pool instance
_connection_pool: Optional[ThreadedConnectionPool] = None


def get_pool() -> ThreadedConnectionPool:
    """Initializes or returns the thread-safe connection pool."""
    global _connection_pool
    if _connection_pool is None:
        db_url = DATABASE_URL
        if not db_url:
            raise ValueError(
                "DATABASE_URL environment variable is not configured. "
                "Please provide a Postgres connection URI in your .env file "
                "(e.g. from Supabase Dashboard -> Project Settings -> Database -> Connection string -> URI -> Transaction Pooler port 6543)."
            )
        # Ensure sslmode=require for secure cloud connections
        if "sslmode=" not in db_url:
            separator = "&" if "?" in db_url else "?"
            db_url = f"{db_url}{separator}sslmode=require"

        _connection_pool = ThreadedConnectionPool(
            minconn=1,
            maxconn=10,
            dsn=db_url,
        )
    return _connection_pool


def get_pool_status() -> Dict[str, Any]:
    """Returns the current connection pool status (used vs available connections)."""
    pool = get_pool()
    return {
        "minconn": pool.minconn,
        "maxconn": pool.maxconn,
        "active_checked_out": len(pool._used),
        "idle_available": len(pool._pool),
        "pool_closed": pool.closed,
    }


class PooledConnectionWrapper:
    """
    Wraps a pooled psycopg2 connection so that:
    - calling .close() returns the connection to the pool instead of closing the socket
    - using it as a context manager guarantees pool.putconn() runs on exit
    - unhandled exceptions trigger an automatic rollback before returning to the pool
    """
    def __init__(self, pool: ThreadedConnectionPool, conn):
        self._pool = pool
        self._conn = conn
        self._closed = False

    def cursor(self, *args, **kwargs):
        if "cursor_factory" not in kwargs:
            kwargs["cursor_factory"] = RealDictCursor
        return self._conn.cursor(*args, **kwargs)

    def commit(self):
        return self._conn.commit()

    def rollback(self):
        return self._conn.rollback()

    def close(self):
        if not self._closed and self._pool:
            try:
                if not self._conn.closed:
                    self._conn.rollback()
                self._pool.putconn(self._conn)
            except Exception as e:
                print(f"[DB Warning] Error returning connection to pool: {e}")
            finally:
                self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            try:
                self.rollback()
            except Exception:
                pass
        self.close()


def get_connection() -> PooledConnectionWrapper:
    """Acquire a pooled connection wrapped to safely return to pool on .close() or context exit."""
    pool = get_pool()
    conn = pool.getconn()
    return PooledConnectionWrapper(pool, conn)


def init_db():
    """
    Idempotent initialization: ensures required tables and indexes exist in Postgres.
    SQLite-only PRAGMA checks have been completely removed.
    """
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS employees (
                id BIGSERIAL PRIMARY KEY,
                name TEXT NOT NULL UNIQUE,
                encoding BYTEA NOT NULL,
                created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS presence_log (
                id BIGSERIAL PRIMARY KEY,
                employee_id BIGINT NOT NULL REFERENCES employees(id) ON DELETE CASCADE,
                seen_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                distance DOUBLE PRECISION
            );
        """)

        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_presence_log_employee_id ON presence_log(employee_id);
            CREATE INDEX IF NOT EXISTS idx_presence_log_seen_at ON presence_log(seen_at);
        """)

        conn.commit()


def add_employee(name: str, encoding) -> int:
    """Store or update an employee with their averaged unit-normalized face encoding."""
    enc_arr = np.array(encoding, dtype=np.float64)
    norm = np.linalg.norm(enc_arr)
    if norm > 0:
        enc_arr = enc_arr / norm
    blob = pickle.dumps(enc_arr)
    now_dt = datetime.datetime.now(datetime.timezone.utc)

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO employees (name, encoding, created_at)
            VALUES (%s, %s, %s)
            ON CONFLICT (name) DO UPDATE SET
                encoding = EXCLUDED.encoding,
                created_at = EXCLUDED.created_at
            RETURNING id;
            """,
            (name, psycopg2.Binary(blob), now_dt),
        )
        row = cur.fetchone()
        conn.commit()
        return row["id"]


def delete_employee(employee_id: int):
    """Remove an employee and all their presence records."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("DELETE FROM presence_log WHERE employee_id = %s", (employee_id,))
        cur.execute("DELETE FROM employees WHERE id = %s", (employee_id,))
        conn.commit()


def get_all_employees() -> List[Dict[str, Any]]:
    """Returns list of dicts: id, name, encoding (decoded numpy array, L2 normalized)."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, name, encoding FROM employees")
        rows = cur.fetchall()

    result = []
    for r in rows:
        try:
            raw_data = bytes(r["encoding"])
            enc = pickle.loads(raw_data)
            if isinstance(enc, np.ndarray) and enc.size in (128, 512):
                enc = enc.flatten().astype(np.float64)
                norm = np.linalg.norm(enc)
                if norm > 0:
                    enc = enc / norm
                result.append({"id": r["id"], "name": r["name"], "encoding": enc, "dim": enc.size})
        except Exception as e:
            print(f"Error loading employee encoding id={r['id']}: {e}")
    return result


def get_all_people() -> List[Dict[str, Any]]:
    """Get all enrolled people (without encodings, for the API roster)."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, name, created_at FROM employees ORDER BY name")
        rows = cur.fetchall()

    result = []
    for r in rows:
        created = r["created_at"]
        created_str = created.isoformat() if hasattr(created, "isoformat") else str(created)
        result.append({
            "id": r["id"],
            "name": r["name"],
            "created_at": created_str,
        })
    return result


def log_presence(employee_id: int, distance: Optional[float] = None, min_interval_seconds: int = 30) -> bool:
    """
    Log presence for an employee.
    If seen within min_interval_seconds (default 30s), updates seen_at on the existing record
    to keep active presence fresh while avoiding duplicate activity log spam.
    Otherwise inserts a new record.
    """
    now = datetime.datetime.now(datetime.timezone.utc)

    with get_connection() as conn:
        cur = conn.cursor()

        # Check last logged entry for this employee
        cur.execute(
            "SELECT id, seen_at FROM presence_log WHERE employee_id = %s ORDER BY id DESC LIMIT 1",
            (employee_id,)
        )
        last_row = cur.fetchone()

        if last_row:
            try:
                last_time = last_row["seen_at"]
                if not hasattr(last_time, "tzinfo") or last_time.tzinfo is None:
                    last_time = datetime.datetime.fromisoformat(str(last_time)).replace(tzinfo=datetime.timezone.utc)

                if (now - last_time).total_seconds() < min_interval_seconds:
                    cur.execute(
                        "UPDATE presence_log SET seen_at = %s, distance = %s WHERE id = %s",
                        (now, distance, last_row["id"])
                    )
                    conn.commit()
                    return True
            except Exception as e:
                print(f"Error checking last presence: {e}")

        cur.execute(
            "INSERT INTO presence_log (employee_id, seen_at, distance) VALUES (%s, %s, %s)",
            (employee_id, now, distance),
        )
        conn.commit()
        return True


def get_recently_present(window_seconds: int = 20) -> List[Dict[str, Any]]:
    """
    Employees whose face was detected within the last `window_seconds`.
    Used to populate the 'Currently Present' / 'Active Attendees' panel.
    """
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT DISTINCT ON (e.id)
                e.name AS name,
                p.seen_at AS last_seen,
                p.distance
            FROM presence_log p
            JOIN employees e ON e.id = p.employee_id
            ORDER BY e.id, p.seen_at DESC
            """
        )
        rows = cur.fetchall()

    now_utc = datetime.datetime.now(datetime.timezone.utc)
    result = []
    for r in rows:
        last_val = r["last_seen"]
        if not last_val:
            continue
        try:
            if hasattr(last_val, "tzinfo") and last_val.tzinfo is not None:
                dt = last_val
            else:
                clean_str = str(last_val)
                if not clean_str.endswith("Z") and "+" not in clean_str and "-" not in clean_str[10:]:
                    clean_str += "+00:00"
                dt = datetime.datetime.fromisoformat(clean_str)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=datetime.timezone.utc)

            elapsed = (now_utc - dt).total_seconds()
            if elapsed <= window_seconds:
                result.append({
                    "name": r["name"],
                    "last_seen": dt.isoformat(),
                    "distance": r["distance"],
                })
        except Exception as e:
            print(f"Error parsing last_seen timestamp '{last_val}': {e}")

    result.sort(key=lambda x: x["last_seen"], reverse=True)
    return result


def get_today_summary() -> List[Dict[str, Any]]:
    """
    For each employee seen today, return first-seen and last-seen times.
    """
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT e.name AS name,
                   MIN(p.seen_at) AS first_seen,
                   MAX(p.seen_at) AS last_seen,
                   COUNT(p.id) AS detections
            FROM presence_log p
            JOIN employees e ON e.id = p.employee_id
            WHERE p.seen_at >= CURRENT_DATE
              AND p.seen_at < CURRENT_DATE + INTERVAL '1 day'
            GROUP BY e.id, e.name
            ORDER BY first_seen ASC
            """
        )
        rows = cur.fetchall()

    result = []
    for r in rows:
        f_seen = r["first_seen"]
        l_seen = r["last_seen"]
        result.append({
            "name": r["name"],
            "first_seen": f_seen.isoformat() if hasattr(f_seen, "isoformat") else str(f_seen),
            "last_seen": l_seen.isoformat() if hasattr(l_seen, "isoformat") else str(l_seen),
            "detections": r["detections"],
        })
    return result


def get_activity_log(limit: int = 100, name_filter: Optional[str] = None, date_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Get detailed timestamped activity log records with face match distance and confidence scores.
    """
    query = """
        SELECT p.id, e.name AS name, p.seen_at, p.distance
        FROM presence_log p
        JOIN employees e ON e.id = p.employee_id
        WHERE 1=1
    """
    params = []

    if name_filter:
        query += " AND LOWER(e.name) LIKE %s"
        params.append(f"%{name_filter.lower()}%")

    if date_filter:
        query += " AND p.seen_at::text LIKE %s"
        params.append(f"{date_filter}%")

    query += " ORDER BY p.id DESC LIMIT %s"
    params.append(limit)

    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute(query, tuple(params))
        rows = cur.fetchall()

    result = []
    for r in rows:
        dist = r["distance"]
        conf = distance_to_confidence(dist) if dist is not None else None
        seen = r["seen_at"]
        seen_str = seen.isoformat() if hasattr(seen, "isoformat") else str(seen)
        result.append({
            "id": r["id"],
            "name": r["name"],
            "seen_at": seen_str,
            "distance": round(dist, 4) if dist is not None else None,
            "confidence": conf,
        })
    return result
