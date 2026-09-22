"""
test_db_connection_only.py
Minimal standalone script to verify direct connectivity to Supabase PostgreSQL.
Tests a single psycopg2.connect() call and executes 'SELECT 1;'.
Masks passwords and credentials in all output.
"""

import os
import re
import sys
from pathlib import Path

# Load environment variables
try:
    from dotenv import load_dotenv
except ImportError:
    print("[ERROR] python-dotenv is not installed. Run: pip install python-dotenv")
    sys.exit(1)

try:
    import psycopg2
except ImportError:
    print("[ERROR] psycopg2 is not installed. Run: pip install psycopg2-binary")
    sys.exit(1)

BASE_DIR = Path(__file__).parent
ENV_PATH = BASE_DIR / ".env"


def mask_connection_string(uri: str) -> str:
    """Masks the password component of a PostgreSQL URI with asterisks."""
    if not uri:
        return ""
    # Matches: postgresql://username:password@host...
    return re.sub(r":([^@/]+)@", r":********@", uri)


def validate_format(uri: str):
    """Validates the PostgreSQL URI format against Supabase pooler conventions."""
    checks = []

    # 1. Scheme check
    is_postgres = uri.startswith("postgresql://") or uri.startswith("postgres://")
    checks.append(("[x]" if is_postgres else "[ ]", "Starts with postgresql://", is_postgres))

    # 2. Supabase Pooler username pattern: postgres.[project-ref]
    has_pooler_user = bool(re.search(r"//postgres\.[a-zA-Z0-9_-]+:", uri))
    checks.append(("[x]" if has_pooler_user else "[ ]", "Contains pooler username format (postgres.[project-ref])", has_pooler_user))

    # 3. Port check: 6543 (Transaction Pooler) vs 5432 (Direct)
    has_port_6543 = ":6543" in uri
    has_port_5432 = ":5432" in uri
    if has_port_6543:
        checks.append(("[x]", "Uses port 6543 (Transaction Pooler - correct for Render)", True))
    elif has_port_5432:
        checks.append(("[!]", "Uses port 5432 (Direct connection - NOT recommended for Render)", False))
    else:
        checks.append(("[ ]", "Specifies port 6543 (Transaction Pooler)", False))

    # 4. Database name check
    # Strips query params (?sslmode=...) if present
    base_uri = uri.split("?")[0]
    ends_postgres = base_uri.endswith("/postgres")
    checks.append(("[x]" if ends_postgres else "[ ]", "Targets default database (/postgres)", ends_postgres))

    return checks


def main():
    print("=" * 60)
    print("      Supabase Postgres Single-Connection Test")
    print("=" * 60)

    if not ENV_PATH.exists():
        print(f"\n[STATUS] .env file NOT FOUND at: {ENV_PATH}")
        print("Please copy .env.example to .env and insert your Supabase credentials.")
        return

    load_dotenv(ENV_PATH)
    db_url = os.environ.get("DATABASE_URL", "").strip()

    if not db_url:
        print("\n[STATUS] DATABASE_URL is EMPTY or not set in .env.")
        print("Please set DATABASE_URL in .env.")
        return

    placeholders = ["your_database_password_here", "[YOUR-PASSWORD]", "your-project-ref", "your_supabase"]
    if any(p in db_url for p in placeholders):
        print("\n[STATUS] DATABASE_URL contains placeholder text and has not been filled in yet:")
        print(f"  Current value: {mask_connection_string(db_url)}")
        print("\nPlease replace the placeholder with your actual Supabase database password and project ref.")
        return

    # Validate format
    print("\n--- Connection String Format Checklist ---")
    checklist = validate_format(db_url)
    for mark, desc, passed in checklist:
        print(f"  {mark} {desc}")

    masked = mask_connection_string(db_url)
    print(f"\nMasked Connection URI: {masked}")

    # Attempt connection
    print("\nAttempting single connection test (SELECT 1)...")
    connect_url = db_url
    if "sslmode=" not in connect_url:
        sep = "&" if "?" in connect_url else "?"
        connect_url = f"{connect_url}{sep}sslmode=require"

    conn = None
    try:
        conn = psycopg2.connect(connect_url, connect_timeout=10)
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
            val = cur.fetchone()[0]
        conn.close()
        print(f"\n[RESULT] SUCCESS! 'SELECT 1;' returned {val}.")
        print("Database is reachable and credentials are valid.")
    except Exception as e:
        # Sanitize any accidental password leak in the error string
        err_msg = str(e)
        if db_url in err_msg:
            err_msg = err_msg.replace(db_url, masked)
        print(f"\n[RESULT] CONNECTION FAILED: {err_msg}")
    finally:
        if conn and not conn.closed:
            conn.close()


if __name__ == "__main__":
    main()
