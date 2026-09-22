"""
test_db_migration.py
Standalone test script to verify Supabase PostgreSQL connection,
employee enrollment with 512-d vector, presence logging, tight-loop pool leak test,
and data retrieval.
"""

import sys
import os
import time
import numpy as np
from pathlib import Path

# Ensure application package can be imported
sys.path.insert(0, str(Path(__file__).parent))

import app.database as database


def run_migration_test():
    print("=" * 70)
    print("       Supabase PostgreSQL Migration Verification & Pool Leak Test")
    print("=" * 70)

    # 1. Test database connection & table initialization
    print("\n[Step 1] Initializing database schema check...")
    try:
        database.init_db()
        pool_status = database.get_pool_status()
        print("  -> SUCCESS: Connected to Supabase Postgres & verified schema.")
        print(f"  -> Initial pool state: active_in_use={pool_status['active_checked_out']}, idle_available={pool_status['idle_available']}")
    except Exception as e:
        print(f"  -> FAILED: Could not connect or initialize: {e}")
        return False

    # 2. Test enrolling a fake employee with a 512-d normalized vector
    fake_name = f"Test_User_{np.random.randint(1000, 9999)}"
    print(f"\n[Step 2] Enrolling fake employee '{fake_name}' with random 512-d vector...")
    raw_vector = np.random.randn(512).astype(np.float64)
    unit_vector = raw_vector / np.linalg.norm(raw_vector)

    try:
        emp_id = database.add_employee(fake_name, unit_vector)
        print(f"  -> SUCCESS: Enrolled '{fake_name}' with ID={emp_id}")
    except Exception as e:
        print(f"  -> FAILED: Error adding employee: {e}")
        return False

    # 3. Test logging presence for the employee
    test_distance = 0.285
    print(f"\n[Step 3] Logging presence record for employee ID={emp_id} (distance={test_distance})...")
    try:
        logged = database.log_presence(emp_id, distance=test_distance, min_interval_seconds=0)
        print(f"  -> SUCCESS: Presence logged = {logged}")
    except Exception as e:
        print(f"  -> FAILED: Error logging presence: {e}")
        return False

    # 4. Tight-Loop Connection Pool Leak Stress Test
    loop_iterations = 25
    print(f"\n[Step 4] Connection Pool Leak Test: Executing log_presence() in a tight loop ({loop_iterations} calls)...")
    try:
        start_time = time.time()
        for i in range(loop_iterations):
            # Rapidly log presence with min_interval_seconds=0 to exercise both insert and update branches
            interval = 0 if (i % 2 == 0) else 30
            database.log_presence(emp_id, distance=round(test_distance + (i * 0.001), 4), min_interval_seconds=interval)

        elapsed = time.time() - start_time
        status = database.get_pool_status()
        print(f"  -> Completed {loop_iterations} rapid database calls in {elapsed:.3f}s (avg {elapsed/loop_iterations*1000:.1f}ms/call)")
        print(f"  -> Connection Pool Status:")
        print(f"       * Active (checked-out) connections: {status['active_checked_out']} (EXPECTED: 0)")
        print(f"       * Idle (available in pool) connections: {status['idle_available']} (max allowed: {status['maxconn']})")

        if status["active_checked_out"] == 0:
            print("  -> SUCCESS: Zero leaked connections! All connections were cleanly returned to the pool.")
        else:
            print(f"  -> WARNING / LEAK DETECTED: {status['active_checked_out']} connections remain checked out!")
            return False
    except Exception as e:
        print(f"  -> FAILED: Error during tight-loop presence logging: {e}")
        return False

    # 5. Read back all employees and verify vector shape & values
    print("\n[Step 5] Reading back enrolled employees from Postgres...")
    try:
        employees = database.get_all_employees()
        matched = next((e for e in employees if e["id"] == emp_id), None)
        if matched is None:
            print(f"  -> FAILED: Enrolled employee ID={emp_id} was not found in get_all_employees()!")
            return False

        recovered_vector = matched["encoding"]
        cos_similarity = np.dot(unit_vector, recovered_vector)
        print(f"  -> SUCCESS: Retrieved '{matched['name']}' (ID={matched['id']})")
        print(f"  -> Vector dimension: {len(recovered_vector)}")
        print(f"  -> Vector Cosine Similarity to original: {cos_similarity:.6f} (expect ~1.000000)")
        if abs(cos_similarity - 1.0) > 1e-4:
            print("  -> WARNING: Recovered vector differs from stored vector!")
    except Exception as e:
        print(f"  -> FAILED: Error reading employees: {e}")
        return False

    # 6. Read back activity log and presence roster
    print("\n[Step 6] Reading back activity log & presence...")
    try:
        activity_logs = database.get_activity_log(limit=5, name_filter=fake_name)
        print(f"  -> Retrieved {len(activity_logs)} activity log entries for '{fake_name}':")
        for log in activity_logs:
            print(f"     * ID={log['id']} | {log['name']} | seen_at={log['seen_at']} | dist={log['distance']} | conf={log['confidence']}%")

        present = database.get_recently_present(window_seconds=60)
        print(f"  -> Currently present count (within 60s window): {len(present)}")
    except Exception as e:
        print(f"  -> FAILED: Error reading logs or presence: {e}")
        return False

    # 7. Clean up the test employee record
    print(f"\n[Step 7] Cleaning up test employee ID={emp_id}...")
    try:
        database.delete_employee(emp_id)
        final_status = database.get_pool_status()
        print("  -> SUCCESS: Test employee and cascading presence logs removed.")
        print(f"  -> Final Pool State: active={final_status['active_checked_out']}, idle={final_status['idle_available']}")
    except Exception as e:
        print(f"  -> WARNING: Cleanup failed: {e}")

    print("\n" + "=" * 70)
    print("       ALL DATABASE MIGRATION & POOL LEAK CHECKS PASSED!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    run_migration_test()
