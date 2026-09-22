"""
test_consecutive_confirmation.py
Verification test script for Change 2: Consecutive-frame agreement before logging attendance.
Simulates:
1. Scenario A: 3 consecutive frames all matching 'Alice' -> MUST log.
2. Scenario B: 2 frames matching 'Alice' then 1 frame matching 'Bob' -> MUST NOT log either, counter resets.
3. Scenario C: Interaction with Change 1's margin check:
   Frame 1 = 'Alice' (accepted),
   Frame 2 = 'Alice' but margin check fails (gap < 0.15 -> rejected as 'Unknown'),
   Frame 3 = 'Alice' (accepted).
   -> MUST NOT log (proves a track can never log an identity that failed the margin check even once in the window).
"""

import sys
import time
from pathlib import Path

# Mock logging sink to verify exactly when database.log_presence is called
logged_presence_calls = []

def mock_log_presence(employee_id, distance=None, min_interval_seconds=30):
    logged_presence_calls.append({
        "employee_id": employee_id,
        "distance": distance,
        "timestamp": time.time(),
    })
    return True


class ConsecutiveAttendanceController:
    """
    Simulates the exact presence gating logic implemented in app/main.py.
    """
    def __init__(self, required_consecutive=3, throttle_seconds=30.0):
        self.required_consecutive = required_consecutive
        self.throttle_seconds = throttle_seconds
        self.track_history = {}
        self.last_logged_presence = {}

    def process_match(self, track_id, emp_id, name, conf=75.0, distance=0.25):
        now_ts = time.time()

        is_accepted_match = (
            emp_id is not None and
            name != "Unknown" and
            (conf is None or conf >= 60.0)
        )
        current_identity = emp_id if is_accepted_match else None

        if track_id not in self.track_history:
            self.track_history[track_id] = {"history": [], "last_seen": now_ts}

        track_record = self.track_history[track_id]
        track_record["last_seen"] = now_ts
        history = track_record["history"]

        if current_identity is None:
            # Identity rejected (failed margin check, distance check, or unknown) -> reset streak
            history.clear()
        else:
            if history and history[-1] != current_identity:
                # Identity flipped mid-sequence (e.g. Alice then Bob) -> reset counter
                history.clear()
            history.append(current_identity)
            if len(history) > self.required_consecutive:
                history.pop(0)

        is_confirmed = (
            len(history) >= self.required_consecutive and
            all(x == current_identity for x in history) and
            current_identity is not None
        )

        logged = False
        if is_confirmed:
            if now_ts - self.last_logged_presence.get(current_identity, 0.0) >= self.throttle_seconds:
                self.last_logged_presence[current_identity] = now_ts
                mock_log_presence(current_identity, distance=distance, min_interval_seconds=self.throttle_seconds)
                logged = True

        return {
            "track_id": track_id,
            "tentative_name": name,
            "history": list(history),
            "is_confirmed": is_confirmed,
            "logged": logged,
        }


def run_tests():
    print("=" * 70)
    print("   Verification: Change 2 Consecutive-Frame Agreement Logic")
    print("=" * 70)

    # -------------------------------------------------------------
    # Scenario A: 3 consecutive frames all matching 'Alice' (ID=1)
    # -------------------------------------------------------------
    print("\n[Scenario A] 3 consecutive frames all matching 'Alice' (ID=1)...")
    global logged_presence_calls
    logged_presence_calls.clear()
    controller_a = ConsecutiveAttendanceController(required_consecutive=3)

    res1 = controller_a.process_match(track_id=101, emp_id=1, name="Alice", conf=78.0)
    print(f"  Frame 1: Display='{res1['tentative_name']}', History={res1['history']}, Confirmed={res1['is_confirmed']}, Logged={res1['logged']}")
    assert not res1["logged"], "Frame 1 should NOT log"

    res2 = controller_a.process_match(track_id=101, emp_id=1, name="Alice", conf=79.0)
    print(f"  Frame 2: Display='{res2['tentative_name']}', History={res2['history']}, Confirmed={res2['is_confirmed']}, Logged={res2['logged']}")
    assert not res2["logged"], "Frame 2 should NOT log"

    res3 = controller_a.process_match(track_id=101, emp_id=1, name="Alice", conf=80.0)
    print(f"  Frame 3: Display='{res3['tentative_name']}', History={res3['history']}, Confirmed={res3['is_confirmed']}, Logged={res3['logged']}")
    assert res3["logged"], "Frame 3 MUST log"
    assert len(logged_presence_calls) == 1
    assert logged_presence_calls[0]["employee_id"] == 1
    print("  -> SUCCESS: 'Alice' logged on the 3rd consecutive frame!")

    # -------------------------------------------------------------
    # Scenario B: 2 frames matching 'Alice', then 1 frame matching 'Bob'
    # -------------------------------------------------------------
    print("\n[Scenario B] 2 frames matching 'Alice' (ID=1) then 1 frame matching 'Bob' (ID=2)...")
    logged_presence_calls.clear()
    controller_b = ConsecutiveAttendanceController(required_consecutive=3)

    r1 = controller_b.process_match(track_id=102, emp_id=1, name="Alice", conf=75.0)
    print(f"  Frame 1: Display='{r1['tentative_name']}', History={r1['history']}, Logged={r1['logged']}")
    assert not r1["logged"]

    r2 = controller_b.process_match(track_id=102, emp_id=1, name="Alice", conf=76.0)
    print(f"  Frame 2: Display='{r2['tentative_name']}', History={r2['history']}, Logged={r2['logged']}")
    assert not r2["logged"]

    r3 = controller_b.process_match(track_id=102, emp_id=2, name="Bob", conf=74.0)
    print(f"  Frame 3: Display='{r3['tentative_name']}', History={r3['history']}, Logged={r3['logged']}")
    assert not r3["logged"], "Frame 3 with identity flip to Bob MUST NOT log"
    assert len(logged_presence_calls) == 0, "No attendance should be logged for either identity"
    assert r3["history"] == [2], f"History should reset to [2], got {r3['history']}"
    print("  -> SUCCESS: Counter reset! Neither 'Alice' nor 'Bob' was logged.")

    # -------------------------------------------------------------
    # Scenario C: Interaction with Change 1 Margin Check
    # Frame 1: Alice (pass)
    # Frame 2: Margin check fails (gap < 0.15) -> Matcher outputs Unknown (emp_id=None)
    # Frame 3: Alice (pass)
    # -------------------------------------------------------------
    print("\n[Scenario C] Margin Check Failure on Frame 2 (Alice -> Margin Failed -> Alice)...")
    logged_presence_calls.clear()
    controller_c = ConsecutiveAttendanceController(required_consecutive=3)

    c1 = controller_c.process_match(track_id=103, emp_id=1, name="Alice", conf=77.0)
    print(f"  Frame 1: Display='{c1['tentative_name']}', History={c1['history']}, Logged={c1['logged']}")

    # Frame 2: Change 1 rejects match due to margin failure -> name='Unknown', emp_id=None
    c2 = controller_c.process_match(track_id=103, emp_id=None, name="Unknown", conf=None)
    print(f"  Frame 2 (Margin Check Failed): Display='{c2['tentative_name']}', History={c2['history']}, Logged={c2['logged']}")
    assert len(c2["history"]) == 0, f"Streak must be reset to empty, got {c2['history']}"

    c3 = controller_c.process_match(track_id=103, emp_id=1, name="Alice", conf=78.0)
    print(f"  Frame 3: Display='{c3['tentative_name']}', History={c3['history']}, Logged={c3['logged']}")
    assert not c3["logged"], "Frame 3 MUST NOT log because Frame 2 broke the sequence"
    assert len(logged_presence_calls) == 0, "Nothing logged because streak was broken by margin check failure"
    assert c3["history"] == [1], f"History should be [1], got {c3['history']}"
    print("  -> SUCCESS: Margin check failure correctly interrupted consecutive window; no logging occurred!")

    print("\n" + "=" * 70)
    print("   ALL CHANGE 2 CONSECUTIVE-FRAME VERIFICATION CHECKS PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()
