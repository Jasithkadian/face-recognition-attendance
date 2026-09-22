"""
test_real_integration.py
Real end-to-end integration test exercising the live FastAPI recognize_frame() endpoint
with actual image frames, real YOLOv8-Face detection, ArcFace embedding, tracking,
and Supabase Postgres presence logging.
"""

import sys
import os
import cv2
import base64
from pathlib import Path

# Ensure application package can be imported
sys.path.insert(0, str(Path(__file__).parent))

import app.database as database
from app.main import (
    recognize_frame,
    RecognizeRequest,
    get_recognizer,
    _track_history,
    _last_logged_presence,
    REQUIRED_CONSECUTIVE_MATCHES,
)
from recognizer import compute_encoding_from_bgr


def run_real_integration_test():
    print("=" * 70)
    print("   Real Integration Test: Change 2 Consecutive-Frame Agreement")
    print("=" * 70)

    image_path = r"C:\Users\Umang Kadian\Desktop\me.jpg"
    if not os.path.exists(image_path):
        print(f"[ERROR] Test image not found at: {image_path}")
        return False

    # 1. Read real image and create Base64 Data URL
    img_bgr = cv2.imread(image_path)
    _, buffer = cv2.imencode(".jpg", img_bgr)
    b64_str = base64.b64encode(buffer).decode("utf-8")
    b64_data_url = f"data:image/jpeg;base64,{b64_str}"

    # 2. Extract real 512-d embedding and enroll a temporary test employee
    print("\n[Step 1] Enrolling real face embedding in Supabase Postgres...")
    real_encoding = compute_encoding_from_bgr(img_bgr)
    if real_encoding is None:
        print("[ERROR] Could not extract embedding from test image.")
        return False

    test_employee_name = "RealTestPerson_999"
    emp_id = database.add_employee(test_employee_name, real_encoding)
    print(f"  -> Enrolled '{test_employee_name}' with ID={emp_id}")

    # Reload recognizer known faces cache from Supabase Postgres
    recognizer = get_recognizer()
    recognizer.refresh_known_faces()
    print(f"  -> Recognizer refreshed with {len(recognizer.known_ids)} enrolled identities.")

    # Reset any prior in-memory tracking / throttle state
    _track_history.clear()
    _last_logged_presence.clear()

    try:
        # 3. Verify initial presence log count for this employee
        init_logs = database.get_activity_log(name_filter=test_employee_name)
        print(f"\n[Step 2] Initial presence log count: {len(init_logs)} (expect 0)")
        assert len(init_logs) == 0

        req = RecognizeRequest(image=b64_data_url)

        # 4. Feed Real Frame 1
        print("\n--- Feeding Real Frame 1 ---")
        res1 = recognize_frame(req)
        m1 = res1["matches"][0]
        track_id = m1["track_id"]
        logs1 = database.get_activity_log(name_filter=test_employee_name)
        track_data1 = _track_history.get(track_id, {})
        print(f"  Result 1: track_id={track_id}, name='{m1['name']}', conf={m1['confidence']}%, history={track_data1.get('history')}")
        print(f"  Presence logs in DB: {len(logs1)}")
        assert len(logs1) == 0, "Frame 1 must NOT write to presence_log"
        assert m1["name"] == test_employee_name, "Tentative name should be displayed immediately on UI"

        # 5. Feed Real Frame 2
        print("\n--- Feeding Real Frame 2 ---")
        res2 = recognize_frame(req)
        m2 = res2["matches"][0]
        logs2 = database.get_activity_log(name_filter=test_employee_name)
        track_data2 = _track_history.get(track_id, {})
        print(f"  Result 2: track_id={m2['track_id']}, name='{m2['name']}', conf={m2['confidence']}%, history={track_data2.get('history')}")
        print(f"  Presence logs in DB: {len(logs2)}")
        assert len(logs2) == 0, "Frame 2 must NOT write to presence_log"

        # 6. Feed Real Frame 3 (Should trigger database write!)
        print("\n--- Feeding Real Frame 3 (Consecutive Confirmation Threshold Reached) ---")
        res3 = recognize_frame(req)
        m3 = res3["matches"][0]
        logs3 = database.get_activity_log(name_filter=test_employee_name)
        track_data3 = _track_history.get(track_id, {})
        print(f"  Result 3: track_id={m3['track_id']}, name='{m3['name']}', conf={m3['confidence']}%, history={track_data3.get('history')}")
        print(f"  Presence logs in DB: {len(logs3)}")
        assert len(logs3) == 1, f"Frame 3 MUST write exactly 1 row to presence_log! Got {len(logs3)}"
        logged_row = logs3[0]
        print(f"  -> DB ROW CONFIRMED: id={logged_row['id']}, name='{logged_row['name']}', seen_at={logged_row['seen_at']}, dist={logged_row['distance']}, conf={logged_row['confidence']}%")

        # 7. Feed Real Frame 4 (Throttle should prevent duplicate write)
        print("\n--- Feeding Real Frame 4 (Throttle Verification) ---")
        res4 = recognize_frame(req)
        logs4 = database.get_activity_log(name_filter=test_employee_name)
        print(f"  Presence logs in DB after frame 4: {len(logs4)} (expect still 1 due to 30s throttle)")
        assert len(logs4) == 1

        print("\n" + "=" * 70)
        print("   REAL INTEGRATION TEST PASSED PERFECTLY!")
        print("=" * 70)
        return True

    finally:
        # Clean up the test employee record
        print(f"\n[Cleanup] Deleting temporary test employee ID={emp_id}...")
        database.delete_employee(emp_id)
        recognizer.refresh_known_faces()
        print("  -> Cleanup complete.")


if __name__ == "__main__":
    success = run_real_integration_test()
    sys.exit(0 if success else 1)
