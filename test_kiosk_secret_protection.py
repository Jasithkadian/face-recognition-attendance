"""
test_kiosk_secret_protection.py
Verifies lightweight X-Kiosk-Client header enforcement on POST /api/recognize:
1. POST /api/recognize WITHOUT X-Kiosk-Client header -> 403 Forbidden
2. POST /api/recognize WITH INVALID X-Kiosk-Client header -> 403 Forbidden
3. POST /api/recognize WITH VALID X-Kiosk-Client header -> 200 OK
4. GET /api/present WITHOUT any header -> 200 OK (unaffected)
"""

import base64
import numpy as np
import cv2
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

from fastapi.testclient import TestClient
from app.main import app, KIOSK_SECRET

client = TestClient(app)

# Helper to create a valid base64 dummy frame (blank image)
def make_dummy_frame_base64():
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', img)
    return f"data:image/jpeg;base64,{base64.b64encode(buffer).decode('utf-8')}"

def test_kiosk_protection():
    payload = {"image": make_dummy_frame_base64()}

    print("\n=======================================================")
    print("Testing /api/recognize Kiosk Header Protection")
    print(f"Server KIOSK_SECRET: '{KIOSK_SECRET}'")
    print("=======================================================")

    # Test 1: Request WITHOUT header
    print("\n[Test 1] POST /api/recognize WITHOUT X-Kiosk-Client header...")
    r_no_header = client.post("/api/recognize", json=payload)
    print(f"  -> HTTP Status Code: {r_no_header.status_code}")
    print(f"  -> Response Body:    {r_no_header.json()}")
    assert r_no_header.status_code == 403, f"Expected 403, got {r_no_header.status_code}"
    assert "Invalid or missing X-Kiosk-Client header" in r_no_header.json().get("detail", "")
    print("  -> Result: [PASS] Correctly rejected with 403 Forbidden")

    # Test 2: Request with WRONG header
    print("\n[Test 2] POST /api/recognize with WRONG X-Kiosk-Client header ('bad-client')...")
    r_bad_header = client.post("/api/recognize", json=payload, headers={"X-Kiosk-Client": "bad-client"})
    print(f"  -> HTTP Status Code: {r_bad_header.status_code}")
    print(f"  -> Response Body:    {r_bad_header.json()}")
    assert r_bad_header.status_code == 403, f"Expected 403, got {r_bad_header.status_code}"
    print("  -> Result: [PASS] Correctly rejected with 403 Forbidden")

    # Test 3: Request with CORRECT header
    print(f"\n[Test 3] POST /api/recognize with VALID X-Kiosk-Client header ('{KIOSK_SECRET}')...")
    r_valid = client.post("/api/recognize", json=payload, headers={"X-Kiosk-Client": KIOSK_SECRET})
    print(f"  -> HTTP Status Code: {r_valid.status_code}")
    resp_data = r_valid.json()
    print(f"  -> Response Keys:    {list(resp_data.keys())}")
    print(f"  -> Matches:          {resp_data.get('matches')}")
    print(f"  -> Present Count:    {len(resp_data.get('present', []))}")
    assert r_valid.status_code == 200, f"Expected 200, got {r_valid.status_code}"
    assert "matches" in resp_data
    assert "present" in resp_data
    print("  -> Result: [PASS] Correctly accepted and processed with 200 OK")

    # Test 4: Confirm /api/present does NOT require the header
    print("\n[Test 4] GET /api/present WITHOUT any header...")
    r_present = client.get("/api/present")
    print(f"  -> HTTP Status Code: {r_present.status_code}")
    print(f"  -> Response Body:    {r_present.json()}")
    assert r_present.status_code == 200, f"Expected 200, got {r_present.status_code}"
    print("  -> Result: [PASS] /api/present remains public and accessible (200 OK)")

    print("\n=======================================================")
    print("ALL KIOSK HEADER PROTECTION TESTS PASSED (100%)")
    print("=======================================================")

if __name__ == "__main__":
    test_kiosk_protection()
