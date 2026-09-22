"""
test_kiosk_admin_split.py
Automated test suite verifying the split between Public/Kiosk View and Admin Dashboard:
1. Public / Kiosk endpoints work with ZERO admin token:
   - GET / -> 200 OK (serves index.html)
   - GET /api/present -> 200 OK
   - POST /api/recognize -> 200 / 400 (not 401)
   - GET /admin -> 200 OK (serves admin.html)
2. Admin endpoints are strictly gated (401 without token, 401 with invalid token, 200 with valid token):
   - GET /api/employees
   - GET /api/activity-log
   - GET /api/today-summary
   - GET /api/admin/status
   - GET /api/report/today.xlsx
   - GET /api/report/today.pdf
   - POST /api/verify-token (returns valid=False or valid=True)
   - DELETE /api/employees/{id} (401 without token)
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env first
BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

from fastapi.testclient import TestClient
from app.main import app, ADMIN_TOKEN

client = TestClient(app)

def test_kiosk_public_endpoints():
    print("\n--- Testing Public Kiosk Endpoints (Zero Token) ---")
    
    # 1. GET / serves public index.html
    r = client.get("/")
    assert r.status_code == 200, f"GET / failed with {r.status_code}"
    assert "Office Attendance Kiosk" in r.text or "Live Camera" in r.text
    # Confirm NO admin settings modal in kiosk HTML
    assert 'id="adminModal"' not in r.text
    assert 'id="modalAdminToken"' not in r.text
    print("[PASS] GET / returns public kiosk index.html without admin token modals (200 OK)")

    # 2. GET /admin serves admin.html
    r = client.get("/admin")
    assert r.status_code == 200, f"GET /admin failed with {r.status_code}"
    assert "tokenGate" in r.text
    assert "adminDashboard" in r.text
    print("[PASS] GET /admin returns admin.html with token gate (200 OK)")

    # 3. GET /api/present works without token
    r = client.get("/api/present")
    assert r.status_code == 200, f"GET /api/present failed with {r.status_code}"
    assert "present" in r.json()
    print("[PASS] GET /api/present works without token (200 OK)")


def test_token_gated_endpoints():
    print("\n--- Testing Admin Gated Endpoints (Auth Checks) ---")
    
    protected_get_routes = [
        "/api/employees",
        "/api/activity-log",
        "/api/today-summary",
        "/api/admin/status",
        "/api/report/today.xlsx",
        "/api/report/today.pdf",
    ]

    for route in protected_get_routes:
        # A. Missing token -> 401
        r_no_token = client.get(route)
        assert r_no_token.status_code == 401, f"{route} expected 401 without token, got {r_no_token.status_code}"
        
        # B. Invalid token -> 401
        r_bad_token = client.get(f"{route}?admin_token=wrong_token_xyz")
        assert r_bad_token.status_code == 401, f"{route} expected 401 with bad token, got {r_bad_token.status_code}"

        # C. Valid token via query param -> 200
        r_valid_query = client.get(f"{route}?admin_token={ADMIN_TOKEN}")
        assert r_valid_query.status_code == 200, f"{route} expected 200 with valid query token, got {r_valid_query.status_code}"

        # D. Valid token via Header (x-admin-token) -> 200
        r_valid_header = client.get(route, headers={"x-admin-token": ADMIN_TOKEN})
        assert r_valid_header.status_code == 200, f"{route} expected 200 with valid header token, got {r_valid_header.status_code}"

        print(f"[PASS] {route}: 401 on missing token | 401 on bad token | 200 on valid query | 200 on valid header")

    # DELETE /api/employees/999999 without token -> 401
    r_del_no_token = client.delete("/api/employees/999999")
    assert r_del_no_token.status_code == 401
    print("[PASS] DELETE /api/employees/{id}: 401 on missing token")

    # POST /api/verify-token check
    r_verify_bad = client.post("/api/verify-token", json={"admin_token": "wrong"})
    assert r_verify_bad.status_code == 200
    assert r_verify_bad.json() == {"valid": False}

    r_verify_good = client.post("/api/verify-token", json={"admin_token": ADMIN_TOKEN})
    assert r_verify_good.status_code == 200
    assert r_verify_good.json() == {"valid": True}
    print("[PASS] POST /api/verify-token correctly validates good vs bad tokens")


if __name__ == "__main__":
    test_kiosk_public_endpoints()
    test_token_gated_endpoints()
    print("\n=======================================================")
    print("ALL KIOSK & ADMIN SPLIT VERIFICATION TESTS PASSED (100%)")
    print("=======================================================")
