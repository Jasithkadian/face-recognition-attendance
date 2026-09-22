"""
test_report_generation.py
Test report generation endpoints for Excel (.xlsx) and PDF (.pdf).
Verifies:
1. Admin token authorization requirement (rejects invalid token with 401).
2. Attendance summary data retrieval from database.
3. Excel generation returns non-empty bytes starting with zip signature b"PK\\x03\\x04".
4. PDF generation returns non-empty bytes starting with b"%PDF".
5. Proper cleanup of enrolled test user.
"""

import sys
import os
import numpy as np
from pathlib import Path

# Ensure application package can be imported
sys.path.insert(0, str(Path(__file__).parent))

import app.database as database
from app.main import (
    app,
    download_today_report_excel,
    download_today_report_pdf,
    ADMIN_TOKEN,
)
from fastapi import HTTPException


def run_report_test():
    print("=" * 70)
    print("       Testing Attendance Report Generation (.xlsx & .pdf)")
    print("=" * 70)

    # Step 1: Enroll a temporary test employee and log presence
    fake_name = f"Report_Test_User_{np.random.randint(1000, 9999)}"
    print(f"\n[Step 1] Enrolling fake employee '{fake_name}' with 512-d vector...")
    raw_vec = np.random.randn(512).astype(np.float64)
    unit_vec = raw_vec / np.linalg.norm(raw_vec)

    emp_id = database.add_employee(fake_name, unit_vec)
    print(f"  -> Enrolled '{fake_name}' with ID={emp_id}")

    print(f"\n[Step 2] Logging presence for '{fake_name}' today...")
    database.log_presence(emp_id, distance=0.18, min_interval_seconds=0)

    summary = database.get_today_summary()
    print(f"  -> get_today_summary() returned {len(summary)} record(s).")
    test_record = next((s for s in summary if s["name"] == fake_name), None)
    if not test_record:
        print(f"  -> WARNING: '{fake_name}' not found in today's summary.")
    else:
        print(f"  -> Found record: {test_record}")

    try:
        # Step 3: Test Auth Protection
        print("\n[Step 3] Testing Admin Auth Protection on report endpoints...")
        try:
            download_today_report_excel(admin_token="wrong_token")
            print("  -> FAILED: Excel endpoint allowed access with invalid token!")
            return False
        except HTTPException as e:
            assert e.status_code == 401
            print("  -> SUCCESS: Excel endpoint correctly rejected invalid token (HTTP 401).")

        try:
            download_today_report_pdf(admin_token="wrong_token")
            print("  -> FAILED: PDF endpoint allowed access with invalid token!")
            return False
        except HTTPException as e:
            assert e.status_code == 401
            print("  -> SUCCESS: PDF endpoint correctly rejected invalid token (HTTP 401).")

        # Step 4: Test Excel Generation (.xlsx)
        print("\n[Step 4] Generating Excel (.xlsx) Report...")
        excel_resp = download_today_report_excel(admin_token=ADMIN_TOKEN)
        excel_bytes = excel_resp.body
        excel_media = excel_resp.media_type
        excel_headers = dict(excel_resp.headers)

        print(f"  -> Content-Type: {excel_media}")
        print(f"  -> Content-Disposition: {excel_headers.get('content-disposition')}")
        print(f"  -> File size: {len(excel_bytes)} bytes")
        print(f"  -> Magic header (first 4 bytes): {excel_bytes[:4]}")

        assert len(excel_bytes) > 0, "Excel output must not be empty"
        assert excel_bytes.startswith(b"PK\x03\x04"), f"Excel file must start with PK\\x03\\x04, got: {excel_bytes[:4]}"
        print("  -> SUCCESS: Valid Excel spreadsheet (.xlsx) generated completely in-memory!")

        # Step 5: Test PDF Generation (.pdf)
        print("\n[Step 5] Generating PDF (.pdf) Report...")
        pdf_resp = download_today_report_pdf(admin_token=ADMIN_TOKEN)
        pdf_bytes = pdf_resp.body
        pdf_media = pdf_resp.media_type
        pdf_headers = dict(pdf_resp.headers)

        print(f"  -> Content-Type: {pdf_media}")
        print(f"  -> Content-Disposition: {pdf_headers.get('content-disposition')}")
        print(f"  -> File size: {len(pdf_bytes)} bytes")
        print(f"  -> Magic header (first 4 bytes): {pdf_bytes[:4]}")

        assert len(pdf_bytes) > 0, "PDF output must not be empty"
        assert pdf_bytes.startswith(b"%PDF"), f"PDF file must start with %PDF, got: {pdf_bytes[:4]}"
        print("  -> SUCCESS: Valid PDF document (.pdf) generated completely in-memory!")

        print("\n" + "=" * 70)
        print("       ALL REPORT GENERATION TESTS PASSED!")
        print("=" * 70)
        return True

    finally:
        # Step 6: Cleanup
        print(f"\n[Step 6] Cleaning up test employee ID={emp_id}...")
        database.delete_employee(emp_id)
        print("  -> Cleanup complete.")


if __name__ == "__main__":
    success = run_report_test()
    sys.exit(0 if success else 1)
