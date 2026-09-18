"""
app/main.py
FastAPI web application for Face Recognition Attendance System.
Provides endpoints for webcam frame recognition, employee enrollment, active presence monitoring,
health checks (for Render keep-alive pings), and static frontend serving.
"""

import os
import sys
import datetime
import time
import sqlite3
import traceback
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Header, Depends, status, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import app.database as database
from recognizer import (
    FaceRecognizer,
    decode_base64_image,
    compute_encoding_from_bgr,
    compute_encoding_and_box,
    compute_averaged_encoding_from_images,
    validate_enrollment_sample,
)


ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN", "admin123")
if os.environ.get("ADMIN_TOKEN") is None:
    print("[WARNING] ADMIN_TOKEN environment variable is not set. Using default fallback token: 'admin123'")

app = FastAPI(
    title="Face Pulse Attendance API",
    description="Live face recognition attendance web service",
    version="1.0.0",
)

@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    err_tb = traceback.format_exc()
    print(f"[Unhandled Server Error] {request.method} {request.url.path}: {err_tb}")
    return JSONResponse(
        status_code=500,
        content={"detail": f"Server error: {str(exc)}", "type": type(exc).__name__}
    )

# Enable CORS for web client access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global face recognizer instance
recognizer_instance: Optional[FaceRecognizer] = None

# In-memory throttles to prevent synchronous SQLite lock contention on high-frequency frame streams
_last_logged_presence = {}  # employee_id -> epoch timestamp
_presence_cache = []
_presence_cache_time = 0.0


@app.on_event("startup")
def startup_event():
    get_recognizer()
    print("Database initialized and face recognizer loaded successfully.")


def get_recognizer() -> FaceRecognizer:
    global recognizer_instance
    if recognizer_instance is None:
        database.init_db()
        recognizer_instance = FaceRecognizer()
    return recognizer_instance


# Request models
class RecognizeRequest(BaseModel):
    image: str  # Base64 data URL string


class EnrollRequest(BaseModel):
    name: str
    image: Optional[str] = None  # Single Base64 data URL string
    images: Optional[List[str]] = None  # List of 15 Base64 data URL strings for multi-sample guided enrollment
    admin_token: str


class VerifySampleRequest(BaseModel):
    image: str  # Base64 data URL string
    existing_images: Optional[List[str]] = None  # Already collected sample base64 strings for duplicate check


class VerifyTokenRequest(BaseModel):
    admin_token: str


# Helper dependency to check admin token
def verify_admin_token(x_admin_token: Optional[str] = Header(None), token: Optional[str] = None):
    provided = x_admin_token or token
    if not provided or provided != ADMIN_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing admin token.",
        )
    return True


# ---------------- API Endpoints ----------------

@app.get("/health")
def health_check():
    """
    Keep-alive endpoint for ping services (e.g. cron-job.org or UptimeRobot)
    to prevent Render free tier instance spin-down.
    """
    return {
        "status": "ok",
        "service": "face-pulse-attendance",
        "timestamp": datetime.datetime.now().isoformat(),
    }


@app.post("/api/verify-token")
def verify_token(req: VerifyTokenRequest):
    """Verify if the supplied admin token is valid."""
    is_valid = req.admin_token == ADMIN_TOKEN
    return {"valid": is_valid}


@app.post("/api/recognize")
def recognize_frame(req: RecognizeRequest):
    """
    Accepts a base64 camera frame, performs fast face detection + IoU/centroid tracking + recognition,
    logs presence in DB with in-memory throttling, and returns tracked bounding boxes and names.
    """
    recognizer = get_recognizer()

    try:
        frame_bgr = decode_base64_image(req.image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {str(e)}")

    result = recognizer.process_bgr_frame(frame_bgr)

    # In-memory throttle: avoid executing synchronous SQLite transactions on every video frame
    global _presence_cache, _presence_cache_time
    now_ts = time.time()
    for m in result.get("matches", []):
        emp_id = m.get("employee_id")
        if emp_id is not None:
            if now_ts - _last_logged_presence.get(emp_id, 0.0) >= 30.0:
                _last_logged_presence[emp_id] = now_ts
                try:
                    database.log_presence(emp_id, distance=m.get("distance"), min_interval_seconds=30)
                except Exception as e:
                    print(f"Error logging presence: {e}")

    # Cache get_recently_present for 2.0s to avoid expensive DB queries on every video frame
    if now_ts - _presence_cache_time >= 2.0:
        try:
            _presence_cache = database.get_recently_present(window_seconds=20)
            _presence_cache_time = now_ts
        except Exception as e:
            print(f"Error refreshing presence cache: {e}")

    result["present"] = _presence_cache
    return result


@app.post("/api/enroll-check")
def verify_enrollment_sample(req: VerifySampleRequest):
    """
    Verifies if a candidate camera frame contains exactly 1 detectable face
    and rejects duplicate/near-identical pose frames compared to existing samples.
    """
    try:
        frame_bgr = decode_base64_image(req.image)
        valid, reason, box = validate_enrollment_sample(frame_bgr, req.existing_images)
        return {"valid": valid, "reason": reason, "box": box}
    except Exception as e:
        return {"valid": False, "reason": f"Sample verification failed: {str(e)}"}


@app.post("/api/enroll")
def enroll_employee(req: EnrollRequest):
    """
    Enrolls a new person into the system using single or multi-sample guided enrollment.
    Requires valid admin_token. Computes averaged 128-d face encoding from samples.
    """
    print(f"[Enroll] Request received for name='{req.name}', admin_token_provided={'Yes' if req.admin_token else 'No'}")

    if req.admin_token != ADMIN_TOKEN:
        print(f"[Enroll REJECTED] Admin token mismatch. Received '{req.admin_token}', expected '{ADMIN_TOKEN}'")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Incorrect admin token. Please enter valid admin token.",
        )

    clean_name = req.name.strip()
    if not clean_name:
        raise HTTPException(status_code=400, detail="Name cannot be empty.")

    encoding = None
    samples_used = 1

    if req.images and len(req.images) > 0:
        print(f"[Enroll] Processing multi-sample enrollment for '{clean_name}' with {len(req.images)} images...")
        avg_enc, count = compute_averaged_encoding_from_images(req.images)
        if avg_enc is None or count == 0:
            print(f"[Enroll ERROR] Could not detect faces in multi-sample set for '{clean_name}'.")
            raise HTTPException(
                status_code=400,
                detail="Could not detect faces in the provided sample set. Please ensure good lighting and re-enroll.",
            )
        encoding = avg_enc
        samples_used = count
    elif req.image:
        print(f"[Enroll] Processing single-sample enrollment for '{clean_name}'...")
        try:
            frame_bgr = decode_base64_image(req.image)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

        encoding = compute_encoding_from_bgr(frame_bgr)
        if encoding is None:
            print(f"[Enroll ERROR] No face detected in single image for '{clean_name}'.")
            raise HTTPException(
                status_code=400,
                detail="No face detected in the image. Please position your face clearly in front of the camera and try again.",
            )
    else:
        raise HTTPException(status_code=400, detail="No image data provided for enrollment.")

    try:
        emp_id = database.add_employee(clean_name, encoding)
        print(f"[Enroll DB SUCCESS] Saved employee '{clean_name}' with ID={emp_id} to database.")
    except sqlite3.IntegrityError as e:
        print(f"[Enroll DB INTEGRITY ERROR] Duplicate employee name '{clean_name}': {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Employee '{clean_name}' already exists in the database. Please choose a different name.",
        )
    except sqlite3.OperationalError as e:
        print(f"[Enroll DB OPERATIONAL ERROR] Database error saving '{clean_name}': {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Database write failed due to lock/concurrency: {str(e)}",
        )
    except Exception as e:
        print(f"[Enroll DB ERROR] Failed to save '{clean_name}': {e}")
        raise HTTPException(
            status_code=400,
            detail=f"Could not enroll user: {str(e)}",
        )

    # Refresh recognizer with the new employee encoding (non-fatal if reload fails)
    try:
        if recognizer_instance:
            recognizer_instance.refresh_known_faces()
            print(f"[Enroll RECOGNIZER SUCCESS] Refreshed recognizer with new employee '{clean_name}'.")
    except Exception as e:
        print(f"[Enroll RECOGNIZER WARNING] Failed to refresh known faces cache for '{clean_name}': {e}")

    return {
        "status": "success",
        "message": f"Enrolled '{clean_name}' successfully using {samples_used} face samples!",
        "employee_id": emp_id,
        "name": clean_name,
        "samples_used": samples_used,
    }



@app.get("/api/present")
def get_currently_present(window_seconds: int = 20):
    """Get list of employees recognized within the last `window_seconds`."""
    present = database.get_recently_present(window_seconds=window_seconds)
    return {"present": present}


@app.get("/api/employees")
def get_employees():
    """Get list of all enrolled employees."""
    employees = database.get_all_people()
    return {"employees": employees}


@app.delete("/api/employees/{employee_id}")
def delete_employee(employee_id: int, admin_token: str):
    """Delete an enrolled employee by ID. Gated by admin_token."""
    if admin_token != ADMIN_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Incorrect admin token.",
        )

    database.delete_employee(employee_id)
    if recognizer_instance:
        recognizer_instance.refresh_known_faces()

    return {"status": "success", "message": "Employee deleted."}


@app.get("/api/activity-log")
def get_activity_log(limit: int = 100, name: Optional[str] = None, date: Optional[str] = None):
    """Get detailed timestamped activity log records with ML match distance & confidence scores."""
    logs = database.get_activity_log(limit=limit, name_filter=name, date_filter=date)
    return {"logs": logs}


@app.get("/api/today-summary")
def get_today_summary():
    """Get attendance log summary for today."""
    summary = database.get_today_summary()
    return {"summary": summary}


# ---------------- Static Files & UI Mounting ----------------

BASE_DIR = Path(__file__).parent.parent
STATIC_DIR = BASE_DIR / "static"

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
def read_root():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "Face Pulse API is running. Static frontend not found."}
