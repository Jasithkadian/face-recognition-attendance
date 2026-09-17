"""
app/main.py
FastAPI web application for Face Recognition Attendance System.
Provides endpoints for webcam frame recognition, employee enrollment, active presence monitoring,
health checks (for Render keep-alive pings), and static frontend serving.
"""

import os
import sys
import datetime
from pathlib import Path
from typing import Optional, List

from fastapi import FastAPI, HTTPException, Header, Depends, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
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


ADMIN_TOKEN = os.environ.get("ADMIN_TOKEN")
if not ADMIN_TOKEN:
    sys.exit("FATAL: ADMIN_TOKEN environment variable is not set. Refusing to start with no admin protection.")

app = FastAPI(
    title="Face Pulse Attendance API",
    description="Live face recognition attendance web service",
    version="1.0.0",
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


@app.on_event("startup")
def startup_event():
    global recognizer_instance
    database.init_db()
    recognizer_instance = FaceRecognizer()
    print("Database initialized and face recognizer loaded successfully.")


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
    Accepts a base64 camera frame, performs face detection + IoU tracking + recognition,
    logs presence in DB for recognized individuals (max once per 60s), and returns tracked bounding boxes and names.
    """
    if recognizer_instance is None:
        raise HTTPException(status_code=500, detail="Recognizer not initialized.")

    try:
        frame_bgr = decode_base64_image(req.image)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image data: {str(e)}")

    result = recognizer_instance.process_bgr_frame(frame_bgr)

    # Log presence for any recognized faces with distance score (keeps seen_at fresh, throttled to 30s session log)
    for m in result.get("matches", []):
        if m.get("employee_id") is not None:
            try:
                database.log_presence(m["employee_id"], distance=m.get("distance"), min_interval_seconds=30)
            except Exception as e:
                print(f"Error logging presence: {e}")

    # Return updated active presence list directly in frame recognition response for instant frontend sync
    result["present"] = database.get_recently_present(window_seconds=20)

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
    if req.admin_token != ADMIN_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unauthorized: Incorrect admin token.",
        )

    clean_name = req.name.strip()
    if not clean_name:
        raise HTTPException(status_code=400, detail="Name cannot be empty.")

    encoding = None
    samples_used = 1

    if req.images and len(req.images) > 0:
        # Multi-sample guided enrollment path
        avg_enc, count = compute_averaged_encoding_from_images(req.images)
        if avg_enc is None or count == 0:
            raise HTTPException(
                status_code=400,
                detail="Could not detect faces in the provided sample set. Please ensure good lighting and re-enroll.",
            )
        encoding = avg_enc
        samples_used = count
    elif req.image:
        # Single-sample legacy enrollment path
        try:
            frame_bgr = decode_base64_image(req.image)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image format: {str(e)}")

        encoding = compute_encoding_from_bgr(frame_bgr)
        if encoding is None:
            raise HTTPException(
                status_code=400,
                detail="No face detected in the image. Please position your face clearly in front of the camera and try again.",
            )
    else:
        raise HTTPException(status_code=400, detail="No image data provided for enrollment.")

    try:
        emp_id = database.add_employee(clean_name, encoding)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not enroll user (name may already exist): {str(e)}",
        )

    # Refresh recognizer with the new employee encoding
    recognizer_instance.refresh_known_faces()

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
