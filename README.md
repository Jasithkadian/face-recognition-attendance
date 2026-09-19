---
title: Attendance
emoji: 👤
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
pinned: false
---

# FacePulse AI — Face Recognition Attendance System

A modern, high-performance web application built with **FastAPI**, **YOLOv8-Face (ONNX)**, and **ArcFace MobileFaceNet (ONNX)** for real-time face detection, recognition, and attendance logging.

Designed for seamless deployment on **Render** (Free Tier with 512MB RAM limit) and **Hugging Face Spaces** with persistent SQLite storage and a modern glassmorphism UI.

---

## 🚀 Key Upgrades & Architecture

- **Face Detector**: **YOLOv8-Face (ONNX)** — High-precision detection with 5-point facial landmark estimation (eyes, nose, mouth corners). Robust against angled faces, partial occlusion, and low light.
- **Face Embeddings**: **ArcFace MobileFaceNet (ONNX)** — SOTA 512-dimensional deep facial metric embeddings trained on WebFace600K with Cosine Similarity verification.
- **Runtime**: **ONNX Runtime (CPU)** — Pure prebuilt wheels, no PyTorch overhead (~25MB runtime, under 150MB total RAM).
- **Render Ready**: Zero C++ compilation (`dlib` eliminated), builds in under 45 seconds, binds dynamically to Render's `$PORT`.

---

## 🌟 Features

- **Modern Responsive UI**: Clean glassmorphism dark theme built with vanilla HTML5, CSS3, and ES6 JS.
- **Webcam Face Recognition**: Real-time box overlays with IoU + centroid multi-face tracking and EMA jitter smoothing.
- **"Currently Present" Panel**: Live listing of recognized attendees with active pulse indicators.
- **Guided Multi-Angle Enrollment**: Step-by-step guided enrollment (15 poses) with duplicate angle rejection and averaged embeddings.
- **Admin-Gated Security**: Face enrollment and deletion protected behind an `ADMIN_TOKEN`.
- **Persistent DB Storage**: SQLite database stored at `/home/user/app/data/office.db` (compatible with Render persistent disks).
- **Keep-Alive Endpoint**: Built-in `/health` endpoint for free uptime pings.

---

## 🛠 Project Structure

```
├── Dockerfile              # Lean Docker build file for Render & Hugging Face Spaces
├── render.yaml             # Render Blueprint specification
├── requirements.txt        # Pure prebuilt Python dependencies (zero compilation)
├── download_models.py      # Automated model downloader
├── recognizer.py          # YOLOv8-Face & ArcFace ONNX inference and tracking
├── models/
│   ├── yolov8n-face.onnx   # Lightweight YOLOv8 face detector (~12MB)
│   └── w600k_mbf.onnx      # ArcFace MobileFaceNet embedding extractor (~13MB)
├── app/
│   ├── __init__.py
│   ├── database.py        # SQLite database layer with 512-d ArcFace support
│   └── main.py            # FastAPI REST application & endpoints
└── static/
    ├── index.html         # Web application UI
    ├── style.css          # Responsive dark theme stylesheet
    └── app.js             # Client webcam & tracking controller
```

---

## 🔐 Using Admin Features

1. Open your deployed Web App URL in your browser.
2. Click **"Admin Mode"** in the top right header (or under **Enroll New Person**).
3. Enter your `ADMIN_TOKEN` (configured in your Render environment variables or repository secret).
4. Enroll new faces directly using your webcam with the guided 15-pose sequence, or manage records from the roster.
