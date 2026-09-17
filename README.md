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

A modern, responsive web application built with **FastAPI**, **OpenCV**, and **face_recognition** (dlib) for real-time face detection, recognition, and attendance logging. Designed for seamless 1-click cloud deployment on **Hugging Face Spaces** with persistent SQLite storage and a modern dark UI.

---

## 🌟 Features

- **Modern Responsive UI**: Clean glassmorphism dark theme built with vanilla HTML5, CSS3, and ES6 JS.
- **Webcam Face Recognition**: Live box overlays with smooth rounded borders, custom color accents, and avatar name badges.
- **"Currently Present" Panel**: Real-time listing of recognized individuals with active pulse indicators.
- **Admin-Gated Enrollment**: Simple 1-click face capture & enrollment gated behind an `ADMIN_TOKEN`.
- **Persistent DB Storage**: SQLite database stored at `/home/user/app/data/office.db` so enrolled faces persist across app restarts.
- **Keep-Alive Endpoint**: Built-in `/health` endpoint for free pings to prevent free-tier cold-starts.

---

## 🛠 Project Structure

```
├── Dockerfile              # Docker build file for Hugging Face Spaces containerization
├── requirements.txt        # Python package dependencies (dlib-bin precompiled)
├── recognizer.py          # Shared face recognition & image decoding logic
├── app/
│   ├── __init__.py
│   ├── database.py        # Web-adapted SQLite database layer
│   └── main.py            # FastAPI application & API endpoints
└── static/
    ├── index.html         # Modern web application UI
    ├── style.css          # Responsive dark theme stylesheet
    └── app.js             # Client JS (Webcam, Canvas overlays, API fetch)
```

---

## 🔐 Using Admin Features

1. Open your deployed Web App URL in your browser.
2. Click **"Admin Mode"** in the top right header (or under **Enroll New Person**).
3. Enter your `ADMIN_TOKEN` (matches the repository secret set in Hugging Face Space Settings).
4. You can now enroll new faces directly using your device webcam, and manage/delete enrolled records from the roster!
