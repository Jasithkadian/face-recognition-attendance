"""
dashboard.py
Live office face-monitoring and employee enrollment dashboard built with Streamlit.

Features:
- Optimized live face monitoring feed with frame-skipping and compressed JPEG streaming (no lagging)
- Cloud-compatible browser camera snapshot recognition for Streamlit Community Cloud
- Add new people / employee enrollment directly inside Streamlit using st.camera_input or file upload
- Face detection preview and verification before saving
- Sidebar performance controls and enrolled roster management
"""

import time
import sqlite3
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

import database
from recognizer import FaceRecognizer, compute_encoding_and_box

st.set_page_config(
    page_title="Office Face Monitor & Enrollment",
    page_icon="👤",
    layout="wide",
)

database.init_db()

# Initialize session state objects
if "recognizer" not in st.session_state:
    st.session_state.recognizer = FaceRecognizer()
if "running" not in st.session_state:
    st.session_state.running = False

st.title("🟢 Office Face Monitoring & Enrollment Dashboard")

# ---------------- Sidebar: Settings & Enrolled Roster ----------------
with st.sidebar:
    st.header("⚙️ Performance Controls")
    st.caption("Tweak performance settings to eliminate video lag:")
    
    frame_skip = st.slider(
        "Face Detection Frequency",
        min_value=1,
        max_value=6,
        value=3,
        help="Process face recognition every N frames. Higher values dramatically reduce CPU lag.",
    )
    jpeg_quality = st.slider(
        "Stream Compression Quality",
        min_value=40,
        max_value=95,
        value=80,
        help="Lower quality reduces data transmission latency across Streamlit.",
    )
    cam_res = st.selectbox(
        "Webcam Resolution",
        ["640x480 (Fast)", "1280x720 (HD)"],
        index=0,
    )

    st.divider()
    st.header("👥 Enrolled People")
    employees = database.get_all_people()
    if not employees:
        st.info("No people enrolled yet. Use the **Add New Person** tab to register someone!")
    else:
        st.write(f"Total Enrolled: **{len(employees)}**")
        for e in employees:
            col1, col2 = st.columns([3, 1])
            col1.write(f"👤 {e['name']}")
            if col2.button("🗑️", key=f"del_{e['id']}", help=f"Delete {e['name']}"):
                database.delete_employee(e["id"])
                st.session_state.recognizer.refresh_known_faces()
                st.toast(f"Deleted {e['name']} successfully!", icon="🗑️")
                st.rerun()

    st.divider()
    if st.button("🔄 Reload Faces from DB", use_container_width=True):
        st.session_state.recognizer.refresh_known_faces()
        st.success("Face database reloaded successfully.")

# ---------------- Main Layout Tabs ----------------
tab_monitor, tab_enroll, tab_logs = st.tabs([
    "🎥 Live Monitor Feed",
    "➕ Add New Person",
    "📋 Attendance Logs & Roster",
])

# ================= TAB 1: LIVE MONITOR =================
with tab_monitor:
    st.subheader("Face Monitoring & Recognition")

    feed_mode = st.radio(
        "Select Monitoring Mode",
        ["📹 Live Stream (Local Webcam)", "📷 Browser Camera Snapshot (Cloud Compatible)"],
        horizontal=True,
        help="Use 'Browser Camera Snapshot' when deployed on Streamlit Cloud.",
    )

    if feed_mode == "📹 Live Stream (Local Webcam)":
        col_a, col_b, _ = st.columns([1.2, 1.2, 4])
        if col_a.button("▶️ Start Monitoring", disabled=st.session_state.running, type="primary", use_container_width=True):
            st.session_state.running = True
            st.rerun()
        if col_b.button("⏹️ Stop Feed", disabled=not st.session_state.running, use_container_width=True):
            st.session_state.running = False
            st.rerun()

        video_col, info_col = st.columns([2, 1])
        frame_slot = video_col.empty()
        present_slot = info_col.empty()
        log_slot = st.empty()

        if st.session_state.running:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("⚠️ Local webcam device (`VideoCapture(0)`) is unavailable. If running on Streamlit Community Cloud, please switch to **📷 Browser Camera Snapshot** mode above!")
                st.session_state.running = False
            else:
                res_w = 640 if "640" in cam_res else 1280
                res_h = 480 if "480" in cam_res else 720
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, res_w)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, res_h)

                recognizer = st.session_state.recognizer
                last_logged = {}
                LOG_INTERVAL = 5
                UI_UPDATE_INTERVAL = 1.2

                frame_count = 0
                last_matches = []
                last_ui_update = 0

                try:
                    while st.session_state.running:
                        ok, frame = cap.read()
                        if not ok:
                            st.warning("Lost webcam video stream.")
                            break

                        frame_count += 1

                        # Run heavy face recognition every N frames (frame-skipping)
                        if frame_count % frame_skip == 0 or not last_matches:
                            res = recognizer.process_bgr_frame(frame)
                            last_matches = res["matches"]

                            now = time.time()
                            for m in last_matches:
                                if m["employee_id"] is not None:
                                    if now - last_logged.get(m["employee_id"], 0) > LOG_INTERVAL:
                                        database.log_presence(m["employee_id"])
                                        last_logged[m["employee_id"]] = now

                        # Draw boxes using cached or updated matches
                        annotated = recognizer.draw_boxes(frame.copy(), last_matches)

                        # Encode to compressed JPEG to eliminate Streamlit serialization lag
                        _, jpeg_buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
                        frame_slot.image(jpeg_buf.tobytes(), use_container_width=True)

                        # Throttle sidebar/panel updates so Streamlit UI doesn't lag
                        now = time.time()
                        if now - last_ui_update > UI_UPDATE_INTERVAL:
                            last_ui_update = now

                            present = database.get_recently_present()
                            with present_slot.container():
                                st.subheader("👀 Currently in view")
                                if present:
                                    for p in present:
                                        st.success(f"👤 **{p['name']}**")
                                else:
                                    st.info("No recognized people in view right now.")

                            today = database.get_today_summary()
                            with log_slot.container():
                                st.subheader("📋 Today's Summary Log")
                                if today:
                                    df = pd.DataFrame(today)
                                    st.dataframe(df, use_container_width=True, hide_index=True)
                                else:
                                    st.write("No detections logged today yet.")

                        time.sleep(0.01)
                finally:
                    cap.release()
        else:
            frame_slot.info("Click **▶️ Start Monitoring** above to begin the live webcam feed.")
            today = database.get_today_summary()
            with log_slot.container():
                st.subheader("📋 Today's Summary Log")
                if today:
                    st.dataframe(pd.DataFrame(today), use_container_width=True, hide_index=True)
                else:
                    st.write("No detections logged today yet.")

    else:
        # Browser Camera Snapshot Mode (Cloud Compatible)
        st.write("Take a snapshot from your browser camera to perform face recognition and log attendance:")
        cam_snap = st.camera_input("Capture Face Snapshot for Recognition", key="live_snap_input")

        c_snap_img, c_snap_info = st.columns([1.5, 1])

        if cam_snap is not None:
            pil_img = Image.open(cam_snap).convert("RGB")
            frame_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

            recognizer = st.session_state.recognizer
            res = recognizer.process_bgr_frame(frame_bgr)
            matches = res["matches"]

            recognized_names = []
            for m in matches:
                if m["employee_id"] is not None:
                    database.log_presence(m["employee_id"])
                    recognized_names.append(m["name"])

            annotated = recognizer.draw_boxes(frame_bgr.copy(), matches)

            with c_snap_img:
                st.image(
                    cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB),
                    caption="Recognition Results",
                    use_container_width=True,
                )

            with c_snap_info:
                st.subheader("👀 Detection Results")
                if recognized_names:
                    unique_names = list(set(recognized_names))
                    for name in unique_names:
                        st.success(f"👤 Recognized & Logged: **{name}**")
                else:
                    st.info("No enrolled people recognized in this snapshot.")

        st.divider()
        today = database.get_today_summary()
        st.subheader("📋 Today's Summary Log")
        if today:
            st.dataframe(pd.DataFrame(today), use_container_width=True, hide_index=True)
        else:
            st.write("No detections logged today yet.")

# ================= TAB 2: ADD NEW PERSON =================
with tab_enroll:
    st.subheader("➕ Register New Person")
    st.write("Capture or upload a face photo to register a new person into the attendance database.")

    col_name, col_method = st.columns([1, 1])
    with col_name:
        new_name = st.text_input("Full Name", placeholder="e.g. John Smith", key="enroll_person_name")
    with col_method:
        enroll_source = st.radio(
            "Photo Input Method",
            ["📷 Camera Input (Streamlit Image)", "📁 Upload Image File"],
            horizontal=True,
        )

    captured_bgr = None

    if enroll_source == "📷 Camera Input (Streamlit Image)":
        st.write("Click below to capture a snapshot with your camera:")
        cam_image = st.camera_input("Take a photo of the person", key="enroll_snap_input")
        if cam_image is not None:
            pil_img = Image.open(cam_image).convert("RGB")
            captured_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    elif enroll_source == "📁 Upload Image File":
        uploaded_file = st.file_uploader("Upload a face photo (JPG/PNG)", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            pil_img = Image.open(uploaded_file).convert("RGB")
            captured_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    if captured_bgr is not None:
        st.divider()
        st.subheader("Face Detection & Verification Preview")

        encoding, box, num_faces = compute_encoding_and_box(captured_bgr)

        if num_faces == 0:
            st.error("❌ No face detected in the photo. Please make sure your face is clearly visible and well-lit.")
        elif num_faces > 1:
            st.warning(f"⚠️ Multiple faces detected ({num_faces}). Please ensure ONLY ONE person is in the photo frame.")
        else:
            top, right, bottom, left = box
            preview = captured_bgr.copy()
            cv2.rectangle(preview, (left, top), (right, bottom), (0, 220, 100), 3)
            label = new_name.strip() if new_name.strip() else "Detected Face"
            cv2.putText(
                preview, label, (left, max(24, top - 10)),
                cv2.FONT_HERSHEY_DUPLEX, 0.75, (0, 220, 100), 2,
            )

            c_prev, c_action = st.columns([1, 1])
            with c_prev:
                st.image(
                    cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                    caption="Verified Face Bounding Box",
                    use_container_width=True,
                )

            with c_action:
                st.success("✅ Face detected and verified successfully!")

                if not new_name.strip():
                    st.warning("⚠️ Please enter a **Full Name** above before saving.")
                else:
                    if st.button("💾 Save & Enroll Person", type="primary", use_container_width=True):
                        try:
                            clean_name = new_name.strip()
                            emp_id = database.add_employee(clean_name, encoding)
                            st.session_state.recognizer.refresh_known_faces()
                            st.balloons()
                            st.success(f"🎉 **{clean_name}** has been registered successfully (ID: {emp_id})!")
                            st.info("You can now switch to the **Live Monitor Feed** tab to recognize them.")
                        except sqlite3.IntegrityError:
                            st.error(f"⚠️ A person named '{new_name.strip()}' is already enrolled in the system.")
                        except Exception as ex:
                            st.error(f"Failed to enroll person: {ex}")

# ================= TAB 3: LOGS & ROSTER =================
with tab_logs:
    st.subheader("📋 Attendance Records & Roster Overview")

    col_l1, col_l2 = st.columns([1, 1])
    with col_l1:
        st.markdown("### 👤 Registered People")
        all_ppl = database.get_all_people()
        if all_ppl:
            df_ppl = pd.DataFrame(all_ppl)
            st.dataframe(df_ppl, use_container_width=True, hide_index=True)
        else:
            st.info("No enrolled people found in database.")

    with col_l2:
        st.markdown("### 📊 Today's Recognition Summary")
        summary_data = database.get_today_summary()
        if summary_data:
            df_sum = pd.DataFrame(summary_data)
            st.dataframe(df_sum, use_container_width=True, hide_index=True)
        else:
            st.info("No attendance activity recorded today.")
