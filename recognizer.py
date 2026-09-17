"""
recognizer.py
Wraps face detection and recognition logic using OpenCV and face_recognition.
Provides methods for processing live video frames, base64 images, and Streamlit image feeds.
"""

import cv2
import numpy as np
import face_recognition
import base64
import io
from PIL import Image

try:
    from app import database
except ImportError:
    import database

# Lower = stricter match. 0.55 is a balanced threshold for face_recognition
MATCH_TOLERANCE = 0.55

# Shrinking frames before detection accelerates CPU recognition speed
DETECTION_SCALE = 0.25


class FaceRecognizer:
    def __init__(self):
        self.known_ids = []
        self.known_names = []
        self.known_encodings = []
        self.refresh_known_faces()

    def refresh_known_faces(self):
        """Reload known employees from the SQLite database."""
        employees = database.get_all_employees()
        self.known_ids = [e["id"] for e in employees]
        self.known_names = [e["name"] for e in employees]
        self.known_encodings = [e["encoding"] for e in employees]

    def process_bgr_frame(self, frame_bgr):
        """
        Takes a BGR OpenCV numpy array.
        Returns a dict containing frame dimensions and list of match dicts.
        """
        h, w = frame_bgr.shape[:2]
        small = cv2.resize(frame_bgr, (0, 0), fx=DETECTION_SCALE, fy=DETECTION_SCALE)
        rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        boxes = face_recognition.face_locations(rgb_small, model="hog")
        encodings = face_recognition.face_encodings(rgb_small, boxes)

        matches = []
        for (top, right, bottom, left), face_encoding in zip(boxes, encodings):
            scale = 1 / DETECTION_SCALE
            box = [
                int(top * scale),
                int(right * scale),
                int(bottom * scale),
                int(left * scale),
            ]

            name = "Unknown"
            employee_id = None

            if self.known_encodings:
                distances = face_recognition.face_distance(self.known_encodings, face_encoding)
                best_idx = int(np.argmin(distances))
                if distances[best_idx] <= MATCH_TOLERANCE:
                    name = self.known_names[best_idx]
                    employee_id = self.known_ids[best_idx]

            matches.append({
                "name": name,
                "employee_id": employee_id,
                "box": box,
            })

        return {"width": w, "height": h, "matches": matches}

    def process_frame(self, frame_bgr):
        """
        Takes a BGR OpenCV frame.
        Returns (annotated_frame_bgr, list_of_matches)
        """
        res = self.process_bgr_frame(frame_bgr)
        matches = res["matches"]
        annotated = self.draw_boxes(frame_bgr.copy(), matches)
        return annotated, matches

    @staticmethod
    def draw_boxes(frame, matches):
        for m in matches:
            top, right, bottom, left = m["box"]
            color = (0, 220, 180) if m["employee_id"] is not None else (0, 60, 240)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.rectangle(frame, (left, max(0, top - 24)), (right, top), color, cv2.FILLED)
            cv2.putText(
                frame, m["name"], (left + 6, max(16, top - 6)),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1,
            )
        return frame

    # Keep _draw_boxes alias for backward compatibility
    _draw_boxes = draw_boxes


def decode_base64_image(base64_str: str):
    """Decodes a base64 DataURL string into a BGR numpy array for OpenCV."""
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]
    img_bytes = base64.b64decode(base64_str)
    img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    frame_rgb = np.array(img_pil)
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)


def compute_encoding_from_bgr(frame_bgr):
    """Extracts the 128-d face encoding vector from the first detected face in a frame."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model="hog")
    if not boxes:
        return None
    encodings = face_recognition.face_encodings(rgb, boxes)
    return encodings[0] if encodings else None


def compute_encoding_and_box(frame_bgr):
    """Extracts encoding, box location, and total face count for detected face(s)."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model="hog")
    if not boxes:
        return None, None, 0
    encodings = face_recognition.face_encodings(rgb, boxes)
    if not encodings:
        return None, None, len(boxes)
    return encodings[0], boxes[0], len(boxes)


def compute_encoding_from_frame(frame_bgr):
    return compute_encoding_from_bgr(frame_bgr)

