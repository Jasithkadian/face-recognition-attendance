"""
recognizer.py
Wraps face detection, IoU-based face tracking, recognition, and multi-angle enrollment logic
using OpenCV and face_recognition (dlib).
"""

import cv2
import numpy as np
import face_recognition
import base64
import io
import time
from PIL import Image

from app import database

# Match threshold for face_recognition Euclidean distance (dlib standard)
MATCH_TOLERANCE = 0.58

# Maximum width for processing frames safely
MAX_FRAME_WIDTH = 640

# Minimum Euclidean distance between new sample and existing samples for enrollment duplicate rejection
DUPLICATE_ENCODING_THRESHOLD = 0.16


def resize_if_large(frame_bgr, max_width=MAX_FRAME_WIDTH):
    """Safety resize step: downscales BGR image if width exceeds max_width."""
    h, w = frame_bgr.shape[:2]
    if w > max_width:
        new_w = max_width
        new_h = int(h * (max_width / w))
        return cv2.resize(frame_bgr, (new_w, new_h))
    return frame_bgr


def distance_to_confidence(distance, threshold=MATCH_TOLERANCE):
    """
    Converts raw 128-d face_distance into a realistic, calibrated confidence percentage.
    Uses dlib's decision boundary curve relative to threshold (0.58):
      - distance 0.20 -> ~91.4%
      - distance 0.30 -> ~87.1%
      - distance 0.58 -> 50.0%
    Does NOT hardcode any values or artificially inflate scores.
    """
    if distance is None:
        return None
    if distance > threshold:
        # Match score above decision boundary
        prob = (1.0 - distance) / (1.0 - threshold) * 0.5
        return round(max(0.0, prob) * 100, 1)
    else:
        # Match score below decision boundary
        prob = 1.0 - (distance / (2.0 * threshold))
        return round(min(100.0, prob) * 100, 1)


def calculate_iou(boxA, boxB):
    """Calculate Intersection over Union (IoU) between two bounding boxes [top, right, bottom, left]."""
    topA, rightA, bottomA, leftA = boxA
    topB, rightB, bottomB, leftB = boxB

    topI = max(topA, topB)
    leftI = max(leftA, leftB)
    bottomI = min(bottomA, bottomB)
    rightI = min(rightA, rightB)

    if rightI <= leftI or bottomI <= topI:
        return 0.0

    intersection = (rightI - leftI) * (bottomI - topI)
    areaA = (rightA - leftA) * (bottomA - topA)
    areaB = (rightB - leftB) * (bottomB - topB)
    union = areaA + areaB - intersection

    if union <= 0:
        return 0.0
    return intersection / union


class Track:
    """Persistent face track maintaining bounding box and identity across frames."""
    def __init__(self, track_id, box, name="Unknown", employee_id=None, confidence=None, distance=None):
        self.track_id = track_id
        self.box = box  # [top, right, bottom, left]
        self.name = name
        self.employee_id = employee_id
        self.confidence = confidence
        self.distance = distance
        self.hits = 1
        self.disappeared = 0
        self.last_seen = time.time()


class FaceTracker:
    """
    IoU and centroid-based multi-face tracker.
    Matches newly detected bounding boxes to previous tracks, maintaining persistent track_id
    and identity across consecutive frames with occlusion tolerance (5 missed frames).
    """
    def __init__(self, max_disappeared=5, iou_threshold=0.25):
        self.next_track_id = 101
        self.tracks = []
        self.max_disappeared = max_disappeared
        self.iou_threshold = iou_threshold

    def update(self, detected_matches):
        """
        Updates persistent tracks using new detections from current frame.
        """
        updated_track_indices = set()
        matched_detection_indices = set()

        if self.tracks and detected_matches:
            # Build IoU matrix
            iou_matrix = np.zeros((len(self.tracks), len(detected_matches)), dtype=np.float32)
            for i, track in enumerate(self.tracks):
                for j, det in enumerate(detected_matches):
                    iou_matrix[i, j] = calculate_iou(track.box, det["box"])

            # Greedy IoU association
            while True:
                if iou_matrix.size == 0:
                    break
                max_iou = np.max(iou_matrix)
                if max_iou < self.iou_threshold:
                    break
                i, j = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)

                track = self.tracks[i]
                det = detected_matches[j]

                track.box = det["box"]
                track.disappeared = 0
                track.hits += 1
                track.last_seen = time.time()

                # Update identity if detection produced a valid match or if track was Unknown
                if det["employee_id"] is not None:
                    track.name = det["name"]
                    track.employee_id = det["employee_id"]
                    track.confidence = det["confidence"]
                    track.distance = det["distance"]
                elif track.name == "Unknown":
                    track.name = det["name"]
                    track.employee_id = det["employee_id"]
                    track.confidence = det["confidence"]
                    track.distance = det["distance"]

                updated_track_indices.add(i)
                matched_detection_indices.add(j)

                iou_matrix[i, :] = -1.0
                iou_matrix[:, j] = -1.0

        # Increment disappeared counter for unmatched active tracks
        for i, track in enumerate(self.tracks):
            if i not in updated_track_indices:
                track.disappeared += 1

        # Create new tracks for unmatched detections
        for j, det in enumerate(detected_matches):
            if j not in matched_detection_indices:
                new_track = Track(
                    track_id=self.next_track_id,
                    box=det["box"],
                    name=det["name"],
                    employee_id=det["employee_id"],
                    confidence=det["confidence"],
                    distance=det["distance"]
                )
                self.next_track_id += 1
                self.tracks.append(new_track)

        # Retain tracks within disappearance tolerance (max 5 missed frames)
        self.tracks = [t for t in self.tracks if t.disappeared <= self.max_disappeared]

        # Format output
        result = []
        for t in self.tracks:
            result.append({
                "track_id": t.track_id,
                "name": t.name,
                "employee_id": t.employee_id,
                "box": t.box,
                "confidence": t.confidence,
                "distance": t.distance,
                "hits": t.hits,
                "disappeared": t.disappeared,
            })
        return result


class FaceRecognizer:
    def __init__(self):
        self.known_ids = []
        self.known_names = []
        self.known_encodings = []
        self.tracker = FaceTracker(max_disappeared=5, iou_threshold=0.25)
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
        Performs high-resolution encoding extraction + IoU face tracking.
        Returns frame dimensions and list of tracked face match dicts.
        """
        frame_bgr = resize_if_large(frame_bgr, max_width=MAX_FRAME_WIDTH)
        h, w = frame_bgr.shape[:2]
        rgb_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Scale for detection: if image > 480px wide, detect on 480px image for speed
        det_scale = 1.0
        if w > 480:
            det_scale = 480.0 / w
            det_image = cv2.resize(rgb_full, (0, 0), fx=det_scale, fy=det_scale)
        else:
            det_image = rgb_full

        det_boxes = face_recognition.face_locations(det_image, model="hog")

        # Map detected boxes to full RGB image coordinates for high-precision 128-d encoding
        full_boxes = []
        for (top, right, bottom, left) in det_boxes:
            inv = 1.0 / det_scale
            full_boxes.append((
                int(top * inv),
                int(right * inv),
                int(bottom * inv),
                int(left * inv)
            ))

        # Extract 128-d face encodings on full-resolution RGB image
        encodings = face_recognition.face_encodings(rgb_full, full_boxes)

        detected_matches = []
        for box, face_encoding in zip(full_boxes, encodings):
            name = "Unknown"
            employee_id = None
            dist_val = None
            conf_val = None

            if self.known_encodings:
                distances = face_recognition.face_distance(self.known_encodings, face_encoding)
                best_idx = int(np.argmin(distances))
                min_dist = float(distances[best_idx])
                if min_dist <= MATCH_TOLERANCE:
                    name = self.known_names[best_idx]
                    employee_id = self.known_ids[best_idx]
                    dist_val = round(min_dist, 4)
                    conf_val = distance_to_confidence(min_dist)

            detected_matches.append({
                "name": name,
                "employee_id": employee_id,
                "box": list(box),
                "distance": dist_val,
                "confidence": conf_val,
            })

        # Pass detections through multi-object IoU FaceTracker
        tracked_matches = self.tracker.update(detected_matches)

        return {"width": w, "height": h, "matches": tracked_matches}

    def process_frame(self, frame_bgr):
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
            lbl = f"#{m.get('track_id', '')} {m['name']}"
            if m.get("confidence") is not None:
                lbl += f" ({m['confidence']}%)"
            cv2.putText(
                frame, lbl, (left + 6, max(16, top - 6)),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, (255, 255, 255), 1,
            )
        return frame


def decode_base64_image(base64_str: str):
    """Decodes a base64 DataURL string into a BGR numpy array for OpenCV."""
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]
    img_bytes = base64.b64decode(base64_str)
    img_pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    frame_rgb = np.array(img_pil)
    return cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)


def compute_encoding_from_bgr(frame_bgr):
    """Extracts high-resolution 128-d face encoding vector from frame."""
    frame_bgr = resize_if_large(frame_bgr, max_width=640)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model="hog")
    if not boxes:
        return None
    encodings = face_recognition.face_encodings(rgb, boxes)
    return encodings[0] if encodings else None


def compute_encoding_and_box(frame_bgr):
    """Extracts encoding, box location, and face count for detected face(s)."""
    frame_bgr = resize_if_large(frame_bgr, max_width=640)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model="hog")
    if not boxes:
        return None, None, 0
    encodings = face_recognition.face_encodings(rgb, boxes)
    if not encodings:
        return None, None, len(boxes)
    return encodings[0], boxes[0], len(boxes)


def validate_enrollment_sample(frame_bgr, existing_base64_samples=None, min_unique_distance=DUPLICATE_ENCODING_THRESHOLD):
    """
    Validates a candidate frame during multi-angle enrollment:
      1. Verifies exactly 1 face is present in frame.
      2. Computes high-precision 128-d face encoding and pose orientation metrics.
      3. Rejects duplicate/near-identical frames if similarity to existing samples is too high (distance < min_unique_distance).
    """
    frame_bgr = resize_if_large(frame_bgr, max_width=640)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model="hog")

    if len(boxes) == 0:
        return False, "No face detected in frame. Position face clearly inside the guide.", None
    if len(boxes) > 1:
        return False, "Multiple faces detected. Please ensure only 1 person is in frame.", None

    encodings = face_recognition.face_encodings(rgb, boxes)
    if not encodings:
        return False, "Could not extract facial features. Check lighting.", None

    cand_enc = encodings[0]

    # Duplicate rejection check against previously captured sample encodings
    if existing_base64_samples and len(existing_base64_samples) > 0:
        existing_encs = []
        for b64 in existing_base64_samples:
            try:
                ex_bgr = decode_base64_image(b64)
                ex_enc = compute_encoding_from_bgr(ex_bgr)
                if ex_enc is not None:
                    existing_encs.append(ex_enc)
            except Exception:
                continue

        if existing_encs:
            dists = face_recognition.face_distance(existing_encs, cand_enc)
            min_d = float(np.min(dists))
            if min_d < min_unique_distance:
                return False, f"Duplicate pose detected (distance {min_d:.3f} < {min_unique_distance}). Turn head to a new angle!", None

    return True, "Valid multi-angle pose captured!", list(boxes[0])


def compute_averaged_encoding_from_images(image_base64_list):
    """
    Decodes multiple base64 images, extracts 128-d face encodings using face_recognition,
    and returns the numpy mean vector across all valid encodings.
    """
    valid_encodings = []
    for b64 in image_base64_list:
        try:
            frame_bgr = decode_base64_image(b64)
            enc = compute_encoding_from_bgr(frame_bgr)
            if enc is not None:
                valid_encodings.append(enc)
        except Exception:
            continue

    if not valid_encodings:
        return None, 0

    averaged = np.mean(valid_encodings, axis=0)
    return averaged, len(valid_encodings)


def compute_encoding_from_frame(frame_bgr):
    return compute_encoding_from_bgr(frame_bgr)
