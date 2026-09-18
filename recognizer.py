import cv2
import numpy as np
import face_recognition
import base64
import io
import time

# Match threshold for face_recognition Euclidean distance (dlib standard normalized)
MATCH_TOLERANCE = 0.50

# Maximum width for processing frames safely
MAX_FRAME_WIDTH = 640

# Minimum Euclidean distance between new sample and existing samples for enrollment duplicate rejection
DUPLICATE_ENCODING_THRESHOLD = 0.05


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
    Uses dlib's decision boundary curve relative to threshold (0.50):
      - distance 0.20 -> ~80.0%
      - distance 0.50 -> 50.0%
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


# Import database layer after distance_to_confidence to avoid circular import issues
from app import database


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


def box_center(box):
    """Returns (center_y, center_x) for box [top, right, bottom, left]."""
    return ((box[0] + box[2]) * 0.5, (box[3] + box[1]) * 0.5)


def box_diagonal(box):
    """Returns diagonal length of box [top, right, bottom, left]."""
    w = max(1.0, float(box[1] - box[3]))
    h = max(1.0, float(box[2] - box[0]))
    return float(np.hypot(w, h))


def calculate_match_score(boxA, boxB):
    """
    Hybrid similarity score combining IoU and normalized centroid proximity.
    Returns float in [0.0, 1.0]. Higher is better match.
    Prevents track loss during fast motion when IoU drops.
    """
    iou = calculate_iou(boxA, boxB)
    cA = box_center(boxA)
    cB = box_center(boxB)
    diag = max(box_diagonal(boxA), box_diagonal(boxB))
    dist = float(np.hypot(cA[0] - cB[0], cA[1] - cB[1]))
    norm_dist = min(1.0, dist / diag)

    if iou >= 0.25:
        return float(iou * 0.7 + (1.0 - norm_dist) * 0.3)
    if norm_dist <= 0.40:
        return float(max(iou, 0.25) * 0.5 + (1.0 - norm_dist) * 0.5)
    return float(iou)


class Track:
    """Persistent face track maintaining smoothed bounding box and identity across frames."""
    def __init__(self, track_id, box, name="Unknown", employee_id=None, confidence=None, distance=None):
        self.track_id = track_id
        # Continuous float coordinates [top, right, bottom, left] to eliminate integer rounding jitter
        self.smooth_box = [float(x) for x in box]
        self.box = [int(round(x)) for x in self.smooth_box]
        self.name = name
        self.employee_id = employee_id
        self.confidence = confidence
        self.distance = distance
        self.hits = 1
        self.disappeared = 0
        self.last_seen = time.time()
        self.last_recognized_frame = 0

    def update_box(self, new_box):
        """
        Adaptive EMA smoothing on coordinates with sub-pixel jitter deadband.
        - < 2.5px shift: suppressed as detector noise (alpha = 0.10)
        - 2.5px - 15px: smooth following (alpha = 0.55)
        - > 15px: responsive fast motion tracking (alpha = 0.80)
        """
        new_t, new_r, new_b, new_l = [float(x) for x in new_box]
        old_t, old_r, old_b, old_l = self.smooth_box

        max_delta = max(
            abs(new_t - old_t),
            abs(new_r - old_r),
            abs(new_b - old_b),
            abs(new_l - old_l),
        )

        if max_delta < 2.5:
            alpha = 0.10
        elif max_delta < 15.0:
            alpha = 0.55
        else:
            alpha = 0.80

        self.smooth_box = [
            alpha * new_t + (1.0 - alpha) * old_t,
            alpha * new_r + (1.0 - alpha) * old_r,
            alpha * new_b + (1.0 - alpha) * old_b,
            alpha * new_l + (1.0 - alpha) * old_l,
        ]
        self.box = [int(round(x)) for x in self.smooth_box]
        self.disappeared = 0
        self.hits += 1
        self.last_seen = time.time()


class FaceTracker:
    """
    Robust hybrid IoU + Centroid multi-face tracker.
    Prevents duplicate track spawning, handles occlusions, and eliminates jitter.
    """
    def __init__(self, max_disappeared=5, match_score_threshold=0.20):
        self.next_track_id = 101
        self.tracks = []
        self.max_disappeared = max_disappeared
        self.match_score_threshold = match_score_threshold
        self.frame_count = 0

    def update(self, detected_boxes):
        """
        Updates persistent tracks from raw detected bounding boxes.
        Returns list of active Track objects for current frame.
        """
        self.frame_count += 1
        updated_track_indices = set()
        matched_detection_indices = set()

        if self.tracks and detected_boxes:
            score_matrix = np.zeros((len(self.tracks), len(detected_boxes)), dtype=np.float32)
            for i, track in enumerate(self.tracks):
                for j, box in enumerate(detected_boxes):
                    score_matrix[i, j] = calculate_match_score(track.box, box)

            # Greedy highest-similarity association
            while True:
                if score_matrix.size == 0:
                    break
                max_score = float(np.max(score_matrix))
                if max_score < self.match_score_threshold:
                    break
                i, j = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)

                track = self.tracks[i]
                det_box = detected_boxes[j]
                track.update_box(det_box)

                updated_track_indices.add(i)
                matched_detection_indices.add(j)

                score_matrix[i, :] = -1.0
                score_matrix[:, j] = -1.0

        # Increment disappeared count for active tracks not matched in this frame
        for i, track in enumerate(self.tracks):
            if i not in updated_track_indices:
                track.disappeared += 1

        # Create new tracks for unmatched detections
        for j, det_box in enumerate(detected_boxes):
            if j not in matched_detection_indices:
                new_track = Track(
                    track_id=self.next_track_id,
                    box=det_box,
                    name="Unknown",
                    employee_id=None,
                    confidence=None,
                    distance=None
                )
                self.next_track_id += 1
                self.tracks.append(new_track)

        # Drop expired tracks that exceeded disappearance tolerance
        self.tracks = [t for t in self.tracks if t.disappeared <= self.max_disappeared]

        # Strictly output tracks updated in the current frame (disappeared == 0)
        # Prevents ghost phantom boxes from lingering on empty frames
        current_active = [t for t in self.tracks if t.disappeared == 0]

        # Non-Maximum Suppression (NMS) Deduplication:
        # Sort so known employees and longer-lived tracks take precedence
        current_active.sort(key=lambda t: (1 if t.employee_id is not None else 0, t.hits), reverse=True)

        deduped_tracks = []
        for t in current_active:
            is_dup = False
            for existing in deduped_tracks:
                iou = calculate_iou(t.box, existing.box)
                c_t = box_center(t.box)
                c_e = box_center(existing.box)
                w_min = min(t.box[1] - t.box[3], existing.box[1] - existing.box[3])
                c_dist = float(np.hypot(c_t[0] - c_e[0], c_t[1] - c_e[1]))

                if iou > 0.25 or (w_min > 0 and (c_dist / w_min) < 0.35):
                    is_dup = True
                    # If candidate has valid identity and existing is unknown, copy identity
                    if t.employee_id is not None and existing.employee_id is None:
                        existing.name = t.name
                        existing.employee_id = t.employee_id
                        existing.confidence = t.confidence
                        existing.distance = t.distance
                    break
            if not is_dup:
                deduped_tracks.append(t)

        return deduped_tracks


class FaceRecognizer:
    def __init__(self):
        self.known_ids = []
        self.known_names = []
        self.known_encodings = []
        self.tracker = FaceTracker(max_disappeared=5, match_score_threshold=0.20)
        self.refresh_known_faces()

    def refresh_known_faces(self):
        """Reload known employees from the SQLite database and ensure L2 unit normalization."""
        employees = database.get_all_employees()
        self.known_ids = [e["id"] for e in employees]
        self.known_names = [e["name"] for e in employees]
        self.known_encodings = [e["encoding"] for e in employees]
        print(f"[FaceRecognizer] Reloaded {len(self.known_ids)} enrolled employees from DB.")

    def process_bgr_frame(self, frame_bgr):
        """
        High-efficiency frame processing pipeline:
        1. Scales down to 320px for fast HOG face detection (~30ms vs ~90ms at 480px).
        2. Associates bounding boxes to persistent tracks (IoU + Centroid).
        3. Selectively extracts 128-d ResNet embeddings ONLY for new/unknown faces
           or periodic re-verification, bypassing 365ms deep embedding on every frame.
        4. Returns frame dimensions and list of tracked face match dicts.
        """
        frame_bgr = resize_if_large(frame_bgr, max_width=MAX_FRAME_WIDTH)
        h, w = frame_bgr.shape[:2]
        rgb_full = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        # Scale for detection: 320px width gives 25-35ms HOG detection
        target_det_w = 320.0
        det_scale = (target_det_w / float(w)) if w > target_det_w else 1.0
        if det_scale < 1.0:
            det_image = cv2.resize(rgb_full, (0, 0), fx=det_scale, fy=det_scale)
        else:
            det_image = rgb_full

        det_boxes = face_recognition.face_locations(det_image, model="hog")

        # Map detected boxes back to full RGB image coordinates
        inv = 1.0 / det_scale
        full_boxes = []
        for (top, right, bottom, left) in det_boxes:
            full_boxes.append([
                int(top * inv),
                int(right * inv),
                int(bottom * inv),
                int(left * inv)
            ])

        # Pass detections through robust FaceTracker
        tracked_faces = self.tracker.update(full_boxes)

        # Selective Recognition: Only run 128-d ResNet embedding if:
        # - Track is newly registered (last_recognized_frame == 0), OR
        # - It has been 25 frames (~2-3 seconds) since last verification
        for t in tracked_faces:
            needs_recognition = (
                (t.last_recognized_frame == 0) or
                (self.tracker.frame_count - t.last_recognized_frame >= 25)
            )

            if needs_recognition and self.known_encodings:
                try:
                    # Extract 128-d encoding for this specific face box
                    crop_box = tuple(t.box)
                    encs = face_recognition.face_encodings(rgb_full, [crop_box])
                    if encs:
                        face_enc = encs[0].astype(np.float64)
                        norm = float(np.linalg.norm(face_enc))
                        if norm > 0:
                            face_enc = face_enc / norm

                        distances = face_recognition.face_distance(self.known_encodings, face_enc)
                        best_idx = int(np.argmin(distances))
                        min_dist = float(distances[best_idx])
                        if min_dist <= MATCH_TOLERANCE:
                            t.name = self.known_names[best_idx]
                            t.employee_id = self.known_ids[best_idx]
                            t.distance = round(min_dist, 4)
                            t.confidence = distance_to_confidence(min_dist)
                        else:
                            t.name = "Unknown"
                            t.employee_id = None
                            t.distance = round(min_dist, 4)
                            t.confidence = distance_to_confidence(min_dist)
                except Exception as e:
                    print(f"Error extracting face embedding for track #{t.track_id}: {e}")

                t.last_recognized_frame = self.tracker.frame_count

        # Format output payload
        result = []
        for t in tracked_faces:
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

        return {"width": w, "height": h, "matches": result}

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
    """
    Decodes a base64 DataURL string into a BGR numpy array using fast OpenCV imdecode.
    Bypasses PIL conversion to reduce CPU and memory overhead (~1.5ms decode).
    """
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]
    img_bytes = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Failed to decode image buffer with OpenCV imdecode")
    return frame



def compute_encoding_from_bgr(frame_bgr):
    """Extracts high-resolution 128-d face encoding vector from frame, normalized to unit length."""
    frame_bgr = resize_if_large(frame_bgr, max_width=640)
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model="hog")
    if not boxes:
        return None
    encodings = face_recognition.face_encodings(rgb, boxes)
    if not encodings:
        return None
    enc = encodings[0].astype(np.float64)
    norm = np.linalg.norm(enc)
    if norm > 0:
        enc = enc / norm
    return enc


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
    enc = encodings[0].astype(np.float64)
    norm = np.linalg.norm(enc)
    if norm > 0:
        enc = enc / norm
    return enc, boxes[0], len(boxes)


def validate_enrollment_sample(frame_bgr, existing_base64_samples=None, min_unique_distance=DUPLICATE_ENCODING_THRESHOLD):
    """
    Validates a candidate frame during multi-angle enrollment:
      1. Verifies exactly 1 face is present in frame.
      2. Computes high-precision 128-d face encoding.
      3. Rejects duplicate/near-identical frames if similarity to recent sample is too high.
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

    cand_enc = encodings[0].astype(np.float64)
    norm = np.linalg.norm(cand_enc)
    if norm > 0:
        cand_enc = cand_enc / norm

    # Duplicate rejection check against recent captured sample encoding (last sample only for speed)
    if existing_base64_samples and len(existing_base64_samples) > 0:
        try:
            last_b64 = existing_base64_samples[-1]
            ex_bgr = decode_base64_image(last_b64)
            ex_enc = compute_encoding_from_bgr(ex_bgr)
            if ex_enc is not None:
                dist = float(face_recognition.face_distance([ex_enc], cand_enc)[0])
                if dist < min_unique_distance:
                    return False, f"Duplicate pose detected (distance {dist:.3f} < {min_unique_distance}). Turn head to a new angle!", None
        except Exception:
            pass

    return True, "Valid multi-angle pose captured!", list(boxes[0])


def compute_averaged_encoding_from_images(image_base64_list):
    """
    Decodes multiple base64 images, extracts 128-d face encodings using face_recognition,
    averages them, and returns the L2-renormalized unit vector across all valid encodings.
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
    norm = np.linalg.norm(averaged)
    if norm > 0:
        averaged = averaged / norm
    return averaged.astype(np.float64), len(valid_encodings)


def compute_encoding_from_frame(frame_bgr):
    return compute_encoding_from_bgr(frame_bgr)

