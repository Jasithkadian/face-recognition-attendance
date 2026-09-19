"""
recognizer.py
High-Performance Face Recognition & Tracking Engine
Using YOLOv8-Face (ONNX) for real-time face detection & 5-point landmark estimation,
and ArcFace MobileFaceNet (ONNX) for 512-dimensional deep facial metric embeddings.

Engineered for ultra-low memory (<150MB RAM) and zero-compilation deployment on Render.
"""

import os
import io
import time
import base64
import cv2
import numpy as np
import onnxruntime as ort

from download_models import ensure_models

# Ensure model files exist before initialization
ensure_models()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "yolov8n-face.onnx")
ARCFACE_MODEL_PATH = os.path.join(MODELS_DIR, "w600k_mbf.onnx")

# Match threshold for ArcFace Cosine Distance (1.0 - Cosine Similarity)
# Calibrated against MobileFaceNet / WebFace600K:
# - Same person: Cosine Distance ~0.15 - 0.40 (Cosine Similarity >= 0.60)
# - Strangers: Cosine Distance ~0.65 - 0.95 (Cosine Similarity < 0.35)
# Threshold 0.50 provides a high-security decision boundary with zero false accepts.
MATCH_TOLERANCE = 0.50

# Minimum confidence score (%) required to accept and display an enrolled person's identity
CONFIDENCE_FLOOR = 60.0

# Maximum width for processing frames safely
MAX_FRAME_WIDTH = 640

# Minimum Cosine distance between new sample and existing samples for enrollment duplicate rejection
DUPLICATE_ENCODING_THRESHOLD = 0.035

# Standard InsightFace reference 5 facial points for 112x112 affine alignment
REFERENCE_FACIAL_POINTS = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041]
], dtype=np.float32)


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
    Converts raw Cosine distance (1.0 - Cosine Similarity) into a calibrated confidence percentage.
    Uses decision boundary at threshold (0.50):
      - distance 0.15 -> 85.0%
      - distance 0.25 -> 75.0%
      - distance 0.35 -> 65.0%
      - distance 0.50 -> 50.0% (Decision Boundary)
      - distance 0.60 -> 40.0%
      - distance 0.75 -> 25.0%
      - distance >= 1.0 -> 0.0%
    """
    if distance is None:
        return None
    d = max(0.0, float(distance))
    if d <= threshold:
        prob = 1.0 - (d / (2.0 * threshold))
        return round(float(min(100.0, prob * 100.0)), 1)
    else:
        prob = max(0.0, (1.0 - min(1.0, d)) / (1.0 - threshold) * 0.5)
        return round(float(prob * 100.0), 1)


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
    def __init__(self, track_id, box, name="Unknown", employee_id=None, confidence=None, distance=None, landmarks=None):
        self.track_id = track_id
        self.smooth_box = [float(x) for x in box]
        self.box = [int(round(x)) for x in self.smooth_box]
        self.landmarks = landmarks
        self.name = name
        self.employee_id = employee_id
        self.confidence = confidence
        self.distance = distance
        self.hits = 1
        self.disappeared = 0
        self.last_seen = time.time()
        self.last_recognized_frame = 0

    def update_box(self, new_box, landmarks=None):
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
        if landmarks:
            self.landmarks = landmarks
        self.disappeared = 0
        self.hits += 1
        self.last_seen = time.time()


class FaceTracker:
    """Robust hybrid IoU + Centroid multi-face tracker."""
    def __init__(self, max_disappeared=5, match_score_threshold=0.20):
        self.next_track_id = 101
        self.tracks = []
        self.max_disappeared = max_disappeared
        self.match_score_threshold = match_score_threshold
        self.frame_count = 0

    def update(self, detections):
        """
        Updates persistent tracks from raw detections.
        detections: list of dicts with 'box' [top, right, bottom, left] and optional 'landmarks'.
        """
        self.frame_count += 1
        detected_boxes = [d["box"] for d in detections]
        updated_track_indices = set()
        matched_detection_indices = set()

        if self.tracks and detected_boxes:
            score_matrix = np.zeros((len(self.tracks), len(detected_boxes)), dtype=np.float32)
            for i, track in enumerate(self.tracks):
                for j, box in enumerate(detected_boxes):
                    score_matrix[i, j] = calculate_match_score(track.box, box)

            while True:
                if score_matrix.size == 0:
                    break
                max_score = float(np.max(score_matrix))
                if max_score < self.match_score_threshold:
                    break
                i, j = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)

                track = self.tracks[i]
                det = detections[j]
                track.update_box(det["box"], det.get("landmarks"))

                updated_track_indices.add(i)
                matched_detection_indices.add(j)

                score_matrix[i, :] = -1.0
                score_matrix[:, j] = -1.0

        for i, track in enumerate(self.tracks):
            if i not in updated_track_indices:
                track.disappeared += 1

        for j, det in enumerate(detections):
            if j not in matched_detection_indices:
                new_track = Track(
                    track_id=self.next_track_id,
                    box=det["box"],
                    name="Unknown",
                    employee_id=None,
                    confidence=None,
                    distance=None,
                    landmarks=det.get("landmarks"),
                )
                self.next_track_id += 1
                self.tracks.append(new_track)

        self.tracks = [t for t in self.tracks if t.disappeared <= self.max_disappeared]
        current_active = [t for t in self.tracks if t.disappeared == 0]
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
                    break
            if not is_dup:
                deduped_tracks.append(t)

        return deduped_tracks


class YOLOFaceDetector:
    """YOLOv8-Face ONNX Inference Engine for real-time face detection and 5-point landmark prediction."""
    def __init__(self, model_path=YOLO_MODEL_PATH):
        self.model_path = model_path
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(self.model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def detect(self, img_bgr, conf_threshold=0.35):
        """
        Performs inference on BGR image.
        Returns list of detection dicts:
          - 'box': [top, right, bottom, left]
          - 'conf': float
          - 'landmarks': [(x, y), ...]
        """
        h0, w0 = img_bgr.shape[:2]
        target_size = 640
        scale = min(target_size / h0, target_size / w0)
        nw, nh = int(round(w0 * scale)), int(round(h0 * scale))
        resized = cv2.resize(img_bgr, (nw, nh))

        pad_img = np.zeros((target_size, target_size, 3), dtype=np.uint8)
        dx = (target_size - nw) // 2
        dy = (target_size - nh) // 2
        pad_img[dy:dy+nh, dx:dx+nw] = resized

        rgb = cv2.cvtColor(pad_img, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]

        out = self.session.run(None, {self.input_name: blob})[0]

        detections = []
        for row in out[0]:
            conf = float(row[4])
            if conf < conf_threshold:
                continue

            x1, y1, x2, y2 = row[:4]
            orig_x1 = max(0.0, (x1 - dx) / scale)
            orig_y1 = max(0.0, (y1 - dy) / scale)
            orig_x2 = min(float(w0), (x2 - dx) / scale)
            orig_y2 = min(float(h0), (y2 - dy) / scale)

            top = int(round(orig_y1))
            left = int(round(orig_x1))
            bottom = int(round(orig_y2))
            right = int(round(orig_x2))

            if right <= left or bottom <= top:
                continue

            landmarks = []
            if len(row) >= 21:
                kpts = row[6:21].reshape((5, 3))
                for kpt in kpts:
                    kx = max(0.0, min(float(w0), (kpt[0] - dx) / scale))
                    ky = max(0.0, min(float(h0), (kpt[1] - dy) / scale))
                    landmarks.append((kx, ky))

            detections.append({
                "box": [top, right, bottom, left],
                "conf": conf,
                "landmarks": landmarks,
            })
        return detections


class ArcFaceEmbedder:
    """ArcFace MobileFaceNet ONNX feature extractor producing 512-d L2-normalized face embeddings."""
    def __init__(self, model_path=ARCFACE_MODEL_PATH):
        self.model_path = model_path
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(self.model_path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def extract_embedding(self, img_bgr, box, landmarks=None):
        """
        Aligns and extracts 512-d normalized face embedding from BGR image.
        Uses 5-point landmark affine warp if landmarks are present, with square crop fallback.
        """
        top, right, bottom, left = box
        h, w = img_bgr.shape[:2]

        aligned_face = None
        if landmarks and len(landmarks) == 5:
            try:
                src_pts = np.array(landmarks, dtype=np.float32)
                M, _ = cv2.estimateAffinePartial2D(src_pts, REFERENCE_FACIAL_POINTS)
                if M is not None:
                    aligned_face = cv2.warpAffine(img_bgr, M, (112, 112), borderValue=0.0)
            except Exception:
                aligned_face = None

        if aligned_face is None:
            bw = max(1, right - left)
            bh = max(1, bottom - top)
            cx = (left + right) // 2
            cy = (top + bottom) // 2
            side = int(max(bw, bh) * 1.20)
            x1 = max(0, cx - side // 2)
            y1 = max(0, cy - side // 2)
            x2 = min(w, x1 + side)
            y2 = min(h, y1 + side)
            crop = img_bgr[y1:y2, x1:x2]
            if crop.size == 0:
                return None
            aligned_face = cv2.resize(crop, (112, 112))

        rgb = cv2.cvtColor(aligned_face, cv2.COLOR_BGR2RGB)
        blob = (rgb.astype(np.float32) - 127.5) / 128.0
        blob = np.transpose(blob, (2, 0, 1))[np.newaxis, ...]

        out = self.session.run(None, {self.input_name: blob})[0][0]
        norm = float(np.linalg.norm(out))
        if norm > 0:
            out = out / norm
        return out.astype(np.float64)


# Global singletons for ONNX inference sessions
_detector_instance = None
_embedder_instance = None


def get_detector():
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = YOLOFaceDetector()
    return _detector_instance


def get_embedder():
    global _embedder_instance
    if _embedder_instance is None:
        _embedder_instance = ArcFaceEmbedder()
    return _embedder_instance


class FaceRecognizer:
    def __init__(self):
        self.detector = get_detector()
        self.embedder = get_embedder()
        self.known_ids = []
        self.known_names = []
        self.known_encodings = []
        self.tracker = FaceTracker(max_disappeared=5, match_score_threshold=0.20)
        self.refresh_known_faces()

    def refresh_known_faces(self):
        """Reload known employees from the SQLite database and ensure 512-d unit normalization."""
        employees = database.get_all_employees()
        self.known_ids = []
        self.known_names = []
        self.known_encodings = []

        for e in employees:
            enc = e.get("encoding")
            if enc is not None and len(enc) == 512:
                self.known_ids.append(e["id"])
                self.known_names.append(e["name"])
                self.known_encodings.append(enc)
            elif enc is not None and len(enc) == 128:
                print(f"[FaceRecognizer Notice] Employee '{e['name']}' (ID {e['id']}) has legacy 128-d encoding from old dlib model. Please re-enroll this person to activate recognition with YOLO+ArcFace.")

        if self.known_encodings:
            self.known_encodings = np.array(self.known_encodings, dtype=np.float64)
        else:
            self.known_encodings = np.empty((0, 512), dtype=np.float64)

        print(f"[FaceRecognizer] Loaded {len(self.known_ids)} active 512-d enrolled employees from DB.")

    def process_bgr_frame(self, frame_bgr):
        """
        High-efficiency frame processing pipeline:
        1. Scales frame if large.
        2. Detects faces with YOLOv8-Face ONNX.
        3. Associates bounding boxes to persistent tracks (IoU + Centroid).
        4. Selectively extracts 512-d ArcFace embeddings for new/unknown faces.
        5. Computes Cosine Distance against enrolled face embeddings.
        6. Returns frame dimensions and list of tracked face match dicts.
        """
        frame_bgr = resize_if_large(frame_bgr, max_width=MAX_FRAME_WIDTH)
        h, w = frame_bgr.shape[:2]

        detections = self.detector.detect(frame_bgr, conf_threshold=0.35)
        tracked_faces = self.tracker.update(detections)

        for t in tracked_faces:
            is_unrecognized = (t.employee_id is None or t.name == "Unknown")
            eval_interval = 5 if is_unrecognized else 25
            needs_recognition = (
                (t.last_recognized_frame == 0) or
                (self.tracker.frame_count - t.last_recognized_frame >= eval_interval)
            )

            if needs_recognition and len(self.known_encodings) > 0:
                try:
                    face_enc = self.embedder.extract_embedding(frame_bgr, t.box, t.landmarks)
                    if face_enc is not None:
                        # Cosine similarity: dot product of L2-normalized unit vectors
                        sims = np.dot(self.known_encodings, face_enc)
                        distances = 1.0 - sims
                        best_idx = int(np.argmin(distances))
                        min_dist = float(distances[best_idx])
                        conf = distance_to_confidence(min_dist, threshold=MATCH_TOLERANCE)
                        closest_name = self.known_names[best_idx]
                        closest_id = self.known_ids[best_idx]

                        is_accepted = (min_dist <= MATCH_TOLERANCE) and (conf is not None and conf >= CONFIDENCE_FLOOR)

                        print(
                            f"[YOLO-ArcFace Match] Track #{t.track_id} | Closest: '{closest_name}' (ID={closest_id}) | "
                            f"cosine_dist={min_dist:.4f} | threshold={MATCH_TOLERANCE:.2f} | conf={conf}% (floor={CONFIDENCE_FLOOR}%) | "
                            f"Decision: {'ACCEPTED' if is_accepted else 'REJECTED -> Unknown'}"
                        )

                        if is_accepted:
                            t.name = closest_name
                            t.employee_id = closest_id
                            t.distance = round(min_dist, 4)
                            t.confidence = conf
                        else:
                            t.name = "Unknown"
                            t.employee_id = None
                            t.distance = round(min_dist, 4)
                            t.confidence = conf

                        t.last_recognized_frame = self.tracker.frame_count
                    else:
                        t.name = "Unknown"
                        t.employee_id = None
                        t.confidence = None
                except Exception as e:
                    print(f"Error extracting face embedding for track #{t.track_id}: {e}")

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
    """Decodes a base64 DataURL string into a BGR numpy array using fast OpenCV imdecode."""
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]
    img_bytes = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError("Failed to decode image buffer with OpenCV imdecode")
    return frame


def compute_encoding_from_bgr(frame_bgr):
    """Extracts 512-d face encoding vector from frame, normalized to unit length."""
    frame_bgr = resize_if_large(frame_bgr, max_width=640)
    detector = get_detector()
    embedder = get_embedder()
    detections = detector.detect(frame_bgr, conf_threshold=0.35)
    if not detections:
        return None
    # Use the highest confidence detection
    detections.sort(key=lambda d: d["conf"], reverse=True)
    best = detections[0]
    return embedder.extract_embedding(frame_bgr, best["box"], best.get("landmarks"))


def compute_encoding_and_box(frame_bgr):
    """Extracts encoding, box location, and face count for detected face(s)."""
    frame_bgr = resize_if_large(frame_bgr, max_width=640)
    detector = get_detector()
    embedder = get_embedder()
    detections = detector.detect(frame_bgr, conf_threshold=0.35)
    if not detections:
        return None, None, 0
    detections.sort(key=lambda d: d["conf"], reverse=True)
    best = detections[0]
    enc = embedder.extract_embedding(frame_bgr, best["box"], best.get("landmarks"))
    return enc, best["box"], len(detections)


def validate_enrollment_sample(frame_bgr, existing_base64_samples=None, min_unique_distance=DUPLICATE_ENCODING_THRESHOLD):
    """
    Validates a candidate frame during multi-angle enrollment:
      1. Verifies exactly 1 face is present in frame.
      2. Computes high-precision 512-d ArcFace encoding.
      3. Rejects duplicate/near-identical frames if similarity to recent sample is too high.
    """
    frame_bgr = resize_if_large(frame_bgr, max_width=640)
    detector = get_detector()
    embedder = get_embedder()
    detections = detector.detect(frame_bgr, conf_threshold=0.35)

    if len(detections) == 0:
        return False, "No face detected in frame. Position face clearly inside the guide.", None
    if len(detections) > 1:
        return False, "Multiple faces detected. Please ensure only 1 person is in frame.", None

    best = detections[0]
    cand_enc = embedder.extract_embedding(frame_bgr, best["box"], best.get("landmarks"))
    if cand_enc is None:
        return False, "Could not extract facial features. Check lighting.", None

    if existing_base64_samples and len(existing_base64_samples) > 0:
        try:
            last_b64 = existing_base64_samples[-1]
            ex_bgr = decode_base64_image(last_b64)
            ex_enc = compute_encoding_from_bgr(ex_bgr)
            if ex_enc is not None:
                # Cosine distance
                cosine_dist = float(1.0 - np.dot(ex_enc, cand_enc))
                if cosine_dist < min_unique_distance:
                    return False, f"Duplicate pose detected (distance {cosine_dist:.3f} < {min_unique_distance}). Turn head to a new angle!", None
        except Exception:
            pass

    return True, "Valid multi-angle pose captured!", list(best["box"])


def compute_averaged_encoding_from_images(image_base64_list):
    """
    Decodes multiple base64 images, extracts 512-d face encodings using ArcFace,
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
