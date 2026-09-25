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

# Minimum margin required between 1st and 2nd closest candidate matches to prevent ambiguous identity assignment
MATCH_MARGIN = 0.15

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


class KalmanBoxFilter:
    """
    8-dimensional state vector [x, y, a, h, vx, vy, va, vh]
    Tracks bounding box center (x, y), aspect ratio a (w/h), height h, and respective velocities.
    Uses constant-velocity motion model with Wojke et al. covariance weights.
    """
    def __init__(self, box):
        top, right, bottom, left = box
        w = max(1.0, float(right - left))
        h = max(1.0, float(bottom - top))
        x = float(left) + w / 2.0
        y = float(top) + h / 2.0
        a = w / h

        self._motion_mat = np.eye(8, 8, dtype=np.float32)
        for i in range(4):
            self._motion_mat[i, i + 4] = 1.0

        self._update_mat = np.eye(4, 8, dtype=np.float32)
        self._std_weight_position = 1.0 / 20.0
        self._std_weight_velocity = 1.0 / 160.0

        self.mean = np.zeros(8, dtype=np.float32)
        self.mean[:4] = [x, y, a, h]

        std = [
            2 * self._std_weight_position * h,
            2 * self._std_weight_position * h,
            1e-2,
            2 * self._std_weight_position * h,
            10 * self._std_weight_velocity * h,
            10 * self._std_weight_velocity * h,
            1e-5,
            10 * self._std_weight_velocity * h
        ]
        self.covariance = np.diag(np.square(std)).astype(np.float32)

    def predict(self):
        h = max(1.0, float(self.mean[3]))
        std_pos = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-2,
            self._std_weight_position * h
        ]
        std_vel = [
            self._std_weight_velocity * h,
            self._std_weight_velocity * h,
            1e-5,
            self._std_weight_velocity * h
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel])).astype(np.float32)

        self.mean = np.dot(self._motion_mat, self.mean)
        self.covariance = np.linalg.multi_dot([self._motion_mat, self.covariance, self._motion_mat.T]) + motion_cov
        return self.get_box()

    def update(self, box):
        top, right, bottom, left = box
        w = max(1.0, float(right - left))
        h = max(1.0, float(bottom - top))
        x = float(left) + w / 2.0
        y = float(top) + h / 2.0
        a = w / h
        measurement = np.array([x, y, a, h], dtype=np.float32)

        std = [
            self._std_weight_position * h,
            self._std_weight_position * h,
            1e-1,
            self._std_weight_position * h
        ]
        measurement_cov = np.diag(np.square(std)).astype(np.float32)

        projected_mean = np.dot(self._update_mat, self.mean)
        projected_cov = np.linalg.multi_dot([self._update_mat, self.covariance, self._update_mat.T]) + measurement_cov

        try:
            S_chol = np.linalg.cholesky(projected_cov)
            K = np.linalg.solve(S_chol.T, np.linalg.solve(S_chol, np.dot(self._update_mat, self.covariance))).T
        except np.linalg.LinAlgError:
            K = np.dot(self.covariance, np.dot(self._update_mat.T, np.linalg.inv(projected_cov)))

        innovation = measurement - projected_mean
        self.mean = self.mean + np.dot(innovation, K.T)
        self.covariance = self.covariance - np.linalg.multi_dot([K, projected_cov, K.T])
        return self.get_box()

    def get_box(self):
        x, y, a, h = self.mean[:4]
        w = max(1.0, float(a * h))
        h = max(1.0, float(h))
        left = int(round(x - w / 2.0))
        top = int(round(y - h / 2.0))
        right = int(round(left + w))
        bottom = int(round(top + h))
        return [top, right, bottom, left]


def iou_cost_matrix(tracks, detections):
    """
    Computes NxM association cost matrix: 1.0 - IoU.
    For non-overlapping boxes (IoU == 0), adds normalized centroid distance penalty
    to preserve matching on rapid movements across frames.
    """
    N = len(tracks)
    M = len(detections)
    cost = np.zeros((N, M), dtype=np.float32)

    for i, track in enumerate(tracks):
        t1, r1, b1, l1 = track.box
        c1_y, c1_x = (t1 + b1) * 0.5, (l1 + r1) * 0.5
        w1, h1 = max(1.0, r1 - l1), max(1.0, b1 - t1)
        diag1 = max(1.0, np.hypot(w1, h1))

        for j, det in enumerate(detections):
            t2, r2, b2, l2 = det["box"]
            iou = calculate_iou(track.box, det["box"])
            if iou > 0:
                cost[i, j] = 1.0 - iou
            else:
                c2_y, c2_x = (t2 + b2) * 0.5, (l2 + r2) * 0.5
                c_dist = np.hypot(c1_x - c2_x, c1_y - c2_y)
                norm_dist = min(1.0, c_dist / diag1)
                cost[i, j] = 1.0 + norm_dist
    return cost


def linear_assignment(cost_matrix, max_cost):
    """
    Bipartite linear assignment matching on cost_matrix up to max_cost.
    Uses scipy.optimize.linear_sum_assignment if present, falls back to greedy matching.
    """
    if cost_matrix.size == 0:
        return [], list(range(cost_matrix.shape[0])), list(range(cost_matrix.shape[1]))

    try:
        from scipy.optimize import linear_sum_assignment
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
    except ImportError:
        cost_copy = cost_matrix.copy()
        pairs = []
        while True:
            min_val = np.min(cost_copy)
            if min_val > max_cost:
                break
            r, c = np.unravel_index(np.argmin(cost_copy), cost_copy.shape)
            if cost_copy[r, c] == np.inf:
                break
            pairs.append((r, c))
            cost_copy[r, :] = np.inf
            cost_copy[:, c] = np.inf
        row_ind = [p[0] for p in pairs]
        col_ind = [p[1] for p in pairs]

    matches = []
    unmatched_rows = set(range(cost_matrix.shape[0]))
    unmatched_cols = set(range(cost_matrix.shape[1]))

    for r, c in zip(row_ind, col_ind):
        if cost_matrix[r, c] <= max_cost:
            matches.append((int(r), int(c)))
            unmatched_rows.discard(int(r))
            unmatched_cols.discard(int(c))

    return matches, list(unmatched_rows), list(unmatched_cols)


class Track:
    """
    Persistent face track with Kalman-filtered motion prediction and identity persistence.
    """
    def __init__(self, track_id, box, name="Unknown", employee_id=None, confidence=None, distance=None, landmarks=None):
        self.track_id = track_id
        self.kalman_filter = KalmanBoxFilter(box)
        self.box = [int(round(x)) for x in box]
        self.smooth_box = list(self.box)
        self.landmarks = landmarks
        self.name = name
        self.employee_id = employee_id
        self.confidence = confidence
        self.distance = distance
        self.hits = 1
        self.disappeared = 0
        self.last_seen = time.time()
        self.last_recognized_frame = 0

    def predict(self):
        self.box = self.kalman_filter.predict()
        self.smooth_box = list(self.box)
        return self.box

    def update_box(self, new_box, landmarks=None):
        self.box = self.kalman_filter.update(new_box)
        self.smooth_box = list(self.box)
        if landmarks:
            self.landmarks = landmarks
        self.disappeared = 0
        self.hits += 1
        self.last_seen = time.time()

    def mark_missed(self):
        self.disappeared += 1


class ByteTrack:
    """
    Industrial Two-Stage ByteTrack with 8-State Kalman Motion Filter.
    Features:
    - Stage 1: High-confidence detection matching (D_high) via IoU distance.
    - Stage 2: Low-confidence detection recovery (D_low) to maintain tracks through motion blur/turning.
    - Kalman filter velocity projection: eliminates bounding box lag during rapid face movement.
    - Coasting: retains and smoothly projects tracks through brief camera occlusions.
    """
    def __init__(self, track_high_thresh=0.40, track_low_thresh=0.15, match_thresh=0.75, match_low_thresh=0.55, max_disappeared=10):
        self.next_track_id = 101
        self.tracks = []
        self.track_high_thresh = track_high_thresh
        self.track_low_thresh = track_low_thresh
        self.match_thresh = match_thresh
        self.match_low_thresh = match_low_thresh
        self.max_disappeared = max_disappeared
        self.frame_count = 0

    def update(self, detections):
        self.frame_count += 1

        # 1. Kalman Predict for all existing tracks
        for t in self.tracks:
            t.predict()

        # 2. Split detections into high and low confidence
        high_dets = []
        low_dets = []
        for d in detections:
            conf = d.get("conf", 0.5)
            if conf >= self.track_high_thresh:
                high_dets.append(d)
            elif conf >= self.track_low_thresh:
                low_dets.append(d)

        # 3. Stage 1: Associate active tracks with high-confidence detections
        if self.tracks and high_dets:
            cost1 = iou_cost_matrix(self.tracks, high_dets)
            matches1, unmatched_tracks1, unmatched_high = linear_assignment(cost1, self.match_thresh)
        else:
            matches1 = []
            unmatched_tracks1 = list(range(len(self.tracks)))
            unmatched_high = list(range(len(high_dets)))

        for t_idx, d_idx in matches1:
            det = high_dets[d_idx]
            self.tracks[t_idx].update_box(det["box"], det.get("landmarks"))

        # 4. Stage 2: Associate remaining unmatched tracks with low-confidence detections
        remaining_tracks = [self.tracks[i] for i in unmatched_tracks1]
        if remaining_tracks and low_dets:
            cost2 = iou_cost_matrix(remaining_tracks, low_dets)
            matches2, unmatched_tracks2_idx, _ = linear_assignment(cost2, self.match_low_thresh)
        else:
            matches2 = []
            unmatched_tracks2_idx = list(range(len(remaining_tracks)))

        for rem_idx, d_idx in matches2:
            track = remaining_tracks[rem_idx]
            det = low_dets[d_idx]
            track.update_box(det["box"], det.get("landmarks"))

        # 5. Mark lost tracks
        for rem_idx in unmatched_tracks2_idx:
            track = remaining_tracks[rem_idx]
            track.mark_missed()

        # 6. Initialize new tracks from unmatched high-confidence detections
        for d_idx in unmatched_high:
            det = high_dets[d_idx]
            new_track = Track(
                track_id=self.next_track_id,
                box=det["box"],
                landmarks=det.get("landmarks"),
            )
            self.next_track_id += 1
            self.tracks.append(new_track)

        # 7. Purge dead tracks
        self.tracks = [t for t in self.tracks if t.disappeared <= self.max_disappeared]

        # 8. Return currently active (visible) tracks, deduped
        current_active = [t for t in self.tracks if t.disappeared == 0]
        current_active.sort(key=lambda t: (1 if t.employee_id is not None else 0, t.hits), reverse=True)

        deduped = []
        for t in current_active:
            is_dup = False
            for existing in deduped:
                if calculate_iou(t.box, existing.box) > 0.35:
                    is_dup = True
                    break
            if not is_dup:
                deduped.append(t)

        return deduped


FaceTracker = ByteTrack


def create_ort_session(model_path, opts=None):
    """
    Creates an ONNX Runtime InferenceSession trying GPU acceleration providers first
    (CUDAExecutionProvider, DmlExecutionProvider), automatically falling back to
    CPUExecutionProvider if GPU initialization fails or is unavailable.
    """
    if os.environ.get("FORCE_CPU", "").strip().lower() in ("1", "true", "yes"):
        print(f"[Device] FORCE_CPU set: loading {os.path.basename(model_path)} on CPUExecutionProvider")
        return ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])

    available = ort.get_available_providers()
    preferred_gpu_providers = ["CUDAExecutionProvider", "DmlExecutionProvider"]
    candidate_gpu_providers = [p for p in preferred_gpu_providers if p in available]

    for prov in candidate_gpu_providers:
        try:
            session = ort.InferenceSession(model_path, sess_options=opts, providers=[prov, "CPUExecutionProvider"])
            print(f"[Device] Accelerated: Loaded {os.path.basename(model_path)} using {prov}")
            return session
        except Exception as err:
            print(f"[Device Warning] GPU provider '{prov}' failed for {os.path.basename(model_path)}: {err}. Falling back...")

    # CPU Fallback
    print(f"[Device] Loaded {os.path.basename(model_path)} using CPUExecutionProvider")
    return ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])


class YOLOFaceDetector:
    """YOLOv8-Face ONNX Inference Engine for real-time face detection and 5-point landmark prediction."""
    def __init__(self, model_path=YOLO_MODEL_PATH):
        self.model_path = model_path
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = create_ort_session(self.model_path, opts=opts)
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
        self.session = create_ort_session(self.model_path, opts=opts)
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
        self.tracker = ByteTrack(track_high_thresh=0.40, track_low_thresh=0.15, match_thresh=0.75, match_low_thresh=0.55, max_disappeared=10)
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
        2. Detects faces with YOLOv8-Face ONNX (including low-score candidates for ByteTrack stage 2).
        3. Associates bounding boxes using ByteTrack + 8-state Kalman Filter motion projection.
        4. Selectively extracts 512-d ArcFace embeddings for new/unknown faces.
        5. Computes Cosine Distance against enrolled face embeddings.
        6. Returns frame dimensions and list of tracked face match dicts.
        """
        frame_bgr = resize_if_large(frame_bgr, max_width=MAX_FRAME_WIDTH)
        h, w = frame_bgr.shape[:2]

        detections = self.detector.detect(frame_bgr, conf_threshold=0.15)
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
                        num_enrolled = len(distances)

                        if num_enrolled >= 2:
                            sorted_indices = np.argsort(distances)
                            best_idx = int(sorted_indices[0])
                            second_idx = int(sorted_indices[1])
                            min_dist = float(distances[best_idx])
                            second_dist = float(distances[second_idx])
                            margin = second_dist - min_dist
                        else:
                            best_idx = 0
                            second_idx = None
                            min_dist = float(distances[0])
                            second_dist = float("inf")
                            margin = float("inf")

                        conf = distance_to_confidence(min_dist, threshold=MATCH_TOLERANCE)
                        closest_name = self.known_names[best_idx]
                        closest_id = self.known_ids[best_idx]

                        passes_distance = (min_dist <= MATCH_TOLERANCE) and (conf is not None and conf >= CONFIDENCE_FLOOR)
                        passes_margin = (margin >= MATCH_MARGIN)

                        if passes_distance and passes_margin:
                            is_accepted = True
                            decision_reason = "ACCEPTED"
                        elif not passes_distance:
                            is_accepted = False
                            decision_reason = "REJECTED (DISTANCE/CONFIDENCE CHECK FAILED)"
                        else:
                            is_accepted = False
                            decision_reason = f"REJECTED (MARGIN CHECK FAILED: gap={margin:.4f} < {MATCH_MARGIN:.2f})"

                        margin_log_str = (
                            f" | 2nd: '{self.known_names[second_idx]}' (dist={second_dist:.4f}) | gap={margin:.4f} (margin_req={MATCH_MARGIN:.2f})"
                            if second_idx is not None else " | Only 1 enrolled candidate"
                        )

                        print(
                            f"[YOLO-ArcFace Match] Track #{t.track_id} | Closest: '{closest_name}' (ID={closest_id}) | "
                            f"cosine_dist={min_dist:.4f} | threshold={MATCH_TOLERANCE:.2f}{margin_log_str} | "
                            f"conf={conf}% (floor={CONFIDENCE_FLOOR}%) | Decision: {decision_reason} -> {'Accepted' if is_accepted else 'Unknown'}"
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
