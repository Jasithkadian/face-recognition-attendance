"""
benchmark_frame_cost.py
Measures per-frame cost and latency before vs after optimization:
- Before: Unscaled camera frame (1280x720 / 640x480) with high JPEG quality (0.95)
- After:  Client-side downscaled frame (480px max width) with optimal JPEG quality (0.75)

Evaluates:
- Base64 Payload Size (KB)
- Server-side Processing Time (ms)
- End-to-end Round Trip Time (ms)
- Achievable Frame Rate (FPS)
"""

import os
import time
import base64
import numpy as np
import cv2
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).parent
load_dotenv(BASE_DIR / ".env")

from fastapi.testclient import TestClient
from app.main import app, KIOSK_SECRET

client = TestClient(app)

def create_synthetic_frame(width: int, height: int) -> np.ndarray:
    """Create a realistic frame with a gradient background and synthetic facial feature layout."""
    img = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        img[y, :, 0] = int(180 + 50 * (y / height))
        img[y, :, 1] = int(140 + 60 * (y / height))
        img[y, :, 2] = int(120 + 70 * (y / height))

    # Add an oval simulating a face for the detector
    cx, cy = width // 2, height // 2
    axes = (int(width * 0.15), int(height * 0.22))
    cv2.ellipse(img, (cx, cy), axes, 0, 0, 360, (140, 175, 220), -1)
    # Eyes
    cv2.circle(img, (cx - int(axes[0] * 0.4), cy - int(axes[1] * 0.3)), 8, (40, 40, 40), -1)
    cv2.circle(img, (cx + int(axes[0] * 0.4), cy - int(axes[1] * 0.3)), 8, (40, 40, 40), -1)
    # Mouth
    cv2.ellipse(img, (cx, cy + int(axes[1] * 0.4)), (int(axes[0] * 0.4), int(axes[1] * 0.15)), 0, 0, 180, (50, 50, 160), 3)
    return img

def encode_jpeg_base64(img_bgr: np.ndarray, quality: float) -> str:
    quality_int = int(quality * 100)
    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), quality_int]
    _, buffer = cv2.imencode('.jpg', img_bgr, encode_params)
    b64_str = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{b64_str}"

def benchmark_scenario(name: str, src_w: int, src_h: int, target_max_w: int, quality: float, iterations: int = 15):
    # 1. Simulate camera capture
    raw_frame = create_synthetic_frame(src_w, src_h)

    # 2. Simulate client-side downscale (maintaining aspect ratio)
    if src_w > target_max_w:
        target_w = target_max_w
        target_h = int(round((src_h * target_max_w) / src_w))
        scaled_frame = cv2.resize(raw_frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
    else:
        target_w, target_h = src_w, src_h
        scaled_frame = raw_frame

    # 3. Simulate client-side JPEG conversion to base64
    b64_url = encode_jpeg_base64(scaled_frame, quality)
    payload_bytes = len(b64_url.encode('utf-8'))
    payload_kb = payload_bytes / 1024.0

    payload = {"image": b64_url}
    headers = {"X-Kiosk-Client": KIOSK_SECRET}

    # Warm-up request
    _ = client.post("/api/recognize", json=payload, headers=headers)

    # Timed runs
    e2e_times = []
    server_proc_times = []

    for _ in range(iterations):
        t0 = time.perf_counter()
        resp = client.post("/api/recognize", json=payload, headers=headers)
        t_e2e = (time.perf_counter() - t0) * 1000.0

        if resp.status_code == 200:
            e2e_times.append(t_e2e)
            data = resp.json()
            if "proc_ms" in data:
                server_proc_times.append(data["proc_ms"])

    avg_e2e = float(np.mean(e2e_times))
    std_e2e = float(np.std(e2e_times))
    avg_server = float(np.mean(server_proc_times)) if server_proc_times else avg_e2e
    fps = 1000.0 / avg_e2e if avg_e2e > 0 else 0.0

    return {
        "name": name,
        "raw_res": f"{src_w}x{src_h}",
        "sent_res": f"{target_w}x{target_h}",
        "quality": quality,
        "payload_kb": payload_kb,
        "avg_server_ms": avg_server,
        "avg_e2e_ms": avg_e2e,
        "std_e2e_ms": std_e2e,
        "fps": fps,
    }

def main():
    print("=" * 80)
    print("Benchmarking Frame Cost Reduction: Before vs. After")
    print("=" * 80)

    # Scenario 1: BEFORE - HD 720p (1280x720) unscaled with high quality (0.95)
    s1 = benchmark_scenario("Before (HD 720p @ 0.95)", src_w=1280, src_h=720, target_max_w=1280, quality=0.95)

    # Scenario 2: BEFORE - Standard VGA (640x480) unscaled with high quality (0.95)
    s2 = benchmark_scenario("Before (VGA 640x480 @ 0.95)", src_w=640, src_h=480, target_max_w=640, quality=0.95)

    # Scenario 3: AFTER - HD 720p camera downscaled to 480px width @ 0.75 quality (480x270)
    s3 = benchmark_scenario("After (Downscaled 480px @ 0.75, 16:9)", src_w=1280, src_h=720, target_max_w=480, quality=0.75)

    # Scenario 4: AFTER - VGA camera downscaled to 480px width @ 0.75 quality (480x360)
    s4 = benchmark_scenario("After (Downscaled 480px @ 0.75, 4:3)", src_w=640, src_h=480, target_max_w=480, quality=0.75)

    results = [s1, s2, s3, s4]

    print(f"\n{'Scenario':<35} | {'Sent Res':<10} | {'Payload':<10} | {'Server MS':<10} | {'E2E Latency':<12} | {'Max FPS':<8}")
    print("-" * 95)
    for r in results:
        print(f"{r['name']:<35} | {r['sent_res']:<10} | {r['payload_kb']:>6.1f} KB  | {r['avg_server_ms']:>7.1f} ms | {r['avg_e2e_ms']:>6.1f} ms    | {r['fps']:>5.1f}")

    print("\n" + "=" * 80)
    print("KEY PERFORMANCE IMPROVEMENTS:")
    print("=" * 80)
    payload_reduction_hd = (1.0 - s3['payload_kb'] / s1['payload_kb']) * 100.0
    payload_reduction_vga = (1.0 - s4['payload_kb'] / s2['payload_kb']) * 100.0
    latency_reduction_hd = (1.0 - s3['avg_e2e_ms'] / s1['avg_e2e_ms']) * 100.0
    latency_reduction_vga = (1.0 - s4['avg_e2e_ms'] / s2['avg_e2e_ms']) * 100.0

    print(f"1. HD 720p Stream: Payload reduced by {payload_reduction_hd:.1f}% ({s1['payload_kb']:.1f} KB -> {s3['payload_kb']:.1f} KB)")
    print(f"   Latency dropped from {s1['avg_e2e_ms']:.1f} ms to {s3['avg_e2e_ms']:.1f} ms ({latency_reduction_hd:.1f}% faster, FPS: {s1['fps']:.1f} -> {s3['fps']:.1f})")

    print(f"\n2. VGA 480p Stream: Payload reduced by {payload_reduction_vga:.1f}% ({s2['payload_kb']:.1f} KB -> {s4['payload_kb']:.1f} KB)")
    print(f"   Latency dropped from {s2['avg_e2e_ms']:.1f} ms to {s4['avg_e2e_ms']:.1f} ms ({latency_reduction_vga:.1f}% faster, FPS: {s2['fps']:.1f} -> {s4['fps']:.1f})")
    print("=" * 80)

if __name__ == "__main__":
    main()
