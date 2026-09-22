/**
 * FacePulse — Kiosk Application Script
 * Live camera feed, real-time ML face tracking & bounding box overlays,
 * and active attendee presence monitoring.
 * Strictly public/read-only — zero administrative tokens or employee rosters.
 */

// Kiosk client identification secret (matches backend KIOSK_SECRET)
const KIOSK_SECRET = 'facepulse_kiosk_client_default';

// Global State
let stream = null;
let recognitionLoopActive = false;
let isProcessingFrame = false;
let currentCameraDeviceId = null;
let isFaceGuideVisible = false;
let presentPollInterval = null;

// DOM Elements
const webcam = document.getElementById('webcam');
const overlayCanvas = document.getElementById('overlayCanvas');
const ctx = overlayCanvas.getContext('2d');

const cameraPlaceholder = document.getElementById('cameraPlaceholder');
const scanLine = document.getElementById('scanLine');
const recognizingBadge = document.getElementById('recognizingBadge');
const resTag = document.getElementById('resTag');
const faceGuide = document.getElementById('faceGuide');

const startCamBtn = document.getElementById('startCamBtn');
const stopCamBtn = document.getElementById('stopCamBtn');
const cameraSelect = document.getElementById('cameraSelect');
const statusText = document.getElementById('statusText');

const presentList = document.getElementById('presentList');
const presentCount = document.getElementById('presentCount');
const kpiPresent = document.getElementById('kpiPresent');

// Initialize on DOM ready
document.addEventListener('DOMContentLoaded', () => {
  populateCameras();
  fetchPresentList();

  // Poll active presence list every 3.5 seconds
  presentPollInterval = setInterval(fetchPresentList, 3500);
});

// ---------------- Camera Stream Logic ----------------

async function populateCameras() {
  try {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const videoDevices = devices.filter(d => d.kind === 'videoinput');
    
    cameraSelect.innerHTML = '';
    if (videoDevices.length === 0) {
      cameraSelect.innerHTML = '<option value="">Default Camera</option>';
      return;
    }

    videoDevices.forEach((device, idx) => {
      const option = document.createElement('option');
      option.value = device.deviceId;
      option.text = device.label || `Camera ${idx + 1}`;
      cameraSelect.appendChild(option);
    });

    currentCameraDeviceId = videoDevices[0].deviceId;
  } catch (err) {
    console.warn('Could not list cameras:', err);
  }
}

async function startCamera() {
  try {
    const constraints = {
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: 'user'
      }
    };

    if (cameraSelect.value) {
      constraints.video.deviceId = { exact: cameraSelect.value };
    }

    stream = await navigator.mediaDevices.getUserMedia(constraints);
    webcam.srcObject = stream;

    await new Promise((resolve) => {
      webcam.onloadedmetadata = () => resolve();
    });

    // Display camera resolution
    resTag.textContent = `${webcam.videoWidth}x${webcam.videoHeight}`;
    resTag.classList.remove('hidden');

    // UI Updates
    cameraPlaceholder.classList.add('hidden');
    startCamBtn.classList.add('hidden');
    stopCamBtn.classList.remove('hidden');
    scanLine.classList.remove('hidden');
    recognizingBadge.classList.remove('hidden');

    if (isFaceGuideVisible) faceGuide.classList.remove('hidden');
    statusText.textContent = 'Camera Active';

    resizeCanvas();
    window.addEventListener('resize', resizeCanvas);

    // Real-time recognition loop (smooth ~10-15 FPS)
    recognitionLoopActive = true;
    runRecognitionLoop();

  } catch (err) {
    console.error('Camera access error:', err);
    alert('Could not open camera feed. Please allow camera permissions in your browser.');
  }
}

function stopCamera() {
  recognitionLoopActive = false;
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
  }

  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

  cameraPlaceholder.classList.remove('hidden');
  startCamBtn.classList.remove('hidden');
  stopCamBtn.classList.add('hidden');
  scanLine.classList.add('hidden');
  recognizingBadge.classList.add('hidden');
  resTag.classList.add('hidden');
  faceGuide.classList.add('hidden');

  statusText.textContent = 'System Ready';
}

function switchCamera() {
  if (stream) {
    stopCamera();
    startCamera();
  }
}

function toggleFaceGuide() {
  isFaceGuideVisible = !isFaceGuideVisible;
  if (isFaceGuideVisible && stream) {
    faceGuide.classList.remove('hidden');
  } else {
    faceGuide.classList.add('hidden');
  }
}

function resizeCanvas() {
  if (webcam.videoWidth && webcam.videoHeight) {
    overlayCanvas.width = webcam.clientWidth || 640;
    overlayCanvas.height = webcam.clientHeight || 480;
  }
}

// ---------------- Frame Processing Timing Instrumentation ----------------
const frameProcessingTimes = [];
const PERF_WINDOW_SIZE = 20;

function captureFrameCanvas(maxWidth = 480) {
  const srcWidth = webcam.videoWidth || 640;
  const srcHeight = webcam.videoHeight || 480;

  let targetWidth = srcWidth;
  let targetHeight = srcHeight;
  if (srcWidth > maxWidth) {
    targetWidth = maxWidth;
    targetHeight = Math.round((srcHeight * maxWidth) / srcWidth);
  }

  const captureCanvas = document.createElement('canvas');
  captureCanvas.width = targetWidth;
  captureCanvas.height = targetHeight;
  const captureCtx = captureCanvas.getContext('2d');
  captureCtx.drawImage(webcam, 0, 0, targetWidth, targetHeight);
  return captureCanvas;
}

// ---------------- Real-Time ML Frame Recognition Loop ----------------

async function runRecognitionLoop() {
  if (!recognitionLoopActive || !stream) return;

  if (!webcam.paused && !webcam.ended && webcam.videoWidth) {
    await captureAndRecognizeFrame();
  }

  if (recognitionLoopActive) {
    setTimeout(runRecognitionLoop, 60);
  }
}

async function captureAndRecognizeFrame() {
  if (!stream || isProcessingFrame || webcam.paused || webcam.ended) return;

  isProcessingFrame = true;
  const frameStartTime = performance.now();

  try {
    const captureCanvas = captureFrameCanvas(480);
    const base64Image = captureCanvas.toDataURL('image/jpeg', 0.75);

    const resp = await fetch('/api/recognize', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'X-Kiosk-Client': KIOSK_SECRET,
      },
      body: JSON.stringify({ image: base64Image })
    });

    if (resp.ok) {
      const data = await resp.json();
      drawOverlays(data.width || captureCanvas.width, data.height || captureCanvas.height, data.matches || []);
      if (data.present !== undefined) {
        updatePresentUI(data.present);
      }

      // Record and log rolling frame processing performance
      const frameDurationMs = performance.now() - frameStartTime;
      frameProcessingTimes.push(frameDurationMs);
      if (frameProcessingTimes.length > PERF_WINDOW_SIZE) {
        frameProcessingTimes.shift();
      }
      if (frameProcessingTimes.length % 10 === 0) {
        const avgFrameTime = frameProcessingTimes.reduce((a, b) => a + b, 0) / frameProcessingTimes.length;
        console.log(
          `[Frame Timing] Last: ${frameDurationMs.toFixed(1)}ms | ` +
          `Rolling Avg (N=${frameProcessingTimes.length}): ${avgFrameTime.toFixed(1)}ms ` +
          `(${(1000 / avgFrameTime).toFixed(1)} FPS)`
        );
      }
    }
  } catch (err) {
    console.warn('Frame recognition error:', err);
  } finally {
    isProcessingFrame = false;
  }
}

// ---------------- Overlay Bounding Box Drawing ----------------

function drawOverlays(frameWidth, frameHeight, matches) {
  resizeCanvas();
  ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

  if (!matches || matches.length === 0) return;

  const scaleX = overlayCanvas.width / frameWidth;
  const scaleY = overlayCanvas.height / frameHeight;

  matches.forEach(m => {
    const [top, right, bottom, left] = m.box;
    const x = left * scaleX;
    const y = top * scaleY;
    const w = (right - left) * scaleX;
    const h = (bottom - top) * scaleY;

    const CONFIDENCE_FLOOR = 60.0;
    const isRecognized = (m.employee_id !== null) && 
                         (m.name && m.name !== 'Unknown') && 
                         (m.confidence === undefined || m.confidence === null || m.confidence >= CONFIDENCE_FLOOR);
    const strokeColor = isRecognized ? '#2563eb' : '#dc2626';
    const fillColor = isRecognized ? 'rgba(37, 99, 235, 0.08)' : 'rgba(220, 38, 38, 0.08)';

    // Fill Box
    ctx.fillStyle = fillColor;
    drawRoundedRect(ctx, x, y, w, h, 8, true, false);

    // Stroke Border
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 2.5;
    drawRoundedRect(ctx, x, y, w, h, 8, false, true);

    // Label formatting: Name
    let labelText = isRecognized ? m.name : 'Unknown';
    if (isRecognized && m.confidence !== undefined && m.confidence !== null) {
      labelText += ` • ${m.confidence}%`;
    }

    ctx.font = '600 12px "Plus Jakarta Sans", sans-serif';
    const textWidth = ctx.measureText(labelText).width;

    const tagWidth = textWidth + 24;
    const tagHeight = 22;
    const tagX = Math.max(0, x);
    const tagY = Math.max(0, y - tagHeight - 6);

    ctx.fillStyle = isRecognized ? '#1e293b' : '#991b1b';
    drawRoundedRect(ctx, tagX, tagY, tagWidth, tagHeight, 6, true, false);

    ctx.fillStyle = '#ffffff';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(labelText, tagX + 12, tagY + 11);
  });
}

function drawRoundedRect(ctx, x, y, w, h, r, fill, stroke) {
  if (w < 2 * r) r = w / 2;
  if (h < 2 * r) r = h / 2;
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
  if (fill) ctx.fill();
  if (stroke) ctx.stroke();
}

// ---------------- Active Presence UI ----------------

function updatePresentUI(presentArray) {
  const present = presentArray || [];
  if (presentCount) presentCount.textContent = `${present.length} Active`;
  if (kpiPresent) kpiPresent.textContent = present.length;

  if (!presentList) return;

  if (present.length === 0) {
    presentList.innerHTML = `
      <div class="empty-state">
        <p>Stand in front of the camera to record attendance</p>
      </div>
    `;
    return;
  }

  presentList.innerHTML = present.map(p => {
    const initial = p.name ? p.name.charAt(0).toUpperCase() : '?';
    return `
      <div class="user-chip">
        <div class="user-avatar">${initial}</div>
        <div class="user-info">
          <span class="user-name">${escapeHtml(p.name)}</span>
          <span class="user-time">● Active now</span>
        </div>
      </div>
    `;
  }).join('');
}

async function fetchPresentList() {
  try {
    const resp = await fetch('/api/present?window_seconds=20');
    if (!resp.ok) return;
    const data = await resp.json();
    updatePresentUI(data.present || []);
  } catch (err) {
    console.warn('Error fetching present list:', err);
  }
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str).replace(/[&<>'"]/g, 
    tag => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[tag] || tag)
  );
}
