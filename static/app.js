/**
 * FacePulse PRO — Main Application Script
 * Real-time camera stream control, ML bounding box overlays,
 * guided multi-sample enrollment, presence polling, and activity logging.
 */

// Global State
let stream = null;
let recognitionInterval = null;
let presentPollInterval = null;
let isProcessingFrame = false;
let currentCameraDeviceId = null;
let isFaceGuideVisible = false;

// Guided Enrollment State
let isEnrolling = false;
let enrollmentSamples = [];
let currentPoseIndex = 0;
let enrollmentTimer = null;

// Admin token saved in browser localStorage
let adminToken = localStorage.getItem('adminToken') || '';

// Pose Guidance sequence for 15 samples
const ENROLL_POSES = [
  { title: "Look Straight", prompt: "Look directly into the camera", icon: "👤" },
  { title: "Look Straight", prompt: "Hold position, looking straight", icon: "👤" },
  { title: "Turn Left", prompt: "Turn head slightly to your LEFT", icon: "👈" },
  { title: "Turn Left", prompt: "Hold position, facing slightly LEFT", icon: "👈" },
  { title: "Turn Right", prompt: "Turn head slightly to your RIGHT", icon: "👉" },
  { title: "Turn Right", prompt: "Hold position, facing slightly RIGHT", icon: "👉" },
  { title: "Tilt Up", prompt: "Tilt head slightly UP", icon: "👆" },
  { title: "Tilt Up", prompt: "Hold position, tilted slightly UP", icon: "👆" },
  { title: "Tilt Down", prompt: "Tilt head slightly DOWN", icon: "👇" },
  { title: "Tilt Down", prompt: "Hold position, tilted slightly DOWN", icon: "👇" },
  { title: "Expression", prompt: "Smile or change expression slightly", icon: "😊" },
  { title: "Expression", prompt: "Hold expression or neutral gaze", icon: "😊" },
  { title: "Natural Pose", prompt: "Blink naturally and look straight", icon: "👁️" },
  { title: "Natural Pose", prompt: "Look straight at the camera lens", icon: "👤" },
  { title: "Final Sample", prompt: "Final capture: Hold still looking straight", icon: "✨" }
];

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
const presentTabBadge = document.getElementById('presentTabBadge');
const kpiPresent = document.getElementById('kpiPresent');

const rosterList = document.getElementById('rosterList');
const rosterCount = document.getElementById('rosterCount');
const rosterTabBadge = document.getElementById('rosterTabBadge');
const kpiEnrolled = document.getElementById('kpiEnrolled');

const enrollAdminBadge = document.getElementById('enrollAdminBadge');
const enrollLockedView = document.getElementById('enrollLockedView');
const enrollUnlockedView = document.getElementById('enrollUnlockedView');

// Guided Enroll DOM elements
const guidedEnrollContainer = document.getElementById('guidedEnrollContainer');
const poseIcon = document.getElementById('poseIcon');
const poseTitle = document.getElementById('poseTitle');
const poseInstruction = document.getElementById('poseInstruction');
const enrollProgressBar = document.getElementById('enrollProgressBar');
const sampleCounter = document.getElementById('sampleCounter');
const enrollPercent = document.getElementById('enrollPercent');
const sampleThumbnails = document.getElementById('sampleThumbnails');
const enrollBtn = document.getElementById('enrollBtn');
const cancelEnrollBtn = document.getElementById('cancelEnrollBtn');

// Initialize application on DOM ready
document.addEventListener('DOMContentLoaded', () => {
  updateAdminUI();
  populateCameras();
  fetchPresentList();
  fetchRosterList();
  renderThumbnailSlots();

  // Poll active presence list every 3.5 seconds
  presentPollInterval = setInterval(fetchPresentList, 3500);
});

// ---------------- Tab Navigation ----------------

function switchTab(tabId, btnEl) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));

  document.getElementById(tabId).classList.add('active');
  btnEl.classList.add('active');

  if (tabId === 'logTab') {
    fetchActivityLog();
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

    // Frame recognition tick loop (750ms for responsive face tracking during motion)
    if (recognitionInterval) clearInterval(recognitionInterval);
    recognitionInterval = setInterval(captureAndRecognizeFrame, 750);

  } catch (err) {
    console.error('Camera access error:', err);
    alert('Could not open camera feed. Please allow camera permissions in your browser.');
  }
}

function stopCamera() {
  if (stream) {
    stream.getTracks().forEach(track => track.stop());
    stream = null;
  }

  if (recognitionInterval) {
    clearInterval(recognitionInterval);
    recognitionInterval = null;
  }

  if (isEnrolling) {
    cancelGuidedEnrollment();
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

function resizeCanvas() {
  if (webcam.videoWidth && webcam.videoHeight) {
    overlayCanvas.width = webcam.clientWidth || 640;
    overlayCanvas.height = webcam.clientHeight || 480;
  }
}

// Helper to capture a downscaled frame canvas element
function captureFrameCanvas(maxWidth = 400) {
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

// ---------------- Real ML Frame Recognition ----------------

async function captureAndRecognizeFrame() {
  if (!stream || isProcessingFrame || webcam.paused || webcam.ended || isEnrolling) return;

  isProcessingFrame = true;

  try {
    const captureCanvas = captureFrameCanvas(400);
    const base64Image = captureCanvas.toDataURL('image/jpeg', 0.7);

    const resp = await fetch('/api/recognize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: base64Image })
    });

    if (resp.ok) {
      const data = await resp.json();
      drawOverlays(data.width || captureCanvas.width, data.height || captureCanvas.height, data.matches || []);
      if (data.present !== undefined) {
        updatePresentUI(data.present);
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

    const isRecognized = m.employee_id !== null;
    const strokeColor = isRecognized ? '#2563eb' : '#dc2626';
    const fillColor = isRecognized ? 'rgba(37, 99, 235, 0.08)' : 'rgba(220, 38, 38, 0.08)';

    // Fill Box
    ctx.fillStyle = fillColor;
    drawRoundedRect(ctx, x, y, w, h, 8, true, false);

    // Stroke Border
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 2.5;
    drawRoundedRect(ctx, x, y, w, h, 8, false, true);

    // Label formatting: Name + Confidence Score (e.g. "Jane Doe • 92.4%")
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

// ---------------- Guided Multi-Sample Enrollment ----------------

function renderThumbnailSlots() {
  sampleThumbnails.innerHTML = '';
  for (let i = 0; i < 15; i++) {
    const slot = document.createElement('div');
    slot.className = 'thumb-slot';
    slot.id = `thumb-${i}`;
    slot.innerHTML = `<span class="thumb-num">${i + 1}</span>`;
    sampleThumbnails.appendChild(slot);
  }
}

async function startGuidedEnrollment() {
  const nameInput = document.getElementById('employeeName');
  const feedback = document.getElementById('enrollFeedback');

  const name = nameInput.value.trim();
  if (!name) {
    showFeedback(feedback, 'error', 'Please enter a full name for enrollment.');
    return;
  }

  if (!adminToken) {
    showFeedback(feedback, 'error', '🔒 Admin Token required to enroll personnel. Click "Admin Mode" in header to authenticate.');
    return;
  }

  if (!stream) {
    showFeedback(feedback, 'error', 'Camera must be active to capture face samples. Click "Start Camera" first.');
    return;
  }

  // Reset State
  isEnrolling = true;
  enrollmentSamples = [];
  currentPoseIndex = 0;

  enrollBtn.classList.add('hidden');
  cancelEnrollBtn.classList.remove('hidden');
  renderThumbnailSlots();
  updateEnrollmentUI();
  hideFeedback(feedback);

  // Start sample collection tick loop
  if (enrollmentTimer) clearInterval(enrollmentTimer);
  enrollmentTimer = setInterval(processEnrollmentTick, 700);
}

function updateEnrollmentUI() {
  if (currentPoseIndex >= ENROLL_POSES.length) return;

  const currentPose = ENROLL_POSES[currentPoseIndex];
  poseIcon.textContent = currentPose.icon;
  poseTitle.textContent = `${currentPose.title} (Sample ${currentPoseIndex + 1} of 15)`;
  poseInstruction.textContent = currentPose.prompt;

  const count = enrollmentSamples.length;
  const pct = Math.round((count / 15) * 100);
  sampleCounter.textContent = `Sample ${count} of 15`;
  enrollPercent.textContent = `${pct}%`;
  enrollProgressBar.style.width = `${pct}%`;
}

async function processEnrollmentTick() {
  if (!isEnrolling || enrollmentSamples.length >= 15) return;

  const feedback = document.getElementById('enrollFeedback');

  try {
    const captureCanvas = captureFrameCanvas(400);
    const base64Image = captureCanvas.toDataURL('image/jpeg', 0.85);

    // ML Face Check call to backend: verify face detection and duplicate pose rejection
    const resp = await fetch('/api/enroll-check', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        image: base64Image,
        existing_images: enrollmentSamples
      })
    });

    if (resp.ok) {
      const data = await resp.json();
      if (data.valid) {
        // Face verified by face_recognition.face_locations() on backend!
        const sampleIdx = enrollmentSamples.length;
        enrollmentSamples.push(base64Image);

        // Update Thumbnail slot
        const thumbSlot = document.getElementById(`thumb-${sampleIdx}`);
        if (thumbSlot) {
          thumbSlot.innerHTML = `<img src="${base64Image}" alt="Sample ${sampleIdx + 1}" class="thumb-img">`;
          thumbSlot.classList.add('active');
        }

        currentPoseIndex = Math.min(14, enrollmentSamples.length);
        updateEnrollmentUI();

        if (enrollmentSamples.length === 15) {
          finishGuidedEnrollment();
        }
      } else {
        poseInstruction.textContent = `⚠️ ${data.reason || 'Position face clearly in frame'}`;
      }
    }
  } catch (err) {
    console.warn('Enrollment tick error:', err);
  }
}

async function finishGuidedEnrollment() {
  if (enrollmentTimer) clearInterval(enrollmentTimer);
  enrollmentTimer = null;

  const nameInput = document.getElementById('employeeName');
  const feedback = document.getElementById('enrollFeedback');
  const name = nameInput.value.trim();

  poseTitle.textContent = "Processing Encodings...";
  poseInstruction.textContent = "Averaging 15 128-d encodings and saving employee profile...";
  showFeedback(feedback, 'info', 'Submitting 15 samples for numpy mean encoding calculation...');

  try {
    const resp = await fetch('/api/enroll', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: name,
        images: enrollmentSamples,
        admin_token: adminToken
      })
    });

    const data = await resp.json();

    if (resp.ok) {
      showFeedback(feedback, 'success', `✓ Successfully enrolled ${name} using 15 real ML face samples!`);
      nameInput.value = '';
      fetchRosterList();
      fetchPresentList();
      setTimeout(() => resetEnrollmentUI(), 2500);
    } else {
      showFeedback(feedback, 'error', data.detail || 'Enrollment failed.');
      cancelGuidedEnrollment();
    }
  } catch (err) {
    showFeedback(feedback, 'error', 'Server error during enrollment processing.');
    cancelGuidedEnrollment();
  }
}

function cancelGuidedEnrollment() {
  if (enrollmentTimer) clearInterval(enrollmentTimer);
  enrollmentTimer = null;
  isEnrolling = false;
  enrollmentSamples = [];
  currentPoseIndex = 0;
  resetEnrollmentUI();
}

function resetEnrollmentUI() {
  isEnrolling = false;
  enrollmentSamples = [];
  currentPoseIndex = 0;
  poseIcon.textContent = "👤";
  poseTitle.textContent = "Ready to Start";
  poseInstruction.textContent = "Enter name and click 'Start 15-Sample Guided Enrollment'.";
  sampleCounter.textContent = "Sample 0 of 15";
  enrollPercent.textContent = "0%";
  enrollProgressBar.style.width = "0%";
  enrollBtn.classList.remove('hidden');
  cancelEnrollBtn.classList.add('hidden');
  renderThumbnailSlots();
}

function updatePresentUI(presentArray) {
  const present = presentArray || [];
  presentCount.textContent = `${present.length} Active`;
  presentTabBadge.textContent = present.length;
  kpiPresent.textContent = present.length;

  if (present.length === 0) {
    presentList.innerHTML = `
      <div class="empty-state">
        <p>No active faces in view right now</p>
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

async function fetchRosterList() {
  try {
    const resp = await fetch('/api/employees');
    if (!resp.ok) return;
    const data = await resp.json();

    const employees = data.employees || [];
    rosterCount.textContent = `${employees.length} Enrolled`;
    rosterTabBadge.textContent = employees.length;
    kpiEnrolled.textContent = employees.length;

    if (employees.length === 0) {
      rosterList.innerHTML = `
        <div class="empty-state">
          <p>No individuals enrolled yet.</p>
        </div>
      `;
      return;
    }

    renderRosterItems(employees);
  } catch (err) {
    console.warn('Error fetching roster list:', err);
  }
}

function renderRosterItems(employees) {
  rosterList.innerHTML = employees.map(emp => `
    <div class="roster-item" data-name="${escapeHtml(emp.name).toLowerCase()}">
      <div class="roster-item-info">
        <span class="user-name">${escapeHtml(emp.name)}</span>
      </div>
      <button class="btn-icon-del" title="Delete record" onclick="deleteEmployee(${emp.id}, '${escapeHtml(emp.name)}')">
        🗑️
      </button>
    </div>
  `).join('');
}

function filterRoster() {
  const query = document.getElementById('rosterSearch').value.toLowerCase();
  const items = rosterList.querySelectorAll('.roster-item');
  items.forEach(item => {
    const name = item.getAttribute('data-name') || '';
    item.style.display = name.includes(query) ? 'flex' : 'none';
  });
}

async function deleteEmployee(empId, empName) {
  if (!adminToken) {
    alert('Admin token required to delete enrolled people.');
    toggleAdminModal();
    return;
  }

  if (!confirm(`Are you sure you want to remove "${empName}" from the roster?`)) {
    return;
  }

  try {
    const resp = await fetch(`/api/employees/${empId}?admin_token=${encodeURIComponent(adminToken)}`, {
      method: 'DELETE'
    });

    if (resp.ok) {
      fetchRosterList();
      fetchPresentList();
    } else {
      const data = await resp.json();
      alert(data.detail || 'Failed to delete employee record.');
    }
  } catch (err) {
    alert('Server error while deleting employee.');
  }
}

// ---------------- Activity Log Fetching ----------------

async function fetchActivityLog() {
  const tableBody = document.getElementById('activityLogBody');
  const searchVal = document.getElementById('logSearch').value.trim();
  const dateVal = document.getElementById('logDate').value.trim();

  let url = '/api/activity-log?limit=100';
  if (searchVal) url += `&name=${encodeURIComponent(searchVal)}`;
  if (dateVal) url += `&date=${encodeURIComponent(dateVal)}`;

  try {
    const resp = await fetch(url);
    if (!resp.ok) return;
    const data = await resp.json();
    const logs = data.logs || [];

    if (logs.length === 0) {
      tableBody.innerHTML = `
        <tr>
          <td colspan="5" class="empty-cell">No recognition events logged yet matching criteria.</td>
        </tr>
      `;
      return;
    }

    tableBody.innerHTML = logs.map(l => {
      const timeStr = formatTime(l.seen_at);
      const confStr = l.confidence !== null ? `${l.confidence}%` : 'N/A';
      const distStr = l.distance !== null ? l.distance.toFixed(4) : 'N/A';
      const statusBadge = (l.confidence !== null && l.confidence >= 60) ? 
        '<span class="status-badge success">High Confidence</span>' : 
        '<span class="status-badge warning">Matched</span>';

      return `
        <tr>
          <td class="log-time">${timeStr}</td>
          <td class="log-name">${escapeHtml(l.name)}</td>
          <td><strong class="conf-text">${confStr}</strong></td>
          <td class="dist-text">${distStr}</td>
          <td>${statusBadge}</td>
        </tr>
      `;
    }).join('');
  } catch (err) {
    tableBody.innerHTML = `<tr><td colspan="5" class="empty-cell">Error loading activity log.</td></tr>`;
  }
}

// ---------------- Admin Modal Handlers ----------------

function updateAdminUI() {
  const isUnlocked = adminToken.trim().length > 0;

  if (isUnlocked) {
    enrollAdminBadge.textContent = '🔓 Admin Unlocked';
    enrollAdminBadge.className = 'admin-badge unlocked';
    enrollLockedView.classList.add('hidden');
    enrollUnlockedView.classList.remove('hidden');

    document.getElementById('adminLockText').textContent = 'Admin Mode (Active)';
    document.getElementById('adminLockIcon').textContent = '🔓';
  } else {
    enrollAdminBadge.textContent = '🔒 Admin Gated';
    enrollAdminBadge.className = 'admin-badge locked';
    enrollLockedView.classList.remove('hidden');
    enrollUnlockedView.classList.add('hidden');

    document.getElementById('adminLockText').textContent = 'Admin Settings';
    document.getElementById('adminLockIcon').textContent = '🔒';
  }
}

async function unlockAdminInline() {
  const input = document.getElementById('inlineAdminToken');
  const tokenVal = input.value.trim();
  const errorEl = document.getElementById('inlineTokenError');

  if (!tokenVal) {
    showError(errorEl, 'Please enter an admin token.');
    return;
  }

  try {
    const resp = await fetch('/api/verify-token', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ admin_token: tokenVal })
    });

    const data = await resp.json();
    if (data.valid) {
      adminToken = tokenVal;
      localStorage.setItem('adminToken', adminToken);
      errorEl.classList.add('hidden');
      updateAdminUI();
    } else {
      showError(errorEl, 'Incorrect Admin Token.');
    }
  } catch (err) {
    showError(errorEl, 'Could not connect to server.');
  }
}

function toggleAdminModal() {
  const modal = document.getElementById('adminModal');
  const tokenInput = document.getElementById('modalAdminToken');
  tokenInput.value = adminToken;
  modal.classList.toggle('hidden');
}

function saveAdminTokenModal() {
  const tokenInput = document.getElementById('modalAdminToken');
  const tokenVal = tokenInput.value.trim();
  adminToken = tokenVal;
  localStorage.setItem('adminToken', adminToken);
  updateAdminUI();
  toggleAdminModal();
}

// Utility Helpers
function showFeedback(el, type, msg) {
  el.className = `feedback-banner ${type}`;
  el.textContent = msg;
  el.classList.remove('hidden');
}

function hideFeedback(el) {
  el.classList.add('hidden');
}

function showError(el, msg) {
  el.textContent = msg;
  el.classList.remove('hidden');
}

function escapeHtml(str) {
  return str.replace(/[&<>'"]/g, 
    tag => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[tag] || tag)
  );
}

function formatTime(isoStr) {
  if (!isoStr) return '';
  try {
    let str = isoStr;
    if (typeof str === 'string' && !str.endsWith('Z') && !/[+-]\d{2}:\d{2}$/.test(str)) {
      str += 'Z';
    }
    const d = new Date(str);
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) + 
           ' (' + d.toLocaleDateString([], { month: 'short', day: 'numeric' }) + ')';
  } catch {
    return isoStr;
  }
}
