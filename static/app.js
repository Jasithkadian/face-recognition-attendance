/**
 * FacePulse PRO — Main Application Script
 * Real-time camera stream control, canvas overlay bounding box rendering,
 * tab navigation, presence polling, and employee enrollment.
 */

// Global State
let stream = null;
let recognitionInterval = null;
let presentPollInterval = null;
let isProcessingFrame = false;
let currentCameraDeviceId = null;
let isFaceGuideVisible = false;

// Admin token saved in browser localStorage (defaults to admin123 if empty)
let adminToken = localStorage.getItem('adminToken') || '';

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

// Initialize application on DOM ready
document.addEventListener('DOMContentLoaded', () => {
  updateAdminUI();
  populateCameras();
  fetchPresentList();
  fetchRosterList();

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
    fetchTodayLog();
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

    // Frame recognition tick loop (every 1500ms to reduce CPU load on free tier)
    if (recognitionInterval) clearInterval(recognitionInterval);
    recognitionInterval = setInterval(captureAndRecognizeFrame, 1500);

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

async function captureAndRecognizeFrame() {
  if (!stream || isProcessingFrame || webcam.paused || webcam.ended) return;

  isProcessingFrame = true;

  try {
    const srcWidth = webcam.videoWidth || 640;
    const srcHeight = webcam.videoHeight || 480;
    const maxWidth = 400; // Downscale frame to maximum width of 400px

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

    const base64Image = captureCanvas.toDataURL('image/jpeg', 0.7);

    const resp = await fetch('/api/recognize', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: base64Image })
    });

    if (resp.ok) {
      const data = await resp.json();
      drawOverlays(data.width || captureCanvas.width, data.height || captureCanvas.height, data.matches || []);
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
    const strokeColor = isRecognized ? '#06b6d4' : '#f43f5e';
    const fillColor = isRecognized ? 'rgba(6, 182, 212, 0.12)' : 'rgba(244, 63, 94, 0.1)';

    // Fill Box
    ctx.fillStyle = fillColor;
    drawRoundedRect(ctx, x, y, w, h, 10, true, false);

    // Stroke Border with Glow
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 2.5;
    ctx.shadowColor = strokeColor;
    ctx.shadowBlur = 10;
    drawRoundedRect(ctx, x, y, w, h, 10, false, true);

    ctx.shadowBlur = 0;

    // Draw Floating Tag Above Box
    const labelText = isRecognized ? m.name : 'Unknown';
    ctx.font = '600 13px "Plus Jakarta Sans", sans-serif';
    const textWidth = ctx.measureText(labelText).width;

    const tagWidth = textWidth + 32;
    const tagHeight = 24;
    const tagX = Math.max(0, x);
    const tagY = Math.max(0, y - tagHeight - 6);

    ctx.fillStyle = '#111827';
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 1;
    drawRoundedRect(ctx, tagX, tagY, tagWidth, tagHeight, 12, true, true);

    // Avatar Circle
    const avatarRadius = 7;
    const avatarX = tagX + 13;
    const avatarY = tagY + 12;
    ctx.beginPath();
    ctx.arc(avatarX, avatarY, avatarRadius, 0, 2 * Math.PI);
    ctx.fillStyle = strokeColor;
    ctx.fill();

    // Initial Letter
    ctx.fillStyle = '#ffffff';
    ctx.font = '700 9px sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(labelText.charAt(0).toUpperCase(), avatarX, avatarY);

    // Name Label Text
    ctx.fillStyle = '#f9fafb';
    ctx.font = '600 12px "Plus Jakarta Sans", sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(labelText, tagX + 26, tagY + 12);
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

// ---------------- Data Fetching & Roster ----------------

async function fetchPresentList() {
  try {
    const resp = await fetch('/api/present?window_seconds=20');
    if (!resp.ok) return;
    const data = await resp.json();

    const present = data.present || [];
    presentCount.textContent = `${present.length} Active`;
    presentTabBadge.textContent = present.length;
    kpiPresent.textContent = present.length;

    if (present.length === 0) {
      presentList.innerHTML = `
        <div class="empty-state">
          <span class="empty-icon">👤</span>
          <p>No faces detected in camera feed right now</p>
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
      <span>${escapeHtml(emp.name)}</span>
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
    if (name.includes(query)) {
      item.style.display = 'flex';
    } else {
      item.style.display = 'none';
    }
  });
}

async function fetchTodayLog() {
  const logContainer = document.getElementById('todayLogList');
  try {
    const resp = await fetch('/api/today-summary');
    if (!resp.ok) return;
    const data = await resp.json();
    const summary = data.summary || [];

    if (summary.length === 0) {
      logContainer.innerHTML = `
        <div class="empty-state">
          <p>No detections logged today yet.</p>
        </div>
      `;
      return;
    }

    logContainer.innerHTML = summary.map(item => `
      <div class="log-item">
        <div>
          <div class="user-name">${escapeHtml(item.name)}</div>
          <div class="log-item-time">First seen: ${formatTime(item.first_seen)}</div>
        </div>
        <span class="count-badge">${item.detections} detections</span>
      </div>
    `).join('');
  } catch (err) {
    logContainer.innerHTML = `<div class="empty-state"><p>Error loading today's log</p></div>`;
  }
}

// ---------------- Admin & Enrollment ----------------

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

async function enrollCurrentFace() {
  const nameInput = document.getElementById('employeeName');
  const feedback = document.getElementById('enrollFeedback');
  const enrollBtn = document.getElementById('enrollBtn');

  const name = nameInput.value.trim();
  if (!name) {
    showFeedback(feedback, 'error', 'Please enter a full name.');
    return;
  }

  if (!stream) {
    showFeedback(feedback, 'error', 'Camera must be active to capture a face. Click "Start Camera" first.');
    return;
  }

  const captureCanvas = document.createElement('canvas');
  captureCanvas.width = webcam.videoWidth || 640;
  captureCanvas.height = webcam.videoHeight || 480;
  const captureCtx = captureCanvas.getContext('2d');
  captureCtx.drawImage(webcam, 0, 0, captureCanvas.width, captureCanvas.height);
  const base64Image = captureCanvas.toDataURL('image/jpeg', 0.85);

  enrollBtn.disabled = true;
  showFeedback(feedback, 'success', 'Analyzing face capture...');

  try {
    const resp = await fetch('/api/enroll', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: name,
        image: base64Image,
        admin_token: adminToken
      })
    });

    const data = await resp.json();

    if (resp.ok) {
      showFeedback(feedback, 'success', `✓ Successfully enrolled ${name}!`);
      nameInput.value = '';
      fetchRosterList();
      fetchPresentList();
    } else {
      showFeedback(feedback, 'error', data.detail || 'Enrollment failed.');
    }
  } catch (err) {
    showFeedback(feedback, 'error', 'Server error during enrollment.');
  } finally {
    enrollBtn.disabled = false;
  }
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

// ---------------- Admin Modal Handlers ----------------

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
    const d = new Date(isoStr);
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  } catch {
    return isoStr;
  }
}
