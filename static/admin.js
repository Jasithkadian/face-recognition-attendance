/**
 * FacePulse — Admin Console Script
 * Token authentication gate, today's summary & report downloads (Excel/PDF),
 * searchable activity log with vector distances, roster management & deletion,
 * 15-pose guided employee registration, and Supabase connection pool telemetry.
 */

// State
let adminToken = sessionStorage.getItem('adminToken') || localStorage.getItem('adminToken') || '';
let enrolledEmployees = [];
let enrollStream = null;
let isEnrolling = false;
let enrollmentSamples = [];
let currentPoseIndex = 0;
let enrollmentTimer = null;
let enrollmentResetTimer = null;
let isProcessingEnrollmentTick = false;
let logDebounceTimer = null;

// Pose Guidance sequence for 15 samples
const ENROLL_POSES = [
  { title: "Look Straight", prompt: "Look directly into the camera lens", icon: "👤" },
  { title: "Look Straight", prompt: "Hold position, looking straight ahead", icon: "👤" },
  { title: "Turn Left", prompt: "Turn head slightly to your LEFT", icon: "👈" },
  { title: "Turn Left", prompt: "Hold position, facing slightly LEFT", icon: "👈" },
  { title: "Turn Right", prompt: "Turn head slightly to your RIGHT", icon: "👉" },
  { title: "Turn Right", prompt: "Hold position, facing slightly RIGHT", icon: "👉" },
  { title: "Tilt Up", prompt: "Tilt chin and head slightly UP", icon: "👆" },
  { title: "Tilt Up", prompt: "Hold position, tilted slightly UP", icon: "👆" },
  { title: "Tilt Down", prompt: "Tilt head slightly DOWN", icon: "👇" },
  { title: "Tilt Down", prompt: "Hold position, tilted slightly DOWN", icon: "👇" },
  { title: "Slight Smile", prompt: "Smile or change facial expression slightly", icon: "😊" },
  { title: "Neutral Expression", prompt: "Hold expression or neutral gaze", icon: "😊" },
  { title: "Natural Pose", prompt: "Blink naturally and relax facial muscles", icon: "👁️" },
  { title: "Natural Pose", prompt: "Look straight into the lens again", icon: "👤" },
  { title: "Final Sample", prompt: "Final capture: Hold still looking straight", icon: "✨" }
];

// DOM Ready
document.addEventListener('DOMContentLoaded', () => {
  renderThumbnailSlots();

  if (adminToken) {
    // Attempt automatic unlock with saved token
    verifyAndUnlock(adminToken, true);
  }
});

// ---------------- Token Authentication & Gate ----------------

async function handleTokenSubmit(e) {
  e.preventDefault();
  const tokenInput = document.getElementById('tokenInput');
  const tokenVal = tokenInput.value.trim();
  if (!tokenVal) return;

  const btn = document.getElementById('tokenSubmitBtn');
  btn.disabled = true;
  btn.textContent = 'Verifying...';

  await verifyAndUnlock(tokenVal, false);

  btn.disabled = false;
  btn.textContent = 'Unlock Console';
}

async function verifyAndUnlock(token, isAutoUnlock = false) {
  const errorEl = document.getElementById('tokenError');
  errorEl.classList.add('hidden');

  try {
    const resp = await fetch('/api/verify-token', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ admin_token: token })
    });

    if (!resp.ok) {
      throw new Error(`Server returned HTTP ${resp.status}`);
    }

    const data = await resp.json();
    if (data.valid) {
      adminToken = token;
      sessionStorage.setItem('adminToken', token);
      localStorage.setItem('adminToken', token);

      document.getElementById('tokenGate').classList.add('hidden');
      document.getElementById('adminDashboard').classList.remove('hidden');

      // Load initial admin data
      fetchTodaySummary();
      fetchActivityLog();
      fetchRoster();
      fetchSystemStatus();
    } else {
      if (!isAutoUnlock) {
        errorEl.textContent = 'Invalid ADMIN_TOKEN. Please check your credentials.';
        errorEl.classList.remove('hidden');
      }
      sessionStorage.removeItem('adminToken');
      localStorage.removeItem('adminToken');
      adminToken = '';
    }
  } catch (err) {
    console.error('Token verification error:', err);
    if (!isAutoUnlock) {
      errorEl.textContent = `Connection error verifying token: ${err.message}`;
      errorEl.classList.remove('hidden');
    }
  }
}

function logoutAdmin() {
  adminToken = '';
  sessionStorage.removeItem('adminToken');
  localStorage.removeItem('adminToken');

  stopEnrollCamera();
  cancelGuidedEnrollment();

  document.getElementById('adminDashboard').classList.add('hidden');
  document.getElementById('tokenGate').classList.remove('hidden');
  document.getElementById('tokenInput').value = '';
  document.getElementById('tokenError').classList.add('hidden');
}

// ---------------- Authenticated Fetch Helper ----------------

async function adminFetch(url, options = {}) {
  const headers = options.headers ? { ...options.headers } : {};
  if (adminToken) {
    headers['x-admin-token'] = adminToken;
  }
  options.headers = headers;

  // Also include admin_token query param fallback
  const separator = url.includes('?') ? '&' : '?';
  const authenticatedUrl = `${url}${separator}admin_token=${encodeURIComponent(adminToken)}`;

  const resp = await fetch(authenticatedUrl, options);
  if (resp.status === 401) {
    alert('Session expired or invalid admin token. Please unlock again.');
    logoutAdmin();
    return null;
  }
  return resp;
}

// ---------------- Tab Navigation ----------------

function switchAdminTab(tabId, btnEl) {
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));

  const targetTab = document.getElementById(tabId);
  if (targetTab) targetTab.classList.add('active');
  if (btnEl) btnEl.classList.add('active');

  if (tabId === 'todayTab') fetchTodaySummary();
  if (tabId === 'logTab') fetchActivityLog();
  if (tabId === 'rosterTab') fetchRoster();
  if (tabId === 'systemTab') fetchSystemStatus();

  // If navigating away from enrollment tab, stop enroll camera
  if (tabId !== 'enrollTab' && enrollStream) {
    stopEnrollCamera();
  }
}

// ---------------- Tab 1: Today's Summary & Reports ----------------

async function fetchTodaySummary() {
  try {
    const resp = await adminFetch('/api/today-summary');
    if (!resp || !resp.ok) return;

    const data = await resp.json();
    const summary = data.summary || [];
    renderTodaySummaryTable(summary);

    const count = summary.length;
    const kpiToday = document.getElementById('kpiToday');
    const todayTabBadge = document.getElementById('todayTabBadge');
    if (kpiToday) kpiToday.textContent = count;
    if (todayTabBadge) todayTabBadge.textContent = count;
  } catch (err) {
    console.error('Error fetching today summary:', err);
  }
}

function renderTodaySummaryTable(summary) {
  const tbody = document.getElementById('todaySummaryBody');
  if (!tbody) return;

  if (summary.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="4" class="empty-cell">No attendance records logged yet today.</td>
      </tr>
    `;
    return;
  }

  tbody.innerHTML = summary.map(row => {
    return `
      <tr>
        <td><strong>${escapeHtml(row.name)}</strong></td>
        <td>${formatTime(row.first_seen)}</td>
        <td>${formatTime(row.last_seen)}</td>
        <td><span class="count-badge" style="background: #EBF8FF; color: #1E40AF; border: 1px solid #BFDBFE;">${row.detections} detections</span></td>
      </tr>
    `;
  }).join('');
}

async function downloadReport(format) {
  if (!adminToken) {
    alert('Admin token required.');
    logoutAdmin();
    return;
  }

  try {
    const url = `/api/report/today.${format}?admin_token=${encodeURIComponent(adminToken)}`;
    const resp = await fetch(url, {
      headers: { 'x-admin-token': adminToken }
    });

    if (!resp.ok) {
      if (resp.status === 401) {
        alert('Unauthorized admin token.');
        logoutAdmin();
      } else {
        const data = await resp.json().catch(() => ({}));
        alert(data.detail || `Failed to download ${format.toUpperCase()} report.`);
      }
      return;
    }

    const blob = await resp.blob();
    const blobUrl = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = blobUrl;
    const todayStr = new Date().toISOString().slice(0, 10);
    a.download = `attendance_report_${todayStr}.${format}`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    window.URL.revokeObjectURL(blobUrl);
  } catch (err) {
    console.error(`Error downloading ${format} report:`, err);
    alert(`Network error downloading report: ${err.message}`);
  }
}

// ---------------- Tab 2: Activity Log ----------------

function debounceFetchActivityLog() {
  if (logDebounceTimer) clearTimeout(logDebounceTimer);
  logDebounceTimer = setTimeout(fetchActivityLog, 300);
}

async function fetchActivityLog() {
  const searchInput = document.getElementById('logSearch');
  const dateInput = document.getElementById('logDate');
  const nameQuery = searchInput ? searchInput.value.trim() : '';
  const dateQuery = dateInput ? dateInput.value : '';

  let url = '/api/activity-log?limit=150';
  if (nameQuery) url += `&name=${encodeURIComponent(nameQuery)}`;
  if (dateQuery) url += `&date=${encodeURIComponent(dateQuery)}`;

  try {
    const resp = await adminFetch(url);
    if (!resp || !resp.ok) return;

    const data = await resp.json();
    renderActivityLogTable(data.logs || []);
  } catch (err) {
    console.error('Error fetching activity log:', err);
  }
}

function renderActivityLogTable(logs) {
  const tbody = document.getElementById('activityLogBody');
  if (!tbody) return;

  if (logs.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="5" class="empty-cell">No presence events match the selected criteria.</td>
      </tr>
    `;
    return;
  }

  tbody.innerHTML = logs.map(row => {
    const conf = (row.confidence !== undefined && row.confidence !== null) ? `${row.confidence}%` : 'N/A';
    const dist = (row.distance !== undefined && row.distance !== null) ? Number(row.distance).toFixed(3) : 'N/A';
    return `
      <tr>
        <td>${formatTime(row.timestamp)}</td>
        <td><strong>${escapeHtml(row.name)}</strong></td>
        <td>${conf}</td>
        <td><code style="font-size: 0.82rem; background: #F8FAFC; padding: 2px 5px; border-radius: 4px; border: 1px solid #E2E8F0;">${dist}</code></td>
        <td><span class="admin-badge unlocked" style="display: inline-block;">✓ Confirmed</span></td>
      </tr>
    `;
  }).join('');
}

// ---------------- Tab 3: Enrolled Roster ----------------

async function fetchRoster() {
  try {
    const resp = await adminFetch('/api/employees');
    if (!resp || !resp.ok) return;

    const data = await resp.json();
    enrolledEmployees = data.employees || [];
    renderRosterList(enrolledEmployees);

    const count = enrolledEmployees.length;
    const kpiEnrolled = document.getElementById('kpiEnrolled');
    const rosterTabBadge = document.getElementById('rosterTabBadge');
    const rosterCount = document.getElementById('rosterCount');
    if (kpiEnrolled) kpiEnrolled.textContent = count;
    if (rosterTabBadge) rosterTabBadge.textContent = count;
    if (rosterCount) rosterCount.textContent = `${count} Enrolled`;
  } catch (err) {
    console.error('Error fetching roster:', err);
  }
}

function filterRoster() {
  const searchInput = document.getElementById('rosterSearch');
  const query = searchInput ? searchInput.value.trim().toLowerCase() : '';
  const filtered = enrolledEmployees.filter(emp => emp.name.toLowerCase().includes(query));
  renderRosterList(filtered);
}

function renderRosterList(employees) {
  const container = document.getElementById('rosterList');
  if (!container) return;

  if (employees.length === 0) {
    container.innerHTML = `
      <div class="empty-state">
        <p>No registered employees found.</p>
      </div>
    `;
    return;
  }

  container.innerHTML = employees.map(emp => {
    const initial = emp.name ? emp.name.charAt(0).toUpperCase() : '?';
    const regDate = emp.created_at ? formatTime(emp.created_at) : 'Enrolled';
    return `
      <div class="roster-item" style="display: flex; justify-content: space-between; align-items: center; padding: 0.75rem 1rem; border-bottom: 1px solid var(--border-color); background: #FFFFFF; border-radius: var(--radius-md); margin-bottom: 0.5rem; border: 1px solid var(--border-color);">
        <div style="display: flex; align-items: center; gap: 0.75rem;">
          <div class="user-avatar" style="width: 36px; height: 36px; border-radius: 50%; background: #E2E8F0; display: flex; align-items: center; justify-content: center; font-weight: 700;">${initial}</div>
          <div>
            <div style="font-weight: 600; font-size: 0.92rem; color: var(--text-main);">${escapeHtml(emp.name)}</div>
            <div style="font-size: 0.75rem; color: var(--text-dim);">${regDate} • ID #${emp.id}</div>
          </div>
        </div>
        <div>
          <button class="btn-danger btn-sm" onclick="confirmDeleteEmployee(${emp.id}, '${escapeHtml(emp.name).replace(/'/g, "\\'")}')">
            Delete
          </button>
        </div>
      </div>
    `;
  }).join('');
}

async function confirmDeleteEmployee(id, name) {
  if (!confirm(`Are you sure you want to delete "${name}" from the system?\n\nThis will remove their face encodings and cannot be undone.`)) {
    return;
  }

  try {
    const resp = await adminFetch(`/api/employees/${id}`, { method: 'DELETE' });
    if (!resp) return;

    if (resp.ok) {
      fetchRoster();
      fetchSystemStatus();
    } else {
      const data = await resp.json().catch(() => ({}));
      alert(data.detail || 'Failed to delete employee.');
    }
  } catch (err) {
    console.error('Delete error:', err);
    alert(`Error deleting employee: ${err.message}`);
  }
}

// ---------------- Tab 4: Register Person (15-Pose Flow) ----------------

function renderThumbnailSlots() {
  const container = document.getElementById('sampleThumbnails');
  if (!container) return;
  container.innerHTML = '';
  for (let i = 0; i < 15; i++) {
    const slot = document.createElement('div');
    slot.className = 'thumb-slot';
    slot.id = `thumb-${i}`;
    slot.innerHTML = `<span class="thumb-num">${i + 1}</span>`;
    container.appendChild(slot);
  }
}

async function startEnrollCamera() {
  try {
    const constraints = {
      video: { width: { ideal: 640 }, height: { ideal: 480 }, facingMode: 'user' }
    };
    enrollStream = await navigator.mediaDevices.getUserMedia(constraints);
    const video = document.getElementById('enrollWebcam');
    video.srcObject = enrollStream;
    document.getElementById('enrollCamPlaceholder').classList.add('hidden');
  } catch (err) {
    console.error('Enroll camera error:', err);
    alert('Could not access webcam for registration. Please enable camera permissions.');
  }
}

function stopEnrollCamera() {
  if (enrollStream) {
    enrollStream.getTracks().forEach(track => track.stop());
    enrollStream = null;
  }
  const placeholder = document.getElementById('enrollCamPlaceholder');
  if (placeholder) placeholder.classList.remove('hidden');
}

async function startGuidedEnrollment() {
  const nameInput = document.getElementById('employeeName');
  const name = nameInput ? nameInput.value.trim() : '';
  const feedback = document.getElementById('enrollFeedback');

  if (!name) {
    showFeedback(feedback, 'error', 'Please enter employee name before starting registration.');
    return;
  }

  // Ensure camera is active
  if (!enrollStream) {
    await startEnrollCamera();
    if (!enrollStream) return;
  }

  isEnrolling = true;
  enrollmentSamples = [];
  currentPoseIndex = 0;
  isProcessingEnrollmentTick = false;

  document.getElementById('enrollBtn').classList.add('hidden');
  document.getElementById('cancelEnrollBtn').classList.remove('hidden');
  showFeedback(feedback, 'info', 'Guided capture starting: look directly at the camera.');

  renderThumbnailSlots();
  updateEnrollmentUI();

  if (enrollmentTimer) clearInterval(enrollmentTimer);
  enrollmentTimer = setInterval(enrollmentTick, 350);
}

function updateEnrollmentUI() {
  const currentCount = enrollmentSamples.length;
  const percent = Math.round((currentCount / 15) * 100);
  const pose = ENROLL_POSES[Math.min(currentCount, 14)];

  const poseIcon = document.getElementById('poseIcon');
  const poseTitle = document.getElementById('poseTitle');
  const poseInstruction = document.getElementById('poseInstruction');
  const sampleCounter = document.getElementById('sampleCounter');
  const enrollPercent = document.getElementById('enrollPercent');
  const enrollProgressBar = document.getElementById('enrollProgressBar');

  if (poseIcon) poseIcon.textContent = pose.icon;
  if (poseTitle) poseTitle.textContent = `${pose.title} (${currentCount + 1}/15)`;
  if (poseInstruction) poseInstruction.textContent = pose.prompt;
  if (sampleCounter) sampleCounter.textContent = `Sample ${currentCount} of 15`;
  if (enrollPercent) enrollPercent.textContent = `${percent}%`;
  if (enrollProgressBar) enrollProgressBar.style.width = `${percent}%`;
}

async function enrollmentTick() {
  if (!isEnrolling || enrollmentSamples.length >= 15) {
    if (enrollmentTimer) {
      clearInterval(enrollmentTimer);
      enrollmentTimer = null;
    }
    return;
  }

  if (isProcessingEnrollmentTick) return;
  isProcessingEnrollmentTick = true;

  const video = document.getElementById('enrollWebcam');
  if (!video || video.paused || video.ended || !video.videoWidth) {
    isProcessingEnrollmentTick = false;
    return;
  }

  try {
    const captureCanvas = document.createElement('canvas');
    captureCanvas.width = 360;
    captureCanvas.height = Math.round((video.videoHeight * 360) / video.videoWidth);
    const cctx = captureCanvas.getContext('2d');
    cctx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);
    const base64Image = captureCanvas.toDataURL('image/jpeg', 0.85);

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
        const sampleIdx = enrollmentSamples.length;
        enrollmentSamples.push(base64Image);

        const thumbSlot = document.getElementById(`thumb-${sampleIdx}`);
        if (thumbSlot) {
          thumbSlot.innerHTML = `<img src="${base64Image}" alt="Sample ${sampleIdx + 1}" style="width: 100%; height: 100%; object-fit: cover; border-radius: 4px;">`;
          thumbSlot.classList.add('active');
        }

        updateEnrollmentUI();

        if (enrollmentSamples.length >= 15) {
          if (enrollmentTimer) {
            clearInterval(enrollmentTimer);
            enrollmentTimer = null;
          }
          finishGuidedEnrollment();
        }
      } else {
        const poseInstruction = document.getElementById('poseInstruction');
        if (poseInstruction) {
          poseInstruction.textContent = `⚠️ ${data.reason || 'Position face clearly in frame'}`;
        }
      }
    }
  } catch (err) {
    console.warn('Enrollment tick error:', err);
  } finally {
    isProcessingEnrollmentTick = false;
  }
}

async function finishGuidedEnrollment() {
  if (enrollmentTimer) {
    clearInterval(enrollmentTimer);
    enrollmentTimer = null;
  }
  isEnrolling = false;

  const nameInput = document.getElementById('employeeName');
  const feedback = document.getElementById('enrollFeedback');
  const name = nameInput ? nameInput.value.trim() : '';

  showFeedback(feedback, 'info', 'Averaging 15 128-d encodings and saving employee profile to Supabase...');

  try {
    const resp = await fetch('/api/enroll', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: name,
        images: enrollmentSamples.slice(0, 15),
        admin_token: adminToken
      })
    });

    const data = await resp.json().catch(() => ({}));
    if (resp.ok) {
      showFeedback(feedback, 'success', `✓ Successfully enrolled "${name}" using 15 facial samples!`);
      if (nameInput) nameInput.value = '';
      fetchRoster();
      fetchSystemStatus();

      if (enrollmentResetTimer) clearTimeout(enrollmentResetTimer);
      enrollmentResetTimer = setTimeout(resetEnrollmentUI, 3000);
    } else {
      showFeedback(feedback, 'error', data.detail || `Enrollment failed with status ${resp.status}`);
      cancelGuidedEnrollment();
    }
  } catch (err) {
    showFeedback(feedback, 'error', `Enrollment failed: ${err.message}`);
    cancelGuidedEnrollment();
  }
}

function cancelGuidedEnrollment() {
  if (enrollmentTimer) {
    clearInterval(enrollmentTimer);
    enrollmentTimer = null;
  }
  if (enrollmentResetTimer) {
    clearTimeout(enrollmentResetTimer);
    enrollmentResetTimer = null;
  }
  isEnrolling = false;
  enrollmentSamples = [];
  resetEnrollmentUI();
}

function resetEnrollmentUI() {
  isEnrolling = false;
  enrollmentSamples = [];
  const poseIcon = document.getElementById('poseIcon');
  const poseTitle = document.getElementById('poseTitle');
  const poseInstruction = document.getElementById('poseInstruction');
  const sampleCounter = document.getElementById('sampleCounter');
  const enrollPercent = document.getElementById('enrollPercent');
  const enrollProgressBar = document.getElementById('enrollProgressBar');
  const enrollBtn = document.getElementById('enrollBtn');
  const cancelEnrollBtn = document.getElementById('cancelEnrollBtn');

  if (poseIcon) poseIcon.textContent = "👤";
  if (poseTitle) poseTitle.textContent = "Ready to Register";
  if (poseInstruction) poseInstruction.textContent = 'Start camera, enter name, and click "Start 15-Pose Capture".';
  if (sampleCounter) sampleCounter.textContent = "Sample 0 of 15";
  if (enrollPercent) enrollPercent.textContent = "0%";
  if (enrollProgressBar) enrollProgressBar.style.width = "0%";
  if (enrollBtn) enrollBtn.classList.remove('hidden');
  if (cancelEnrollBtn) cancelEnrollBtn.classList.add('hidden');
  renderThumbnailSlots();
}

// ---------------- Tab 5: System & DB Pool Telemetry ----------------

async function fetchSystemStatus() {
  try {
    const resp = await adminFetch('/api/admin/status');
    if (!resp || !resp.ok) return;

    const data = await resp.json();
    const pool = data.pool || {};

    const activeEl = document.getElementById('metricActiveConn');
    const idleEl = document.getElementById('metricIdleConn');
    const maxEl = document.getElementById('metricMaxConn');
    const enrolledEl = document.getElementById('metricEnrolledCount');
    const kpiPool = document.getElementById('kpiPool');

    const active = pool.active_checked_out !== undefined ? pool.active_checked_out : '-';
    const idle = pool.idle_available !== undefined ? pool.idle_available : '-';
    const max = pool.maxconn !== undefined ? pool.maxconn : '-';
    const enrolled = data.enrolled_count !== undefined ? data.enrolled_count : '-';

    if (activeEl) activeEl.textContent = active;
    if (idleEl) idleEl.textContent = idle;
    if (maxEl) maxEl.textContent = max;
    if (enrolledEl) enrolledEl.textContent = enrolled;

    if (kpiPool) {
      kpiPool.textContent = `${active} active / ${idle} idle`;
    }
  } catch (err) {
    console.error('Error fetching system status:', err);
  }
}

// ---------------- Utility Helpers ----------------

function showFeedback(el, type, msg) {
  if (!el) return;
  el.className = `feedback-banner ${type}`;
  el.textContent = msg;
  el.classList.remove('hidden');
}

function escapeHtml(str) {
  if (!str) return '';
  return String(str).replace(/[&<>'"]/g, 
    tag => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[tag] || tag)
  );
}

function formatTime(isoStr) {
  if (!isoStr) return 'N/A';
  try {
    let str = String(isoStr);
    if (!str.endsWith('Z') && !/[+-]\d{2}:\d{2}$/.test(str)) {
      str += 'Z';
    }
    const d = new Date(str);
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }) + 
           ' (' + d.toLocaleDateString([], { month: 'short', day: 'numeric' }) + ')';
  } catch {
    return String(isoStr);
  }
}
