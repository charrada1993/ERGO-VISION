// ═══════════════════════════════════════════════════════════
//  ErgoVision — Dashboard JS
//  Handles: Socket.IO, all Chart.js graphs, body map, IMU
// ═══════════════════════════════════════════════════════════

const socket = io();

// ── Chart colour palette ──────────────────────────────────
const C = {
  neck:    '#ff4d6d',
  trunk:   '#00d4ff',
  armL:    '#00e5a0',
  armR:    '#9b59ff',
  forearm: '#ffc94d',
  wrist:   '#ff7c3e',
  rula:    '#00d4ff',
  reba:    '#9b59ff',
  imuAx:   '#ff4d6d',
  imuAy:   '#00e5a0',
  imuAz:   '#00d4ff',
};

// ── Shared chart helpers ──────────────────────────────────
const MAX_POINTS = 60;

function makeTimestamp() {
  return new Date().toLocaleTimeString('en-GB', { hour12: false });
}

function pushRolling(chart, label, ...values) {
  chart.data.labels.push(label);
  if (chart.data.labels.length > MAX_POINTS) chart.data.labels.shift();
  chart.data.datasets.forEach((ds, i) => {
    ds.data.push(values[i] ?? null);
    if (ds.data.length > MAX_POINTS) ds.data.shift();
  });
  chart.update('none'); // 'none' = no animation for live data
}

function baseLineOptions(yLabel = 'Degrees (°)', yMax = null) {
  return {
    responsive: true,
    maintainAspectRatio: true,
    animation: false,
    interaction: { mode: 'index', intersect: false },
    scales: {
      x: {
        ticks: { color: '#6b7a99', maxTicksLimit: 8, font: { size: 11 } },
        grid:  { color: 'rgba(255,255,255,0.04)' },
      },
      y: {
        beginAtZero: true,
        max: yMax || undefined,
        title: { display: true, text: yLabel, color: '#6b7a99', font: { size: 11 } },
        ticks: { color: '#6b7a99', font: { size: 11 } },
        grid:  { color: 'rgba(255,255,255,0.05)' },
      },
    },
    plugins: {
      legend: { display: false },
      tooltip: {
        backgroundColor: '#111827',
        borderColor: 'rgba(255,255,255,0.1)',
        borderWidth: 1,
        titleColor: '#e8edf5',
        bodyColor: '#6b7a99',
        padding: 10,
      },
    },
  };
}

function ds(label, colour, fill = true) {
  return {
    label,
    data: [],
    borderColor: colour,
    backgroundColor: fill ? colour + '18' : 'transparent',
    borderWidth: 2,
    pointRadius: 0,
    tension: 0.35,
    fill,
  };
}

// ══════════════════════════════════════════════════════════
//  1. JOINT ANGLES chart (per body-part — 6 datasets)
// ══════════════════════════════════════════════════════════
const angleCtx = document.getElementById('angleChart').getContext('2d');
const angleChart = new Chart(angleCtx, {
  type: 'line',
  data: {
    labels: [],
    datasets: [
      ds('Neck',      C.neck,    true),
      ds('Trunk',     C.trunk,   true),
      ds('Arm L',     C.armL,    false),
      ds('Arm R',     C.armR,    false),
      ds('Forearm',   C.forearm, false),
      ds('Wrist',     C.wrist,   false),
    ],
  },
  options: baseLineOptions('Angle (°)', 180),
});

// ══════════════════════════════════════════════════════════
//  2. RULA / REBA history chart
// ══════════════════════════════════════════════════════════
const scoreCtx = document.getElementById('scoreChart').getContext('2d');
const scoreChart = new Chart(scoreCtx, {
  type: 'line',
  data: {
    labels: [],
    datasets: [
      { ...ds('RULA', C.rula, true),  yAxisID: 'yRula' },
      { ...ds('REBA', C.reba, false), yAxisID: 'yReba' },
    ],
  },
  options: {
    ...baseLineOptions(),
    scales: {
      x: {
        ticks: { color: '#6b7a99', maxTicksLimit: 8, font: { size: 11 } },
        grid:  { color: 'rgba(255,255,255,0.04)' },
      },
      yRula: {
        type: 'linear', position: 'left',
        beginAtZero: true, max: 8,
        title: { display: true, text: 'RULA (0–7)', color: C.rula, font: { size: 11 } },
        ticks: { color: C.rula, font: { size: 11 } },
        grid:  { color: 'rgba(255,255,255,0.05)' },
      },
      yReba: {
        type: 'linear', position: 'right',
        beginAtZero: true, max: 16,
        title: { display: true, text: 'REBA (0–15)', color: C.reba, font: { size: 11 } },
        ticks: { color: C.reba, font: { size: 11 } },
        grid:  { drawOnChartArea: false },
      },
    },
    plugins: {
      ...baseLineOptions().plugins,
      legend: {
        display: true,
        labels: { color: '#6b7a99', font: { size: 11 }, boxWidth: 12, padding: 16 },
      },
    },
    animation: false,
    interaction: { mode: 'index', intersect: false },
  },
});


// ══════════════════════════════════════════════════════════
//  4. Per-body-part sparkline charts (small, per joint)
//     These are rendered inside the body panel readouts
// ══════════════════════════════════════════════════════════
// (driven by same data as angleChart — no extra canvases needed)

// ══════════════════════════════════════════════════════════
//  Tab switching
// ══════════════════════════════════════════════════════════
const chartPanels = {
  angles: document.getElementById('angleChart').closest('.card'),
  scores: document.getElementById('scoreTrendPanel'),
};

document.getElementById('chartTabs').addEventListener('click', (e) => {
  const btn = e.target.closest('.tab-btn');
  if (!btn) return;
  const key = btn.dataset.chart;
  document.querySelectorAll('#chartTabs .tab-btn').forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  Object.entries(chartPanels).forEach(([k, el]) => {
    if (el) el.style.display = k === key ? '' : 'none';
  });
});

// ══════════════════════════════════════════════════════════
//  Risk helpers
// ══════════════════════════════════════════════════════════
function classifyRisk(text) {
  const t = (text || '').toLowerCase();
  if (t.includes('high') || t.includes('élevé') || t.includes('critical')) return 'high';
  if (t.includes('medium') || t.includes('moyen') || t.includes('moderate')) return 'medium';
  return 'low';
}

function updateChip(el, text, riskClass) {
  if (!el) return;
  el.textContent = text;
  el.className = `risk-chip ${riskClass}`;
}

function setJointRisk(dotId, angle, warn = 20, danger = 45) {
  const el = document.getElementById(dotId);
  if (!el) return;
  el.className = 'joint-dot ' + (angle >= danger ? 'danger' : angle >= warn ? 'warn' : 'ok');
}

// ══════════════════════════════════════════════════════════
//  FPS counter
// ══════════════════════════════════════════════════════════
let frameCount = 0, totalFrames = 0;
setInterval(() => {
  document.getElementById('fpsStat').textContent = frameCount;
  frameCount = 0;
}, 1000);

// ══════════════════════════════════════════════════════════
//  Socket.IO — pose_update
// ══════════════════════════════════════════════════════════
socket.on('pose_update', (data) => {
  frameCount++;
  totalFrames++;
  document.getElementById('frameStat').textContent = totalFrames;

  const a = data.angles || {};
  const ts = makeTimestamp();

  // ── 1. Metric cards ──
  const neck  = +(a.neck          || 0).toFixed(1);
  const trunk = +(a.trunk         || 0).toFixed(1);
  const armL  = +(a.upper_arm_left|| 0).toFixed(1);
  const armR  = +(a.upper_arm_right||0).toFixed(1);
  const frm   = +(a.lower_arm_left || a.forearm || 0).toFixed(1);
  const wrist = +(a.wrist_left    || a.wrist || 0).toFixed(1);

  document.getElementById('neckVal').innerHTML  = neck  + '<small style="font-size:1.2rem">°</small>';
  document.getElementById('trunkVal').innerHTML = trunk + '<small style="font-size:1.2rem">°</small>';
  document.getElementById('armVal').innerHTML   = armL  + '<small style="font-size:1.2rem">°</small>';
  document.getElementById('rulaVal').textContent = data.rula ?? '--';
  document.getElementById('rebaVal').textContent = data.reba ?? '--';

  // Progress bars
  setBar('rulaBar',  data.rula, 7);
  setBar('rebaBar',  data.reba, 15);
  setBar('neckBar',  neck,  90);
  setBar('trunkBar', trunk, 90);
  setBar('armBar',   armL,  90);

  // Risk chips
  const riskClass = classifyRisk(data.risk_level);
  updateChip(document.getElementById('rulaChip'), data.risk_level || 'Low', riskClass);
  updateChip(document.getElementById('rebaChip'), data.risk_level || 'Low', riskClass);

  // ── 2. Joint readouts ──
  setJR('jr-neck',    neck);
  setJR('jr-trunk',   trunk);
  setJR('jr-arml',    armL);
  setJR('jr-armr',    armR);
  setJR('jr-forearm', frm);
  setJR('jr-wrist',   wrist);

  // ── 3. Body risk dots ──
  setJointRisk('jd-neck',   neck,  15, 30);
  setJointRisk('jd-trunk',  trunk, 20, 60);
  setJointRisk('jd-shl',    armL,  45, 90);
  setJointRisk('jd-shr',    armR,  45, 90);
  setJointRisk('jd-ell',    frm,   60, 100);
  setJointRisk('jd-elr',    frm,   60, 100);
  setJointRisk('jd-wrl',    wrist, 15, 30);
  setJointRisk('jd-wrr',    wrist, 15, 30);

  // ── 4. Angle chart (6 body-part lines) ──
  pushRolling(angleChart, ts, neck, trunk, armL, armR, frm, wrist);

  // ── 5. RULA/REBA chart ──
  pushRolling(scoreChart, ts, +(data.rula || 0), +(data.reba || 0));


  // ── 7. Anomaly feed ──
  const feed = document.getElementById('anomalyFeed');
  if (data.anomalies && data.anomalies.length) {
    const items = data.anomalies.slice(0, 6).map(a =>
      `<div class="anomaly-item"><span class="anomaly-icon">⚡</span>${a}</div>`
    ).join('');
    feed.innerHTML = items;
  } else {
    feed.innerHTML = '<div class="anomaly-item anomaly-ok"><span class="anomaly-icon">✅</span> All joints within safe range</div>';
  }
});

// ══════════════════════════════════════════════════════════
//  Socket.IO — config
// ══════════════════════════════════════════════════════════
socket.on('config', (cfg) => {
  const modeMap = { 1: 'Single-view', 2: 'Dual-view Fusion', 3: 'Multi-view 3D' };
  const mode = modeMap[cfg.mode] || 'Unknown';
  setText('sysMode', mode);
  setText('camStatus', 'Active (' + cfg.mode + ' cam' + (cfg.mode > 1 ? 's' : '') + ')');
  // Camera info panel
  setText('camModeBadge', mode);
  setText('camInfoCount', cfg.mode);
  // Page header live mode label
  setText('headerCamMode', mode + ' mode');
  document.getElementById('liveStatus').style.display = 'flex';
});

socket.on('connect', () => {
  setText('camStatus', 'Connected');
});
socket.on('disconnect', () => {
  setText('camStatus', 'Disconnected');
  document.getElementById('camDot').classList.add('warn');
});

// ══════════════════════════════════════════════════════════
//  Utilities
// ══════════════════════════════════════════════════════════
function setText(id, val) {
  const el = document.getElementById(id);
  if (el) el.textContent = val;
}
function setBar(id, val, max) {
  const el = document.getElementById(id);
  if (el) el.style.width = Math.min((val / max) * 100, 100) + '%';
}
function setJR(id, angle) {
  const el = document.getElementById(id);
  if (!el) return;
  // Replace only text node (keep the <span class="jr-unit"> child)
  el.childNodes[0].textContent = isNaN(angle) ? '--' : angle.toFixed(1);
}

// ══════════════════════════════════════════════════════════
//  Theme toggle (persisted)
// ══════════════════════════════════════════════════════════
const themeBtn = document.getElementById('themeBtn');
function applyTheme(isLight) {
  document.body.classList.toggle('light-mode', isLight);
  themeBtn.innerHTML = isLight ? '☀️' : '🌙';
  localStorage.setItem('ergo-theme', isLight ? 'light' : 'dark');
}
themeBtn.addEventListener('click', () => {
  const isNowLight = !document.body.classList.contains('light-mode');
  applyTheme(isNowLight);
  updateChartsTheme(isNowLight);
});

function updateChartsTheme(isLight) {
  const gridCol = isLight ? 'rgba(0,0,0,0.06)' : 'rgba(255,255,255,0.04)';
  const tickCol = isLight ? '#5d6d7e' : '#6b7a99';
  
  const allCharts = [angleChart, scoreChart];
  allCharts.forEach(c => {
    if (!c) return;
    Object.values(c.options.scales).forEach(s => {
      if (s.grid) s.grid.color = gridCol;
      if (s.ticks) s.ticks.color = tickCol;
    });
    if (c.options.plugins.legend && c.options.plugins.legend.labels) {
      c.options.plugins.legend.labels.color = tickCol;
    }
    c.update();
  });
}

applyTheme(localStorage.getItem('ergo-theme') === 'light');
updateChartsTheme(document.body.classList.contains('light-mode'));

// ══════════════════════════════════════════════════════════
//  SKELETON OVERLAY  —  canvas drawn over #videoFeed
//  Driven by Socket.IO 'skeleton_3d' events (33×3 normalised
//  MediaPipe landmarks emitted by socket_events.py)
// ══════════════════════════════════════════════════════════

const skelCanvas    = document.getElementById('skelCanvas');
const skelCtx       = skelCanvas.getContext('2d');
const videoImg      = document.getElementById('videoFeed');

let skelVisible     = true;
let latestLandmarks = null;

// ── MediaPipe landmark indices ────────────────────────────────────────
const LM = {
  NOSE:0, L_EAR:7, R_EAR:8,
  L_SHOULDER:11, R_SHOULDER:12,
  L_ELBOW:13,    R_ELBOW:14,
  L_WRIST:15,    R_WRIST:16,
  L_HIP:23,      R_HIP:24,
  L_KNEE:25,     R_KNEE:26,
  L_ANKLE:27,    R_ANKLE:28,
  L_FOOT:31,     R_FOOT:32,
};

// ── Bone segments [fromIdx, toIdx, colorA, colorB] ────────────────────
// Two-color linear gradient = glow-bone look matching the reference image
const BONES = [
  // Face / head
  [LM.NOSE,       LM.L_EAR,       '#ffffff', '#00eaff'],
  [LM.NOSE,       LM.R_EAR,       '#ffffff', '#00eaff'],
  // Neck
  [LM.L_SHOULDER, LM.NOSE,        '#ffffff', '#ffffff'],
  [LM.R_SHOULDER, LM.NOSE,        '#ffffff', '#ffffff'],
  // Shoulder bar
  [LM.L_SHOULDER, LM.R_SHOULDER,  '#ffd166', '#ffd166'],
  // Left arm
  [LM.L_SHOULDER, LM.L_ELBOW,     '#00eaff', '#06d6a0'],
  [LM.L_ELBOW,    LM.L_WRIST,     '#06d6a0', '#a8ff78'],
  // Right arm
  [LM.R_SHOULDER, LM.R_ELBOW,     '#ff6b6b', '#f7971e'],
  [LM.R_ELBOW,    LM.R_WRIST,     '#f7971e', '#ffd166'],
  // Torso sides
  [LM.L_SHOULDER, LM.L_HIP,       '#00eaff', '#ffd166'],
  [LM.R_SHOULDER, LM.R_HIP,       '#ff6b6b', '#ffd166'],
  // Hip bar
  [LM.L_HIP,      LM.R_HIP,       '#ffd166', '#ffd166'],
  // Left leg
  [LM.L_HIP,      LM.L_KNEE,      '#00eaff', '#06d6a0'],
  [LM.L_KNEE,     LM.L_ANKLE,     '#06d6a0', '#a8ff78'],
  [LM.L_ANKLE,    LM.L_FOOT,      '#a8ff78', '#a8ff78'],
  // Right leg
  [LM.R_HIP,      LM.R_KNEE,      '#ff6b6b', '#f7971e'],
  [LM.R_KNEE,     LM.R_ANKLE,     '#f7971e', '#ffd166'],
  [LM.R_ANKLE,    LM.R_FOOT,      '#ffd166', '#ffd166'],
];

// ── Per-joint dot colours (key joints = warm, rest = cyan) ─────────
const JOINT_COLORS = {};
JOINT_COLORS[LM.L_SHOULDER] = '#ffd166';
JOINT_COLORS[LM.R_SHOULDER] = '#ffd166';
JOINT_COLORS[LM.L_HIP]      = '#f7971e';
JOINT_COLORS[LM.R_HIP]      = '#f7971e';
JOINT_COLORS[LM.NOSE]        = '#ffffff';
JOINT_COLORS[LM.L_EAR]      = '#00eaff';
JOINT_COLORS[LM.R_EAR]      = '#00eaff';
JOINT_COLORS[LM.L_ELBOW]    = '#06d6a0';
JOINT_COLORS[LM.R_ELBOW]    = '#ff6b6b';
JOINT_COLORS[LM.L_WRIST]    = '#a8ff78';
JOINT_COLORS[LM.R_WRIST]    = '#f7971e';
JOINT_COLORS[LM.L_KNEE]     = '#06d6a0';
JOINT_COLORS[LM.R_KNEE]     = '#f7971e';
JOINT_COLORS[LM.L_ANKLE]    = '#a8ff78';
JOINT_COLORS[LM.R_ANKLE]    = '#ffd166';

const DRAW_JOINTS = [
  LM.NOSE, LM.L_EAR, LM.R_EAR,
  LM.L_SHOULDER, LM.R_SHOULDER,
  LM.L_ELBOW,    LM.R_ELBOW,
  LM.L_WRIST,    LM.R_WRIST,
  LM.L_HIP,      LM.R_HIP,
  LM.L_KNEE,     LM.R_KNEE,
  LM.L_ANKLE,    LM.R_ANKLE,
  LM.L_FOOT,     LM.R_FOOT,
];

// ── Sync canvas pixel size to the rendered <img> element ─────────────
function syncCanvasSize() {
  const rect = videoImg.getBoundingClientRect();
  if (rect.width > 0 && rect.height > 0) {
    skelCanvas.width  = rect.width;
    skelCanvas.height = rect.height;
  }
}

// ── Map normalised [0-1] landmark to canvas pixels ────────────────────
function lmPx(lm, W, H) {
  return [lm[0] * W, lm[1] * H];
}

// ── Main draw call ────────────────────────────────────────────────────
function drawSkeleton(landmarks) {
  syncCanvasSize();
  const W = skelCanvas.width;
  const H = skelCanvas.height;
  skelCtx.clearRect(0, 0, W, H);
  if (!skelVisible || !landmarks || landmarks.length < 29) return;

  // 1. Draw bones with gradient + soft glow halo
  for (const [a, b, colA, colB] of BONES) {
    if (a >= landmarks.length || b >= landmarks.length) continue;
    const [x1, y1] = lmPx(landmarks[a], W, H);
    const [x2, y2] = lmPx(landmarks[b], W, H);
    if (x1 === 0 && y1 === 0) continue;
    if (x2 === 0 && y2 === 0) continue;

    const grad = skelCtx.createLinearGradient(x1, y1, x2, y2);
    grad.addColorStop(0, colA);
    grad.addColorStop(1, colB);

    // Soft glow halo (wide transparent stroke)
    skelCtx.beginPath();
    skelCtx.moveTo(x1, y1); skelCtx.lineTo(x2, y2);
    skelCtx.strokeStyle = colA + '44';
    skelCtx.lineWidth   = 8;
    skelCtx.lineCap     = 'round';
    skelCtx.stroke();

    // Sharp core line
    skelCtx.beginPath();
    skelCtx.moveTo(x1, y1); skelCtx.lineTo(x2, y2);
    skelCtx.strokeStyle = grad;
    skelCtx.lineWidth   = 2.5;
    skelCtx.stroke();
  }

  // 2. Draw joint dots (outer glow ring + white border + filled core)
  for (const idx of DRAW_JOINTS) {
    if (idx >= landmarks.length) continue;
    const [cx, cy] = lmPx(landmarks[idx], W, H);
    if (cx === 0 && cy === 0) continue;
    const color = JOINT_COLORS[idx] || '#00eaff';

    // Glow halo
    skelCtx.beginPath();
    skelCtx.arc(cx, cy, 9, 0, Math.PI * 2);
    skelCtx.fillStyle = color + '28';
    skelCtx.fill();

    // White outer ring
    skelCtx.beginPath();
    skelCtx.arc(cx, cy, 5.5, 0, Math.PI * 2);
    skelCtx.strokeStyle = 'rgba(255,255,255,0.9)';
    skelCtx.lineWidth   = 1.8;
    skelCtx.stroke();

    // Filled colour core
    skelCtx.beginPath();
    skelCtx.arc(cx, cy, 4, 0, Math.PI * 2);
    skelCtx.fillStyle = color;
    skelCtx.fill();
  }
}

// ── Socket.IO: receive 33×3 normalised landmarks ──────────────────────
socket.on('skeleton_3d', (data) => {
  if (data && data.landmarks) {
    latestLandmarks = data.landmarks;
    drawSkeleton(latestLandmarks);
  }
});

// ── Re-draw on resize so skeleton tracks the image ───────────────────
window.addEventListener('resize', () => {
  if (latestLandmarks) drawSkeleton(latestLandmarks);
});

// ── Sync once image has loaded ────────────────────────────────────────
videoImg.addEventListener('load', syncCanvasSize);

// ── Toggle button: SKEL ON / SKEL OFF ────────────────────────────────
document.getElementById('skelToggleBtn').addEventListener('click', function () {
  skelVisible = !skelVisible;
  this.textContent      = skelVisible ? 'SKEL ON'  : 'SKEL OFF';
  this.style.background = skelVisible ? 'rgba(0,212,255,0.15)' : 'rgba(255,255,255,0.05)';
  this.style.color      = skelVisible ? 'var(--cyan)'           : 'var(--text-muted)';
  this.style.borderColor= skelVisible ? 'var(--cyan)'           : 'rgba(255,255,255,0.2)';
  if (!skelVisible) skelCtx.clearRect(0, 0, skelCanvas.width, skelCanvas.height);
  else if (latestLandmarks) drawSkeleton(latestLandmarks);
});