'use strict';

// Processing resolution — detection runs on a downscaled grayscale frame.
// Same pipeline the real device would run: background subtraction -> blobs -> track.
const PROC_W = 320;
let PROC_H = 180;

// Heuristic mosquito confidence must reach this before a snapshot is taken.
const CONF_SNAPSHOT = 0.99;
const SNAPSHOT_COOLDOWN_MS = 4000;

const video = document.getElementById('video');
const overlay = document.getElementById('overlay');
const octx = overlay.getContext('2d');
const debugCv = document.getElementById('debug');
const dctx = debugCv.getContext('2d');

const proc = document.createElement('canvas');
const pctx = proc.getContext('2d', { willReadFrequently: true });

const ui = {
  camera: document.getElementById('camera'),
  beamStyle: document.getElementById('beamStyle'),
  emitter: document.getElementById('emitter'),
  sensitivity: document.getElementById('sensitivity'),
  divergence: document.getElementById('divergence'),
  minSize: document.getElementById('minSize'),
  maxSize: document.getElementById('maxSize'),
  demoBtn: document.getElementById('demoBtn'),
  debugBtn: document.getElementById('debugBtn'),
  resetBtn: document.getElementById('resetBtn'),
  status: document.getElementById('status'),
  conf: document.getElementById('conf'),
  fps: document.getElementById('fps'),
  resInfo: document.getElementById('resInfo'),
  gallerySection: document.getElementById('gallerySection'),
  gallery: document.getElementById('gallery'),
  clearShots: document.getElementById('clearShots'),
  cameraBtn: document.getElementById('cameraBtn'),
  warning: document.getElementById('warning'),
  controls: document.getElementById('controls'),
  controlsToggle: document.getElementById('controlsToggle'),
  dock: document.getElementById('dock'),
  clearTarget: document.getElementById('clearTarget'),
};

// Camera-quality warnings. lowRes is decided once a camera starts; lowFps is
// measured continuously. The banner text is only rewritten when it changes.
const warn = { lowRes: false, lowFps: false };
let lastWarnText = '';

function refreshWarning() {
  const msgs = [];
  if (warn.lowRes) msgs.push('Low-resolution camera detected — detection is limited and small bugs may be missed.');
  if (warn.lowFps) msgs.push('Low frame rate on this device — fast-moving bugs may be missed.');
  const text = msgs.join(' ');
  if (text === lastWarnText) return;
  lastWarnText = text;
  if (text) {
    ui.warning.textContent = '⚠ ' + text;
    ui.warning.hidden = false;
  } else {
    ui.warning.hidden = true;
  }
}

const BEAM_PRESETS = {
  laser:      { color: [57, 255, 110],  originW: 3,  divDefault: 12, core: true },
  flashlight: { color: [255, 213, 145], originW: 16, divDefault: 55, core: false },
};

let stream = null;
let bg = null, gray = null, bin = null, skin = null, skinInt = null;
let debugOn = false;
let manual = null; // manual target in proc coords
let sceneBusy = false; // large foreground (person walking, lighting change) — detection suppressed
let lastBig = [];
let lastSnapshotT = -Infinity;

// Tracker: search -> locked -> coast (briefly, on miss) -> search
const track = {
  state: 'search', x: 0, y: 0, vx: 0, vy: 0, sx: 0, sy: 0, miss: 0,
  frames: 0, conf: 0, areaEma: 0, speedEma: 0, turnEma: 0, heading: null,
  snapshotTaken: false, startX: 0, startY: 0, maxDisp: 0, stale: 0,
};

const demo = { on: false, x: 80, y: 60, vx: 25, vy: 12, t: 0 };

// Self-contained "simulated room" shown on load (no camera). A bug flies around a
// drawn room while the virtual laser tracks it — a preview of what the tool does.
const sim = { active: true, x: 0, y: 0, vx: 60, vy: 30, t: 0, tx: 0, ty: 0, started: false };

const isTouch = matchMedia('(hover: none) and (pointer: coarse)').matches;

// ---------- camera ----------

async function startCamera(deviceId) {
  if (stream) stream.getTracks().forEach(t => t.stop());
  const videoConstraints = {
    width: { ideal: 1920 },
    height: { ideal: 1080 },
  };
  if (deviceId) videoConstraints.deviceId = { exact: deviceId };
  else videoConstraints.facingMode = isTouch ? { ideal: 'environment' } : 'user';

  stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: videoConstraints });
  video.srcObject = stream;
  await video.play();

  PROC_H = Math.max(2, Math.round(PROC_W * video.videoHeight / video.videoWidth));
  proc.width = PROC_W;
  proc.height = PROC_H;
  debugCv.width = PROC_W;
  debugCv.height = PROC_H;
  bg = null; // rebuild background model
  resetTrack();
  const vw = video.videoWidth, vh = video.videoHeight;
  ui.resInfo.textContent = `${vw}×${vh}`;
  // A poor camera (or a heavily downscaled mobile stream) hurts detection — flag it.
  warn.lowRes = (vw > 0 && vw < 640) || (vh > 0 && vh < 480);
  refreshWarning();
  await listCameras(deviceId);
}

async function listCameras(currentId) {
  const devices = await navigator.mediaDevices.enumerateDevices();
  const cams = devices.filter(d => d.kind === 'videoinput');
  ui.camera.innerHTML = '';
  for (const cam of cams) {
    const opt = document.createElement('option');
    opt.value = cam.deviceId;
    opt.textContent = cam.label || `Camera ${ui.camera.length + 1}`;
    ui.camera.appendChild(opt);
  }
  const activeId = currentId || (stream && stream.getVideoTracks()[0].getSettings().deviceId);
  if (activeId) ui.camera.value = activeId;
}

// ---------- detection ----------

function processFrame(dt, now) {
  pctx.drawImage(video, 0, 0, PROC_W, PROC_H);
  if (demo.on) stepDemo(dt); // draws the demo bug into the proc frame so the detector sees it

  const d = pctx.getImageData(0, 0, PROC_W, PROC_H).data;
  const n = PROC_W * PROC_H;

  if (!bg || bg.length !== n) {
    bg = new Float32Array(n);
    gray = new Float32Array(n);
    bin = new Uint8Array(n);
    skin = new Uint8Array(n);
    skinInt = new Int32Array((PROC_W + 1) * (PROC_H + 1));
    for (let i = 0; i < n; i++) bg[i] = (d[i * 4] + 2 * d[i * 4 + 1] + d[i * 4 + 2]) / 4;
    return;
  }

  const thr = 65 - Number(ui.sensitivity.value);
  let fgCount = 0;
  for (let i = 0; i < n; i++) {
    const r = d[i * 4], gg = d[i * 4 + 1], bb = d[i * 4 + 2];
    const g = (r + 2 * gg + bb) / 4;
    gray[i] = g;
    const fg = Math.abs(g - bg[i]) > thr;
    bin[i] = fg ? 1 : 0;
    if (fg) fgCount++;
    skin[i] = isSkin(r, gg, bb) ? 1 : 0;
    // adapt background slowly under foreground so a moving target isn't absorbed
    bg[i] += (fg ? 0.005 : 0.06) * (g - bg[i]);
  }
  buildSkinIntegral();

  const all = findBlobs(bin, PROC_W, PROC_H, Number(ui.minSize.value));
  const maxA = Number(ui.maxSize.value);
  const bigThr = Math.max(1200, maxA * 3);
  const big = all.filter(bl => bl.area > bigThr);
  lastBig = big;

  // Rejection of people and other non-insect motion:
  //  - a big chunk of the frame moving = someone walked by -> ignore everything
  //  - blobs adjacent to a big moving region = fragments of it (hands, hair)
  sceneBusy = fgCount > n * 0.05;
  let assoc = [];   // blobs eligible to CONTINUE an existing lock
  let acquire = []; // blobs eligible to START a new lock
  if (!sceneBusy) {
    assoc = all.filter(bl => bl.area <= maxA && !isNearBig(bl, big));
    // Acquisition is stricter: never start a lock on skin. Eyes/nostrils/mouth
    // aren't skin-colored themselves but sit inside a skin region, so we test
    // the neighborhood. A bug flying over a face is still *kept* via `assoc`.
    acquire = assoc.filter(bl => skinRatioAround(bl.x, bl.y, 7) < 0.25);
  }

  // The demo bug is ground truth: it must survive every filter. It routinely
  // flies inside the user's silhouette bbox (isNearBig) and over skin, and a
  // busy frame would suppress it entirely. Use the detected blob when there is
  // one; if detection merged it into body-fringe motion, synthesize it at the
  // known position.
  let demoBlob = null;
  if (demo.on) {
    demoBlob = all.find(bl => bl.area <= maxA && Math.hypot(bl.x - demo.x, bl.y - demo.y) < 8) || {
      x: demo.x, y: demo.y, area: 7,
      minx: demo.x - 2, maxx: demo.x + 2, miny: demo.y - 2, maxy: demo.y + 2,
    };
    if (!assoc.includes(demoBlob)) assoc.push(demoBlob);
    if (!acquire.includes(demoBlob)) acquire.push(demoBlob);
  }

  updateTrack(assoc, acquire, now, demoBlob);
  if (debugOn) renderDebug(acquire, big);
}

// Skin test in normalized RGB (lighting-robust). Catches faces and hands
// across a wide range of tones; deliberately a bit greedy so facial features
// near skin are excluded too.
function isSkin(r, g, b) {
  const sum = r + g + b;
  if (sum < 90) return false; // too dark to judge
  const rn = r / sum, gn = g / sum;
  return r > 70 && r >= g && g >= b * 0.85 &&
         rn > 0.33 && rn < 0.5 && gn > 0.26 && gn < 0.40 &&
         (r - b) > 8;
}

function buildSkinIntegral() {
  const W = PROC_W, H = PROC_H, S = skinInt;
  for (let y = 0; y < H; y++) {
    let rowSum = 0;
    const row = (y + 1) * (W + 1);
    const prev = y * (W + 1);
    for (let x = 0; x < W; x++) {
      rowSum += skin[y * W + x];
      S[row + x + 1] = S[prev + x + 1] + rowSum;
    }
  }
}

function skinRatioAround(cx, cy, rad) {
  const W = PROC_W, H = PROC_H, S = skinInt;
  const x0 = Math.max(0, (cx | 0) - rad), y0 = Math.max(0, (cy | 0) - rad);
  const x1 = Math.min(W, (cx | 0) + rad + 1), y1 = Math.min(H, (cy | 0) + rad + 1);
  const area = (x1 - x0) * (y1 - y0);
  if (area <= 0) return 0;
  const sum = S[y1 * (W + 1) + x1] - S[y0 * (W + 1) + x1] - S[y1 * (W + 1) + x0] + S[y0 * (W + 1) + x0];
  return sum / area;
}

function isNearBig(bl, big) {
  for (const g of big) {
    if (bl.x > g.minx - 14 && bl.x < g.maxx + 14 && bl.y > g.miny - 14 && bl.y < g.maxy + 14) return true;
  }
  return false;
}

function findBlobs(b, w, h, minArea) {
  const labels = new Int32Array(w * h);
  const blobs = [];
  const stack = [];
  let next = 1;
  for (let i = 0; i < w * h; i++) {
    if (!b[i] || labels[i]) continue;
    let area = 0, sx = 0, sy = 0;
    let minx = w, maxx = 0, miny = h, maxy = 0;
    stack.length = 0;
    stack.push(i);
    labels[i] = next;
    while (stack.length) {
      const p = stack.pop();
      const px = p % w, py = (p / w) | 0;
      area++; sx += px; sy += py;
      if (px < minx) minx = px;
      if (px > maxx) maxx = px;
      if (py < miny) miny = py;
      if (py > maxy) maxy = py;
      if (px > 0 && b[p - 1] && !labels[p - 1]) { labels[p - 1] = next; stack.push(p - 1); }
      if (px < w - 1 && b[p + 1] && !labels[p + 1]) { labels[p + 1] = next; stack.push(p + 1); }
      if (py > 0 && b[p - w] && !labels[p - w]) { labels[p - w] = next; stack.push(p - w); }
      if (py < h - 1 && b[p + w] && !labels[p + w]) { labels[p + w] = next; stack.push(p + w); }
    }
    if (area >= minArea) blobs.push({ x: sx / area, y: sy / area, area, minx, maxx, miny, maxy });
    if (++next > 4000) break;
  }
  return blobs;
}

function angleDiff(a, b) {
  let d = a - b;
  while (d > Math.PI) d -= 2 * Math.PI;
  while (d < -Math.PI) d += 2 * Math.PI;
  return d;
}

function resetTrack() {
  track.state = 'search';
  track.frames = 0;
  track.conf = 0;
  track.areaEma = 0;
  track.speedEma = 0;
  track.turnEma = 0;
  track.heading = null;
  track.snapshotTaken = false;
  track.startX = 0;
  track.startY = 0;
  track.maxDisp = 0;
  track.stale = 0;
}

function updateTrack(assoc, acquire, now, demoBlob) {
  if (track.state === 'locked' || track.state === 'coast') {
    const px = track.x + track.vx, py = track.y + track.vy;
    let best = null, bestD = 45;
    if (demoBlob) {
      best = demoBlob; // demo mode: the lock always rides the demo bug
    } else {
      for (const b of assoc) {
        const dd = Math.hypot(b.x - px, b.y - py);
        if (dd < bestD) { bestD = dd; best = b; }
      }
    }
    if (best) {
      const dx = best.x - track.x, dy = best.y - track.y;
      const speed = Math.hypot(dx, dy);
      track.speedEma += 0.2 * (speed - track.speedEma);
      if (speed > 0.4) {
        const h = Math.atan2(dy, dx);
        if (track.heading !== null) {
          track.turnEma += 0.2 * (Math.abs(angleDiff(h, track.heading)) - track.turnEma);
        }
        track.heading = h;
      }
      track.areaEma += 0.25 * (best.area - track.areaEma);
      track.vx = 0.6 * track.vx + 0.4 * dx;
      track.vy = 0.6 * track.vy + 0.4 * dy;
      track.x = best.x;
      track.y = best.y;
      track.miss = 0;
      track.frames++;
      track.state = 'locked';
      track.maxDisp = Math.max(track.maxDisp, Math.hypot(best.x - track.startX, best.y - track.startY));
      // A real flyer keeps wandering; a facial feature flickers in one spot.
      // If a lock never travels, count it stale and let it expire so a moving
      // target elsewhere (e.g. the demo bug) can win the next search.
      track.stale = speed < 0.5 ? track.stale + 1 : 0;
      if (track.stale > 50 && track.maxDisp < 6) { resetTrack(); return; }
    } else {
      track.miss++;
      track.vx *= 0.94;
      track.vy *= 0.94;
      track.x = Math.min(PROC_W - 1, Math.max(0, px));
      track.y = Math.min(PROC_H - 1, Math.max(0, py));
      if (track.miss > 18) resetTrack();
      else track.state = 'coast';
    }
  }

  const pool = demoBlob ? [demoBlob] : acquire;
  if (track.state === 'search' && pool.length) {
    let best = pool[0];
    for (const b of pool) if (b.area > best.area) best = b;
    resetTrack();
    track.x = track.sx = track.startX = best.x;
    track.y = track.sy = track.startY = best.y;
    track.vx = track.vy = 0;
    track.areaEma = best.area;
    track.frames = 1;
    track.state = 'locked';
  }

  // Heuristic mosquito confidence: insect-sized blob + insect-like flight
  // (moving, erratic heading) sustained over time. Saturates only after
  // ~2-3 s of consistent evidence; large objects never get here at all.
  if (track.state === 'locked') {
    const a = track.areaEma;
    const sizeScore = clamp01((a - 1.5) / 2) * clamp01((70 - a) / 25);
    const s = track.speedEma;
    const speedScore = clamp01((s - 0.1) / 0.4) * clamp01((14 - s) / 4);
    const erraticScore = clamp01(track.turnEma / 0.1);
    const persist = clamp01(track.frames / 45);
    const travelScore = clamp01((track.maxDisp - 5) / 15); // must wander, not flicker in place
    const instant = Math.min(1, 0.5 * sizeScore + 0.3 * speedScore + 0.3 * erraticScore)
      * persist * travelScore;
    track.conf += 0.12 * (instant - track.conf);
  } else {
    track.conf *= 0.95;
  }

  if (!manual && track.state === 'locked' && track.conf >= CONF_SNAPSHOT
      && !track.snapshotTaken && now - lastSnapshotT > SNAPSHOT_COOLDOWN_MS) {
    track.snapshotTaken = true;
    lastSnapshotT = now;
    takeSnapshot(track.conf);
  }

  // smoothed display position
  track.sx += 0.45 * (track.x - track.sx);
  track.sy += 0.45 * (track.y - track.sy);
}

function clamp01(v) {
  return Math.min(1, Math.max(0, v));
}

// ---------- snapshots ----------

function takeSnapshot(conf) {
  const sw = Math.min(1280, video.videoWidth);
  const sh = Math.round(sw * video.videoHeight / video.videoWidth);
  const c = document.createElement('canvas');
  c.width = sw;
  c.height = sh;
  const cc = c.getContext('2d');
  cc.drawImage(video, 0, 0, sw, sh);

  // the demo bug is synthetic, so paint it into the snapshot where the video can't show it
  if (demo.on) {
    cc.fillStyle = 'rgb(25,25,28)';
    cc.beginPath();
    cc.arc(demo.x / PROC_W * sw, demo.y / PROC_H * sh, 6, 0, Math.PI * 2);
    cc.fill();
  }

  const x = track.x / PROC_W * sw, y = track.y / PROC_H * sh;
  cc.strokeStyle = '#3dff7a';
  cc.lineWidth = 3;
  cc.beginPath();
  cc.arc(x, y, 28, 0, Math.PI * 2);
  cc.stroke();
  cc.beginPath();
  cc.moveTo(x - 44, y); cc.lineTo(x - 32, y);
  cc.moveTo(x + 32, y); cc.lineTo(x + 44, y);
  cc.moveTo(x, y - 44); cc.lineTo(x, y - 32);
  cc.moveTo(x, y + 32); cc.lineTo(x, y + 44);
  cc.stroke();

  const stamp = new Date();
  const label = `mosquito ${(conf * 100).toFixed(1)}% · ${stamp.toLocaleString()}`;
  cc.font = '16px -apple-system, sans-serif';
  const tw = cc.measureText(label).width;
  cc.fillStyle = 'rgba(0,0,0,0.65)';
  cc.fillRect(10, sh - 36, tw + 16, 26);
  cc.fillStyle = '#3dff7a';
  cc.fillText(label, 18, sh - 18);

  const url = c.toDataURL('image/jpeg', 0.85);
  const shot = document.createElement('a');
  shot.className = 'shot';
  shot.href = url;
  shot.download = `mosquito-${stamp.toISOString().replace(/[:.]/g, '-')}.jpg`;
  const img = new Image();
  img.src = url;
  shot.appendChild(img);
  const cap = document.createElement('span');
  cap.textContent = label;
  shot.appendChild(cap);
  ui.gallery.prepend(shot);
  ui.gallerySection.hidden = false;
  while (ui.gallery.children.length > 24) ui.gallery.lastChild.remove();
}

// ---------- demo bug ----------

function stepDemo(dt) {
  demo.t += dt;
  demo.vx += (Math.random() - 0.5) * 140 * dt;
  demo.vy += (Math.random() - 0.5) * 140 * dt;
  const sp = Math.hypot(demo.vx, demo.vy);
  const maxSp = 50, minSp = 12;
  if (sp > maxSp) { demo.vx *= maxSp / sp; demo.vy *= maxSp / sp; }
  else if (sp > 0 && sp < minSp) { demo.vx *= minSp / sp; demo.vy *= minSp / sp; }
  demo.x += demo.vx * dt + Math.sin(demo.t * 9) * 0.4;
  demo.y += demo.vy * dt + Math.cos(demo.t * 7) * 0.4;
  const m = 6;
  if (demo.x < m) { demo.x = m; demo.vx = Math.abs(demo.vx); }
  if (demo.x > PROC_W - m) { demo.x = PROC_W - m; demo.vx = -Math.abs(demo.vx); }
  if (demo.y < m) { demo.y = m; demo.vy = Math.abs(demo.vy); }
  if (demo.y > PROC_H - m) { demo.y = PROC_H - m; demo.vy = -Math.abs(demo.vy); }

  pctx.fillStyle = 'rgb(28,28,30)';
  pctx.beginPath();
  pctx.arc(demo.x, demo.y, 1.5, 0, Math.PI * 2);
  pctx.fill();
}

function drawDemoBug(kx, ky) {
  const x = demo.x * kx, y = demo.y * ky;
  octx.fillStyle = 'rgba(15,15,18,0.92)';
  octx.beginPath();
  octx.ellipse(x, y, 5, 3.4, Math.atan2(demo.vy, demo.vx), 0, Math.PI * 2);
  octx.fill();
  octx.strokeStyle = 'rgba(40,40,46,0.8)';
  octx.lineWidth = 1.5;
  octx.beginPath();
  octx.moveTo(x - 3, y); octx.lineTo(x - 9, y - 6);
  octx.moveTo(x + 3, y); octx.lineTo(x + 9, y - 6);
  octx.stroke();
}

// ---------- rendering ----------

function fitOverlay() {
  const w = video.clientWidth, h = video.clientHeight;
  if (w && (overlay.width !== w || overlay.height !== h)) {
    overlay.width = w;
    overlay.height = h;
  }
}

function emitterPoint(W, H) {
  switch (ui.emitter.value) {
    case 'bottom-left': return { x: 10, y: H - 10 };
    case 'bottom-right': return { x: W - 10, y: H - 10 };
    case 'top-left': return { x: 10, y: 10 };
    case 'top-right': return { x: W - 10, y: 10 };
    default: return { x: W / 2, y: H - 10 };
  }
}

function drawBeam(t, now) {
  const W = overlay.width, H = overlay.height;
  const e = emitterPoint(W, H);
  const preset = BEAM_PRESETS[ui.beamStyle.value];
  const [r, g, b] = preset.color;

  const dx = t.x - e.x, dy = t.y - e.y;
  const dist = Math.hypot(dx, dy) || 1;
  const ux = dx / dist, uy = dy / dist;
  const px = -uy, py = ux;

  const div = (Number(ui.divergence.value) / 100) * 0.28; // beam half-spread per px of throw
  const w0 = preset.originW;
  const wT = w0 + div * dist;

  octx.save();
  octx.globalCompositeOperation = 'lighter';

  // cone
  const grad = octx.createLinearGradient(e.x, e.y, t.x, t.y);
  grad.addColorStop(0, `rgba(${r},${g},${b},0.40)`);
  grad.addColorStop(1, `rgba(${r},${g},${b},0.07)`);
  octx.beginPath();
  octx.moveTo(e.x + px * w0 / 2, e.y + py * w0 / 2);
  octx.lineTo(t.x + px * wT / 2, t.y + py * wT / 2);
  octx.lineTo(t.x - px * wT / 2, t.y - py * wT / 2);
  octx.lineTo(e.x - px * w0 / 2, e.y - py * w0 / 2);
  octx.closePath();
  octx.fillStyle = grad;
  octx.fill();

  // bright core for the laser style
  if (preset.core) {
    octx.strokeStyle = `rgba(${r},${g},${b},0.9)`;
    octx.lineWidth = 1.5;
    octx.shadowColor = `rgb(${r},${g},${b})`;
    octx.shadowBlur = 8;
    octx.beginPath();
    octx.moveTo(e.x, e.y);
    octx.lineTo(t.x, t.y);
    octx.stroke();
    octx.shadowBlur = 0;
  }

  // spot where the beam lands
  const spotR = Math.max(wT * 0.6, 10);
  const rad = octx.createRadialGradient(t.x, t.y, 0, t.x, t.y, spotR);
  rad.addColorStop(0, `rgba(${r},${g},${b},0.5)`);
  rad.addColorStop(1, `rgba(${r},${g},${b},0)`);
  octx.fillStyle = rad;
  octx.beginPath();
  octx.arc(t.x, t.y, spotR, 0, Math.PI * 2);
  octx.fill();
  octx.restore();

  // pulsing targeting ring around the bug
  const ringR = Math.max(spotR * 1.15, 16) + 2.5 * Math.sin(now / 180);
  octx.strokeStyle = `rgba(${r},${g},${b},0.95)`;
  octx.lineWidth = 2;
  octx.beginPath();
  octx.arc(t.x, t.y, ringR, 0, Math.PI * 2);
  octx.stroke();
}

function renderDebug(candidates, big) {
  const img = dctx.createImageData(PROC_W, PROC_H);
  for (let i = 0; i < bin.length; i++) {
    const v = bin[i] ? 255 : gray[i] * 0.25;
    // skin (excluded face/hand regions) tinted blue
    img.data[i * 4] = skin[i] ? v * 0.3 : v;
    img.data[i * 4 + 1] = skin[i] ? v * 0.5 : v;
    img.data[i * 4 + 2] = skin[i] ? Math.max(70, v) : v;
    img.data[i * 4 + 3] = 255;
  }
  dctx.putImageData(img, 0, 0);
  dctx.strokeStyle = '#ff5d5d';
  for (const g of big) {
    dctx.strokeRect(g.minx, g.miny, g.maxx - g.minx + 1, g.maxy - g.miny + 1);
  }
  dctx.strokeStyle = '#3dff7a';
  for (const bl of candidates) {
    dctx.strokeRect(bl.x - 3.5, bl.y - 3.5, 7, 7);
  }
}

function render(now) {
  fitOverlay();
  const W = overlay.width, H = overlay.height;
  octx.clearRect(0, 0, W, H);
  const kx = W / PROC_W, ky = H / PROC_H;

  if (demo.on) drawDemoBug(kx, ky);

  let target = null, status = 'searching…', cls = 'search';
  if (manual) {
    target = { x: manual.x * kx, y: manual.y * ky };
    status = 'manual target'; cls = 'manual';
  } else if (track.state === 'locked') {
    target = { x: track.sx * kx, y: track.sy * ky };
    status = 'locked'; cls = 'locked';
  } else if (track.state === 'coast') {
    target = { x: track.sx * kx, y: track.sy * ky };
    status = 'coasting'; cls = 'coast';
  } else if (sceneBusy) {
    status = 'large object — ignoring'; cls = 'coast';
  }

  if (target) drawBeam(target, now);
  ui.status.textContent = status;
  ui.status.className = `status ${cls}`;
  ui.conf.textContent = (!manual && (track.state === 'locked' || track.state === 'coast'))
    ? `bug confidence ${(track.conf * 100).toFixed(0)}%`
    : '';
}

// ---------- simulated room (demo on load) ----------

// A simple drawn room: wall, floor, window with a moon, a framed picture and a
// lamp. Purely decorative — gives the flying bug + laser something to fly over.
function drawRoom(W, H) {
  const floorY = H * 0.68;

  const wall = octx.createLinearGradient(0, 0, 0, floorY);
  wall.addColorStop(0, '#202a3a');
  wall.addColorStop(1, '#192232');
  octx.fillStyle = wall;
  octx.fillRect(0, 0, W, floorY);

  const floor = octx.createLinearGradient(0, floorY, 0, H);
  floor.addColorStop(0, '#33271a');
  floor.addColorStop(1, '#211810');
  octx.fillStyle = floor;
  octx.fillRect(0, floorY, W, H - floorY);

  octx.fillStyle = 'rgba(0,0,0,0.25)';
  octx.fillRect(0, floorY - 4, W, 5); // baseboard shadow

  // window with night sky + moon
  const wx = W * 0.1, wy = H * 0.16, ww = W * 0.26, wh = H * 0.34;
  const sky = octx.createLinearGradient(0, wy, 0, wy + wh);
  sky.addColorStop(0, '#0d1830');
  sky.addColorStop(1, '#16233f');
  octx.fillStyle = sky;
  octx.fillRect(wx, wy, ww, wh);
  octx.fillStyle = 'rgba(245,240,210,0.9)';
  octx.beginPath();
  octx.arc(wx + ww * 0.72, wy + wh * 0.3, Math.min(ww, wh) * 0.16, 0, Math.PI * 2);
  octx.fill();
  octx.strokeStyle = '#2c3850';
  octx.lineWidth = Math.max(3, W * 0.005);
  octx.strokeRect(wx, wy, ww, wh);
  octx.beginPath();
  octx.moveTo(wx + ww / 2, wy); octx.lineTo(wx + ww / 2, wy + wh);
  octx.moveTo(wx, wy + wh / 2); octx.lineTo(wx + ww, wy + wh / 2);
  octx.stroke();

  // framed picture on the right
  const px = W * 0.66, py = H * 0.2, pw = W * 0.2, ph = H * 0.26;
  octx.fillStyle = '#11161f';
  octx.fillRect(px, py, pw, ph);
  octx.strokeStyle = '#3a455a'; // matches --border-hi
  octx.lineWidth = Math.max(3, W * 0.006);
  octx.strokeRect(px, py, pw, ph);
  octx.fillStyle = 'rgba(61,255,122,0.18)';
  octx.beginPath();
  octx.moveTo(px + pw * 0.5, py + ph * 0.28);
  octx.lineTo(px + pw * 0.78, py + ph * 0.72);
  octx.lineTo(px + pw * 0.22, py + ph * 0.72);
  octx.closePath();
  octx.fill();

  // floor lamp, bottom-right
  const lx = W * 0.86;
  octx.strokeStyle = '#2c3548';
  octx.lineWidth = Math.max(3, W * 0.006);
  octx.beginPath();
  octx.moveTo(lx, floorY); octx.lineTo(lx, H * 0.46);
  octx.stroke();
  const glow = octx.createRadialGradient(lx, H * 0.42, 2, lx, H * 0.42, W * 0.09);
  glow.addColorStop(0, 'rgba(255,224,160,0.55)');
  glow.addColorStop(1, 'rgba(255,224,160,0)');
  octx.fillStyle = glow;
  octx.beginPath();
  octx.arc(lx, H * 0.42, W * 0.09, 0, Math.PI * 2);
  octx.fill();
  octx.fillStyle = '#d9b877';
  octx.beginPath();
  octx.moveTo(lx - W * 0.035, H * 0.46);
  octx.lineTo(lx + W * 0.035, H * 0.46);
  octx.lineTo(lx + W * 0.022, H * 0.4);
  octx.lineTo(lx - W * 0.022, H * 0.4);
  octx.closePath();
  octx.fill();
}

function drawSimBug(x, y, ang) {
  octx.save();
  octx.translate(x, y);
  octx.rotate(ang);
  // wings
  octx.fillStyle = 'rgba(180,210,255,0.35)';
  octx.beginPath(); octx.ellipse(-2, -5, 6, 3, -0.5, 0, Math.PI * 2); octx.fill();
  octx.beginPath(); octx.ellipse(-2, 5, 6, 3, 0.5, 0, Math.PI * 2); octx.fill();
  // body
  octx.fillStyle = '#0e0e12';
  octx.beginPath(); octx.ellipse(0, 0, 6, 2.6, 0, 0, Math.PI * 2); octx.fill();
  // legs
  octx.strokeStyle = 'rgba(20,20,24,0.8)';
  octx.lineWidth = 1;
  octx.beginPath();
  octx.moveTo(-1, 1); octx.lineTo(-6, 6);
  octx.moveTo(1, 1); octx.lineTo(6, 6);
  octx.moveTo(0, 1); octx.lineTo(0, 7);
  octx.stroke();
  octx.restore();
}

function renderSim(now, dt) {
  fitOverlay();
  const W = overlay.width, H = overlay.height;
  if (!W || !H) return;

  if (!sim.started) { sim.x = W * 0.5; sim.y = H * 0.4; sim.tx = sim.x; sim.ty = sim.y; sim.started = true; }

  // erratic, mosquito-like flight bounded to the room
  sim.t += dt;
  sim.vx += (Math.random() - 0.5) * 420 * dt;
  sim.vy += (Math.random() - 0.5) * 420 * dt;
  const sp = Math.hypot(sim.vx, sim.vy);
  const maxSp = Math.min(W, H) * 0.6, minSp = Math.min(W, H) * 0.15;
  if (sp > maxSp) { sim.vx *= maxSp / sp; sim.vy *= maxSp / sp; }
  else if (sp > 0 && sp < minSp) { sim.vx *= minSp / sp; sim.vy *= minSp / sp; }
  sim.x += sim.vx * dt + Math.sin(sim.t * 9) * 0.8;
  sim.y += sim.vy * dt + Math.cos(sim.t * 7) * 0.8;
  const m = Math.max(20, W * 0.05);
  if (sim.x < m) { sim.x = m; sim.vx = Math.abs(sim.vx); }
  if (sim.x > W - m) { sim.x = W - m; sim.vx = -Math.abs(sim.vx); }
  if (sim.y < m) { sim.y = m; sim.vy = Math.abs(sim.vy); }
  if (sim.y > H - m) { sim.y = H - m; sim.vy = -Math.abs(sim.vy); }

  // the laser lags slightly toward the bug, like a real tracker catching up
  sim.tx += 0.16 * (sim.x - sim.tx);
  sim.ty += 0.16 * (sim.y - sim.ty);

  octx.clearRect(0, 0, W, H);
  drawRoom(W, H);
  drawSimBug(sim.x, sim.y, Math.atan2(sim.vy, sim.vx));
  drawBeam({ x: sim.tx, y: sim.ty }, now);

  ui.status.textContent = 'demo';
  ui.status.className = 'status locked';
  ui.conf.textContent = 'simulated room';
  ui.resInfo.textContent = '';
}

// ---------- main loop ----------

let lastT = performance.now();
let fpsEma = 0;
let frameCount = 0;

function tick(now) {
  const dt = Math.min(0.1, (now - lastT) / 1000);
  lastT = now;
  if (video.readyState >= 2 && video.videoWidth) {
    processFrame(dt, now);
    render(now);
  } else if (sim.active) {
    renderSim(now, dt);
  }
  if (dt > 0) fpsEma = fpsEma ? fpsEma * 0.92 + (1 / dt) * 0.08 : 1 / dt;
  ui.fps.textContent = `${fpsEma.toFixed(0)} fps`;

  // After a short warm-up, flag a sustained low frame rate (hysteresis avoids flicker).
  if (++frameCount > 120) {
    if (fpsEma < 12 && !warn.lowFps) { warn.lowFps = true; refreshWarning(); }
    else if (fpsEma > 18 && warn.lowFps) { warn.lowFps = false; refreshWarning(); }
  }
  requestAnimationFrame(tick);
}

// ---------- events ----------

function setManualFromEvent(clientX, clientY) {
  const rect = overlay.getBoundingClientRect();
  manual = {
    x: (clientX - rect.left) / rect.width * PROC_W,
    y: (clientY - rect.top) / rect.height * PROC_H,
  };
}

overlay.addEventListener('click', (e) => setManualFromEvent(e.clientX, e.clientY));
overlay.addEventListener('dblclick', () => { manual = null; });

// Touch: a tap sets a manual target without waiting on the 300ms click delay.
overlay.addEventListener('touchstart', (e) => {
  if (e.touches.length === 1) {
    const t = e.touches[0];
    setManualFromEvent(t.clientX, t.clientY);
    e.preventDefault(); // suppress the synthetic click / page scroll
  }
}, { passive: false });

ui.clearTarget.addEventListener('click', () => { manual = null; });

ui.controlsToggle.addEventListener('click', () => {
  const collapsed = ui.dock.classList.toggle('collapsed');
  ui.controlsToggle.setAttribute('aria-expanded', String(!collapsed));
});

// Start collapsed on small screens so the docked bar doesn't eat the viewport.
if (matchMedia('(max-width: 760px)').matches) {
  ui.dock.classList.add('collapsed');
  ui.controlsToggle.setAttribute('aria-expanded', 'false');
}

ui.camera.addEventListener('change', () => startCamera(ui.camera.value).catch(showError));

ui.beamStyle.addEventListener('change', () => {
  ui.divergence.value = BEAM_PRESETS[ui.beamStyle.value].divDefault;
});

ui.demoBtn.addEventListener('click', () => {
  demo.on = !demo.on;
  demo.x = PROC_W / 2;
  demo.y = PROC_H / 2;
  ui.demoBtn.classList.toggle('active', demo.on);
});

ui.debugBtn.addEventListener('click', () => {
  debugOn = !debugOn;
  debugCv.style.display = debugOn ? 'block' : 'none';
  ui.debugBtn.classList.toggle('active', debugOn);
});

ui.resetBtn.addEventListener('click', () => { bg = null; resetTrack(); });

ui.clearShots.addEventListener('click', () => {
  ui.gallery.innerHTML = '';
  ui.gallerySection.hidden = true;
});

video.addEventListener('playing', () => {
  sim.active = false; // real frames flowing — leave the simulated room
  startLoop();
});

function showError(err) {
  console.error(err);
  ui.status.textContent = `camera error: ${err.name || err.message}`;
  ui.status.className = 'status coast';
  // fall back to the simulated demo so the page is never blank
  sim.active = true;
  ui.cameraBtn.disabled = false;
  ui.cameraBtn.classList.remove('live');
  ui.cameraBtn.textContent = '📷 Use my camera';
}

let loopRunning = false;
function startLoop() {
  if (loopRunning) return;
  loopRunning = true;
  requestAnimationFrame(tick);
}

async function boot(deviceId) {
  ui.cameraBtn.disabled = true;
  ui.cameraBtn.textContent = 'Starting camera…';
  ui.status.textContent = 'starting…';
  ui.status.className = 'status search';
  try {
    await startCamera(deviceId);
    sim.active = false;
    ui.cameraBtn.textContent = '📷 Camera on';
    ui.cameraBtn.classList.add('live');
    ui.cameraBtn.disabled = false;
  } catch (err) {
    showError(err);
  }
}

ui.cameraBtn.addEventListener('click', () => boot(ui.camera.value || undefined));

// Start in the simulated-room demo; the user opts into their camera with the button.
startLoop();
