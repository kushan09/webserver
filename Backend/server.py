import io
import os
import base64
import asyncio
import json
import threading
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image, ImageDraw
from flask import Flask, jsonify, request, Response
from flask_socketio import SocketIO

import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions

# ---------- Paths ----------
BASE_DIR = Path(__file__).parent.parent
EYE_MODEL_PATH = BASE_DIR / "models" / "eye_model.pth"
FM_PATH = BASE_DIR / "face_landmarker.task"

# ---------- Constants ----------
IMAGE_SIZE = 224
EYE_LABELS = ["awake", "sleepy"]

LEFT_EYE_IDX = [33, 133, 159, 145, 158, 153, 160, 144, 161, 163, 246, 7]
RIGHT_EYE_IDX = [362, 263, 386, 374, 385, 380, 387, 373, 388, 390, 466, 249]
MOUTH_IDX = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
             291, 409, 270, 269, 267, 0, 37, 39, 40, 185]

MAR_VERTICAL_PAIRS = [(13, 14), (82, 87), (312, 317)]
MAR_HORIZONTAL = (78, 308)
MAR_CLOSED = 0.05
MAR_OPEN = 0.55

EYE_PAD_FRAC = 0.4
EYE_WEIGHT = 0.7
YAWN_WEIGHT = 0.3
DROWSY_THRESHOLD = 0.5

# ---------- Face-validity gate ----------
# A reading is only allowed to reach the ESP32 (and therefore the buzzer) when
# the face was actually captured well enough to trust. Every rejection carries
# a reason code, and the measured numbers behind it are reported alongside, so
# the gate can be tuned from real data rather than by guesswork.
#
# Defaults are set for a 320x240 ESP32-CAM feed, which is small and JPEG-noisy
# compared to a webcam — a stricter gate rejects a genuinely present face.
# Every value can be overridden without editing this file, e.g.
#     FACE_MIN_WIDTH_FRAC=0.10 python Backend/server.py


def _tune(env_name, default):
    """Read a tuning value from the environment, falling back to the default."""
    return type(default)(os.getenv(env_name, default))


FACE_DETECTION_CONFIDENCE = _tune("FACE_DET_CONF", 0.4)
MIN_FACE_WIDTH_FRAC = _tune("FACE_MIN_WIDTH_FRAC", 0.12)
MIN_BRIGHTNESS = _tune("FACE_MIN_BRIGHTNESS", 30.0)   # mean luma of the FACE region
MAX_BRIGHTNESS = _tune("FACE_MAX_BRIGHTNESS", 235.0)
MAX_CLIP_FRAC = _tune("FACE_MAX_CLIP_FRAC", 0.25)   # of a crop may fall outside frame
STABLE_FRAMES = _tune("FACE_STABLE_FRAMES", 3)      # good frames needed to lock on
STATS_EVERY = _tune("FACE_STATS_EVERY", 50)         # frames per reason-histogram print

# ---------- Device ----------
if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"
device = torch.device(DEVICE)

# ---------- Load eye model ----------
def _build_eye_model():
    m = models.efficientnet_b0(weights=None)
    in_feat = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_feat, 2),
    )
    return m


print(f"Loading eye model on {DEVICE} ...")
eye_model = _build_eye_model()
eye_model.load_state_dict(torch.load(
    EYE_MODEL_PATH, map_location=device, weights_only=True))
eye_model.to(device).eval()
print(f"Eye model loaded from {EYE_MODEL_PATH}")

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ---------- Load face landmarker ----------
print("Loading face landmarker ...")
_lm_opts = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(FM_PATH)),
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=FACE_DETECTION_CONFIDENCE,
)
landmarker = vision.FaceLandmarker.create_from_options(_lm_opts)
print("Face landmarker ready")

# The landmarker (RunningMode.IMAGE) and the torch model are shared between the
# Flask HTTP worker threads and the ESP32 WebSocket thread, and neither is
# thread-safe. Every inference call goes through this lock.
_infer_lock = threading.Lock()


# ---------- Helper functions ----------
def landmark_bbox(landmarks, idx_list, w, h, pad_frac):
    """Square crop box around the given landmarks.

    Returns (bbox, clip_frac). `clip_frac` is how much of the wanted box fell
    outside the frame, as a fraction of its area. A little clipping is normal
    once padding is added and is harmless; a lot means that part of the face is
    genuinely out of shot and its crop is squashed beyond use.
    """
    xs = [landmarks[i].x * w for i in idx_list]
    ys = [landmarks[i].y * h for i in idx_list]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    side = max(x1 - x0, y1 - y0) * (1 + pad_frac)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    raw = (cx - side / 2, cy - side / 2, cx + side / 2, cy + side / 2)
    bbox = (
        int(max(0, raw[0])),
        int(max(0, raw[1])),
        int(min(w, raw[2])),
        int(min(h, raw[3])),
    )
    kept = max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])
    wanted = side * side
    clip_frac = 1.0 - (kept / wanted) if wanted > 0 else 1.0
    return bbox, clip_frac


def face_metrics(landmarks, arr, w, h, clip_fracs):
    """The raw numbers the gate decides on — reported so tuning is data-driven.

    Brightness is measured over the face region, not the whole frame. In a car
    the two often disagree badly: a driver backlit by a bright windscreen gives
    a high frame average while the face itself is too dark to read.
    """
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    fx0, fx1 = max(0, int(min(xs))), min(w, int(max(xs)))
    fy0, fy1 = max(0, int(min(ys))), min(h, int(max(ys)))
    region = arr[fy0:fy1, fx0:fx1]
    brightness = float(region.mean()) if region.size else float(arr.mean())
    return {
        "faceWidthFrac": round((fx1 - fx0) / w, 3),
        "brightness": round(brightness, 1),
        "clipFrac": round(max(clip_fracs), 3),
    }


def face_quality(m):
    """Reason code for whether this face is captured well enough to trust."""
    if m["faceWidthFrac"] < MIN_FACE_WIDTH_FRAC:
        return "too_far"
    if m["clipFrac"] > MAX_CLIP_FRAC:
        return "partial"
    if m["brightness"] < MIN_BRIGHTNESS:
        return "too_dark"
    if m["brightness"] > MAX_BRIGHTNESS:
        return "too_bright"
    return "ok"


def classify_eye(crop: Image.Image):
    tensor = transform(crop).unsqueeze(0).to(device)
    with _infer_lock, torch.no_grad():
        probs = F.softmax(eye_model(tensor), dim=1)[0].tolist()
    return {"awake": round(probs[0], 4), "sleepy": round(probs[1], 4)}


def mouth_aspect_ratio(landmarks):
    left, right = MAR_HORIZONTAL
    width = abs(landmarks[right].x - landmarks[left].x)
    if width < 1e-6:
        return 0.0
    vert = sum(
        abs(landmarks[lo].y - landmarks[up].y)
        for up, lo in MAR_VERTICAL_PAIRS
    ) / len(MAR_VERTICAL_PAIRS)
    return vert / width


def yawn_prob_from_mar(mar):
    if mar <= MAR_CLOSED:
        return 0.0
    if mar >= MAR_OPEN:
        return 1.0
    return (mar - MAR_CLOSED) / (MAR_OPEN - MAR_CLOSED)


def draw_detections(img, left_bbox, right_bbox, mouth_bbox,
                    left_probs, right_probs, yawn_prob, score):
    out = img.copy()
    draw = ImageDraw.Draw(out)
    left_sleepy = left_probs["sleepy"] >= 0.5
    right_sleepy = right_probs["sleepy"] >= 0.5
    yawning = yawn_prob >= 0.5
    drowsy = score > DROWSY_THRESHOLD

    for bbox, sleepy, label in [
        (left_bbox,  left_sleepy,  f"L {left_probs['sleepy']*100:.0f}%"),
        (right_bbox, right_sleepy, f"R {right_probs['sleepy']*100:.0f}%"),
    ]:
        color = "#ef4444" if sleepy else "#22c55e"
        draw.rectangle(bbox, outline=color, width=3)
        draw.rectangle([bbox[0], bbox[1]-18, bbox[0]+60, bbox[1]], fill=color)
        draw.text((bbox[0]+3, bbox[1]-16), label, fill="white")

    m_color = "#ef4444" if yawning else "#22c55e"
    m_label = f"YAWN {yawn_prob*100:.0f}%" if yawning else f"OK {(1-yawn_prob)*100:.0f}%"
    draw.rectangle(mouth_bbox, outline=m_color, width=3)
    draw.rectangle([mouth_bbox[0], mouth_bbox[1]-18,
                   mouth_bbox[0]+80, mouth_bbox[1]], fill=m_color)
    draw.text((mouth_bbox[0]+3, mouth_bbox[1]-16), m_label, fill="white")

    banner_color = "#ef4444" if drowsy else "#22c55e"
    status_text = f"DROWSY  score={score:.2f}" if drowsy else f"ALERT   score={score:.2f}"
    draw.rectangle([0, 0, out.width, 26], fill=banner_color)
    draw.text((8, 5), status_text, fill="white")
    return out


def _to_b64_jpeg(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


_reason_stats = Counter()


def _record_reason(reason, metrics):
    """Periodically print what the gate has been rejecting, and on what numbers.

    Without this, tuning the thresholds is guesswork — this says exactly which
    check is firing and how far off the measurement was.
    """
    _reason_stats[reason] += 1
    total = sum(_reason_stats.values())
    if total % STATS_EVERY:
        return
    breakdown = "  ".join(f"{k}={v}" for k, v in _reason_stats.most_common())
    detail = ""
    if metrics:
        detail = (f"   last: width={metrics['faceWidthFrac']} "
                  f"(min {MIN_FACE_WIDTH_FRAC})  bright={metrics['brightness']} "
                  f"(range {MIN_BRIGHTNESS}-{MAX_BRIGHTNESS})  "
                  f"clip={metrics['clipFrac']} (max {MAX_CLIP_FRAC})")
    # flush: stdout is block-buffered when the server is piped to a log file,
    # which would otherwise hide this exact diagnostic for minutes at a time.
    print(f"[gate] {total} frames: {breakdown}{detail}", flush=True)


def _no_reading(img: Image.Image, reason: str, metrics=None):
    """Fail-safe payload: the face was not captured well enough to judge.

    Every field that could raise an alarm is zeroed, so a consumer that ignores
    faceValid still degrades to "no alarm" rather than to a false alarm.
    """
    out = img.copy()
    draw = ImageDraw.Draw(out)
    draw.rectangle([0, 0, out.width, 26], fill="#f59e0b")
    draw.text((8, 5), f"NO READING  ({reason})", fill="white")
    return {
        "face": reason != "no_face",
        "faceValid": 0,
        "reason": reason,
        "drowsyLevel": 0,
        "confidence": 0.0,
        "eyeStatus": 0,
        "yawningStatus": 0,
        "metrics": metrics or {},
        "annotated_b64": _to_b64_jpeg(out),
    }


def _decay(state):
    """One step back toward unlocked, rather than all the way to zero."""
    state["good_streak"] = max(state.get("good_streak", 0) - 1, 0)


def _drowsy_level_from_score(score):
    """Bucket the continuous drowsiness score into a graded 0-4 level.

    Calibrated to land exactly on the Main Hub/backend's own alert-tier
    breakpoints (drowsyLevel >= 2/3/4 in triggerAlerts()/_TIER_THRESHOLDS),
    not an arbitrary scale — a single binary threshold here left the MEDIUM
    and HIGH alert tiers unreachable from camera data alone.
    """
    if score >= 0.90:
        return 4
    if score >= 0.75:
        return 3
    if score >= DROWSY_THRESHOLD:  # 0.5 — same "drowsy" boundary as before
        return 2
    if score >= 0.25:
        return 1
    return 0


def detect_from_image(img: Image.Image, state=None):
    """Run detection, gated on the face actually being captured properly.

    `state` carries the per-connection stability streak; pass the same dict for
    every frame of one camera. Omit it for one-shot calls.
    """
    if state is None:
        state = {}
    w, h = img.size
    arr = np.array(img)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
    with _infer_lock:
        res = landmarker.detect(mp_img)

    if not res.face_landmarks:
        _decay(state)
        _record_reason("no_face", None)
        return _no_reading(img, "no_face")

    lms = res.face_landmarks[0]
    left_bbox, left_clip = landmark_bbox(lms, LEFT_EYE_IDX,  w, h, EYE_PAD_FRAC)
    right_bbox, right_clip = landmark_bbox(lms, RIGHT_EYE_IDX, w, h, EYE_PAD_FRAC)
    mouth_bbox, mouth_clip = landmark_bbox(lms, MOUTH_IDX,     w, h, 0.35)

    metrics = face_metrics(lms, arr, w, h, (left_clip, right_clip, mouth_clip))
    reason = face_quality(metrics)
    if reason != "ok":
        _decay(state)
        _record_reason(reason, metrics)
        return _no_reading(img, reason, metrics)

    # The streak decays by one on a bad frame rather than resetting to zero, so
    # a single dropped detection costs one frame of recovery instead of three.
    # Without that, a camera that flickers even occasionally never locks on at
    # all and sits in "unstable" forever.
    state["good_streak"] = min(state.get("good_streak", 0) + 1, STABLE_FRAMES)
    if state["good_streak"] < STABLE_FRAMES:
        _record_reason("unstable", metrics)
        return _no_reading(img, "unstable", metrics)
    _record_reason("ok", metrics)

    left_probs = classify_eye(img.crop(left_bbox))
    right_probs = classify_eye(img.crop(right_bbox))

    mar = mouth_aspect_ratio(lms)
    yawn_prob = yawn_prob_from_mar(mar)

    eye_sleepy_prob = max(left_probs["sleepy"], right_probs["sleepy"])
    score = EYE_WEIGHT * eye_sleepy_prob + YAWN_WEIGHT * yawn_prob

    annotated = draw_detections(img, left_bbox, right_bbox, mouth_bbox,
                                left_probs, right_probs, yawn_prob, score)

    return {
        "face": True,
        "faceValid": 1,
        "reason": "ok",
        "drowsyLevel": _drowsy_level_from_score(score),
        "confidence": round(score, 4),
        "eyeStatus": 1 if eye_sleepy_prob >= 0.5 else 0,
        "yawningStatus": 1 if yawn_prob >= 0.5 else 0,
        "metrics": metrics,
        "annotated_b64": _to_b64_jpeg(annotated),
    }


# ---------- Flask + Socket.IO ----------
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")


def _esp32_payload(result):
    """The contract the ESP32-CAM firmware parses. Keep both paths identical."""
    return {
        "faceValid":     result["faceValid"],
        "reason":        result["reason"],
        "drowsyLevel":   result["drowsyLevel"],
        "confidence":    result["confidence"],
        "eyeStatus":     result["eyeStatus"],
        "yawningStatus": result["yawningStatus"],
    }


def _preview_payload(result):
    """What the browser preview page renders. Carries the gate's raw
    measurements too, so thresholds can be tuned against a live face."""
    return {
        "frame":   result["annotated_b64"],
        "face":    result["face"],
        "metrics": result.get("metrics", {}),
        "limits":  {
            "minWidth": MIN_FACE_WIDTH_FRAC,
            "minBright": MIN_BRIGHTNESS,
            "maxBright": MAX_BRIGHTNESS,
            "maxClip": MAX_CLIP_FRAC,
        },
        **_esp32_payload(result),
    }


@app.post("/detect")
def detect():
    raw = request.get_data()
    if not raw:
        return jsonify(error="empty body"), 400
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return jsonify(error="invalid image"), 400

    result = detect_from_image(img)
    socketio.emit("frame", _preview_payload(result))
    return jsonify(**_esp32_payload(result))


@app.get("/preview")
def preview():
    return Response(PREVIEW_HTML, mimetype="text/html")


# ---------- Embedded preview page ----------
PREVIEW_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>ESP32-CAM Live Preview</title>
<script src="https://cdn.socket.io/4.7.5/socket.io.min.js"></script>
<style>
  :root {
    --bg:#0f172a; --panel:#1e293b; --text:#e2e8f0;
    --muted:#94a3b8; --alert:#22c55e; --drowsy:#ef4444; --accent:#38bdf8;
  }
  * { box-sizing:border-box; margin:0; padding:0; }
  body {
    background:var(--bg); color:var(--text);
    font-family:-apple-system,system-ui,sans-serif;
    min-height:100vh; display:flex; flex-direction:column;
    align-items:center; padding:24px; gap:16px;
  }
  h1 { font-size:22px; }
  .sub { color:var(--muted); font-size:13px; }
  .conn { font-size:12px; padding:3px 10px; border-radius:999px; font-weight:600; }
  .conn.live       { background:#14532d; color:var(--alert); }
  .conn.connecting { background:#1e3a5f; color:var(--accent); }
  .conn.offline    { background:#450a0a; color:var(--drowsy); }
  .grid {
    display:grid; grid-template-columns:1fr 300px;
    gap:20px; width:100%; max-width:1050px;
  }
  @media(max-width:800px){ .grid{grid-template-columns:1fr;} }
  .card { background:var(--panel); border-radius:12px; padding:16px; }
  #frame { width:100%; border-radius:8px; display:none; background:#000; }
  #no-frame { text-align:center; color:var(--muted); padding:60px 0; font-size:14px; }
  .badge {
    display:inline-block; padding:4px 14px; border-radius:999px;
    font-size:28px; font-weight:700; margin-bottom:10px;
  }
  .badge.alert  { background:#14532d; color:var(--alert); }
  .badge.drowsy { background:#450a0a; color:var(--drowsy); }
  .badge.idle   { background:#1e293b; color:var(--muted); font-size:18px; }
  .row { display:flex; justify-content:space-between; font-size:13px; margin-bottom:4px; }
  .row span:last-child { font-family:monospace; color:var(--accent); }
  .bar-wrap { height:8px; background:#334155; border-radius:4px; overflow:hidden; margin-bottom:12px; }
  .bar-fill { height:100%; transition:width .2s; border-radius:4px; }
  .fill-alert  { background:var(--alert); }
  .fill-drowsy { background:var(--drowsy); }
  .chips { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:12px; }
  .chip { padding:4px 10px; border-radius:6px; font-size:12px; font-weight:600; }
  .chip.on  { background:#450a0a; color:var(--drowsy); }
  .chip.off { background:#14532d; color:var(--alert); }
  .chip.na  { background:#1e293b; color:var(--muted); }
  .meta { font-size:12px; color:var(--muted); line-height:1.8; }
  .meta code { background:#0f172a; padding:1px 6px; border-radius:4px; color:var(--accent); }
</style>
</head>
<body>
  <h1>ESP32-CAM Live Preview</h1>
  <div style="display:flex;align-items:center;gap:10px;">
    <span class="sub">Frames pushed via WebSocket &#8212; zero polling</span>
    <span id="conn-badge" class="conn connecting">Connecting...</span>
  </div>

  <div class="grid">
    <div class="card">
      <img id="frame" alt="frame"/>
      <div id="no-frame">Waiting for first frame from ESP32-CAM&#8230;</div>
    </div>

    <div class="card">
      <div id="badge" class="badge idle">&#8212; IDLE &#8212;</div>

      <div class="row"><span>Face gate</span><span id="reason">&#8212;</span></div>
      <div class="row"><span>Face width</span><span id="m-width">&#8212;</span></div>
      <div class="row"><span>Brightness</span><span id="m-bright">&#8212;</span></div>
      <div class="row"><span>Crop clipped</span><span id="m-clip">&#8212;</span></div>

      <div class="row"><span>Confidence score</span><span id="score">&#8212;</span></div>
      <div class="bar-wrap"><div id="score-bar" class="bar-fill fill-alert" style="width:0%"></div></div>

      <div class="row"><span>Eye status</span><span id="eye">&#8212;</span></div>
      <div class="bar-wrap"><div id="eye-bar" class="bar-fill fill-drowsy" style="width:0%"></div></div>

      <div class="row"><span>Yawn status</span><span id="yawn">&#8212;</span></div>
      <div class="bar-wrap"><div id="yawn-bar" class="bar-fill fill-drowsy" style="width:0%"></div></div>

      <div class="chips">
        <div id="chip-drowsy" class="chip na">DROWSY</div>
        <div id="chip-eye"    class="chip na">EYES CLOSED</div>
        <div id="chip-yawn"   class="chip na">YAWNING</div>
        <div id="chip-face"   class="chip na">NO READING</div>
      </div>

      <div class="meta">
        Frames received: <code id="count">0</code><br>
        Last update: <code id="ts">&#8212;</code>
      </div>
    </div>
  </div>

<script>
let count = 0;
const connBadge = document.getElementById('conn-badge');
const socket = io();

socket.on('connect',      () => { connBadge.className = 'conn live';    connBadge.textContent = 'Live'; });
socket.on('disconnect',   () => { connBadge.className = 'conn offline'; connBadge.textContent = 'Offline'; });
socket.on('connect_error',() => { connBadge.className = 'conn offline'; connBadge.textContent = 'Error'; });

socket.on('frame', (d) => {
  if (d.frame) {
    const frameEl = document.getElementById('frame');
    frameEl.src = 'data:image/jpeg;base64,' + d.frame;
    frameEl.style.display = 'block';
    document.getElementById('no-frame').style.display = 'none';
    count++;
    document.getElementById('count').textContent = count;
    document.getElementById('ts').textContent = new Date().toLocaleTimeString();
  }

  const score   = d.confidence ?? 0;
  const drowsy  = d.drowsyLevel === 1;
  const eyeOn   = d.eyeStatus === 1;
  const yawnOn  = d.yawningStatus === 1;
  const valid   = d.faceValid === 1;

  document.getElementById('reason').textContent = d.reason ?? '—';

  // Show each measurement against the threshold it is judged on, and colour it
  // red when that is the check currently failing — makes tuning obvious.
  const m = d.metrics ?? {}, lim = d.limits ?? {};
  function metric(id, value, ok, suffix) {
    const el = document.getElementById(id);
    if (value === undefined) { el.textContent = '—'; el.style.color = ''; return; }
    el.textContent = value + suffix;
    el.style.color = ok ? '' : 'var(--drowsy)';
  }
  metric('m-width',  m.faceWidthFrac, m.faceWidthFrac >= lim.minWidth,
         ` / min ${lim.minWidth ?? '—'}`);
  metric('m-bright', m.brightness,
         m.brightness >= lim.minBright && m.brightness <= lim.maxBright,
         ` / ${lim.minBright ?? '—'}–${lim.maxBright ?? '—'}`);
  metric('m-clip',   m.clipFrac, m.clipFrac <= lim.maxClip,
         ` / max ${lim.maxClip ?? '—'}`);

  // Nothing downstream of the gate means anything until faceValid is 1 — the
  // panel mirrors what the ESP32 is allowed to act on, not the raw model output.
  const badge = document.getElementById('badge');
  if (!valid)        { badge.className = 'badge idle';   badge.textContent = 'NO READING'; }
  else if (drowsy)   { badge.className = 'badge drowsy'; badge.textContent = 'DROWSY'; }
  else               { badge.className = 'badge alert';  badge.textContent = 'ALERT'; }

  document.getElementById('score').textContent         = score.toFixed(3);
  document.getElementById('score-bar').style.width     = (score * 100).toFixed(1) + '%';
  document.getElementById('score-bar').className       = 'bar-fill ' + (drowsy ? 'fill-drowsy' : 'fill-alert');
  document.getElementById('eye').textContent           = eyeOn  ? 'CLOSED'  : 'OPEN';
  document.getElementById('yawn').textContent          = yawnOn ? 'YAWNING' : 'NORMAL';
  document.getElementById('eye-bar').style.width       = eyeOn  ? '100%' : '15%';
  document.getElementById('yawn-bar').style.width      = yawnOn ? '100%' : '10%';

  function chip(id, active) {
    document.getElementById(id).className = 'chip ' + (active === null ? 'na' : (active ? 'on' : 'off'));
  }
  chip('chip-drowsy', valid ? drowsy : null);
  chip('chip-eye',    valid ? eyeOn  : null);
  chip('chip-yawn',   valid ? yawnOn : null);
  chip('chip-face',   valid ? false  : true);
});
</script>
</body>
</html>
"""

# ---------- WebSocket server for ESP32-CAM (port 5002) ----------
# The ESP32 sends raw JPEG binary frames; we run detection and reply
# with a JSON text message that the Arduino WebSocketsClient parses.

async def _esp32_ws_handler(websocket):
    print(f"[WS] ESP32 connected from {websocket.remote_address}")
    # Per-connection stability streak. Keeping this out of module scope stops
    # two cameras (or the /detect route) from corrupting each other's counter.
    state = {"good_streak": 0}
    loop = asyncio.get_running_loop()
    try:
        async for message in websocket:
            if not isinstance(message, bytes):
                continue
            try:
                img = Image.open(io.BytesIO(message)).convert("RGB")
            except Exception:
                await websocket.send(json.dumps({"error": "bad image"}))
                continue

            # Inference runs off the event loop so ping/pong keeps flowing and
            # the ESP32 is not disconnected mid-frame.
            result = await loop.run_in_executor(
                None, detect_from_image, img, state)

            # Reply to ESP32 with compact JSON
            await websocket.send(json.dumps(_esp32_payload(result)))

            # Push annotated frame to browser preview
            socketio.emit("frame", _preview_payload(result))
    except Exception as e:
        print(f"[WS] Connection closed: {e}")


def _start_ws_thread():
    import websockets

    async def _serve():
        async with websockets.serve(_esp32_ws_handler, "0.0.0.0", 5002):
            print("[WS] ESP32 WebSocket server listening on ws://0.0.0.0:5002")
            await asyncio.get_running_loop().create_future()  # run forever

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(_serve())


if __name__ == "__main__":
    # Start WebSocket server for ESP32 in a background thread
    ws_thread = threading.Thread(target=_start_ws_thread, daemon=True)
    ws_thread.start()

    print(f"\nStarting servers (device={DEVICE})")
    print("ESP32 WebSocket : ws://0.0.0.0:5002")
    print("HTTP POST       : http://0.0.0.0:5001/detect  (alternative)")
    print("Preview         : http://localhost:5001/preview\n")
    socketio.run(app, host="0.0.0.0", port=5001, debug=False, allow_unsafe_werkzeug=True)
