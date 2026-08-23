"""
Drowsiness Dashboard
--------------------
Flask app on a live webcam feed:

  - Eye model  : EfficientNet-B0, awake (0) / sleepy (1)  -> left + right eye crops
  - Yawn       : geometric MAR (no model) on mouth landmarks

Pipeline per frame:
  1. MediaPipe FaceLandmarker -> 468 landmarks (one face).
  2. Crop left eye, right eye, and mouth bboxes.
  3. Classify each crop.
  4. Combined drowsiness score:
        score = 0.7 * eye_sleepy_prob + 0.3 * yawn_prob
        DROWSY if score > 0.5
     where eye_sleepy_prob = max(P(sleepy | left), P(sleepy | right))

Run:
    python app.py
Then open http://127.0.0.1:5050
"""

import base64
import io
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
from flask import Flask, jsonify, render_template, request
from flask_sock import Sock

import mediapipe as mp
from mediapipe.tasks.python import vision, BaseOptions


BASE_DIR        = Path(__file__).parent
EYE_MODEL_PATH  = BASE_DIR / "models" / "eye_model.pth"
FM_PATH         = BASE_DIR / "face_landmarker.task"

IMAGE_SIZE   = 224
EYE_LABELS   = ["awake", "sleepy"]

# MediaPipe Face Mesh canonical indices
LEFT_EYE_IDX  = [33, 133, 159, 145, 158, 153, 160, 144, 161, 163, 246, 7]
RIGHT_EYE_IDX = [362, 263, 386, 374, 385, 380, 387, 373, 388, 390, 466, 249]

# Outer lip contour (used to draw the mouth box on the overlay)
MOUTH_IDX = [
    61, 146, 91, 181, 84, 17, 314, 405, 321, 375,
    291, 409, 270, 269, 267, 0, 37, 39, 40, 185,
]

# Mouth Aspect Ratio (MAR) landmarks: pairs of upper/lower lip points,
# normalized by horizontal mouth width. Open mouth -> MAR rises.
MAR_VERTICAL_PAIRS = [(13, 14), (82, 87), (312, 317)]  # inner-lip vertical distances
MAR_HORIZONTAL     = (78, 308)                          # inner-lip corners
MAR_CLOSED = 0.05   # below -> definitely closed (p_yawn = 0)
MAR_OPEN   = 0.55   # above -> definitely yawning (p_yawn = 1)

EYE_PAD_FRAC   = 0.4
MOUTH_PAD_FRAC = 0.35

# Verdict
EYE_WEIGHT       = 0.7
YAWN_WEIGHT      = 0.3
DROWSY_THRESHOLD = 0.5

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


# ── Classifier helpers ────────────────────────────────────────────────────────
def build_eye_head():
    # EfficientNet-B0 backbone, 2-class head — must match train_eye.py
    m = models.efficientnet_b0(weights=None)
    in_feat = m.classifier[1].in_features
    m.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_feat, 2),
    )
    return m


def load_eye_classifier(path, device):
    m = build_eye_head()
    m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    m.to(device).eval()
    return m


print(f"Loading eye model on {DEVICE} ...")
device    = torch.device(DEVICE)
eye_model = load_eye_classifier(EYE_MODEL_PATH, device)
print(f"Eye model loaded from {EYE_MODEL_PATH}")
print("Yawn detection: geometric MAR (no model)")

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Face landmarker ───────────────────────────────────────────────────────────
print("Loading face landmarker ...")
_lm_opts = vision.FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=str(FM_PATH)),
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
)
landmarker = vision.FaceLandmarker.create_from_options(_lm_opts)
print("Face landmarker ready")


def mouth_aspect_ratio(landmarks):
    """Average vertical lip-pair distances divided by horizontal mouth width.
    Uses normalized landmark coords (already in [0,1]), so it's resolution-invariant.
    """
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
    """Map MAR into a [0, 1] yawn probability via linear ramp between MAR_CLOSED and MAR_OPEN."""
    if mar <= MAR_CLOSED:
        return 0.0
    if mar >= MAR_OPEN:
        return 1.0
    return (mar - MAR_CLOSED) / (MAR_OPEN - MAR_CLOSED)


def landmark_bbox(landmarks, idx_list, w, h, pad_frac):
    xs = [landmarks[i].x * w for i in idx_list]
    ys = [landmarks[i].y * h for i in idx_list]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    side = max(x1 - x0, y1 - y0) * (1 + pad_frac)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return (
        int(max(0, cx - side / 2)),
        int(max(0, cy - side / 2)),
        int(min(w, cx + side / 2)),
        int(min(h, cy + side / 2)),
    )


def classify(crop: Image.Image, model, labels):
    tensor = transform(crop).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)[0].tolist()
    idx = 0 if probs[0] >= probs[1] else 1
    return {
        "label":      labels[idx],
        "confidence": round(probs[idx], 4),
        "probs":      {labels[i]: round(p, 4) for i, p in enumerate(probs)},
    }


# ── Flask app ─────────────────────────────────────────────────────────────────
app  = Flask(__name__)
sock = Sock(app)

# Latest result received from the ESP32-CAM (shared across requests)
_esp32_latest: dict = {"connected": False}


@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        device=DEVICE,
        eye_weight=EYE_WEIGHT,
        yawn_weight=YAWN_WEIGHT,
        threshold=DROWSY_THRESHOLD,
    )


@app.post("/predict")
def predict():
    data = request.get_json(silent=True) or {}
    image_b64 = data.get("image", "")
    if "," in image_b64:
        image_b64 = image_b64.split(",", 1)[1]
    if not image_b64:
        return jsonify(error="no image"), 400

    img = Image.open(io.BytesIO(base64.b64decode(image_b64))).convert("RGB")
    w, h = img.size
    arr  = np.array(img)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
    res = landmarker.detect(mp_img)

    if not res.face_landmarks:
        return jsonify(
            face=False, overall="no face",
            score=0.0, threshold=DROWSY_THRESHOLD,
            left=None, right=None, mouth=None,
        )

    lms = res.face_landmarks[0]
    left_bbox  = landmark_bbox(lms, LEFT_EYE_IDX,  w, h, EYE_PAD_FRAC)
    right_bbox = landmark_bbox(lms, RIGHT_EYE_IDX, w, h, EYE_PAD_FRAC)
    mouth_bbox = landmark_bbox(lms, MOUTH_IDX,     w, h, MOUTH_PAD_FRAC)

    left_pred  = classify(img.crop(left_bbox),  eye_model, EYE_LABELS)
    right_pred = classify(img.crop(right_bbox), eye_model, EYE_LABELS)

    # Geometric yawn detection via Mouth Aspect Ratio
    mar       = mouth_aspect_ratio(lms)
    yawn_prob = yawn_prob_from_mar(mar)
    mouth_pred = {
        "label":      "yawning" if yawn_prob >= 0.5 else "not_yawning",
        "confidence": round(yawn_prob if yawn_prob >= 0.5 else 1 - yawn_prob, 4),
        "probs":      {"not_yawning": round(1 - yawn_prob, 4),
                       "yawning":     round(yawn_prob, 4)},
        "mar":        round(mar, 4),
    }

    eye_sleepy_prob = max(left_pred["probs"]["sleepy"], right_pred["probs"]["sleepy"])
    score           = round(EYE_WEIGHT * eye_sleepy_prob + YAWN_WEIGHT * yawn_prob, 4)
    overall         = "drowsy" if score > DROWSY_THRESHOLD else "alert"

    return jsonify(
        face      = True,
        overall   = overall,
        score     = score,
        threshold = DROWSY_THRESHOLD,
        left      = {**left_pred,  "bbox": list(left_bbox)},
        right     = {**right_pred, "bbox": list(right_bbox)},
        mouth     = {**mouth_pred, "bbox": list(mouth_bbox)},
    )


@app.post("/esp32/predict")
def esp32_predict():
    """
    ESP32-CAM endpoint — accepts a raw JPEG as multipart/form-data OR
    raw binary body (Content-Type: image/jpeg).

    Returns a compact JSON the ESP32 can parse easily:
    {
      "drowsy":      0 or 1,
      "eye":         0=open 1=closed,
      "yawn":        0 or 1,
      "score":       0.0-1.0,
      "face":        true/false,
      "label":       "alert" | "drowsy" | "no face"
    }
    """
    # Accept multipart field "image" OR raw body
    if request.content_type and "multipart" in request.content_type:
        f = request.files.get("image")
        if f is None:
            return jsonify(error="no image field"), 400
        raw = f.read()
    else:
        raw = request.get_data()
        if not raw:
            return jsonify(error="empty body"), 400

    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return jsonify(error="invalid image"), 400

    w, h = img.size
    arr  = np.array(img)
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
    res = landmarker.detect(mp_img)

    if not res.face_landmarks:
        return jsonify(drowsy=0, eye=0, yawn=0, score=0.0, face=False, label="no face")

    lms = res.face_landmarks[0]
    left_bbox  = landmark_bbox(lms, LEFT_EYE_IDX,  w, h, EYE_PAD_FRAC)
    right_bbox = landmark_bbox(lms, RIGHT_EYE_IDX, w, h, EYE_PAD_FRAC)

    left_pred  = classify(img.crop(left_bbox),  eye_model, EYE_LABELS)
    right_pred = classify(img.crop(right_bbox), eye_model, EYE_LABELS)

    mar        = mouth_aspect_ratio(lms)
    yawn_prob  = yawn_prob_from_mar(mar)

    eye_sleepy_prob = max(left_pred["probs"]["sleepy"], right_pred["probs"]["sleepy"])
    score           = round(EYE_WEIGHT * eye_sleepy_prob + YAWN_WEIGHT * yawn_prob, 4)
    overall         = "drowsy" if score > DROWSY_THRESHOLD else "alert"

    eye_closed = 1 if eye_sleepy_prob >= 0.5 else 0
    yawning    = 1 if yawn_prob >= 0.5 else 0

    return jsonify(
        drowsy = 1 if overall == "drowsy" else 0,
        eye    = eye_closed,
        yawn   = yawning,
        score  = score,
        face   = True,
        label  = overall,
    )


@sock.route("/ws")
def esp32_ws(ws):
    """WebSocket endpoint for ESP32-CAM.
    Receives raw JPEG binary frames, returns compact JSON.
    """
    import json, time
    global _esp32_latest
    _esp32_latest = {"connected": True, "ts": time.time()}
    print("ESP32-CAM WebSocket connected")
    try:
        while True:
            raw = ws.receive()
            if raw is None:
                break
            if isinstance(raw, str):
                raw = raw.encode()

            try:
                img = Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception:
                ws.send(json.dumps({"error": "invalid image"}))
                continue

            w, h = img.size
            arr    = np.array(img)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=arr)
            res    = landmarker.detect(mp_img)

            if not res.face_landmarks:
                result = {"drowsy": 0, "eye": 0, "yawn": 0,
                          "score": 0.0, "face": False, "label": "no face",
                          "drowsyLevel": 0, "confidence": 0.0,
                          "eyeStatus": 0, "yawningStatus": 0,
                          "connected": True, "ts": time.time()}
                _esp32_latest = result
                ws.send(json.dumps(result))
                continue

            lms        = res.face_landmarks[0]
            left_bbox  = landmark_bbox(lms, LEFT_EYE_IDX,  w, h, EYE_PAD_FRAC)
            right_bbox = landmark_bbox(lms, RIGHT_EYE_IDX, w, h, EYE_PAD_FRAC)

            left_pred  = classify(img.crop(left_bbox),  eye_model, EYE_LABELS)
            right_pred = classify(img.crop(right_bbox), eye_model, EYE_LABELS)

            mar       = mouth_aspect_ratio(lms)
            yawn_prob = yawn_prob_from_mar(mar)

            eye_sleepy_prob = max(left_pred["probs"]["sleepy"], right_pred["probs"]["sleepy"])
            score           = round(EYE_WEIGHT * eye_sleepy_prob + YAWN_WEIGHT * yawn_prob, 4)
            overall         = "drowsy" if score > DROWSY_THRESHOLD else "alert"
            eye_closed      = 1 if eye_sleepy_prob >= 0.5 else 0
            yawning         = 1 if yawn_prob >= 0.5 else 0

            result = {
                "drowsy":        1 if overall == "drowsy" else 0,
                "eye":           eye_closed,
                "yawn":          yawning,
                "score":         score,
                "face":          True,
                "label":         overall,
                "drowsyLevel":   1 if overall == "drowsy" else 0,
                "confidence":    score,
                "eyeStatus":     eye_closed,
                "yawningStatus": yawning,
                "connected":     True,
                "ts":            time.time(),
            }
            _esp32_latest = result
            ws.send(json.dumps(result))
    finally:
        _esp32_latest = {"connected": False, "ts": __import__("time").time()}
        print("ESP32-CAM WebSocket disconnected")


@app.get("/esp32/status")
def esp32_status():
    return jsonify(_esp32_latest)


if __name__ == "__main__":
    # 0.0.0.0 so VS Code port forwarding (and local network) can reach it
    app.run(host="0.0.0.0", port=5050, debug=False)
