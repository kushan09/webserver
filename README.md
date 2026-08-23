# Drowsiness Detection — Eye & Yawn Classifiers + Live Dashboard

Three MobileNetV2 classifiers (eye-state, yawn, full-face drowsiness) plus a Flask
dashboard that runs the eye-state model on a live webcam feed, using MediaPipe to
crop the left and right eye regions before classification.

## Project layout

```
.
├── app.py                  # Flask dashboard (loads models/eye_model.pth)
├── templates/index.html    # Dashboard UI (webcam → /predict → bars + bounding boxes)
├── face_landmarker.task    # MediaPipe FaceLandmarker model (used by the dashboard)
│
├── train_eye.py            # Train eye-state model       → models/eye_model.pth
├── train_yawn.py           # Train yawn detector         → models/yawn_model.pth
├── train_drowsy.py         # Train full-face drowsiness  → models/drowsy_model.pth
│
├── MRL Eye/                # Eye dataset (train/val/test × awake/sleepy)
├── YAWDD/frames_cache/     # Yawn dataset (subject folders, name ends in Yawning|Talking|Normal)
├── NTHU-DDD/               # Drowsy/notdrowsy face dataset
│
├── models/                 # Trained weights + training history JSONs
├── requirements.txt
└── README.md
```

## Requirements

- macOS or Linux (training uses CUDA on Linux, MPS on Apple Silicon, CPU otherwise)
- Python **3.12** (3.11 also works; 3.13/3.14 currently lack stable torch wheels)
- ~5 GB free disk for the datasets

## Setup

```bash
cd "/path/to/New folder"
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

The dashboard needs `face_landmarker.task` in the project root. If it's missing,
download it:

```bash
curl -fsSL -o face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

## Training

Each script loads its dataset from the project folder, picks the best device
(`cuda` → `mps` → `cpu`), and writes the best-val-acc checkpoint plus a history JSON
to `models/`.

### Eye-state model (`train_eye.py`)

- Dataset: **MRL Eye** (predefined train/val/test, ~84k images, balanced).
- Labels: `0=awake`, `1=sleepy`.
- 20 epochs, Adam (lr=1e-4, wd=1e-5), CosineAnnealingLR.
- Output: `models/eye_model.pth`, `models/eye_history.json`.

```bash
python train_eye.py
```

### Yawn detector (`train_yawn.py`)

- Dataset: **YAWDD frames_cache**. Folder name ending in `Yawning` → label 1,
  everything else → label 0. 70/15/15 random split.
- 20 epochs max, **early stopping** (patience=4, min_delta=1e-4 on val_acc).
- Output: `models/yawn_model.pth`, `models/yawn_history.json` (includes
  `stopped_epoch` so you can see whether training ran to completion).

```bash
python train_yawn.py
```

Tune early stopping at the top of the file:

```python
PATIENCE  = 4
MIN_DELTA = 1e-4
```

### Full-face drowsiness model (`train_drowsy.py`)

- Dataset: **NTHU-DDD** (`drowsy/` vs `notdrowsy/`).
- Output: `models/drowsy_model.pth`.

```bash
python train_drowsy.py
```

## Running the dashboard

The dashboard runs the **eye-state** model on live webcam frames. It needs
`models/eye_model.pth` to exist.

```bash
source .venv/bin/activate
python app.py
```

Then open **http://127.0.0.1:5050** in Chrome or Safari and click **Start camera**.

> **Why port 5050?** macOS Control Center uses port 5000 for AirPlay Receiver and
> returns a 403 to browsers. To free up port 5000, turn off
> *System Settings → General → AirDrop & Handoff → AirPlay Receiver* and change
> the port back to 5000 in `app.py`.

### What the dashboard does

Per frame (every ~500 ms):

1. Browser captures a 640×480 webcam frame and POSTs it as base64 JPEG to `/predict`.
2. Server detects face landmarks with MediaPipe FaceLandmarker.
3. Bounding boxes for the left and right eyes are computed from canonical landmark
   indices (with 40% padding) and the eye crops are run through MobileNetV2.
4. The response contains per-eye label + probabilities + bounding box; the overall
   status is *sleepy* if **either** eye is classified sleepy.
5. The UI shows the overall status, two per-eye cards, probability bars, and draws
   green/red rectangles around each eye on the live video.

### API

`POST /predict`

```json
{ "image": "data:image/jpeg;base64,..." }
```

Response (face detected):

```json
{
  "face": true,
  "overall": "awake",
  "left":  { "label": "awake",  "confidence": 0.97,
             "probs": {"awake": 0.97, "sleepy": 0.03},
             "bbox": [x0, y0, x1, y1] },
  "right": { "label": "awake",  "confidence": 0.94, "probs": {...}, "bbox": [...] }
}
```

Response (no face):

```json
{ "face": false, "overall": "no face", "left": null, "right": null }
```

## Datasets

| Dataset    | Used by             | Notes |
|------------|---------------------|-------|
| MRL Eye    | `train_eye.py`      | Predefined `train/val/test × awake/sleepy` |
| YAWDD      | `train_yawn.py`     | Subject folders; name contains `Yawning` ⇒ label 1 |
| NTHU-DDD   | `train_drowsy.py`   | `drowsy/` vs `notdrowsy/` |

Datasets are not included here. See `MRL Eye/readme.md` for source links to MRL Eye.

## Notes

- `torch.load(..., weights_only=True)` is used everywhere checkpoints are loaded,
  so a malicious checkpoint can't execute code on load.
- Apple Silicon (M-series) uses the **MPS** backend automatically. Training on MPS
  is several times slower than CUDA on a discrete GPU but much faster than CPU.
- The eye model is trained on **cropped eye images**. Pointing a webcam at a full
  face only works because the dashboard crops the eyes server-side via MediaPipe.
