"""
Yawn Detection Model Trainer
-----------------------------
Labels  : 0 = not yawning,  1 = yawning
Dataset : YAWDD frames_cache  (balanced: 13,100 each class)
          Folder name ending in 'Yawning' -> label 1, everything else -> label 0

Output  : models/yawn_model.pth

Run:
    python train_yawn.py
"""

import time, json, random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.model_selection import train_test_split
from PIL import Image
from tqdm import tqdm

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
YAWDD_DIR   = BASE_DIR / "YAWDD" / "frames_cache"
MODELS_DIR  = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

SAVE_PATH = MODELS_DIR / "yawn_model.pth"

# ── Hyperparameters ───────────────────────────────────────────────────────────
IMAGE_SIZE     = 224
BATCH_SIZE     = 32
EPOCHS         = 20
LR             = 1e-4
WEIGHT_DECAY   = 1e-5
NUM_WORKERS    = 0
SEED           = 42

# Early stopping: stop if val_acc hasn't improved by MIN_DELTA in PATIENCE epochs
PATIENCE       = 4
MIN_DELTA      = 1e-4

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# ── Transforms ────────────────────────────────────────────────────────────────
TRAIN_TF = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

VAL_TF = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Dataset ───────────────────────────────────────────────────────────────────
class YawnDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples   = samples    # list of (Path, label)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


def collect(folder, label):
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Folder not found: {folder}")
    return [(p, label) for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS]


def build_datasets():
    yawning     = []
    not_yawning = []

    for folder in sorted(YAWDD_DIR.iterdir()):
        if not folder.is_dir():
            continue
        imgs = collect(folder, label=0)   # label assigned below
        if "Yawning" in folder.name:
            yawning.extend([(p, 1) for p, _ in imgs])
        else:
            not_yawning.extend([(p, 0) for p, _ in imgs])

    all_samples = yawning + not_yawning
    random.shuffle(all_samples)

    # Split: 70% train / 15% val / 15% test  (subject-agnostic random split)
    train_val, test_samples = train_test_split(all_samples, test_size=0.15, random_state=SEED)
    train_samples, val_samples = train_test_split(train_val, test_size=0.1765, random_state=SEED)
    # 0.1765 of 0.85 = 0.15 of total -> gives 70/15/15

    y_tr  = sum(1 for _, l in train_samples if l == 1)
    y_val = sum(1 for _, l in val_samples   if l == 1)
    y_te  = sum(1 for _, l in test_samples  if l == 1)

    print(f"[YAWDD]  train={len(train_samples)} (yawn={y_tr})  "
          f"val={len(val_samples)} (yawn={y_val})  "
          f"test={len(test_samples)} (yawn={y_te})")

    return (
        YawnDataset(train_samples, TRAIN_TF),
        YawnDataset(val_samples,   VAL_TF),
        YawnDataset(test_samples,  VAL_TF),
    )


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model():
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1
    model   = models.mobilenet_v2(weights=weights)
    in_feat = model.classifier[1].in_features   # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_feat, 2),                  # 2 classes: not-yawning / yawning
    )
    return model


# ── Training helpers ──────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = correct = total = 0
    for imgs, labels in tqdm(loader, leave=False, desc="  train"):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    for imgs, labels in tqdm(loader, leave=False, desc="  eval "):
        imgs, labels = imgs.to(device), labels.to(device)
        out  = model(imgs)
        loss = criterion(out, labels)
        total_loss += loss.item() * imgs.size(0)
        correct    += (out.argmax(1) == labels).sum().item()
        total      += imgs.size(0)
    return total_loss / total, correct / total


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    torch.manual_seed(SEED)
    random.seed(SEED)

    print(f"Device : {DEVICE}")
    print(f"Epochs : {EPOCHS}  |  Batch : {BATCH_SIZE}  |  LR : {LR}\n")

    train_ds, val_ds, test_ds = build_datasets()

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    device    = torch.device(DEVICE)
    model     = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss()

    best_val_acc   = 0.0
    epochs_no_imp  = 0
    stopped_epoch  = EPOCHS
    history        = []

    print(f"\n{'='*55}")
    print(f"  Training Yawn Detection Model  (0=not yawning, 1=yawning)")
    print(f"  Early stop: patience={PATIENCE}, min_delta={MIN_DELTA}")
    print(f"{'='*55}")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc   = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = eval_epoch(model,  val_loader,   criterion, device)
        scheduler.step()

        history.append({
            "epoch":    epoch,
            "tr_loss":  round(tr_loss,  4),
            "tr_acc":   round(tr_acc,   4),
            "val_loss": round(val_loss, 4),
            "val_acc":  round(val_acc,  4),
        })

        saved = ""
        if val_acc > best_val_acc + MIN_DELTA:
            best_val_acc  = val_acc
            epochs_no_imp = 0
            torch.save(model.state_dict(), SAVE_PATH)
            saved = "  <- saved"
        else:
            epochs_no_imp += 1

        print(f"  Epoch {epoch:3d}/{EPOCHS} | "
              f"tr  loss={tr_loss:.4f}  acc={tr_acc:.4f} | "
              f"val loss={val_loss:.4f}  acc={val_acc:.4f} | "
              f"{time.time()-t0:.1f}s | no_imp={epochs_no_imp}/{PATIENCE}{saved}")

        if epochs_no_imp >= PATIENCE:
            stopped_epoch = epoch
            print(f"\n  Early stopping triggered at epoch {epoch} "
                  f"(no improvement in {PATIENCE} epochs).")
            break

    model.load_state_dict(torch.load(SAVE_PATH, map_location=device, weights_only=True))
    _, test_acc = eval_epoch(model, test_loader, criterion, device)

    print(f"\n  Best val accuracy : {best_val_acc:.4f}")
    print(f"  Test accuracy     : {test_acc:.4f}")
    print(f"  Stopped at epoch  : {stopped_epoch}/{EPOCHS}")
    print(f"  Model saved to    : {SAVE_PATH}")

    hist_path = MODELS_DIR / "yawn_history.json"
    with open(hist_path, "w") as f:
        json.dump({
            "history":       history,
            "best_val_acc":  best_val_acc,
            "test_acc":      round(test_acc, 4),
            "stopped_epoch": stopped_epoch,
            "max_epochs":    EPOCHS,
            "patience":      PATIENCE,
        }, f, indent=2)
    print(f"  History saved to  : {hist_path}")


if __name__ == "__main__":
    main()
