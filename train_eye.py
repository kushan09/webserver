"""
Eye Status Model Trainer (EfficientNet-B0)
------------------------------------------
Labels  : 0 = awake (open),  1 = sleepy (closed)
Dataset : MRL Eye  (84,898 images, balanced, predefined splits)
          MRL Eye/{train|val|test}/{awake|sleepy}/

Backbone: EfficientNet-B0 (ImageNet pretrained)

Anti-overfit changes vs. the previous MobileNetV2 run
(prev run val_loss bottomed at epoch 7 = 0.0267 then rose to 0.0420 by ep 20):
  - Stronger dropout (0.4) in the classifier head
  - Heavier augmentation (RandAugment + RandomErasing)
  - Label smoothing (0.05) in CrossEntropyLoss
  - Early stopping on val_loss (patience=4) so we keep the best generalizer,
    not the most-overfit checkpoint
  - Save on best val_loss, not best val_acc — val_acc is saturated on MRL Eye
    so val_loss is the better signal for generalization

Output : models/eye_model.pth     (state_dict for EfficientNet-B0 head)
         models/eye_history.json

Run:
    python train_eye.py
"""

import time, json, random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm


# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
MRL_DIR    = BASE_DIR / "MRL Eye"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

SAVE_PATH = MODELS_DIR / "eye_model.pth"

# ── Hyperparameters ───────────────────────────────────────────────────────────
IMAGE_SIZE     = 224
BATCH_SIZE     = 32
EPOCHS         = 20
LR             = 1e-4
WEIGHT_DECAY   = 1e-4          # bumped from 1e-5 for more regularization
DROPOUT_P      = 0.4           # bumped from 0.3
LABEL_SMOOTH   = 0.05
EARLY_STOP_PAT = 4             # stop if val_loss doesn't improve for N epochs
NUM_WORKERS    = 0             # keep 0 on Windows / macOS spawn quirks
SEED           = 42

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
    transforms.RandAugment(num_ops=2, magnitude=7),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
])

VAL_TF = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ── Dataset ───────────────────────────────────────────────────────────────────
class EyeDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples   = samples
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
        raise FileNotFoundError(f"Dataset folder not found: {folder}")
    return [(p, label) for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS]


def build_datasets():
    # awake=0 (open), sleepy=1 (closed)
    train_samples = collect(MRL_DIR / "train" / "awake", 0) + collect(MRL_DIR / "train" / "sleepy", 1)
    val_samples   = collect(MRL_DIR / "val"   / "awake", 0) + collect(MRL_DIR / "val"   / "sleepy", 1)
    test_samples  = collect(MRL_DIR / "test"  / "awake", 0) + collect(MRL_DIR / "test"  / "sleepy", 1)

    print(f"[MRL Eye]  train={len(train_samples)}  val={len(val_samples)}  test={len(test_samples)}")

    return (
        EyeDataset(train_samples, TRAIN_TF),
        EyeDataset(val_samples,   VAL_TF),
        EyeDataset(test_samples,  VAL_TF),
    )


# ── Model ─────────────────────────────────────────────────────────────────────
def build_model():
    """EfficientNet-B0 pretrained on ImageNet, 2-class head."""
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
    model   = models.efficientnet_b0(weights=weights)
    in_feat = model.classifier[1].in_features   # 1280
    model.classifier = nn.Sequential(
        nn.Dropout(p=DROPOUT_P, inplace=True),
        nn.Linear(in_feat, 2),                  # 2 classes: awake / sleepy
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
    print(f"Backbone : EfficientNet-B0")
    print(f"Epochs : {EPOCHS}  |  Batch : {BATCH_SIZE}  |  LR : {LR}  |  WD : {WEIGHT_DECAY}")
    print(f"Dropout : {DROPOUT_P}  |  Label smooth : {LABEL_SMOOTH}  |  Early stop patience : {EARLY_STOP_PAT}\n")

    train_ds, val_ds, test_ds = build_datasets()

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=NUM_WORKERS)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    device    = torch.device(DEVICE)
    model     = build_model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

    best_val_loss = float("inf")
    best_val_acc  = 0.0
    epochs_no_improve = 0
    history       = []

    print(f"\n{'='*60}")
    print(f"  Training Eye Status Model — EfficientNet-B0  (0=awake, 1=sleepy)")
    print(f"{'='*60}")

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
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc  = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), SAVE_PATH)
            saved = "  ← saved (best val_loss)"
        else:
            epochs_no_improve += 1

        print(f"  Epoch {epoch:3d}/{EPOCHS} | "
              f"tr  loss={tr_loss:.4f}  acc={tr_acc:.4f} | "
              f"val loss={val_loss:.4f}  acc={val_acc:.4f} | "
              f"{time.time()-t0:.1f}s{saved}")

        if epochs_no_improve >= EARLY_STOP_PAT:
            print(f"\n  Early stop: val_loss hasn't improved in {EARLY_STOP_PAT} epochs.")
            break

    # Final test evaluation on the best (lowest val_loss) checkpoint
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
    test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)

    print(f"\n  Best val loss : {best_val_loss:.4f}  (acc at that epoch: {best_val_acc:.4f})")
    print(f"  Test loss     : {test_loss:.4f}")
    print(f"  Test accuracy : {test_acc:.4f}")
    print(f"  Model saved to: {SAVE_PATH}")

    hist_path = MODELS_DIR / "eye_history.json"
    with open(hist_path, "w") as f:
        json.dump({
            "backbone":     "efficientnet_b0",
            "history":      history,
            "best_val_loss": round(best_val_loss, 4),
            "best_val_acc":  round(best_val_acc,  4),
            "test_loss":     round(test_loss,     4),
            "test_acc":      round(test_acc,      4),
        }, f, indent=2)
    print(f"  History saved : {hist_path}")


if __name__ == "__main__":
    main()
