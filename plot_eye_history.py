"""
Plot the eye model training history saved by train_eye.py.

Reads:  models/eye_history.json
Writes: models/eye_history.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt


BASE_DIR  = Path(__file__).parent
HIST_PATH = BASE_DIR / "models" / "eye_history.json"
OUT_PATH  = BASE_DIR / "models" / "eye_history.png"


def main():
    data    = json.loads(HIST_PATH.read_text())
    history = data["history"]
    best    = data.get("best_val_acc")
    test    = data.get("test_acc")

    epochs   = [r["epoch"]    for r in history]
    tr_loss  = [r["tr_loss"]  for r in history]
    val_loss = [r["val_loss"] for r in history]
    tr_acc   = [r["tr_acc"]   for r in history]
    val_acc  = [r["val_acc"]  for r in history]

    best_epoch = max(history, key=lambda r: r["val_acc"])["epoch"]

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Loss
    ax_loss.plot(epochs, tr_loss,  "o-", label="train loss",      color="#2563eb")
    ax_loss.plot(epochs, val_loss, "s-", label="validation loss", color="#ef4444")
    ax_loss.axvline(best_epoch, ls="--", color="gray", alpha=0.5,
                    label=f"best epoch ({best_epoch})")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Cross-entropy loss")
    ax_loss.set_title("Eye model — Loss")
    ax_loss.grid(alpha=0.3)
    ax_loss.legend()

    # Accuracy
    ax_acc.plot(epochs, tr_acc,  "o-", label="train acc",      color="#2563eb")
    ax_acc.plot(epochs, val_acc, "s-", label="validation acc", color="#22c55e")
    ax_acc.axvline(best_epoch, ls="--", color="gray", alpha=0.5,
                   label=f"best epoch ({best_epoch})")
    if test is not None:
        ax_acc.axhline(test, ls=":", color="#7c3aed", alpha=0.7,
                       label=f"test acc = {test:.4f}")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_title("Eye model — Accuracy")
    ax_acc.set_ylim(min(min(tr_acc), min(val_acc)) - 0.005, 1.001)
    ax_acc.grid(alpha=0.3)
    ax_acc.legend(loc="lower right")

    suptitle = "Eye Status Model (MobileNetV2 on MRL Eye)"
    if best is not None:
        suptitle += f"   best val acc = {best:.4f}"
    if test is not None:
        suptitle += f"   test acc = {test:.4f}"
    fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved chart to {OUT_PATH}")
    plt.show()


if __name__ == "__main__":
    main()
