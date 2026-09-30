import json
import os

import matplotlib
import matplotlib.pyplot as plt


def visualize_training_process(metrics):
    """Legacy helper kept for backward compat."""
    plt.figure(figsize=(8, 5))
    plt.plot(metrics['epochs'], metrics['train_loss'], 'bo-', label='Train Loss')
    plt.plot(metrics['epochs'], metrics['val_loss'],   'ro-', label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig("final_training_metrics.png")
    plt.close()


def plot_metrics(modelDir):
    """
    Reads epoch_metrics.json and test_metrics.json from modelDir and saves
    two figures to modelDir/figures/:

      training_curves.png  — 2 subplots:
          top:    train loss + val loss over epochs
          bottom: train mIoU + val mIoU over epochs
                  + test mIoU as a horizontal dashed line (if test_metrics.json exists)
    """
    matplotlib.use("Agg")

    epoch_path = os.path.join(modelDir, "metrics", "epoch_metrics.json")
    test_path  = os.path.join(modelDir, "metrics", "test_metrics.json")

    try:
        with open(epoch_path) as f:
            log = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"No valid metrics found at {epoch_path}; skipping plots.")
        return

    epochs_data = log.get("epochs", log) if isinstance(log, dict) else log
    if not epochs_data:
        return

    epochs     = [d["epoch"]      for d in epochs_data]
    train_loss = [d["train_loss"] for d in epochs_data]
    val_loss   = [d["val_loss"]   for d in epochs_data]
    train_miou = [d["train_miou"] for d in epochs_data]
    val_miou   = [d["val_miou"]   for d in epochs_data]

    # Optional test point
    test_miou = None
    if os.path.exists(test_path):
        try:
            with open(test_path) as f:
                test_data = json.load(f)
            test_miou = test_data.get("test_miou")
        except (json.JSONDecodeError, KeyError):
            pass

    # Best epoch marker
    best_epoch = log.get("best", {}).get("epoch") if isinstance(log, dict) else None

    fig, (ax_loss, ax_miou) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # ── Loss ─────────────────────────────────────────────────────────────────
    ax_loss.plot(epochs, train_loss, color="steelblue", label="Train loss")
    ax_loss.plot(epochs, val_loss,   color="tomato",    label="Val loss")
    if best_epoch is not None:
        ax_loss.axvline(best_epoch, color="gray", linestyle=":", linewidth=1,
                        label=f"Best epoch ({best_epoch})")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(); ax_loss.grid(True, alpha=0.4)

    # ── mIoU ─────────────────────────────────────────────────────────────────
    ax_miou.plot(epochs, train_miou, color="steelblue", label="Train mIoU")
    ax_miou.plot(epochs, val_miou,   color="tomato",    label="Val mIoU")
    if test_miou is not None:
        ax_miou.axhline(test_miou, color="seagreen", linestyle="--", linewidth=1.5,
                        label=f"Test mIoU ({test_miou:.4f})")
    if best_epoch is not None:
        ax_miou.axvline(best_epoch, color="gray", linestyle=":", linewidth=1)
    ax_miou.set_ylabel("mIoU"); ax_miou.set_xlabel("Epoch")
    ax_miou.set_ylim(0, 1)
    ax_miou.legend(); ax_miou.grid(True, alpha=0.4)

    # Meta title if available
    meta = log.get("meta", {}) if isinstance(log, dict) else {}
    title_parts = [meta.get("arch", ""), meta.get("encoder", ""), meta.get("dataset", "")]
    title = "  ·  ".join(p for p in title_parts if p)
    if title:
        fig.suptitle(title, fontsize=11)

    plt.tight_layout()
    out_path = os.path.join(modelDir, "figures", "training_curves.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
