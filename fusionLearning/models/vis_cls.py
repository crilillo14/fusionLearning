import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from fusionLearning.models.train_cls import _extended_metrics_from_cm


def _plot_confusion_matrix(ax, cm, title):
    cm = np.array(cm, dtype=np.int64)
    total = cm.sum()
    cm_pct = cm / total if total > 0 else cm.astype(np.float64)

    ax.imshow(cm_pct, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["normal", "lesion"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["normal", "lesion"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(2):
        for j in range(2):
            pct = cm_pct[i, j] * 100
            ax.text(j, i, f"{cm[i, j]}\n({pct:.1f}%)", ha="center", va="center",
                     color="white" if cm_pct[i, j] > 0.5 else "black", fontsize=10)


def plot_metrics_cls(modelDir):
    """
    Reads epoch_metrics.json and test_metrics.json from modelDir and saves two
    figures to modelDir/figures/:

      training_curves_cls.png — 2 subplots:
          top:    train loss + val loss over epochs
          bottom: train acc + val acc over epochs
                  + test acc as a horizontal dashed line (if test_metrics.json exists)

      confusion_matrix.png — heatmaps of the final train/val epoch's confusion
          matrix plus the test confusion matrix (if test_metrics.json exists)
    """
    matplotlib.use("Agg")

    epoch_path = os.path.join(modelDir, "metrics", "epoch_metrics.json")
    test_path = os.path.join(modelDir, "metrics", "test_metrics.json")

    try:
        with open(epoch_path) as f:
            log = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"No valid metrics found at {epoch_path}; skipping plots.")
        return

    epochs_data = log.get("epochs", log) if isinstance(log, dict) else log
    if not epochs_data:
        return

    epochs = [d["epoch"] for d in epochs_data]
    train_loss = [d["train_loss"] for d in epochs_data]
    val_loss = [d["val_loss"] for d in epochs_data]
    train_acc = [d["train_acc"] for d in epochs_data]
    val_acc = [d["val_acc"] for d in epochs_data]
    lr = [d["lr"] for d in epochs_data]

    test_data = None
    if os.path.exists(test_path):
        try:
            with open(test_path) as f:
                test_data = json.load(f)
        except json.JSONDecodeError:
            pass

    best_epoch = log.get("best", {}).get("epoch") if isinstance(log, dict) else None

    fig, (ax_loss, ax_acc) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax_loss.plot(epochs, train_loss, color="steelblue", label="Train loss")
    ax_loss.plot(epochs, val_loss, color="tomato", label="Val loss")
    if best_epoch is not None:
        ax_loss.axvline(best_epoch, color="gray", linestyle=":", linewidth=1,
                         label=f"Best epoch ({best_epoch})")
    ax_loss.set_ylabel("Loss")
    ax_loss.legend(); ax_loss.grid(True, alpha=0.4)

    ax_acc.plot(epochs, train_acc, color="steelblue", label="Train acc")
    ax_acc.plot(epochs, val_acc, color="tomato", label="Val acc")
    if test_data is not None:
        ax_acc.axhline(test_data["test_acc"], color="seagreen", linestyle="--", linewidth=1.5,
                        label=f"Test acc ({test_data['test_acc']:.4f})")
        if "test_auc" in test_data:
            ax_acc.axhline(test_data["test_auc"], color="darkorchid", linestyle=":", linewidth=1.5,
                            label=f"Test AUC ({test_data['test_auc']:.4f})")
    if best_epoch is not None:
        ax_acc.axvline(best_epoch, color="gray", linestyle=":", linewidth=1)
    ax_acc.set_ylabel("Accuracy"); ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylim(0, 1)
    ax_acc.legend(); ax_acc.grid(True, alpha=0.4)

    meta = log.get("meta", {}) if isinstance(log, dict) else {}
    title_parts = [meta.get("arch", ""), meta.get("dataset", ""),
                    f"final lr={lr[-1]:.2e}" if lr else ""]
    title = "  ·  ".join(p for p in title_parts if p)
    if title:
        fig.suptitle(title, fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(modelDir, "figures", "training_curves_cls.png"), dpi=120)
    plt.close(fig)

    # ── Confusion matrices ───────────────────────────────────────────────────
    n_panels = 2 + (1 if test_data is not None else 0)
    fig2, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    last = epochs_data[-1]
    _plot_confusion_matrix(axes[0], last["train_cm"], f"Train CM (epoch {last['epoch']})")
    _plot_confusion_matrix(axes[1], last["val_cm"], f"Val CM (epoch {last['epoch']})")
    if test_data is not None:
        _plot_confusion_matrix(axes[2], test_data["test_cm"], "Test CM")

    plt.tight_layout()
    plt.savefig(os.path.join(modelDir, "figures", "confusion_matrix.png"), dpi=120)
    plt.close(fig2)


def plot_extended_metrics_cls(modelDir):
    """
    Reads epoch_metrics.json and derives precision/recall/F1/MCC per epoch from the
    train_cm/val_cm already saved by train_dist_cls (no re-evaluation needed). Saves
    extended_metrics_cls.png to modelDir/figures/ - 5 stacked subplots (loss, F1,
    precision, recall, MCC), each with train vs val curves over epochs.
    """
    matplotlib.use("Agg")

    epoch_path = os.path.join(modelDir, "metrics", "epoch_metrics.json")
    try:
        with open(epoch_path) as f:
            log = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"No valid metrics found at {epoch_path}; skipping extended metrics plot.")
        return

    epochs_data = log.get("epochs", log) if isinstance(log, dict) else log
    if not epochs_data:
        return

    epochs = [d["epoch"] for d in epochs_data]
    train_loss = [d["train_loss"] for d in epochs_data]
    val_loss = [d["val_loss"] for d in epochs_data]

    train_stats = [_extended_metrics_from_cm(d["train_cm"]) for d in epochs_data]
    val_stats = [_extended_metrics_from_cm(d["val_cm"]) for d in epochs_data]

    best_epoch = log.get("best", {}).get("epoch") if isinstance(log, dict) else None

    panels = [
        ("Loss", train_loss, val_loss, None),
    ]
    # train_auc/val_auc are computed live (BinaryAUROC), not derived from the CM -
    # only present in epoch_metrics.json written after the AUC pivot, so guard for
    # older/archived logs that predate it.
    if all("train_auc" in d and "val_auc" in d for d in epochs_data):
        panels.append(("AUC", [d["train_auc"] for d in epochs_data], [d["val_auc"] for d in epochs_data], (0, 1)))
    panels += [
        ("F1", [s["f1"] for s in train_stats], [s["f1"] for s in val_stats], (0, 1)),
        ("Precision", [s["precision"] for s in train_stats], [s["precision"] for s in val_stats], (0, 1)),
        ("Sensitivity (Recall)", [s["sensitivity"] for s in train_stats], [s["sensitivity"] for s in val_stats], (0, 1)),
        ("Specificity", [s["specificity"] for s in train_stats], [s["specificity"] for s in val_stats], (0, 1)),
        ("MCC", [s["mcc"] for s in train_stats], [s["mcc"] for s in val_stats], (-1, 1)),
    ]

    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 4 * len(panels)), sharex=True)

    for ax, (name, train_vals, val_vals, ylim) in zip(axes, panels):
        ax.plot(epochs, train_vals, color="steelblue", label=f"Train {name}")
        ax.plot(epochs, val_vals, color="tomato", label=f"Val {name}")
        if best_epoch is not None:
            ax.axvline(best_epoch, color="gray", linestyle=":", linewidth=1)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.set_ylabel(name)
        ax.legend(); ax.grid(True, alpha=0.4)

    axes[-1].set_xlabel("Epoch")

    meta = log.get("meta", {}) if isinstance(log, dict) else {}
    title_parts = [meta.get("arch", ""), meta.get("dataset", "")]
    title = "  ·  ".join(p for p in title_parts if p)
    if title:
        fig.suptitle(title, fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(modelDir, "figures", "extended_metrics_cls.png"), dpi=120)
    plt.close(fig)
