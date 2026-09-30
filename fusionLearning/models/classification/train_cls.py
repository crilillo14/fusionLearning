import math
import os
import json
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torchmetrics.classification import BinaryConfusionMatrix, BinaryAUROC
from tqdm import tqdm

DEBUG_TRAIN_CLS = False


def _cm_to_nested(cm: torch.Tensor) -> list:
    """[[tn, fp], [fn, tp]] as plain ints, for JSON serialization."""
    return [[int(cm[0, 0]), int(cm[0, 1])], [int(cm[1, 0]), int(cm[1, 1])]]


def _accuracy_from_cm(cm: torch.Tensor) -> float:
    total = cm.sum().item()
    if total == 0:
        return 0.0
    correct = (cm[0, 0] + cm[1, 1]).item()
    return correct / total


def _extended_metrics_from_cm(cm) -> dict:
    """
    precision/recall(=sensitivity)/specificity/F1/MCC from a [[tn, fp], [fn, tp]]
    confusion matrix (nested list or tensor) - the metrics your PI's guidelines
    name explicitly (AUC/ACC/sensitivity/specificity/F1) beyond what a plain
    accuracy/AUC pair covers. Shared by test_cls.py (test set) and vis_cls.py
    (train/val plotting) so the definitions can't drift between the two.
    """
    if hasattr(cm, "tolist"):
        cm = cm.tolist()
    (tn, fp), (fn, tp) = cm
    tn, fp, fn, tp = float(tn), float(fp), float(fn), float(tp)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    mcc_denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / mcc_denom if mcc_denom > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "specificity": specificity,
        "f1": f1,
        "mcc": mcc,
    }


def _reduce_avg(val: float, device) -> float:
    t = torch.tensor(val, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return t.item()


def train_dist_cls(
    modelDir,
    model,
    optimizer,
    scheduler,
    lossFunc,
    training_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    rank: int,
    maxepochs: int,
    arch: str = "",
    dataset: str = "TOMPEI-CMMD",
    config: dict | None = None,
):
    """
    `config`, if given, is the full roster_cls.py config dict for this variant
    (variant_id/family/timm_name/depth_tier/resolution_tier/resolution_px/lr/
    batch_size/params_m) - recorded verbatim into meta so each model's full
    hyperparameter configuration lives alongside its results, not only in the
    roster module.
    """
    device = f"cuda:{rank}"

    if rank == 0:
        for sub in ("weights", "metrics", "figures"):
            os.makedirs(os.path.join(modelDir, sub), exist_ok=True)

        metrics_path = os.path.join(modelDir, "metrics", "epoch_metrics.json")
        log = {
            "meta": {
                "arch": arch,
                "dataset": dataset,
                "num_classes": 1,
                "total_params": sum(p.numel() for p in model.parameters()),
                "config": config or {},
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "completed_at": None,
            },
            "epochs": [],
            "best": {"epoch": None, "val_loss": None},
        }
        with open(metrics_path, "w") as f:
            json.dump(log, f, indent=2)

    best_val_loss = float("inf")

    for epoch in range(1, maxepochs + 1):
        t0 = time.time()

        # ── Train ─────────────────────────────────────────────────────────────
        training_dataloader.sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        train_cm = BinaryConfusionMatrix().to(device)
        train_auroc = BinaryAUROC().to(device)

        loader = (
            tqdm(training_dataloader, desc=f"Epoch {epoch}/{maxepochs} [Train]",
                 leave=False, ncols=80)
            if rank == 0 else training_dataloader
        )
        for images, labels, _ in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(images)
            loss = lossFunc(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            preds = torch.sigmoid(logits.detach())
            train_cm.update(preds.reshape(-1), labels.reshape(-1).long())
            train_auroc.update(preds.reshape(-1), labels.reshape(-1).long())

        scheduler.step()

        avg_train_loss = train_loss / len(training_dataloader)

        # ── Val ───────────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_cm = BinaryConfusionMatrix().to(device)
        val_auroc = BinaryAUROC().to(device)

        vloader = (
            tqdm(validation_dataloader, desc=f"Epoch {epoch}/{maxepochs} [Val]",
                 leave=False, ncols=80)
            if rank == 0 else validation_dataloader
        )
        with torch.no_grad():
            for images, labels, _ in vloader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                logits = model(images)
                val_loss += lossFunc(logits, labels).item()
                preds = torch.sigmoid(logits)
                val_cm.update(preds.reshape(-1), labels.reshape(-1).long())
                val_auroc.update(preds.reshape(-1), labels.reshape(-1).long())

        avg_val_loss = val_loss / len(validation_dataloader)

        # ── Reduce across ranks ───────────────────────────────────────────────
        # BinaryConfusionMatrix defaults to sync_on_compute=True, so .compute()
        # already all-gathers and sums state across every DDP rank internally -
        # each rank gets back the same globally-aggregated matrix. Only the plain
        # float losses (not torchmetrics-tracked) need a manual reduction.
        avg_train_loss = _reduce_avg(avg_train_loss, device)
        avg_val_loss = _reduce_avg(avg_val_loss, device)
        train_cm_t = train_cm.compute()
        val_cm_t = val_cm.compute()
        train_auc = train_auroc.compute().item()
        val_auc = val_auroc.compute().item()

        is_best = avg_val_loss < best_val_loss

        if rank == 0:
            if is_best:
                best_val_loss = avg_val_loss
                torch.save(
                    model.state_dict(),
                    os.path.join(modelDir, "weights", "best_model.pth"),
                )

            record = {
                "epoch": epoch,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "duration_s": round(time.time() - t0, 1),
                "train_loss": round(avg_train_loss, 6),
                "train_acc": round(_accuracy_from_cm(train_cm_t), 6),
                "train_auc": round(train_auc, 6),
                "train_cm": _cm_to_nested(train_cm_t),
                "val_loss": round(avg_val_loss, 6),
                "val_acc": round(_accuracy_from_cm(val_cm_t), 6),
                "val_auc": round(val_auc, 6),
                "val_cm": _cm_to_nested(val_cm_t),
                "lr": round(scheduler.get_last_lr()[0], 8),
                "best": is_best,
            }
            with open(metrics_path, "r") as f:
                log = json.load(f)
            log["epochs"].append(record)
            if is_best:
                log["best"] = {"epoch": epoch, "val_loss": round(best_val_loss, 6)}
            with open(metrics_path, "w") as f:
                json.dump(log, f, indent=2)

            print(
                f"Epoch {epoch:2d} | "
                f"Train  loss={avg_train_loss:.4f}  acc={record['train_acc']:.4f}  auc={train_auc:.4f} | "
                f"Val    loss={avg_val_loss:.4f}  acc={record['val_acc']:.4f}  auc={val_auc:.4f}"
                + ("  ✓" if is_best else "")
            )

    if rank == 0:
        with open(metrics_path, "r") as f:
            log = json.load(f)
        log["meta"]["completed_at"] = datetime.now().isoformat(timespec="seconds")
        with open(metrics_path, "w") as f:
            json.dump(log, f, indent=2)
        print(f"Training complete. Best val loss: {best_val_loss:.4f}  (epoch {log['best']['epoch']})")
