import os
import json
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torchmetrics.classification import BinaryJaccardIndex, MulticlassJaccardIndex
from tqdm import tqdm

from fusionLearning.models.consts import MAXEPOCHS, LEARNING_RATE

DEBUG_TRAIN = False


def _make_iou(num_classes: int, device):
    if num_classes == 1:
        return BinaryJaccardIndex().to(device)
    return MulticlassJaccardIndex(
        num_classes=num_classes, ignore_index=255, average="macro"
    ).to(device)


def _preds_targets(logits, masks, num_classes: int):
    if num_classes == 1:
        return torch.sigmoid(logits).squeeze(1), masks.squeeze(1).long()
    return logits, masks.long()


def _reduce(val: float, device) -> float:
    t = torch.tensor(val, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return t.item()


def train_dist(
    modelDir,
    model,
    optimizer,
    lossFunc,
    training_dataloader: DataLoader,
    validation_dataloader: DataLoader,
    rank: int,
    num_classes: int = 1,
    arch: str = "",
    encoder: str = "",
    dataset: str = "",
):
    device = f"cuda:{rank}"

    if rank == 0:
        for sub in ("weights", "metrics", "figures"):
            os.makedirs(os.path.join(modelDir, sub), exist_ok=True)

        metrics_path = os.path.join(modelDir, "metrics", "epoch_metrics.json")
        log = {
            "meta": {
                "arch": arch,
                "encoder": encoder,
                "dataset": dataset,
                "num_classes": num_classes,
                "total_params": sum(p.numel() for p in model.parameters()),
                "started_at": datetime.now().isoformat(timespec="seconds"),
                "completed_at": None,
            },
            "epochs": [],
            "best": {"epoch": None, "val_miou": None},
        }
        with open(metrics_path, "w") as f:
            json.dump(log, f, indent=2)

    best_val_miou = -1.0

    for epoch in range(1, MAXEPOCHS + 1):
        t0 = time.time()

        # ── Train ─────────────────────────────────────────────────────────────
        training_dataloader.sampler.set_epoch(epoch)
        model.train()
        train_loss = 0.0
        train_iou = _make_iou(num_classes, device)

        loader = (
            tqdm(training_dataloader, desc=f"Epoch {epoch}/{MAXEPOCHS} [Train]",
                 leave=False, ncols=80)
            if rank == 0 else training_dataloader
        )
        for images, masks, _ in loader:
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(images)
            loss   = lossFunc(logits, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            p, t = _preds_targets(logits.detach(), masks, num_classes)
            train_iou.update(p, t)

        avg_train_loss = train_loss / len(training_dataloader)
        avg_train_miou = train_iou.compute().item()

        # ── Val ───────────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_iou = _make_iou(num_classes, device)

        vloader = (
            tqdm(validation_dataloader, desc=f"Epoch {epoch}/{MAXEPOCHS} [Val]",
                 leave=False, ncols=80)
            if rank == 0 else validation_dataloader
        )
        with torch.no_grad():
            for images, masks, _ in vloader:
                images = images.to(device, non_blocking=True)
                masks  = masks.to(device, non_blocking=True)
                logits = model(images)
                val_loss += lossFunc(logits, masks).item()
                p, t = _preds_targets(logits, masks, num_classes)
                val_iou.update(p, t)

        avg_val_loss = val_loss / len(validation_dataloader)
        avg_val_miou = val_iou.compute().item()

        # ── Reduce across ranks ───────────────────────────────────────────────
        avg_train_loss = _reduce(avg_train_loss, device)
        avg_train_miou = _reduce(avg_train_miou, device)
        avg_val_loss   = _reduce(avg_val_loss,   device)
        avg_val_miou   = _reduce(avg_val_miou,   device)

        is_best = avg_val_miou > best_val_miou

        if rank == 0:
            if is_best:
                best_val_miou = avg_val_miou
                torch.save(
                    model.state_dict(),
                    os.path.join(modelDir, "weights", "best_model.pth"),
                )

            record = {
                "epoch":      epoch,
                "timestamp":  datetime.now().isoformat(timespec="seconds"),
                "duration_s": round(time.time() - t0, 1),
                "train_loss": round(avg_train_loss, 6),
                "train_miou": round(avg_train_miou, 6),
                "val_loss":   round(avg_val_loss,   6),
                "val_miou":   round(avg_val_miou,   6),
                "lr":         LEARNING_RATE,
                "best":       is_best,
            }
            with open(metrics_path, "r") as f:
                log = json.load(f)
            log["epochs"].append(record)
            if is_best:
                log["best"] = {"epoch": epoch, "val_miou": round(best_val_miou, 6)}
            with open(metrics_path, "w") as f:
                json.dump(log, f, indent=2)

            print(
                f"Epoch {epoch:2d} | "
                f"Train  loss={avg_train_loss:.4f}  mIoU={avg_train_miou:.4f} | "
                f"Val    loss={avg_val_loss:.4f}  mIoU={avg_val_miou:.4f}"
                + ("  ✓" if is_best else "")
            )

    if rank == 0:
        with open(metrics_path, "r") as f:
            log = json.load(f)
        log["meta"]["completed_at"] = datetime.now().isoformat(timespec="seconds")
        with open(metrics_path, "w") as f:
            json.dump(log, f, indent=2)
        print(f"Training complete. Best val mIoU: {best_val_miou:.4f}  (epoch {log['best']['epoch']})")
