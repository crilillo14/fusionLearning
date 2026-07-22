import os
import json
import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torchmetrics.classification import BinaryConfusionMatrix
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


def _reduce_avg(val: float, device) -> float:
    t = torch.tensor(val, device=device)
    dist.all_reduce(t, op=dist.ReduceOp.AVG)
    return t.item()


def _reduce_sum_cm(cm: torch.Tensor, device) -> torch.Tensor:
    t = cm.to(device)
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t


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
):
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

        scheduler.step()

        avg_train_loss = train_loss / len(training_dataloader)

        # ── Val ───────────────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        val_cm = BinaryConfusionMatrix().to(device)

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

        avg_val_loss = val_loss / len(validation_dataloader)

        # ── Reduce across ranks ───────────────────────────────────────────────
        avg_train_loss = _reduce_avg(avg_train_loss, device)
        avg_val_loss = _reduce_avg(avg_val_loss, device)
        train_cm_t = _reduce_sum_cm(train_cm.compute(), device)
        val_cm_t = _reduce_sum_cm(val_cm.compute(), device)

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
                "train_cm": _cm_to_nested(train_cm_t),
                "val_loss": round(avg_val_loss, 6),
                "val_acc": round(_accuracy_from_cm(val_cm_t), 6),
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
                f"Train  loss={avg_train_loss:.4f}  acc={record['train_acc']:.4f} | "
                f"Val    loss={avg_val_loss:.4f}  acc={record['val_acc']:.4f}"
                + ("  ✓" if is_best else "")
            )

    if rank == 0:
        with open(metrics_path, "r") as f:
            log = json.load(f)
        log["meta"]["completed_at"] = datetime.now().isoformat(timespec="seconds")
        with open(metrics_path, "w") as f:
            json.dump(log, f, indent=2)
        print(f"Training complete. Best val loss: {best_val_loss:.4f}  (epoch {log['best']['epoch']})")
