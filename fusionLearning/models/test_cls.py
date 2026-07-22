"""Run testing process for TOMPEI-CMMD binary classification models."""

import json
import os
from datetime import datetime

import torch
import torch.distributed as dist
from torchmetrics.classification import BinaryConfusionMatrix
from tqdm import tqdm

from fusionLearning.models.train_cls import _accuracy_from_cm, _cm_to_nested


def test_dist_cls(modelDir, model, test_dataloader, lossFunc, rank):
    device = f"cuda:{rank}"
    model.eval()

    test_loss = 0.0
    test_cm = BinaryConfusionMatrix().to(device)

    loader = (
        tqdm(test_dataloader, desc="[TEST]", leave=False, ncols=80)
        if rank == 0 else test_dataloader
    )

    with torch.no_grad():
        for images, labels, _ in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            test_loss += lossFunc(logits, labels).item()
            preds = torch.sigmoid(logits)
            test_cm.update(preds.reshape(-1), labels.reshape(-1).long())

    avg_test_loss = test_loss / len(test_dataloader)

    # Reduce across ranks: loss averaged, confusion matrix summed
    loss_t = torch.tensor(avg_test_loss, device=device)
    dist.all_reduce(loss_t, op=dist.ReduceOp.AVG)
    avg_test_loss = loss_t.item()

    cm_t = test_cm.compute().to(device)
    dist.all_reduce(cm_t, op=dist.ReduceOp.SUM)

    if rank == 0:
        result = {
            "test_loss": round(avg_test_loss, 6),
            "test_acc": round(_accuracy_from_cm(cm_t), 6),
            "test_cm": _cm_to_nested(cm_t),
            "tested_at": datetime.now().isoformat(timespec="seconds"),
        }
        with open(os.path.join(modelDir, "metrics", "test_metrics.json"), "w") as f:
            json.dump(result, f, indent=2)
        print(f"Test | loss={avg_test_loss:.4f}  acc={result['test_acc']:.4f}  cm={result['test_cm']}")

    return avg_test_loss if rank == 0 else None
