"""Run testing process"""

import json
from datetime import datetime

import torch
import torch.distributed as dist
from torchmetrics.classification import BinaryJaccardIndex, MulticlassJaccardIndex
from tqdm import tqdm


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


def test_dist(modelDir, model, test_dataloader, lossFunc, rank, num_classes: int = 1):
    device = f"cuda:{rank}"
    model.eval()

    test_loss = 0.0
    test_iou  = _make_iou(num_classes, device)

    loader = (
        tqdm(test_dataloader, desc="[TEST]", leave=False, ncols=80)
        if rank == 0 else test_dataloader
    )

    with torch.no_grad():
        for images, masks, _ in loader:
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device, non_blocking=True)
            logits = model(images)
            test_loss += lossFunc(logits, masks).item()
            p, t = _preds_targets(logits, masks, num_classes)
            test_iou.update(p, t)

    avg_test_loss = test_loss / len(test_dataloader)
    avg_test_miou = test_iou.compute().item()

    # Reduce across ranks
    for name, val in [("loss", avg_test_loss), ("miou", avg_test_miou)]:
        t = torch.tensor(val, device=device)
        dist.all_reduce(t, op=dist.ReduceOp.AVG)
        if name == "loss":
            avg_test_loss = t.item()
        else:
            avg_test_miou = t.item()

    if rank == 0:
        result = {
            "test_loss": round(avg_test_loss, 6),
            "test_miou": round(avg_test_miou, 6),
            "tested_at": datetime.now().isoformat(timespec="seconds"),
        }
        with open(modelDir + "metrics/test_metrics.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"Test | loss={avg_test_loss:.4f}  mIoU={avg_test_miou:.4f}")

    return avg_test_loss if rank == 0 else None
