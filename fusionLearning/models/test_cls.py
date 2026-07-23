"""Run testing process for TOMPEI-CMMD binary classification models."""

import json
import os
from datetime import datetime

import torch
import torch.distributed as dist
from torchmetrics.classification import BinaryConfusionMatrix, BinaryAUROC
from tqdm import tqdm

from fusionLearning.models.train_cls import _accuracy_from_cm, _cm_to_nested, _extended_metrics_from_cm


def test_dist_cls(modelDir, model, test_dataloader, lossFunc, rank):
    device = f"cuda:{rank}"
    model.eval()

    test_loss = 0.0
    test_cm = BinaryConfusionMatrix().to(device)
    test_auroc = BinaryAUROC().to(device)
    sample_records = []

    loader = (
        tqdm(test_dataloader, desc="[TEST]", leave=False, ncols=80)
        if rank == 0 else test_dataloader
    )

    with torch.no_grad():
        for images, labels, filenames in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(images)
            test_loss += lossFunc(logits, labels).item()
            preds = torch.sigmoid(logits)
            test_cm.update(preds.reshape(-1), labels.reshape(-1).long())
            test_auroc.update(preds.reshape(-1), labels.reshape(-1).long())

            probs = preds.reshape(-1).cpu().tolist()
            trues = labels.reshape(-1).cpu().tolist()
            for filename, prob, true in zip(filenames, probs, trues):
                sample_records.append({
                    "filename": filename,
                    "true_label": int(true),
                    "pred_prob": round(float(prob), 6),
                    "pred_label": int(prob >= 0.5),
                })

    avg_test_loss = test_loss / len(test_dataloader)

    # Reduce across ranks: loss is a plain float, needs a manual all_reduce. The
    # confusion matrix doesn't - BinaryConfusionMatrix defaults to
    # sync_on_compute=True, so .compute() already all-gathers and sums state
    # across every DDP rank internally; summing it again here would double-count.
    loss_t = torch.tensor(avg_test_loss, device=device)
    dist.all_reduce(loss_t, op=dist.ReduceOp.AVG)
    avg_test_loss = loss_t.item()

    cm_t = test_cm.compute()
    test_auc = test_auroc.compute().item()

    # Per-sample records are plain Python objects (not torchmetrics-tracked), so
    # they need an explicit gather. DistributedSampler(shuffle=False, drop_last=False)
    # pads the dataset to be evenly divisible across ranks by repeating a few
    # leading samples, so the gathered lists may contain duplicate filenames -
    # deduped below by filename, keeping the first occurrence (predictions for a
    # duplicated sample are identical across ranks anyway since the model/weights
    # are the same).
    world_size = dist.get_world_size()
    gathered = [None] * world_size
    dist.all_gather_object(gathered, sample_records)

    if rank == 0:
        seen = set()
        all_records = []
        for records in gathered:
            for r in records:
                if r["filename"] in seen:
                    continue
                seen.add(r["filename"])
                all_records.append(r)
        all_records.sort(key=lambda r: r["filename"])

        test_cm_nested = _cm_to_nested(cm_t)
        ext = _extended_metrics_from_cm(test_cm_nested)

        result = {
            "test_loss": round(avg_test_loss, 6),
            "test_acc": round(_accuracy_from_cm(cm_t), 6),
            "test_auc": round(test_auc, 6),
            "test_precision": round(ext["precision"], 6),
            "test_sensitivity": round(ext["sensitivity"], 6),
            "test_specificity": round(ext["specificity"], 6),
            "test_f1": round(ext["f1"], 6),
            "test_mcc": round(ext["mcc"], 6),
            "test_cm": test_cm_nested,
            "tested_at": datetime.now().isoformat(timespec="seconds"),
        }
        with open(os.path.join(modelDir, "metrics", "test_metrics.json"), "w") as f:
            json.dump(result, f, indent=2)

        predictions_out = {
            "n_samples": len(all_records),
            "tested_at": result["tested_at"],
            "predictions": all_records,
        }
        with open(os.path.join(modelDir, "metrics", "test_predictions.json"), "w") as f:
            json.dump(predictions_out, f, indent=2)

        print(f"Test | loss={avg_test_loss:.4f}  acc={result['test_acc']:.4f}  "
              f"auc={test_auc:.4f}  f1={result['test_f1']:.4f}  "
              f"sens={result['test_sensitivity']:.4f}  spec={result['test_specificity']:.4f}  "
              f"cm={result['test_cm']}  (n={len(all_records)} predictions saved)")

    return avg_test_loss if rank == 0 else None
