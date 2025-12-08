"""Run testing process"""

import json

import torch
import torch.distributed as dist
from tqdm import tqdm
from torchmetrics.classification import BinaryAUROC


def test_dist(modelDir, model, test_dataloader, lossFunc, rank):

    device = f"cuda:{rank}"

    model.eval()
    tloss = 0.0

    loader = tqdm(test_dataloader, desc=f"[TEST]", leave=False, ncols=80) if rank == 0 else test_dataloader
    
    test_auc = BinaryAUROC(thresholds=32).to(device)


    with torch.no_grad():
        for image, segmentation_mask, _ in loader:
            image = image.to(device)
            segmentation_mask = segmentation_mask.to(device)

            logits = model(image)
            tloss += lossFunc(logits, segmentation_mask).item()

            # Ensure correct shape and type for ROC AUC
            probs = torch.sigmoid(logits).reshape(-1)
            targets = segmentation_mask.reshape(-1)
            test_auc.update(probs, targets)

    avg_test_loss = tloss / len(test_dataloader)

    test_loss_tensor = torch.tensor(avg_test_loss).to(device)

    dist.all_reduce(test_loss_tensor, op=dist.ReduceOp.AVG)
    avg_test_loss = test_loss_tensor.item()

    test_auc_tensor = torch.tensor(test_auc.compute().item()).to(device)
    dist.all_reduce(test_auc_tensor, op=dist.ReduceOp.AVG)
    avg_test_auc = test_auc_tensor.item()

    if rank == 0:
        
        test_metrics = {
            'test_loss': avg_test_loss,
            'test_auc': avg_test_auc
        }
        with open(modelDir + 'metrics/test_metrics.json', 'w') as f:
            json.dump(test_metrics, f)

        print(f"Test: Loss={avg_test_loss:.4f} | AUC={avg_test_auc:.4f}")


    return avg_test_loss if rank == 0 else None 