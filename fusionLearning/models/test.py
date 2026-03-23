"""Run testing process"""

import json

import torch
import torch.distributed as dist
from tqdm import tqdm
from torchmetrics.segmentation import DiceScore, MeanIoU


def test_dist(modelDir, model, test_dataloader, lossFunc, rank):

    device = f"cuda:{rank}"

    model.eval()
    tloss = 0.0

    loader = tqdm(test_dataloader, desc=f"[TEST]", leave=False, ncols=80) if rank == 0 else test_dataloader
    
    test_dice = DiceScore().to(device)
    test_iou = MeanIoU().to(device)


    with torch.no_grad():
        for image, segmentation_mask, _ in loader:
            image = image.to(device)
            segmentation_mask = segmentation_mask.to(device)

            logits = model(image)
            tloss += lossFunc(logits, segmentation_mask).item()

            # Ensure correct shape and type for ROC AUC
            probs = torch.sigmoid(logits).reshape(-1)
            targets = segmentation_mask.reshape(-1)
            test_dice.update(probs, targets)
            test_iou.update(probs, targets)

    avg_test_loss = tloss / len(test_dataloader)

    test_loss_tensor = torch.tensor(avg_test_loss).to(device)

    dist.all_reduce(test_loss_tensor, op=dist.ReduceOp.AVG)
    avg_test_loss = test_loss_tensor.item()

    test_dice_tensor = torch.tensor(test_dice.compute().item()).to(device)
    dist.all_reduce(test_dice_tensor, op=dist.ReduceOp.AVG)
    avg_test_dice = test_dice_tensor.item()

    test_iou_tensor = torch.tensor(test_iou.compute().item()).to(device)
    dist.all_reduce(test_iou_tensor, op=dist.ReduceOp.AVG)
    avg_test_iou = test_iou_tensor.item()

    if rank == 0:
        
        test_metrics = {
            'test_loss': avg_test_loss,
            'test_dice': avg_test_dice,
            'test_iou': avg_test_iou
        }
        with open(modelDir + 'metrics/test_metrics.json', 'w') as f:
            json.dump(test_metrics, f)

        print(f"Test: Loss={avg_test_loss:.4f} | Dice={avg_test_dice:.4f} | IoU={avg_test_iou:.4f}")


    return avg_test_loss if rank == 0 else None 