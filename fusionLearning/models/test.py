"""Run testing process"""

import json

import torch
import torch.distributed as dist
from tqdm import tqdm


def test_dist(modelDir, model, test_dataloader, lossFunc, rank):

    device = f"cuda:{rank}"

    model.eval()
    tloss = 0.0

    loader = tqdm(test_dataloader, desc=f"[TEST]", leave=False, ncols=80) if rank == 0 else test_dataloader

    with torch.no_grad():
        for image, segmentation_mask, _ in loader:
            image = image.to(device)
            segmentation_mask = segmentation_mask.to(device)

            logits = model(image)
            tloss += lossFunc(logits, segmentation_mask).item()

    avg_test_loss = tloss / len(test_dataloader)

    test_loss_tensor = torch.tensor(avg_test_loss).to(device)

    dist.all_reduce(test_loss_tensor, op=dist.ReduceOp.AVG)
    avg_test_loss = test_loss_tensor.item()


    if rank == 0:
        
        test_metrics = {
            'test_loss': avg_test_loss,
        }
        with open(modelDir + 'outputs/test_metrics.json', 'w') as f:
            json.dump(test_metrics, f)

        print(f"Test: Loss={avg_test_loss:.4f}")


    return avg_test_loss if rank == 0 else None 