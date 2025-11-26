
import os
import json

import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
from torchmetrics.classification import BinaryAUROC
from tqdm import tqdm

from fusionLearning.models.consts import MAXEPOCHS, LEARNING_RATE

DEBUG_TRAIN = False


def train_dist(modelDir, 
          model, 
          optimizer, 
          lossFunc, 
          training_dataloader : DataLoader, 
          validation_dataloader : DataLoader,
          rank):
    
    device = f"cuda:{rank}"
    
    
    if rank == 0:
        output_dir = modelDir + "outputs"
        os.makedirs(output_dir, exist_ok=True)
        metrics_path = os.path.join(output_dir, "epoch_metrics.json")
        # Initialize metrics file
        with open(metrics_path, 'w') as f:
            json.dump([], f)

    best_val_loss = float('inf')
    val_auc_metric = BinaryAUROC(thresholds=32).to(device)

    # –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    for epoch in range(1, MAXEPOCHS + 1):

        # --- Training Phase ---

        
        training_dataloader.sampler.set_epoch(epoch) # for distr.
        
        model.train()
        train_loss = 0.0
        
        # no cluttering io
        loader = tqdm(training_dataloader, desc=f"Epoch {epoch}/{MAXEPOCHS} [Train]", 
                     leave=False, ncols=80) if rank == 0 else training_dataloader
        
        
        for images, masks, _ in loader:


            if DEBUG_TRAIN:
                print("images shape:", images.shape)
                print("images dtype:", images.dtype)
                print("images device:", images.device)

                print("masks shape:", masks.shape)
                print("masks dtype:", masks.dtype)
                print("masks device:", masks.device)

                print("device:", device)


            images = images.to(device)
            masks  = masks.to(device)
            
            if DEBUG_TRAIN:
                print("mask unique values:", torch.unique(masks))
                print("mask dtype:", masks.dtype)
                print("mask shape:", masks.shape)


            optimizer.zero_grad()
            logits = model(images)
            loss = lossFunc(logits, masks)

            if DEBUG_TRAIN:
                print(f"logits: {logits.shape}, dtype: {logits.dtype}, device: {logits.device}")
                print(f"masks: {masks.shape}, dtype: {masks.dtype}, device: {masks.device}")

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        avg_train_loss = train_loss / len(training_dataloader)
        
        
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_auc_metric.reset()
        
        vloader = tqdm(validation_dataloader, desc=f"Epoch {epoch}/{MAXEPOCHS} [Val]", leave=False, ncols=80) if rank == 0 else validation_dataloader
        
        with torch.no_grad():
            for images, masks, _ in tqdm(validation_dataloader, desc=f"Epoch {epoch}/{MAXEPOCHS} [Val]", leave=False, ncols=80):
                images = images.to(device, non_blocking=True)
                masks  = masks.to(device, non_blocking=True)
                logits = model(images)
                val_loss += lossFunc(logits, masks).item()

                # Ensure correct shape and type for ROC AUC
                probs = torch.sigmoid(logits).reshape(-1)
                targets = masks.reshape(-1)
                val_auc_metric.update(probs, targets)

        avg_val_loss = val_loss / len(validation_dataloader)
        val_auc = val_auc_metric.compute().item()
        
        # !!! all reduce and such ...
        
        avg_train_loss_tensor = torch.tensor(avg_train_loss).to(device)
        avg_val_loss_tensor = torch.tensor(avg_val_loss).to(device)
        val_auc_tensor = torch.tensor(val_auc).to(device)
        
        dist.all_reduce(avg_train_loss_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(avg_val_loss_tensor, op=dist.ReduceOp.AVG)
        dist.all_reduce(val_auc_tensor, op=dist.ReduceOp.AVG)
        
        avg_train_loss = avg_train_loss_tensor.item()
        avg_val_loss = avg_val_loss_tensor.item()
        val_auc = val_auc_tensor.item()

        # Learning rate -- maybe swap to scheduler for higher convergence
        lr = LEARNING_RATE

        if rank == 0:   
            # save to .pth
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), os.path.join(output_dir, 'best_model.pth'))

            # write to jason
            with open(metrics_path, 'r+') as f:
                data = json.load(f)
                data.append({
                    'epoch': epoch,
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'val_auc': val_auc,
                    'lr': lr
                })
                f.seek(0)
                json.dump(data, f, indent=2)
                f.truncate()

            print(f"Epoch {epoch:2d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} "
                f"| Val AUC: {val_auc:.4f} | LR: {lr:.6f}")

    if rank == 0:
        print(f"Training complete. Metrics written to {metrics_path}")
