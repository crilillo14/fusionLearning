

import os
import sys
parent_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if parent_parent_dir not in sys.path:
    sys.path.insert(0, parent_parent_dir)

from fusionLearning.models.consts import MAXEPOCHS, BATCHSIZE, MOMENTUM, LEARNING_RATE, NUM_CLASSES
from fusionLearning.config import CUB, CUB_IMAGES, CUB_SEGMENTATIONS
from fusionLearning.data.dataloaders import CUBDataset
from fusionLearning.data.aug import geoTransforms, photometricTransforms

import segmentation_models_pytorch as smp
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, random_split
from torchmetrics.classification import BinaryAUROC
import matplotlib.pyplot as plt
import numpy as np


import json
import pprint
import random
import shutil
from tqdm import tqdm
import argparse


# ## Post training: visualizing prediction masks
# debug_viz determines if logits are outputted

debug_viz = 0
DEBUG_TRAIN = False

def warmup(device) -> None:
    try:
        images = torch.randn(1, 3, 352, 512, device=device)
        masks = torch.randint(0, 2, (1, 352, 512), dtype=torch.int64, device=device)
        if dist.get_rank() == 0:
            print("Tensors created and moved to CUDA successfully")
    except RuntimeError as e:
        if dist.get_rank() == 0:
            print("RuntimeError:", e)
    

def get_dataloaders(image_dir, segmentation_dir, batch_size, train_ratio=0.7, val_ratio=0.2, gTransforms=None, pTransforms=None):
    """
    Create train, validation, and test DataLoaders with DistributedSampler
    """
    full_dataset = CUBDataset(image_dir, 
                              segmentation_dir, 
                              gTransforms=gTransforms, 
                              pTransforms=pTransforms)
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    # Set a fixed seed for reproducibility
    generator = torch.Generator().manual_seed(42)
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )
    
    # Create DistributedSamplers
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    # Test sampler is optional, usually we just run test on rank 0 or distributed without shuffle
    test_sampler = DistributedSampler(test_dataset, shuffle=False)

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=False, # Shuffle is handled by sampler
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, train_sampler


def main():
    # Initialize distributed process group
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if global_rank == 0:
        print(f"On torch version: {torch.__version__}")
        print(f"World Size: {world_size}")

    warmup(device)

    # Declare model type and encoder architecture
    MODEL_NAME = "UnetPlusPlus"
    MODEL = smp.UnetPlusPlus
    encoder = "resnet34"

    model = MODEL(
        encoder_name=encoder,  
        encoder_weights=None,  
        in_channels=3,  
        classes=NUM_CLASSES,
    ).to(device)
    
    # Wrap model with DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    optimizer = torch.optim.SGD(model.parameters(),
                            lr=LEARNING_RATE,
                            momentum=MOMENTUM)

    lossFunc = torch.nn.BCEWithLogitsLoss()

    path_images_folder = os.path.join(CUB_IMAGES)
    path_segmentations_folder = os.path.join(CUB_SEGMENTATIONS)

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # Debugging only
    # os.environ["TORCH_USE_CUDA_DSA"] = "1" # Debugging only

    # EDIT BELOW
    modelName = f"{MODEL_NAME}_{encoder}"
    modelDir = f"models/{modelName}/"
    
    # Create outputs directory if it doesn't exist (only rank 0 needs to do this effectively, but safe to run on all)
    if global_rank == 0:
        os.makedirs(modelDir + "outputs", exist_ok=True)

    # Data Loading
    training_dataloader, validation_dataloader, test_dataloader, train_sampler = get_dataloaders(
        path_images_folder,
        path_segmentations_folder,
        batch_size=BATCHSIZE,
        train_ratio=0.7,
        val_ratio=0.2,   
        gTransforms=geoTransforms,  
        pTransforms=photometricTransforms
    )

    if os.path.exists(modelDir + "outputs/best_model.pth"):
        if global_rank == 0:
            print("Model already trained. To retrain, delete the 'outputs/best_model.pth' file.\n Going ahead with testing...")

        # Load best model
        # We need to unwrap or load into DDP model. 
        # Since we are in DDP context, we can load state dict. 
        # Note: saved state dict might have 'module.' prefix or not depending on how it was saved.
        # If we save model.module.state_dict(), it won't have prefix.
        
        map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
        state_dict = torch.load(modelDir + f"outputs/best_model.pth", map_location=map_location)
        
        # If saved from DDP (module.), and loading into DDP, it should match.
        # If saved from single GPU (no module.), and loading into DDP, we might need to adjust.
        # Assuming we save model.module.state_dict() in train(), so it's clean.
        # But here 'model' is DDP wrapped. So we should load into model.module or expect keys to not have 'module.' if we load into model.module.
        
        model.module.load_state_dict(state_dict)
        
        test_metrics = test(modelDir, model, test_dataloader, lossFunc, device, global_rank)
        
        if global_rank == 0:
            print("\nTesting completed successfully.")
            plot_metrics(modelDir)
            inference_from_paths(model=model.module, modelDir=modelDir, test_dataloader=test_dataloader, device=device, n=20)
            copy_best_model_to_weights(modelDir, modelName)

    else:
        if global_rank == 0:
            print("Starting training process...")
            
        train(modelDir, model, optimizer, lossFunc, training_dataloader, validation_dataloader, train_sampler, device, global_rank)
        
        if global_rank == 0:
            print("\nStarting testing process...")
            
        test_metrics = test(modelDir, model, test_dataloader, lossFunc, device, global_rank)
        
        if global_rank == 0:
            print("\nTraining and testing completed successfully.")
            print("\t * Results and visualizations saved in the 'outputs' directory. * ")
            plot_metrics(modelDir)
            inference_from_paths(model=model.module, modelDir=modelDir, test_dataloader=test_dataloader, device=device, n=20)
            copy_best_model_to_weights(modelDir, modelName)

    dist.destroy_process_group()


def visualize_training_process(metrics):
    """Visualize the final training process metrics"""
    plt.figure(figsize=(8, 5))
    plt.plot(metrics['epochs'], metrics['train_loss'], 'bo-', label='Training Loss')
    plt.plot(metrics['epochs'], metrics['val_loss'], 'ro-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("outputs/final_training_metrics.png")
    plt.close()

def train(modelDir, 
          model, 
          optimizer, 
          lossFunc, 
          training_dataloader, 
          validation_dataloader,
          train_sampler,
          device,
          rank):
    
    output_dir = modelDir + "outputs"
    metrics_path = os.path.join(output_dir, "epoch_metrics.json")
    
    if rank == 0:
        os.makedirs(output_dir, exist_ok=True)
        # Initialize metrics file
        with open(metrics_path, 'w') as f:
            json.dump([], f)

    best_val_loss = float('inf')
    val_auc_metric = BinaryAUROC(thresholds=32).to(device)

    # –––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
    for epoch in range(1, MAXEPOCHS + 1):
        train_sampler.set_epoch(epoch)

        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        
        # Only show progress bar on rank 0
        iterator = tqdm(training_dataloader, desc=f"Epoch {epoch}/{MAXEPOCHS} [Train]", leave=False, ncols=80) if rank == 0 else training_dataloader
        
        for images, masks, _ in iterator:
            images = images.contiguous().to(device, non_blocking=True)
            masks  = masks.contiguous().to(device, non_blocking=True)
            
            optimizer.zero_grad()
            logits = model(images)
            loss = lossFunc(logits, masks)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            
        # Average loss across all batches on this rank
        avg_train_loss_local = train_loss / len(training_dataloader)
        
        # Reduce loss across all ranks for logging
        avg_train_loss_tensor = torch.tensor(avg_train_loss_local, device=device)
        dist.all_reduce(avg_train_loss_tensor, op=dist.ReduceOp.SUM)
        avg_train_loss = avg_train_loss_tensor.item() / dist.get_world_size()
        
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_auc_metric.reset()
        
        iterator = tqdm(validation_dataloader, desc=f"Epoch {epoch}/{MAXEPOCHS} [Val]", leave=False, ncols=80) if rank == 0 else validation_dataloader

        with torch.no_grad():
            for images, masks, _ in iterator:
                images = images.to(device, non_blocking=True)
                masks  = masks.to(device, non_blocking=True)
                logits = model(images)
                val_loss += lossFunc(logits, masks).item()

                # Ensure correct shape and type for ROC AUC
                probs = torch.sigmoid(logits).reshape(-1)
                targets = masks.reshape(-1)
                val_auc_metric.update(probs, targets)

        avg_val_loss_local = val_loss / len(validation_dataloader)
        
        # Reduce val loss
        avg_val_loss_tensor = torch.tensor(avg_val_loss_local, device=device)
        dist.all_reduce(avg_val_loss_tensor, op=dist.ReduceOp.SUM)
        avg_val_loss = avg_val_loss_tensor.item() / dist.get_world_size()

        # Metric computation (AUC) - torchmetrics handles DDP synchronization if configured, 
        # but here we are using it per rank and might need manual sync or use compute() which might sync.
        # BinaryAUROC default behavior: computes locally. We should sync.
        # For simplicity, let's just print rank 0's AUC or try to sync if critical. 
        # Ideally use torchmetrics with dist_sync_on_step=True or compute on gathered data.
        # Here we will just let rank 0 compute its own AUC as a proxy or assume data is i.i.d.
        # Better: use metric.compute() which should be roughly correct if data is shuffled.
        val_auc = val_auc_metric.compute().item()

        # Learning rate
        lr = LEARNING_RATE

        if rank == 0:
            # save to .pth
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                # Save the underlying model, not the DDP wrapper
                torch.save(model.module.state_dict(), os.path.join(output_dir, 'best_model.pth'))

            # write to json
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

def test(modelDir, model, test_dataloader, lossFunc, device, rank):
    model.eval()
    tloss = 0.0
    
    iterator = tqdm(test_dataloader, desc=f"[TEST]", leave=False, ncols=80) if rank == 0 else test_dataloader

    with torch.no_grad():
        for image, segmentation_mask, _ in iterator:
            image = image.to(device, non_blocking=True)
            segmentation_mask = segmentation_mask.to(device, non_blocking=True)

            logits = model(image)
            tloss += lossFunc(logits, segmentation_mask).item()

    avg_test_loss_local = tloss / len(test_dataloader)
    
    avg_test_loss_tensor = torch.tensor(avg_test_loss_local, device=device)
    dist.all_reduce(avg_test_loss_tensor, op=dist.ReduceOp.SUM)
    avg_test_loss = avg_test_loss_tensor.item() / dist.get_world_size()

    test_metrics = {
        'test_loss': avg_test_loss,
    }
    
    if rank == 0:
        with open(modelDir + 'outputs/test_metrics.json', 'w') as f:
            json.dump(test_metrics, f)
        print(f"Test: Loss={avg_test_loss:.4f}")
        
    return test_metrics

def plot_metrics(modelDir):
    matplotlib.use('Agg')  # non interactive
    
    try:
        with open(modelDir + "outputs/epoch_metrics.json", 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Metrics file not found, skipping plot.")
        return

    epochs = [d['epoch'] for d in data]
    train_loss = [d['train_loss'] for d in data]
    val_loss = [d['val_loss'] for d in data]
    val_auc = [d['val_auc'] for d in data]
    
    plt.figure(figsize=(12, 8))
    plt.subplot(2,1,1)
    plt.plot(epochs, train_loss, 'b-', label='Train Loss')
    plt.plot(epochs, val_loss, 'r-', label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.subplot(2,1,2)
    plt.plot(epochs, val_auc, 'r-', label='Validation ROC AUC')
    plt.xlabel('Epoch')
    plt.ylabel('Validation ROC AUC')
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(modelDir + "outputs/final_training_metrics.png")
    plt.close()

def inference_from_paths(model, 
                         modelDir, 
                         test_dataloader, 
                         device,
                         n=5):
    global debug_viz
    
    # Only run on rank 0
    
    # Try to access the dataset object and its image paths
    # test_dataloader.dataset is a Subset
    dataset = test_dataloader.dataset
    if hasattr(dataset, 'dataset'):
        base_dataset = dataset.dataset
        indices = dataset.indices
        # We need to map subset indices to original dataset indices to get paths if needed
        # But dataset[i] works on the subset directly.
    else:
        pass

    total_samples = len(dataset)
    n = min(n, 50)                                          
    # Random sample from the subset
    sample_indices = random.sample(range(total_samples), n)

    model.eval()
    
    fns = []
    plt.figure(figsize=(6, n * 3))
    
    for i, idx in enumerate(sample_indices):
        # Get sample from dataset by index
        image, true_mask, filename = dataset[idx]
        fns.append(filename)
        image_input = image.to(device).unsqueeze(0)
        true_mask_np = true_mask.cpu().squeeze(0).numpy()

        with torch.no_grad():
            logits = model(image_input)
            prob = torch.sigmoid(logits)
            pred_mask = prob.squeeze(1)
            pred_mask_continuous = pred_mask.cpu().numpy().astype(np.float32)

        plt.subplot(n, 3, i * 3 + 1)
        plt.imshow(image.cpu().permute(1, 2, 0))
        plt.title(f"Sample {i+1}")
        plt.axis('off') 

        plt.subplot(n, 3, i * 3 + 2)
        plt.imshow(true_mask_np, cmap='gray')
        plt.title(f"True Mask")
        plt.axis('off') 

        plt.subplot(n, 3, i * 3 + 3)
        im = plt.imshow(pred_mask_continuous[0], cmap='viridis', vmin=0, vmax=1)
        plt.title(f"Predicted Probability")
        plt.axis('off')
        plt.colorbar(im, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(modelDir + "outputs/inference.png", dpi=150, bbox_inches='tight')
    plt.close()

    if debug_viz:
        pprint.pprint(fns)

def copy_best_model_to_weights(model_dir, modelName):
    src = os.path.join(model_dir, "outputs", "best_model.pth")
    dst = os.path.join(model_dir, "..", "..", "weights", modelName, f"{modelName}.pth")
    os.makedirs(os.path.dirname(dst), exist_ok = True)
    shutil.copy(src, dst)

if __name__ == "__main__":
    # Environment variables set by torchrun
    # RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT
    
    # Basic check
    if "RANK" not in os.environ:
        # Fallback for single process run without torchrun (debugging)
        os.environ["RANK"] = "0"
        os.environ["LOCAL_RANK"] = "0"
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "12355"
        print("WARNING: Running without torchrun. Setting default DDP environment variables.")

    main()