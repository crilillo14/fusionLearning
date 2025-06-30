import os
#change to root of src
os.chdir(os.path.join(os.path.dirname(__file__), "../.."))

from config import CUB, CUB_IMAGES, CUB_SEGMENTATIONS
from models.consts import MAXEPOCHS, BATCHSIZE, MOMENTUM, LEARNING_RATE, NUM_CLASSES
from data.dataloaders import create_train_val_test_loaders
from data.aug import geoTransforms, photometricTransforms

import segmentation_models_pytorch as smp 
from torchmetrics.classification import BinaryAUROC
import torch 

import matplotlib.pyplot as plt
from tqdm import tqdm
import json



# ## Configuring HPs, model, device, compiling.



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# initialize model and compile it


model = smp.FPN(
    encoder_name="resnet34",  
    encoder_weights=None,  
    in_channels=3,  
    classes=NUM_CLASSES,
).to(device)

optimizer = torch.optim.SGD(model.parameters(),
                           lr=LEARNING_RATE,
                           momentum=MOMENTUM)

lossFunc = torch.nn.CrossEntropyLoss().to(device)


if device.type == "cuda":
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA current device: {torch.cuda.current_device()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
    print(f"Cuda version: {torch.version.cuda}")

else:
    print("No CUDA device found")


# ## Init Dataloaders w/ transforms
# ---

path_images_folder = os.path.join(CUB_IMAGES)
path_segmentations_folder = os.path.join(CUB_SEGMENTATIONS)
print(path_images_folder)
print(path_segmentations_folder)



training_dataloader, validation_dataloader, test_dataloader = create_train_val_test_loaders(
    path_images_folder,
    path_segmentations_folder,
    batch_size=BATCHSIZE,
    train_ratio=0.7,
    val_ratio=0.2,
    gTransforms=geoTransforms,
    pTransforms=photometricTransforms
)
    

# ## Training loop:

DEBUG_TRAIN = True



# !!! TODO: Fix ROC AUC metric calculation in validation phase
# think it's fixed now

def train(modelDir, modelName : str):
    
    output_dir = modelDir + "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize metrics file
    metrics_path = os.path.join(output_dir, "epoch_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump([], f)

    best_val_loss = float('inf')
    val_auc_metric = BinaryAUROC(thresholds=64)

    for epoch in range(1, MAXEPOCHS + 1):
        
        # --- Training Phase ---
        
        model.train()
        train_loss : float = 0.0

        for images, masks, _ in tqdm(training_dataloader, desc=f"Epoch {epoch}/{MAXEPOCHS} [Train]", leave=False, ncols=80):

            images = images.to(device)

            # transfer to GPU, remove channel dim, convert uint8 to float32, normalize to [0, 1]
            masks  = masks.to(device).squeeze(1).long()

            if DEBUG_TRAIN:
                print(f"images shape: {images.shape}, masks shape: {masks.shape}")
                print(f"images dtype: {images.dtype}, masks dtype: {masks.dtype}")
                print(f"Mask min: {masks.min()}, max: {masks.max()}")
                print(f"Unique mask values: {torch.unique(masks)}")


            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = lossFunc(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        avg_train_loss = train_loss / len(training_dataloader)
        
        # --- Validation Phase ---
        model.eval()
        val_loss = 0.0
        val_auc_metric.reset()
        with torch.no_grad():
            for images, masks, _ in tqdm(validation_dataloader, desc=f"Epoch {epoch}/{MAXEPOCHS} [Val]", leave=False, ncols=80):
                images = images.to(device)
                masks  = masks.to(device).squeeze(1).long()
                logits = model(images)
                val_loss += lossFunc(logits, masks.long()).item()

                # Ensure correct shape and type for ROC AUC
                probs = torch.softmax(logits, dim=1)[:, 1].reshape(-1)
                targets = masks.reshape(-1).float()
                val_auc_metric.update(probs, targets)

        avg_val_loss = val_loss / len(validation_dataloader)
        val_auc = val_auc_metric.compute().item()

        # Learning rate -- maybe swap to scheduler for higher convergence
        lr = LEARNING_RATE

        # save to .pth
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(output_dir, f'{modelDir + modelName}_best_model.pth'))

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

    print(f"Training complete. Metrics written to {metrics_path}")


def test(modelDir, modelName : str):
    model.eval()
    tloss = 0.0

    with torch.no_grad():
        for image, segmentation_mask, _ in tqdm(test_dataloader, desc=f"[TEST]", leave=False, ncols=80):
            image = image.to(device)
            segmentation_mask = segmentation_mask.to(device).squeeze(1).long()

            logits = model(image)
            tloss += lossFunc(logits, segmentation_mask.long()).item()

    avg_test_loss = tloss / len(test_dataloader)

    test_metrics = {
        'test_loss': avg_test_loss,
    }
    with open(modelDir + f'outputs/{modelName}_test_metrics.json', 'w') as f:
        json.dump(test_metrics, f)

    print(f"Test: Loss={avg_test_loss:.4f}")
    return test_metrics

# ## Visualization Helpers >

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



def plot_metrics(modelDir, modelName : str):
    with open(modelDir + f"{modelName}_outputs/epoch_metrics.json", 'r') as f:
        data = json.load(f)
    epochs = [d['epoch'] for d in data]
    train_loss = [d['train_loss'] for d in data]
    val_loss = [d['val_loss'] for d in data]
    val_auc = [d['val_auc'] for d in data]
    # lr = [d['lr'] for d in data]
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
    plt.show()

torch.cuda.empty_cache()

# # **Change modelDir and modelName!**


modelDir = "models/FPN/"
modelName = "FPN"

# Create outputs directory if it doesn't exist
os.makedirs(modelDir + "outputs", exist_ok=True)

if os.path.exists(modelDir + f"outputs/{modelName}_best_model.pth"):
    print("Model already trained. To retrain, delete the 'outputs/{modelName}_best_model.pth' file.\n Going ahead with testing...")

    # Load best model
    model.load_state_dict(torch.load(modelDir + f"outputs/{modelName}_best_model.pth"))
    model.to(device)

    test_metrics = test(modelDir)
    print("\nTesting completed successfully.")
    plot_metrics(modelDir)
else:
    if device.type == "cuda":
        print("Starting training process...")
        train(modelDir, modelName)
        print("\nStarting testing process...")
        test_metrics = test(modelDir, modelName)
        print("\nTraining and testing completed successfully.")
        print("\t * Results and visualizations saved in the 'outputs' directory. * ")

        plot_metrics(modelDir)

    else:
        print(f"CUDA device not found: {device.type}")
        print("No GPU available, exiting...")
