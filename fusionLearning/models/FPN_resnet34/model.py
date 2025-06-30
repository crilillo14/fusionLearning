import os
#change to root of src
os.chdir(os.path.join(os.path.dirname(__file__), "../.."))

from models.consts import MAXEPOCHS, BATCHSIZE, MOMENTUM, LEARNING_RATE, NUM_CLASSES


import torch 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


import segmentation_models_pytorch as smp 

model = smp.FPN(
    encoder_name="resnet34",  
    encoder_weights=None,  
    in_channels=3,  
    classes=NUM_CLASSES,
).to(device)

optimizer = torch.optim.SGD(model.parameters(),
                           lr=LEARNING_RATE,
                           momentum=MOMENTUM)

lossFunc = torch.nn.CrossEntropyLoss()



from config import CUB, CUB_IMAGES, CUB_SEGMENTATIONS

path_images_folder = os.path.join(CUB_IMAGES)
path_segmentations_folder = os.path.join(CUB_SEGMENTATIONS)

print(path_images_folder)
print(path_segmentations_folder)


from data.dataloaders import create_train_val_test_loaders
from data.aug import geoTransforms, photometricTransforms


training_dataloader, validation_dataloader, test_dataloader = create_train_val_test_loaders(
    path_images_folder,
    path_segmentations_folder,
    batch_size=BATCHSIZE,
    train_ratio=0.7,
    val_ratio=0.2,
    gTransforms=geoTransforms,       # to change later
    pTransforms=photometricTransforms        # to change later
)
    


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



DEBUG_TRAIN = False




from tqdm import tqdm
import os
import json
from torchmetrics.classification import BinaryAUROC
# !!! TODO: Fix ROC AUC metric calculation in validation phase

def train(modelDir):
    output_dir = modelDir + "outputs"
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, "epoch_metrics.json")
    # Initialize metrics file
    with open(metrics_path, 'w') as f:
        json.dump([], f)

    best_val_loss = float('inf')
    val_auc_metric = BinaryAUROC(thresholds=32).to(device)

    for epoch in range(1, MAXEPOCHS + 1):
        # --- Training Phase ---
        model.train()
        train_loss = 0.0
        for images, masks, _ in tqdm(training_dataloader, desc=f"Epoch {epoch}/{MAXEPOCHS} [Train]", leave=False, ncols=80):


            if DEBUG_TRAIN:
                print("images shape:", images.shape)
                print("images dtype:", images.dtype)
                print("images device:", images.device)

                print("masks shape:", masks.shape)
                print("masks dtype:", masks.dtype)
                print("masks device:", masks.device)

                print("device:", device)


            images = images.contiguous().to(device)
            masks  = masks.contiguous().to(device).squeeze(1)
            
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
        with torch.no_grad():
            for images, masks, _ in tqdm(validation_dataloader, desc=f"Epoch {epoch}/{MAXEPOCHS} [Val]", leave=False, ncols=80):
                images = images.to(device)
                masks  = masks.long().squeeze(1).to(device)
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

    print(f"Training complete. Metrics written to {metrics_path}")




def test(modelDir):
    model.eval()
    tloss = 0.0

    with torch.no_grad():
        for image, segmentation_mask, _ in tqdm(test_dataloader, desc=f"[TEST]", leave=False, ncols=80):
            image = image.to(device)
            segmentation_mask = segmentation_mask.long().squeeze(1).to(device)

            logits = model(image)
            tloss += lossFunc(logits, segmentation_mask).item()

    avg_test_loss = tloss / len(test_dataloader)

    test_metrics = {
        'test_loss': avg_test_loss,
    }
    with open(modelDir + 'outputs/test_metrics.json', 'w') as f:
        json.dump(test_metrics, f)

    print(f"Test: Loss={avg_test_loss:.4f}")
    return test_metrics


import matplotlib.pyplot as plt
import json

def plot_metrics(modelDir):
    with open(modelDir + "outputs/epoch_metrics.json", 'r') as f:
        data = json.load(f)
    epochs = [d['epoch'] for d in data]
    train_loss = [d['train_loss'] for d in data]
    val_loss = [d['val_loss'] for d in data]
    val_auc = [d['val_auc'] for d in data]
    lr = [d['lr'] for d in data]
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


import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


torch.cuda.is_available()


import torch

device = torch.device("cuda")

try:
    images = torch.randn(1, 3, 352, 512, device=device)
    masks = torch.randint(0, 2, (1, 352, 512), dtype=torch.int64, device=device)
    print("Tensors created and moved to CUDA successfully")
except RuntimeError as e:
    print("RuntimeError:", e)




import torch
torch.cuda.empty_cache()
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'





modelDir = "models/FPN/"
modelName = "FPN"
# Create outputs directory if it doesn't exist
os.makedirs(modelDir + "outputs", exist_ok=True)

if os.path.exists(modelDir + "outputs/best_model.pth"):
    print("Model already trained. To retrain, delete the 'outputs/best_model.pth' file.\n Going ahead with testing...")

    # Load best model

    model = smp.FPN(
        encoder_name="resnet34",  
        encoder_weights=None,  
        in_channels=3,  
        classes=NUM_CLASSES,
    )
    model.load_state_dict(torch.load(modelDir + f"outputs/best_model.pth"))
    model.to(device)

    test_metrics = test(modelDir)
    print("\nTesting completed successfully.")
    plot_metrics(modelDir)
else:
    if device.type == "cuda":
        print("Starting training process...")
        train(modelDir)
        print("\nStarting testing process...")
        test_metrics = test(modelDir)
        print("\nTraining and testing completed lsuccessfully.")
        print("\t * Results and visualizations saved in the 'outputs' directory. * ")

        plot_metrics(modelDir)

    else:
        print("No GPU available, exiting...")


import random



def inference_from_paths(model, modelDir, test_dataloader, n=5):
    # limit to 50 samples
    """
    Samples n random datapoints from the test set using the underlying dataset's image paths,
    loads and visualizes both the true and predicted segmentation masks.
    """


    # Try to access the dataset object and its image paths
    dataset = test_dataloader.dataset
    if hasattr(dataset, 'dataset'):
        # This is a Subset, get the original dataset
        base_dataset = dataset.dataset
        indices = dataset.indices
        image_paths = [base_dataset.image_paths[i] for i in indices]
    else:
        image_paths = dataset.image_paths

    total_samples = len(image_paths)
    n = min(n, 50)                                          
    indices = random.sample(range(total_samples), n)


    model.to(device)
    model.load_state_dict(torch.load(modelDir + "outputs/best_model.pth", map_location=device))
    model.eval()

    plt.figure(figsize=(6, n * 3))
    for i, idx in enumerate(indices):
        # Get sample from dataset by index
        image, true_mask, _ = dataset[idx]
        image_input = image.to(device).unsqueeze(0)     # think indexing the test dataset doesnt apply transforms
        true_mask_np = true_mask.cpu().numpy()

        with torch.no_grad():
            logits = model(image_input)
            pred_mask = torch.argmax(logits, dim=1)[0].cpu().numpy()

        plt.subplot(n, 3, i * 3 + 1)
        plt.imshow(image.cpu().permute(1, 2, 0))
        plt.title(f"Sample {i+1}: Image")
        plt.axis('off')

        plt.subplot(n, 3, i * 3 + 2)
        plt.imshow(true_mask_np, cmap='gray')
        plt.title("True Mask")
        plt.axis('off')

        plt.subplot(n, 3, i * 3 + 3)
        plt.imshow(pred_mask, cmap='gray')
        plt.title("Predicted Mask")
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(modelDir + "outputs/inference.png")
    plt.show()

    plt.close()


# %%
model = smp.FPN(
    encoder_name="resnet34",  
    encoder_weights=None,  
    in_channels=3,  
    classes=NUM_CLASSES,
)


inference_from_paths(model=model, modelDir="models/FPN/", test_dataloader=test_dataloader, n=20)


