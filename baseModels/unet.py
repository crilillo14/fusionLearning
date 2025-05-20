
"""
Initializing an empty-weighted Unet and training it on the CUB dataset.
Serves as a baseline model for the
"""


import torch
import segmentation_models_pytorch as smp 
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import json
from dataloaders import create_train_val_test_loaders

#MACROS
MAXEPOCHS = 10
BATCHSIZE = 1
MOMENTUM = 0.99
LEARNING_RATE = 0.01
NUM_CLASSES = 2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model, optimizer, loss definitions -------------------------------------------------------------
# vanilla unet, untrained
unet = smp.Unet(
    encoder_name="resnet34",  
    encoder_weights=None,  
    in_channels=3,  
    classes=NUM_CLASSES,
)
unet.to(device)

optimizer = torch.optim.SGD(unet.parameters(),
                           lr=LEARNING_RATE,
                           momentum=MOMENTUM)

lossFunc = torch.nn.CrossEntropyLoss()

# Data loaders ------------------------------------------------------------------------------------
path_images_folder = os.path.join("CUBdata/CUB_200_2011/images/")
path_segmentations_folder = os.path.join("CUBdata/segmentations/")

training_dataloader, validation_dataloader, testing_dataloader = create_train_val_test_loaders(
    image_dir=path_images_folder, 
    segmentation_dir=path_segmentations_folder,
    batch_size=BATCHSIZE
)

# Training ------------------------------------------------------------------------------------------
def train(): 
    # Lists to store metrics for visualization
    train_losses = []
    val_losses = []
    train_aucs = []
    val_aucs = []
    epochs = []
    
    # Create output directory for saving metrics and plots
    os.makedirs("outputs", exist_ok=True)
    
    for epoch in range(1, MAXEPOCHS + 1):
        epochs.append(epoch)
        
        # Training phase
        unet.train()
        tloss = 0.0
        train_preds = []
        train_targets = []
        
        for image, segmentation_mask in training_dataloader: 
            # batch size is 1, outputs individual image seg pairs
            image = image.to(device)
            segmentation_mask = segmentation_mask.to(device).long()
            
            optimizer.zero_grad()
            
            logits = unet(image)
            loss = lossFunc(logits, segmentation_mask)
            
            loss.backward()
            optimizer.step()
            
            tloss += loss.item()
            
            # Store predictions and targets for AUC calculation
            probs = torch.softmax(logits, dim=1)[:, 1]  # Probability of class 1 (foreground)
            train_preds.extend(probs.flatten().cpu().detach().numpy())
            train_targets.extend(segmentation_mask.flatten().cpu().numpy())
        
        # Calculate training metrics
        avg_train = tloss / len(training_dataloader)
        train_losses.append(avg_train)
        
        # Calculate training AUC if possible
        try:
            if len(np.unique(train_targets)) > 1:  # Ensure both classes are present
                train_auc = roc_auc_score(train_targets, train_preds)
                train_aucs.append(train_auc)
                print(f"Epoch {epoch} \t Training Loss: {avg_train:.4f} \t Training AUC: {train_auc:.4f}")
            else:
                train_aucs.append(float('nan'))
                print(f"Epoch {epoch} \t Training Loss: {avg_train:.4f} \t Training AUC: N/A (need both classes)")
        except ValueError as e:
            train_aucs.append(float('nan'))
            print(f"Epoch {epoch} \t Training Loss: {avg_train:.4f} \t Training AUC: Error ({str(e)})")
        
        # Validation phase
        unet.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for image, segmentation_mask in validation_dataloader:
                image = image.to(device)
                segmentation_mask = segmentation_mask.to(device).long()
                logits = unet(image)
                val_loss += lossFunc(logits, segmentation_mask).item()
                
                # Store predictions and targets for AUC calculation
                probs = torch.softmax(logits, dim=1)[:, 1]  # Probability of class 1 (foreground)
                val_preds.extend(probs.flatten().cpu().numpy())
                val_targets.extend(segmentation_mask.flatten().cpu().numpy())
        
        # Calculate validation metrics
        avg_val = val_loss / len(validation_dataloader)
        val_losses.append(avg_val)
        
        # Calculate validation AUC if possible
        try:
            if len(np.unique(val_targets)) > 1:  # Ensure both classes are present
                val_auc = roc_auc_score(val_targets, val_preds)
                val_aucs.append(val_auc)
                print(f"Epoch {epoch} \t Validation Loss: {avg_val:.4f} \t Validation AUC: {val_auc:.4f}")
            else:
                val_aucs.append(float('nan'))
                print(f"Epoch {epoch} \t Validation Loss: {avg_val:.4f} \t Validation AUC: N/A (need both classes)")
        except ValueError as e:
            val_aucs.append(float('nan'))
            print(f"Epoch {epoch} \t Validation Loss: {avg_val:.4f} \t Validation AUC: Error ({str(e)})")
            
        print("=" * 60)
        
        # Visualize metrics after each epoch
        plt.figure(figsize=(12, 5))
        
        # Plot losses
        plt.subplot(1, 2, 1)
        plt.plot(epochs, train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Plot AUCs
        plt.subplot(1, 2, 2)
        plt.plot(epochs, train_aucs, 'b-', label='Training AUC')
        plt.plot(epochs, val_aucs, 'r-', label='Validation AUC')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.title('Training and Validation AUC')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"outputs/training_metrics_epoch_{epoch}.png")
        plt.close()
    
    # Save metrics to JSON for later visualization
    metrics = {
        'epochs': epochs,
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_auc': train_aucs,
        'val_auc': val_aucs
    }
    
    with open('outputs/training_metrics.json', 'w') as f:
        json.dump(metrics, f)
    
    # Final visualization of training process
    visualize_training_process(metrics)

def test(): 
    unet.eval()
    
    tloss = 0.0
    test_preds = []
    test_targets = []
    
    # Create a figure for sample visualization
    fig, axes = plt.subplots(4, 3, figsize=(15, 10))
    sample_idx = 0
    
    for image, segmentation_mask in testing_dataloader: 
        image = image.to(device)
        segmentation_mask = segmentation_mask.to(device).long()
        
        logits = unet(image)
        tloss += lossFunc(logits, segmentation_mask).item()
        
        # Store predictions and targets for AUC calculation
        probs = torch.softmax(logits, dim=1)[:, 1]  # Probability of class 1 (foreground)
        test_preds.extend(probs.flatten().cpu().detach().numpy())
        test_targets.extend(segmentation_mask.flatten().cpu().numpy())
        
        # Visualize some test samples
        if sample_idx < 4:  # Show 4 samples
            # Original image
            axes[sample_idx, 0].imshow(image[0].cpu().permute(1, 2, 0))
            axes[sample_idx, 0].set_title(f"Sample {sample_idx+1}: Original")
            axes[sample_idx, 0].axis('off')
            
            # Ground truth segmentation
            axes[sample_idx, 1].imshow(segmentation_mask[0].cpu(), cmap='gray')
            axes[sample_idx, 1].set_title(f"Sample {sample_idx+1}: Ground Truth")
            axes[sample_idx, 1].axis('off')
            
            # Predicted segmentation
            pred_mask = torch.argmax(logits, dim=1)[0].cpu()
            axes[sample_idx, 2].imshow(pred_mask, cmap='gray')
            axes[sample_idx, 2].set_title(f"Sample {sample_idx+1}: Prediction")
            axes[sample_idx, 2].axis('off')
            
            sample_idx += 1
    
    # Save the visualization samples
    plt.tight_layout()
    plt.savefig("outputs/test_samples_visualization.png")
    plt.close()
    
    # Calculate test metrics
    avg_test = tloss / len(testing_dataloader)
    
    # Calculate test AUC if possible
    try:
        if len(np.unique(test_targets)) > 1:  # Ensure both classes are present
            test_auc = roc_auc_score(test_targets, test_preds)
            print(f"Test Loss: {avg_test:.4f} \t Test AUC: {test_auc:.4f}")
        else:
            print(f"Test Loss: {avg_test:.4f} \t Test AUC: N/A (need both classes)")
    except ValueError as e:
        print(f"Test Loss: {avg_test:.4f} \t Test AUC: Error ({str(e)})")
    
    # Save test metrics
    test_metrics = {
        'test_loss': avg_test,
        'test_auc': test_auc if 'test_auc' in locals() else float('nan'),
        'test_predictions': test_preds,
        'test_targets': test_targets
    }
    
    with open('outputs/test_metrics.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        test_metrics['test_predictions'] = [float(p) for p in test_metrics['test_predictions']]
        test_metrics['test_targets'] = [int(t) for t in test_metrics['test_targets']]
        json.dump(test_metrics, f)
    
    return test_metrics


def visualize_training_process(metrics):
    """Visualize the final training process metrics"""
    # Create a more detailed final visualization
    plt.figure(figsize=(15, 10))
    
    # Plot losses
    plt.subplot(2, 1, 1)
    plt.plot(metrics['epochs'], metrics['train_loss'], 'bo-', label='Training Loss')
    plt.plot(metrics['epochs'], metrics['val_loss'], 'ro-', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    
    # Plot AUCs
    plt.subplot(2, 1, 2)
    plt.plot(metrics['epochs'], metrics['train_auc'], 'bo-', label='Training AUC')
    plt.plot(metrics['epochs'], metrics['val_auc'], 'ro-', label='Validation AUC')
    plt.xlabel('Epoch')
    plt.ylabel('AUC')
    plt.title('Training and Validation AUC Over Epochs')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("outputs/final_training_metrics.png")
    plt.close()


if __name__ == "__main__":
    # Create outputs directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)
    
    if device.type == "cuda":
        print("Starting training process...")
        train()
        print("\nStarting testing process...")
        test_metrics = test()
        print("\nTraining and testing completed successfully.")
        print("Results and visualizations saved in the 'outputs' directory.")
    else:
        print("No GPU available, exiting...")