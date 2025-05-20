
"""
Initializing an empty-weighted Unet and training it on the CUB dataset.
Serves as a baseline model for the
"""


import torch
import segmentation_models_pytorch as smp
from baseModels.dataloaders import create_train_val_test_loaders
import os
import time
import matplotlib.pyplot as plt

# Training parameters
MAXEPOCHS = 10
BATCHSIZE = 4     # Slightly larger batch size for better training
MOMENTUM = 0.99
LEARNING_RATE = 0.01
NUM_CLASSES = 2
SAVE_INTERVAL = 5  # Save checkpoints every N epochs


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create UNet model
unet = smp.Unet(
    encoder_name="resnet34",  # Use a lighter backbone for CPU training
    encoder_weights=None,     # Start with random weights
    in_channels=3,           # RGB images have 3 channels
    classes=NUM_CLASSES,     # Binary segmentation (background/foreground)
)
unet.to(device)

# Define optimizer with SGD
optimizer = torch.optim.SGD(unet.parameters(),
                           lr=LEARNING_RATE,
                           momentum=MOMENTUM)

# Define loss function for segmentation
lossFunc = torch.nn.CrossEntropyLoss()

# ----------------------------------------------------------------------------------------------------
path_images_folder = os.path.join("CUBdata/CUB_200_2011/images/")
path_segmentations_folder = os.path.join("CUBdata/segmentations/")

# Create data loaders with parameters optimized for CPU training
training_dataloader, validation_dataloader, testing_dataloader = create_train_val_test_loaders(
    image_dir=path_images_folder, 
    segmentation_dir=path_segmentations_folder,
    batch_size=BATCHSIZE,
    num_workers=2,  # Use multiple workers for data loading
    pin_memory=True  # This helps with data transfer to device
)

# ----------------------------------------------------------------------------------------------------

def train(num_epochs=MAXEPOCHS, save_model=True):
    """Train the UNet model
    
    Args:
        num_epochs: Number of epochs to train
        save_model: Whether to save the model after training
        
    Returns:
        Model training history (train_losses, val_losses)
    """
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"Training on {device}")
    print(f"Number of training samples: {len(training_dataloader.dataset)}")
    print(f"Number of validation samples: {len(validation_dataloader.dataset)}")
    print("-" * 50)
    
    for epoch in range(1, num_epochs + 1):
        # Training phase
        unet.train()
        train_loss = 0.0
        
        for batch_idx, (images, masks) in enumerate(training_dataloader):
            # Move data to device
            images = images.to(device)
            masks = masks.to(device).long()
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = unet(images)
            loss = lossFunc(outputs, masks)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track loss
            train_loss += loss.item()
            
            # Print progress every 10 batches
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch}/{num_epochs}], Batch [{batch_idx+1}/{len(training_dataloader)}], Loss: {loss.item():.4f}")
        
        # Calculate average training loss for this epoch
        avg_train_loss = train_loss / len(training_dataloader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        unet.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, masks in validation_dataloader:
                images = images.to(device)
                masks = masks.to(device).long()
                
                outputs = unet(images)
                loss = lossFunc(outputs, masks)
                val_loss += loss.item()
        
        # Calculate average validation loss
        avg_val_loss = val_loss / len(validation_dataloader)
        val_losses.append(avg_val_loss)
        
        # Print epoch stats
        print(f"Epoch {epoch}/{num_epochs}:\t Training Loss: {avg_train_loss:.4f}\t Validation Loss: {avg_val_loss:.4f}")
        print("-" * 50)
        
        # Save model if it's the best so far
        if avg_val_loss < best_val_loss and save_model:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': unet.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, 'unet_best_model.pth')
            print(f"Model saved at epoch {epoch} with validation loss: {avg_val_loss:.4f}")
    
    print("Training completed!")
    
    if save_model:
        # Save final model
        torch.save({
            'epoch': num_epochs,
            'model_state_dict': unet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': train_losses[-1],
            'val_loss': val_losses[-1],
        }, 'unet_final_model.pth')
        print("Final model saved as 'unet_final_model.pth'")
    
    return train_losses, val_losses


def evaluate(dataloader, model):
    """Evaluate the model on a given dataloader
    
    Args:
        dataloader: DataLoader to evaluate on
        model: Model to evaluate
        
    Returns:
        avg_loss, accuracy, iou_score
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_pixels = 0
    total_iou = 0.0
    num_samples = 0
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device).long()
            
            # Forward pass
            outputs = model(images)
            loss = lossFunc(outputs, masks)
            
            # Track metrics
            total_loss += loss.item()
            
            # Convert outputs to predictions
            preds = torch.argmax(outputs, dim=1)
            
            # Calculate accuracy
            correct = (preds == masks).sum().item()
            total_correct += correct
            total_pixels += masks.numel()
            
            # Calculate IoU (Intersection over Union)
            intersection = ((preds == 1) & (masks == 1)).sum().item()
            union = ((preds == 1) | (masks == 1)).sum().item()
            if union > 0:
                iou = intersection / union
                total_iou += iou
                num_samples += 1
    
    # Calculate averages
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_pixels if total_pixels > 0 else 0
    avg_iou = total_iou / num_samples if num_samples > 0 else 0
    
    return avg_loss, accuracy, avg_iou


def plot_training_history(train_losses, val_losses):
    """Plot training and validation loss history"""
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_history.png')
    plt.close()
    print("Training history plot saved as 'training_history.png'")


def predict_and_visualize(image, mask, model):
    """Make prediction and visualize results"""
    model.eval()
    with torch.no_grad():
        image = image.unsqueeze(0).to(device)  # Add batch dimension
        output = model(image)
        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
    
    # Convert tensors to numpy arrays for visualization
    image = image.squeeze().cpu().permute(1, 2, 0).numpy()  # Change from (C,H,W) to (H,W,C)
    mask = mask.cpu().numpy()
    
    # Plot results
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(image)
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    
    axs[1].imshow(mask, cmap='gray')
    axs[1].set_title('True Mask')
    axs[1].axis('off')
    
    axs[2].imshow(pred, cmap='gray')
    axs[2].set_title('Predicted Mask')
    axs[2].axis('off')
    
    plt.savefig('prediction_example.png')
    plt.close()
    print("Prediction visualization saved as 'prediction_example.png'")


if __name__ == "__main__":
    # Train the model
    start_time = time.time()
    train_losses, val_losses = train(num_epochs=MAXEPOCHS)
    training_time = time.time() - start_time
    
    print(f"Total training time: {training_time/60:.2f} minutes")
    
    # Plot training history
    plot_training_history(train_losses, val_losses)
    
    # Evaluate on test set
    test_loss, test_accuracy, test_iou = evaluate(testing_dataloader, unet)
    print(f"\nTest Results:\n"
          f"Loss: {test_loss:.4f}\n"
          f"Accuracy: {test_accuracy:.4f}\n"
          f"IoU Score: {test_iou:.4f}")
    
    # Visualize a prediction example
    # Get a sample from the test set
    for images, masks in testing_dataloader:
        predict_and_visualize(images[0], masks[0], unet)
        break  # Just use the first sample