""" 

Datasets & Dataloaders...

does two things:
1. complies with pytorch dataloader types
2. pairs segmentation mask targets with images

no need for labels, segmentation only.
"""

from torch.utils.data import Dataset, DataLoader, random_split
import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms

# --------------------------------------------------------------------------------------------------------
def load_image(image_path):
    """
    Loads an image from path, converts to RGB, and converts to a PyTorch tensor.
    Normalizes to [0, 1] range.
    
    Args:
        image_path: Path to the image file.
        
    Returns:
        A PyTorch tensor of shape (3, height, width)
    """
    try:
        img = Image.open(image_path).convert('RGB')
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts to tensor and scales to [0, 1]
        ])
        return transform(img)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def load_segmentation_mask(mask_path):
    """
    Loads a segmentation mask and converts it to a class index tensor.
    For binary segmentation (foreground/background), creates a tensor of shape (height, width).
    
    Args:
        mask_path: Path to the segmentation mask.
        
    Returns:
        A PyTorch tensor of shape (height, width) with class indices.
    """
    try:
        mask = Image.open(mask_path)
        # Convert mask to numpy array
        mask_array = np.array(mask)
        
        # If mask is RGB or RGBA, convert to binary (0/1)
        if len(mask_array.shape) == 3:
            # Assuming any non-zero value in any channel represents foreground
            mask_array = (mask_array.sum(axis=2) > 0).astype(np.int64)
        else:
            # For grayscale masks, threshold to convert to binary
            mask_array = (mask_array > 0).astype(np.int64)
        
        # Convert to tensor (no normalization needed for segmentation masks)
        return torch.from_numpy(mask_array)
    except Exception as e:
        print(f"Error loading mask {mask_path}: {e}")
        return None

# --------------------------------------------------------------------------------------------------------
# get images, works for both segmentations and images

def get_file_paths(directory):
    """
    Get all image file paths from a directory structure.
    
    Args:
        directory: Path to directory containing images (possibly in subdirectories)
        
    Returns:
        List of file paths to images
    """
    file_paths = []
    
    for root, _, files in os.walk(directory):
        for file in files:
            # Only include image files (add more extensions if needed)
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                file_paths.append(os.path.join(root, file))
    
    return sorted(file_paths)  # Sort to ensure consistent ordering

# --------------------------------------------------------------------------------------------------------

class CUBDataset(Dataset):
    def __init__(self, image_dir, segmentation_dir, transform=None):
        self.image_paths = get_file_paths(image_dir)
        self.segmentation_paths = get_file_paths(segmentation_dir)
        self.transform = transform
        
        # Ensure matching number of images and segmentation masks
        if len(self.image_paths) != len(self.segmentation_paths):
            raise ValueError(f"Number of images ({len(self.image_paths)}) doesn't match number of segmentations ({len(self.segmentation_paths)})")
            
        print(f"Dataset loaded with {len(self.image_paths)} image-segmentation pairs")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and convert to tensor
        image = load_image(self.image_paths[idx])
        
        # Load segmentation mask and convert to tensor
        segmentation = load_segmentation_mask(self.segmentation_paths[idx])
        
        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)
            
        return image, segmentation


# --------------------------------------------------------------------------------------------------------

# No need for a custom DataLoader class, we'll use the standard DataLoader with improved parameters

# --------------------------------------------------------------------------------------------------------

def create_train_val_test_loaders(image_dir, segmentation_dir, batch_size=32, 
                                 train_ratio=0.7, val_ratio=0.2, test_ratio=0.1,
                                 num_workers=2, pin_memory=True):
    """
    Create train, validation, and test DataLoaders with split
    
    Args:
        image_dir: Directory containing images
        segmentation_dir: Directory containing segmentation masks
        batch_size: Batch size for DataLoaders
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory for faster data transfer to GPU
        
    Returns:
        train_loader, val_loader, test_loader
    """
    # Create full dataset
    full_dataset = CUBDataset(image_dir, segmentation_dir)
    
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
    
    # Create DataLoaders with improved settings for CPU usage
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, val_loader, test_loader



# --------------------------------------------------------------------------------------------------------

"""
Example usage:

image_dir = os.path.join("CUBdata/CUB_200_2011/images/")
segmentation_dir = os.path.join("CUBdata/segmentations/")

# Create data loaders with smaller batch size for CPU training
train_loader, val_loader, test_loader = create_train_val_test_loaders(
    image_dir=image_dir,
    segmentation_dir=segmentation_dir,
    batch_size=1,  # Small batch size for CPU
    num_workers=2  # Use more workers for CPU parallelism
)

# Test loading a batch
for images, masks in train_loader:
    print(f"Image batch shape: {images.shape}")
    print(f"Mask batch shape: {masks.shape}")
    break
"""
