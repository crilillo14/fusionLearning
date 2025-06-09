""" 

Datasets & Dataloaders...

does two things:
1. complies with pytorch dataloader types
2. pairs segmentation mask targets with images

no need for labels, segmentation only.


for create_train_val_test_loaders:

    first load each image
    load_image()

    then load each mask
    load_segmentation_mask()

    then split into train, val, test
    random_split()

    then create dataloaders
    DataLoader()




"""

from torch.utils.data import Dataset, DataLoader, random_split  # split technique up for discussion
import os
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

# --------------------------------------------------------------------------------------------------------

# !!!! returns a pytorch tensor.

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
        return img, os.path.basename(image_path)
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
            mask_array = (mask_array.sum(axis=2) > 0).astype(np.int64)
        else:
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
            if file.endswith(('.png', '.jpg', '.jpeg', '.JPG', '.JPEG', '.PNG')):
                file_paths.append(os.path.join(root, file))
    
    return file_paths

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
        image, image_filename = load_image(self.image_paths[idx])
        
        # Load segmentation mask and convert to tensor
        segmentation = load_segmentation_mask(self.segmentation_paths[idx])
        
        # Apply transformations if specified
        if self.transform:
            image = self.transform(image)
            
        return image, segmentation, image_filename




# --------------------------------------------------------------------------------------------------------

def create_train_val_test_loaders(image_dir, segmentation_dir, batch_size=1, 
                                 train_ratio=0.7, val_ratio=0.2, test_ratio=0.1, transform=None):
    """
    Create train, validation, and test DataLoaders with split
    
    Args:
        image_dir: Directory containing images
        segmentation_dir: Directory containing segmentation masks
        batch_size: Batch size for DataLoaders
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        
    Returns:
        train_loader, val_loader, test_loader
    """





    #           $ pass transform to Dataset $
    full_dataset = CUBDataset(image_dir, segmentation_dir, transform)
    
    # Calculate split sizes
    total_size = len(full_dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size
    
    # Set a fixed seed for reproducibility  (optional)
    # Note: ensures that the split is consistent across runs.
    generator = torch.Generator().manual_seed(42)
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )
    
    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader



# --------------------------------------------------------------------------------------------------------


    """ Generate segmentation masks for all images in the dataset, placing them under images/segmentations.
    
    Args:
        model: The model to use for segmentation.
        model_weights_path: Path to the model weights.
        model_name: Name of the model.
        image_dir: Directory containing the images.
        segmentation_dir: Directory containing the segmentation masks.
        batch_size: Batch size for the dataloader.
        save_dir: Directory to save the segmentation masks.
        
    Returns:
        None
    """
def generateSegmentationMasks( model : torch.nn.Module, 
                               model_weights_path : str, 
                               model_name : str, 
                               image_dir : str, 
                               segmentation_dir : str, 
                               batch_size : int = 1, 
                               save_dir : str = "images/segmentations"):


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model.load_state_dict(torch.load(model_weights_path))
    model.to(device)
    model.eval()
    
    allDataloader = DataLoader(
        CUBDataset(image_dir, segmentation_dir),
        batch_size=batch_size,
        shuffle=False
    )

    os.makedirs(os.path.join(save_dir, model_name), exist_ok=True)

    for images, _, filename in tqdm(allDataloader, desc="Generating segmentation masks", leave=False, ncols=80):
        images = images.to(device)
        
        with torch.no_grad():
            logits = model(images)
            
            # From the logits, construct a png, save it using Pillow
            segmentation_mask = torch.argmax(logits, dim=1).squeeze().cpu().numpy()
            segmentation_mask = (segmentation_mask * 255).astype(np.uint8)
            img = Image.fromarray(segmentation_mask)

            filename = filename[0] #    1 len tuple ???


            img.save(os.path.join(save_dir, model_name, filename))
    
        # print(f"Saved segmentation mask for {filename}")
    
    print(f"Generated all segmentation masks for {model_name}, saved to ./{os.path.join(save_dir, model_name)}")
