""" 

Datasets & Dataloaders...

does two things:
1. complies with pytorch dataloader types
2. pairs segmentation mask targets with images

no need for labels, segmentation only.


1.  create_train_val_test_loaders:

    first load each image
    load_image()

    then load each mask
    load_segmentation_mask()

    then split into train, val, test
    random_split()

    then create dataloaders
    DataLoader()

"""

from torch.utils.data import Dataset, DataLoader, random_split  # TODO > split technique up for discussion (consider k fold or CV)
import os
from PIL import Image
import numpy as np
import torch
from tqdm import tqdm
from typing import Optional, Type
from torchvision.transforms import v2
import random

from data.aug import crop_to_multiple, geom_transform_pair


# --------------------------------------------------------------------------------------------------------
# GET > file paths, then files

def get_file_paths(directory : str):
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

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def load_image(image_path : str):
    """
    Loads an image from path, converts to RGB.
    
    Args:
        image_path: Path to the image file.
        
    Returns:
        PIL image and filename
    """
    try:
        img = Image.open(image_path).convert('RGB')
        return img, os.path.basename(image_path)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def load_segmentation_mask(mask_path : str):
    """
    Loads a segmentation mask preserving original grayscale values.
    """
    try:
        mask = Image.open(mask_path).convert('L')
        return mask
    except Exception as e:
        print(f"Error loading mask {mask_path}: {e}")
        return None

# --------------------------------------------------------------------------------------------------------
# Custom dataset class that implements the torch.utils.data.Dataset interface

class CUBDataset(Dataset):
    def __init__( self, 
                  image_dir : str, 
                  segmentation_dir : str, 
                  gTransforms : Optional[torch.nn.Module] = None, 
                  pTransforms : Optional[torch.nn.Module] = None,
                  ):

        # hold files by reference
        # Ensure deterministic, matching order by sorting paths
        self.image_paths = sorted(get_file_paths(image_dir))
        self.segmentation_paths = sorted(get_file_paths(segmentation_dir))

        self.use_geo = gTransforms is not None
        self.photometricTransforms = pTransforms
        
        # Ensure matching number of images and segmentation masks
        if len(self.image_paths) != len(self.segmentation_paths):
            raise ValueError(f"Number of images ({len(self.image_paths)}) doesn't match number of segmentations ({len(self.segmentation_paths)})")
            
        print(f"Dataset loaded with {len(self.image_paths)} image-segmentation pairs")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx : int):
        # Load image and segmentation mask
        image, image_filename = load_image(self.image_paths[idx])
        segmentation = load_segmentation_mask(self.segmentation_paths[idx])

        # Apply geometric transformations with consistent seeding
        if self.use_geo:
            image, segmentation = geom_transform_pair(image, segmentation)

        # Convert to tensors
        image_tensor = v2.PILToTensor()(image).float() / 255.0
    
        # Handle segmentation: remove interpolation artifacts from geometric transforms
        seg_np = np.array(segmentation)

    
        # Map mask values to binary class indices {0,1}
        # Convert grayscale values 0–255 to probabilities 0–1
        seg_prob = seg_np.astype(np.float32) / 255.0
        segmentation_tensor = torch.as_tensor(seg_prob, dtype=torch.float32).unsqueeze(0)

        # Apply photometric transforms only to image
        if self.photometricTransforms:
            image_tensor = self.photometricTransforms(image_tensor)

        # print(image_tensor.shape)
        # print(segmentation_tensor.shape)
        # assert image_tensor.ndim == 3  # [C, H, W]
        # assert segmentation_tensor.ndim == 2  # [H, W]

        return image_tensor, segmentation_tensor, image_filename


# -------------------------------------------------------------------------------------------------------------------------------
# Vanilla CUB Dataset - no transforms apart from padding to a multiple of 32
class vanillaCUBDataset(Dataset):
    def __init__( self, 
                  image_dir : str, 
                  segmentation_dir : str, 
                  ):

        self.image_paths = sorted(get_file_paths(image_dir))
        self.segmentation_paths = sorted(get_file_paths(segmentation_dir))

        self.geometricTransforms = v2.Compose([
            crop_to_multiple,
        ])
        
        if len(self.image_paths) != len(self.segmentation_paths):
            raise ValueError(f"Number of images ({len(self.image_paths)}) doesn't match number of segmentations ({len(self.segmentation_paths)})")
            
        print(f"Dataset loaded with {len(self.image_paths)} image-segmentation pairs")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx : int):
        # Load image and segmentation mask
        image, image_filename = load_image(self.image_paths[idx])
        segmentation = load_segmentation_mask(self.segmentation_paths[idx])

        # crop to multiple of 32
        image = self.geometricTransforms(image)
        segmentation = self.geometricTransforms(segmentation)

        # Convert to tensors
        image_tensor = v2.PILToTensor()(image).float() / 255.0
        # Handle segmentation: remove interpolation artifacts from geometric transforms
        seg_np = np.array(segmentation)
        seg_np = np.round(seg_np).astype(np.uint8)
    
        # Map mask values to binary class indices {0,1}
        # Convert grayscale values 0–255 to probabilities 0–1
        seg_prob = seg_np.astype(np.float32) / 255.0
        segmentation_tensor = torch.as_tensor(seg_prob, dtype=torch.float32).unsqueeze(0)
        
        # assert image_tensor.ndim == 3  # [C, H, W]
        # assert segmentation_tensor.ndim == 2  # [H, W]

        return image_tensor, segmentation_tensor, image_filename


# -------------------------------------------------------------------------------------------------------------------------------

def create_train_val_test_loaders(image_dir : str, 
                                  segmentation_dir : str, 
                                  batch_size : int = 1, 
                                  train_ratio : float = 0.7, 
                                  val_ratio : float = 0.2, 
                                  gTransforms : Optional[torch.nn.Module] = None, 
                                  pTransforms : Optional[torch.nn.Module] = None):
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





    #        $ pass transform to Dataset $
    full_dataset : CUBDataset = CUBDataset(image_dir, 
                                          segmentation_dir, 
                                          gTransforms=gTransforms, 
                                          pTransforms=pTransforms)
    
    # Calculate split sizes
    total_size : int = len(full_dataset)
    train_size : int = int(train_ratio * total_size)
    val_size : int = int(val_ratio * total_size)
    test_size : int = total_size - train_size - val_size
    
    # Set a fixed seed for reproducibility  (optional)
    # Note: ensures that the split is consistent across runs.
    generator : torch.Generator = torch.Generator().manual_seed(42)
    
    # Split dataset
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )
    
    # Create DataLoaders
    train_loader : DataLoader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    val_loader : DataLoader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    test_loader : DataLoader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )
    
    return train_loader, val_loader, test_loader




# -------------------------------------------------------------------------------------------------------------------------------


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
def generateSegmentationMasks( DATASET : Type[CUBDataset | vanillaCUBDataset],
                               model : torch.nn.Module, 
                               model_weights_path : str, 
                               image_dir : str,
                               segmentation_dir : str,
                               model_name : str, 
                               batch_size : int = 1, 
                               save_dir : str = "images/segmentations"): 


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model.load_state_dict(torch.load(model_weights_path))
    model.to(device)
    model.eval()
    
    allDataloader = DataLoader(
        DATASET(image_dir, segmentation_dir),
        batch_size=batch_size,
        shuffle=False
    )

    if os.path.exists(os.path.join(save_dir, model_name)):
        print(f"Segmentation masks for {model_name} already exist, skipping.")
        return

    os.makedirs(os.path.join(save_dir, model_name), exist_ok=True)

    for images, _, filename in tqdm(allDataloader, desc="Generating segmentation masks", leave=False, ncols=80):
        images = images.to(device)
        
        with torch.no_grad():
            logits = model(images)
            
            # Convert logits to continuous probability mask (0–255 grayscale)
            prob_mask = torch.sigmoid(logits).squeeze(1)  # shape [B, H, W]
            segmentation_mask = (prob_mask.squeeze().cpu().numpy() * 255).astype(np.uint8)
            img = Image.fromarray(segmentation_mask)

            filename = filename[0] #    1 len tuple ???


            img.save(os.path.join(save_dir, model_name, filename))
    
        # print(f"Saved segmentation mask for {filename}")
    
    print(f"Generated all segmentation masks for {model_name}, saved to ./{os.path.join(save_dir, model_name)}")
