"""
does two things:
1. complies with pytorch dataloader types
2. pairs segmentation mask targets with images

no need for labels.
"""

from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms

"""
PyTorch Dataset for segmentation with UNet model
Handles image-mask pairs and provides data in the format expected by UNet
"""

class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        """
        Args:
            image_dir (string): Directory with all the images.
            mask_dir (string): Directory with all the masks.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        
        # Get all image filenames
        self.image_files = []
        for subdir in os.listdir(image_dir):
            subdir_path = os.path.join(image_dir, subdir)
            if os.path.isdir(subdir_path):
                for file in os.listdir(subdir_path):
                    if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_files.append(os.path.join(subdir, file))
        
        # Default transforms if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get full paths
        img_name = os.path.join(self.image_dir, self.image_files[idx])
        mask_name = os.path.join(self.mask_dir, self.image_files[idx])
        
        # Load image and mask
        image = Image.open(img_name).convert('RGB')
        mask = Image.open(mask_name).convert('L')  # Convert to grayscale
        
        # Apply transformations
        image = self.transform(image)
        mask = self.transform(mask)
        
        # Ensure mask is single channel
        mask = mask[0, :, :].unsqueeze(0)
        
        return image, mask

def image_to_3channel_array(image_path):
    """
    Converts an image to a 3-channel 2D array (height, width, 3).

    Args:
        image_path: Path to the image file.

    Returns:
        A NumPy array representing the image with shape (height, width, 3), or None if an error occurs.
    """
    try:
        img = Image.open(image_path)
        img_array = np.array(img)

        if len(img_array.shape) == 2:  # Grayscale image
            # Convert to RGB by duplicating the channel
            img_array_rgb = np.stack([img_array, img_array, img_array], axis=-1)
            return img_array_rgb
        elif len(img_array.shape) == 3 and img_array.shape[2] == 4: #RGBA image
             img_array_rgb = img_array[:,:,:3]
             return img_array_rgb
        elif len(img_array.shape) == 3 and img_array.shape[2] == 3: #RGB image
            return img_array
        else:
            print("Unsupported image format.")
            return None
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except Exception as e:
         print(f"An error occurred: {e}")
         return None

# get images

def getImages(path, arr):    
    for subdir in os.scandir(path):
        if subdir.is_dir():
            for image_path in os.scandir(subdir):
                    i = image_to_3channel_array(image_path=image_path)
                    print(i)
                    arr.append(i)
        
    return arr
            

pimages = os.path.join("CUBdata/CUB_200_2011/images/")
psegmentations = os.path.join("CUBdata/segmentations/")
images = getImages( pimages , [])
segmentations = getImages( psegmentations , [])

print(len(segmentations))
print(len(images))

# Example usage:
# from dataloader import SegmentationDataset
# from torch.utils.data import DataLoader
# 
# # Create dataset
# dataset = SegmentationDataset(
#     image_dir='CUBdata/CUB_200_2011/images',
#     mask_dir='CUBdata/segmentations'
# )
# 
# # Create dataloader
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
