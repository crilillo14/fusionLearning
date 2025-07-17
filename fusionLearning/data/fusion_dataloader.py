from torch.utils.data import Dataset
from torchvision.transforms import v2
from PIL import Image
import torch


from config import BASE_MODELS_SEGMENTATIONS
import os

from data.dataloaders import get_file_paths, load_segmentation_mask
from data.aug import crop_to_multiple





# --------------------------------------------------------------------------------------------------------

def load_preds(pred_paths : list[str]):
    """
    Loads a list of prediction masks from paths.
    
    Args:
        pred_paths: List of paths to prediction masks from base models
        
    Returns:
        List of PIL images
    """
    try:
        preds = []
        pred_filenames : list[str] = []
        for pred_path in pred_paths:
            pred = Image.open(pred_path).convert('RGB')
            preds.append(pred)
            pred_filenames.append(os.path.basename(pred_path))
        return preds, pred_filenames
    except Exception as e:
        print(f"Error loading prediction mask: {e}")
        return None, None
        
# --------------------------------------------------------------------------------------------------------
# LOADING FILEPATHS


def get_model_segmentation_paths(model_prediction_dir: str) -> tuple[list[str], list[str]]:
    """Get all image file paths from a model's prediction directory.
    
    Args:
        model_prediction_dir: Directory containing model's prediction images
        
    Returns:
        tuple: (sorted_paths, sorted_filenames) where:
            - sorted_paths: List of absolute file paths, sorted by filename
            - sorted_filenames: List of corresponding filenames, sorted
    """
    file_paths = []
    filenames = []
    
    for root, _, files in os.walk(model_prediction_dir):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_paths.append(os.path.join(root, file))
                filenames.append(file)
    
    # Sort both lists by filename
    sorted_pairs = sorted(zip(filenames, file_paths))
    sorted_filenames = [pair[0] for pair in sorted_pairs]
    sorted_paths = [pair[1] for pair in sorted_pairs]
    
    return sorted_paths, sorted_filenames


# --------------------------------------------------------------------------------------------------------

class FusionDataset(Dataset):
    def __init__(self, 
                 model_names: list[str],
                 segmentation_dir: str):
        super().__init__()

        self.model_names: list[str] = model_names
        
        # Get and sort segmentation paths and filenames
        seg_paths = get_file_paths(segmentation_dir)
        seg_filenames = [os.path.basename(p) for p in seg_paths]
        
        # Sort both lists by filename
        sorted_pairs = sorted(zip(seg_filenames, seg_paths))
        self.segmentation_filenames = [pair[0] for pair in sorted_pairs]
        self.segmentation_paths = [pair[1] for pair in sorted_pairs]
        
        self.crop_to_multiple = v2.Compose([
            crop_to_multiple,
        ])

        self.model_segmentations_paths: list[list[str]] = []
        self.model_segmentations_filenames: list[list[str]] = []
        self._get_all_prediction_paths()

    def _get_all_prediction_paths(self):
        # Get base filenames from ground truth for validation
        # gt_filenames = set(self.segmentation_filenames)
        
        for model_name in self.model_names:
            mdir = os.path.join(BASE_MODELS_SEGMENTATIONS, model_name)
            model_paths, model_filenames = get_model_segmentation_paths(mdir)
            """
            # Verify that all ground truth files have corresponding predictions
            missing = gt_filenames - set(model_filenames)
            if missing:
                raise ValueError(f"Model {model_name} is missing predictions for files: {missing}")
            """
            self.model_segmentations_paths.append(model_paths)
            self.model_segmentations_filenames.append(model_filenames)


    def __len__(self):
        return len(self.segmentation_paths)

    def __getitem__(self, idx: int):
        # Get paths and filenames for this index
        predictions_paths = [self.model_segmentations_paths[i][idx] 
                          for i in range(len(self.model_segmentations_paths))]
        prediction_filenames = [self.model_segmentations_filenames[i][idx] 
                              for i in range(len(self.model_segmentations_filenames))]
        true_mask_path = self.segmentation_paths[idx]
        true_mask_filename = self.segmentation_filenames[idx]

        # Verify filenames match
        # if not all(f == true_mask_filename for f in prediction_filenames):
        #     raise ValueError("Filename mismatch between predictions and ground truth")

        preds, _ = load_preds(predictions_paths) 
        mask = load_segmentation_mask(true_mask_path)
        
        pred_tensors = []
        for pred in preds:   # transform
            pred = self.crop_to_multiple(pred)
            pred_tensor = v2.functional.to_tensor(pred)
            pred_tensors.append(pred_tensor)
            
        # Convert mask to tensor and normalize
        mask = self.crop_to_multiple(mask)
        mask_tensor = v2.functional.to_tensor(mask)
        
        # Stack predictions along a new dimension
        preds_tensor = torch.stack(pred_tensors, dim=0)  # Shape: [num_models, C, H, W]
        
        return preds_tensor, mask_tensor, true_mask_filename


        
        

