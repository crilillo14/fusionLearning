from dataloaders import CUBDataset
from torch.utils.data import Dataset
from torchvision.transforms import v2

from config import BASE_MODELS_SEGMENTATIONS
import os

from dataloaders import get_file_paths, load_segmentation_mask
from aug import pad_to_multiple



# --------------------------------------------------------------------------------------------------------
# LOADING FILEPATHS


def get_model_segmentation_paths(model_prediction_dir : str):
    file_paths = []
    
    for root, _, files in os.walk(model_prediction_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg', '.JPG', '.JPEG', '.PNG')):
                file_paths.append(os.path.join(root, file))
    return sorted(file_paths)

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
        pred_filenames = []
        for pred_path in pred_paths:
            pred = Image.open(pred_path).convert('RGB')
            preds.append(pred)
            pred_filenames.append(os.path.basename(pred_path))
        return preds, pred_filenames
    except Exception as e:
        print(f"Error loading prediction mask: {e}")
        return None



# --------------------------------------------------------------------------------------------------------

class FusionDataset(Dataset):
    def __init__(self, 
                 model_names : list[str],
                 segmentation_dir : str):
        super().__init__()

        self.model_names : list[str] = sorted(model_names)
        self.segmentation_paths : list[str] = sorted(get_file_paths(segmentation_dir)) 

        self.padding = v2.Compose([
            pad_to_multiple,
        ])

        self.model_segmentations_paths : list[list[str]] = []
        self._get_all_prediction_paths()


    def _get_all_prediction_paths(self):
        for model_name in self.model_names:
            # find the path to the images/segmentations for each model

            mdir = os.path.join(BASE_MODELS_SEGMENTATIONS, model_name)
            
            self.model_segmentations_paths.append(get_model_segmentation_paths(mdir))


            
            
            


    def __len__(self):
        return len(self.segmentation_paths)

    def __getitem__(self, idx : int):
        predictions_paths : list[str] = [(self.model_segmentations_paths[i][idx]) for i in range(len(self.model_segmentations_paths))]
        true_mask_path : str = self.segmentation_paths[idx]

        preds, pred_filenames = load_preds(predictions_paths) 
        mask = load_segmentation_mask(true_mask_path)

        for pred in preds:
            pred = self.padding(pred)

        mask = self.padding(mask)

        return preds, mask, pred_filenames

        
        

