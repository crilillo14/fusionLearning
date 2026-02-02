# prior to imports change cwd to parent directory
import os
import sys

parent_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if parent_parent_dir not in sys.path:
    sys.path.insert(0, parent_parent_dir)

# import configs
from fusionLearning.config import BASE_MODELS_SEGMENTATIONS, CUB_IMAGES, CUB_SEGMENTATIONS, BASE_MODELS
from fusionLearning.models.consts import NUM_CLASSES
# dataloading
from fusionLearning.data.dataloaders import vanillaCUBDataset, generateSegmentationMasks
from tqdm import tqdm
# model classes 
from segmentation_models_pytorch import Unet, UnetPlusPlus, Linknet, FPN

import fusionLearning.models.train as train

# match every available model type with every available encoder

encoders : list[str] = []
archs : list[type] = []

        
arch_encoder_pairings = { 

}

# cross match every available encoder with every available model type
for arch in archs: 
    for encoder in encoders:
        arch_encoder_pairings[arch] = encoder
        

for arch, encoder in arch_encoder_pairings.items():
    result = train
