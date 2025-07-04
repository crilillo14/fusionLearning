

import os
import sys
parent_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if parent_parent_dir not in sys.path:
    sys.path.insert(0, parent_parent_dir)

from fusionLearning.config import BASE_MODELS_SEGMENTATIONS, CUB_IMAGES, CUB_SEGMENTATIONS, BASE_MODELS
from fusionLearning.data.dataloaders import vanillaCUBDataset, generateSegmentationMasks
from fusionLearning.models.consts import NUM_CLASSES

from segmentation_models_pytorch import Unet, UnetPlusPlus, Linknet, FPN

from tqdm import tqdm





arch_encoder_pairings = { 
    Unet : ["resnet34", "resnet18"],
    UnetPlusPlus : ["resnet34", "resnet18"],
    Linknet : ["resnet18"],
    FPN : ["resnet34"],
}

def archToString(arch):
    if arch is Unet:
        return 'Unet'
    elif arch is UnetPlusPlus:
        return 'UnetPlusPlus'
    elif arch is Linknet:
        return 'Linknet'
    elif arch is FPN:
        return 'FPN'
    else:
        raise ValueError(f"Unknown architecture: {arch}")


def GENERATE_ALL_PREDICTIONS():
    for arch, encoders in arch_encoder_pairings.items(): 
        for encoder in encoders:
            model = arch(
                encoder_name=encoder,
                encoder_weights=None,
                in_channels=3,
                classes=NUM_CLASSES,
            )

            archstr = archToString(arch)
            model_weights_path = os.path.join(BASE_MODELS, f"{archstr}_{encoder}", "outputs", "best_model.pth")
            print(f"Model weights expected at: {model_weights_path} (exists={os.path.exists(model_weights_path)})")

            generateSegmentationMasks(
                DATASET = vanillaCUBDataset,
                model=model,
                model_weights_path=model_weights_path,
                image_dir=CUB_IMAGES,
                segmentation_dir=CUB_SEGMENTATIONS,
                model_name=f"{archstr}_{encoder}",
                batch_size=1,
                save_dir=BASE_MODELS_SEGMENTATIONS,
            )
    
    
def debug_print_paths():
    print("=== Path Debug ===")
    paths = {
        "CUB_IMAGES": CUB_IMAGES,
        "CUB_SEGMENTATIONS": CUB_SEGMENTATIONS,
        "BASE_MODELS_SEGMENTATIONS": BASE_MODELS_SEGMENTATIONS,
        "BASE_MODELS": BASE_MODELS,
    }
    for name, path in paths.items():
        print(f"{name}: {path} (exists={os.path.exists(path)})")
    print("==================")


def main(): 
    debug_print_paths()
    print("Going ahead with generating all masks...")
    GENERATE_ALL_PREDICTIONS()
    print("Generation complete.")
        

if __name__ == "__main__":
    main()