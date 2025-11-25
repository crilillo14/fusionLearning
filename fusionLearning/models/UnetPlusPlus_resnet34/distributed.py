

import os
import sys
parent_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if parent_parent_dir not in sys.path:
    sys.path.insert(0, parent_parent_dir)

from fusionLearning.models.consts import MAXEPOCHS, BATCHSIZE, MOMENTUM, LEARNING_RATE, NUM_CLASSES
from fusionLearning.config import CUB, CUB_IMAGES, CUB_SEGMENTATIONS
from fusionLearning.data.dataloaders import create_train_val_test_loaders_distributed
from fusionLearning.data.aug import geoTransforms, photometricTransforms
from fusionLearning.config import MASTER_ADDR, MASTER_PORT, WORLD_SIZE

import segmentation_models_pytorch as smp
import torch
import torch.distributed as dist
from torchmetrics.classification import BinaryAUROC
import matplotlib.pyplot as plt
import numpy as np


import json
import pprint
import random
import shutil
from tqdm import tqdm

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group

from fusionLearning.models.train import train_distr
from fusionLearning.models.test import test
from fusionLearning.models.vis import visualize_training_process, plot_metrics

print("On torch version:", torch.__version__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("On device:", device)


# ## Post training: visualizing prediction masks
# debug_viz determines if logits are outputted

debug_viz = 0
DEBUG_TRAIN = False

def warmup(device) -> None:

    try:
        images = torch.randn(1, 3, 352, 512, device=device)
        masks = torch.randint(0, 2, (1, 352, 512), dtype=torch.int64, device=device)
        print("Tensors created and moved to CUDA successfully")
    except RuntimeError as e:
        print("RuntimeError:", e)


def ddp_setup(rank, world_size):

    os.environ["MASTER_ADDR"] = MASTER_ADDR
    os.environ["MASTER_PORT"] = MASTER_PORT
    
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size
    )

def main(rank, world_size):
    
    # don´t know why ...
    global torch 
    
    ddp_setup(rank, world_size) 


    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    
    # not really necessary, but making sure torch and device working correctly
    warmup(device)
    


    # Declare model type and encoder architecture
    # Available encoders are listed [here](https://smp.readthedocs.io/en/latest/encoders.html) in SMP's documentation

    # TODO : Move MODEL NAME, encoder config to CLI.
    MODEL_NAME = "UnetPlusPlus"
    MODEL = smp.UnetPlusPlus
    encoder = "resnet34"

    # ––––––––––––––––––––– init model, optim, loss func ––––––––––––––––––––– 
    
    model = MODEL(
        encoder_name=encoder,  
        encoder_weights=None,  
        in_channels=3,  
        classes=NUM_CLASSES,
    ).to(device)
    
    model = DDP(model, device_ids=[rank])

    optimizer = torch.optim.SGD(model.parameters(),
                            lr=LEARNING_RATE,
                            momentum=MOMENTUM)

    lossFunc = torch.nn.BCEWithLogitsLoss()

    path_images_folder = os.path.join(CUB_IMAGES)
    path_segmentations_folder = os.path.join(CUB_SEGMENTATIONS)
    
    # os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    # os.environ["TORCH_USE_CUDA_DSA"] = "1"

    # torch.cuda.empty_cache()
    # ––––––––––––––––––––– Edit below hard links to point at model directory –––––––––––––––––––––

    # EDIT BELOW
    modelName = f"{MODEL_NAME}_{encoder}"
    modelDir = f"models/{modelName}/"

    trained = os.path.exists(modelDir + "outputs/best_model.pth") 

    # Create outputs directory if it doesn't exist
    os.makedirs(modelDir + "outputs", exist_ok=True)

    confirmation = input(f"Specified model dir : {modelDir}. Training model {modelName}. Proceed? (y/n)")
    if confirmation.lower() != "y":
        sys.exit("Exiting...")
    
    if trained:
        print("Model already trained. To retrain, delete the 'outputs/best_model.pth' file.\n Going ahead with testing...")

        # Load best model -- EDIT BELOW

        model = MODEL(
            encoder_name=encoder,  
            encoder_weights=None,  
            in_channels=3,  
            classes=NUM_CLASSES,
        ).to(device)
        
        model = DDP(model, device_ids=[])
        state_dict = torch.load(modelDir + f"outputs/best_model.pth", map_location=device)
        
        model.load_state_dict(state_dict)
        model.to(device)

        test(modelDir)
        print("\nTesting completed successfully.")
        plot_metrics(modelDir)
    else:
        if device.type == "cuda":
            print("Starting training process...")
            
            train(modelDir, model, optimizer, lossFunc, training_dataloader, validation_dataloader)
            print("\nStarting testing process...")
            
            test(modelDir, model, test_dataloader, lossFunc)
            print("\nTraining and testing completed lsuccessfully.")
            
            print("\t * Results and visualizations saved in the 'outputs' directory. * ")

            plot_metrics(modelDir)

        else:
            print("No GPU available, exiting...")


    training_dataloader, validation_dataloader, test_dataloader = create_train_val_test_loaders_distributed(
        path_images_folder,
        path_segmentations_folder,
        batch_size=BATCHSIZE,
        train_ratio=0.7,
        val_ratio=0.2,   
        gTransforms=geoTransforms,  
        pTransforms=photometricTransforms
    )
    
    inference_from_paths(model=model, modelDir=f"models/{MODEL_NAME}_{encoder}/", test_dataloader=test_dataloader, n=20)
    copy_best_model_to_weights(modelDir)



if __name__ == "__main__":
    
    # TODO: work on cli
    # parser = argparse.ArgumentParser(description='Distributed Training')
    # args = parser.parse_args()

    mp.spawn(main, args=(WORLD_SIZE,), nprocs=world_size)
    