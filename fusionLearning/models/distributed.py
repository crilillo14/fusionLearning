from __future__ import annotations

import os
import sys

parent_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if parent_parent_dir not in sys.path:
    sys.path.insert(0, parent_parent_dir)

from fusionLearning.models.consts import MAXEPOCHS, BATCHSIZE, MOMENTUM, LEARNING_RATE, NUM_CLASSES_CUB, NUM_CLASSES_CITYSCAPES
from fusionLearning.config import CUB, CUB_IMAGES, CUB_SEGMENTATIONS, BASE_MODELS
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
import traceback
from datetime import datetime
from tqdm import tqdm

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch.distributed import init_process_group, destroy_process_group

from fusionLearning.models.train import train_dist
from fusionLearning.models.test import test_dist
from fusionLearning.models.vis import visualize_training_process, plot_metrics
from fusionLearning.models.inference import copy_best_model_to_weights, inference_from_paths

from fusionLearning.models.ViT.vit import ViT
# debug_viz determines if logits are outputted when graphing

debug_viz = 0
# prints logits in train phase, some other verbose stuff too

DEBUG_TRAIN = False

arch_dict = {
    "UnetPlusPlus" : smp.UnetPlusPlus,
    "Unet": smp.Unet,
    "FPN": smp.FPN,
    "DeepLabV3": smp.DeepLabV3,
    "DeepLabV3Plus": smp.DeepLabV3Plus,
    "PSPNet": smp.PSPNet,
    "MAnet": smp.MAnet,
    "Linknet": smp.Linknet,
    "Segformer": smp.Segformer,
}

# Available encoders are listed [here](https://smp.readthedocs.io/en/latest/encoders.html) in SMP's documentation
# TODO: Read up on MMsegmentation

available_encoder_types = {
    "vgg" : ["vgg16" , "vgg13", "vgg11", "vgg19"],
    "resnet" : ["resnet18", "resnet34", "resnet50", "resnet101", "resnet152"],
    "se_resnet" : ["seresnet18", "seresnet34", "seresnet50", "seresnet101", "seresnet152"],
    "resnext" : ["resnext50", "resnext101"],
    "se_resnext" : ["seresnext50", "seresnext101"],
    "senet154" : ["senet154"],
    "densenet" : ["densenet121", "densenet109", "densenet201"],
    "inception" : ["inceptionv3", "inceptionresnetv2"],
    "mobilenet" : ["mobilenet", "mobilenetv2"],
    "efficientnet" : ["efficientnetb0", "efficientnetb1", "efficientnetb2", "efficientnetb3", "efficientnetb4", "efficientnetb5", "efficientnetb6", "efficientnetb7"],
}

flat_encoders = [item for sublist in available_encoder_types.values() for item in sublist]


# helpers for incompatible arch + encoder pairs. blame is usually on arch.
# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
import json

SKIP_FILE = os.path.join(BASE_MODELS, "skip_pairs.json")

def load_skip_set():
    if os.path.exists(SKIP_FILE):
        with open(SKIP_FILE) as f:
            return set(json.load(f))
    return set()

def record_failure(arch, encoder, error):
    key = f"{arch}__{encoder}"
    # append traceback to per-pair log
    err_dir = os.path.join(BASE_MODELS, "_errors")
    os.makedirs(err_dir, exist_ok=True)
    with open(f"{err_dir}/{key}.txt", "a") as f:
        import traceback
        f.write(f"\n{'='*60}\n{error}\n{traceback.format_exc()}\n")
    # add to skip list
    skip = load_skip_set()
    skip.add(key)
    with open(SKIP_FILE, "w") as f:
        json.dump(sorted(skip), f, indent=2)

# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
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
    
    init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size
    )

# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
def main(rank, world_size, arch_name, encoder, dset):
    
    # don´t know why ...
    global torch 
    
    try:
        ddp_setup(rank, world_size) 

        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        
        # not really necessary, but making sure torch and device working correctly
        warmup(device)
        
        # Available encoders are listed [here](https://smp.readthedocs.io/en/latest/encoders.html) in SMP's documentation

        MODEL = arch_dict[arch_name]

        # ––––––––––––––––––––– init model, optim, loss func ––––––––––––––––––––– 
        # TODO change to dset dependent instantiation
       
        num_classes = NUM_CLASSES_CUB if dset == "CUB" else NUM_CLASSES_CITYSCAPES

        model = MODEL(
            encoder_name=encoder,  
            encoder_weights=None,  
            in_channels=3,  
            classes=num_classes,
        ).to(device)
        
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

        optimizer = torch.optim.SGD(model.parameters(),
                                lr=LEARNING_RATE,
                                momentum=MOMENTUM)

        lossFunc = torch.nn.BCEWithLogitsLoss()

        path_images_folder = os.path.join(CUB_IMAGES)
        path_segmentations_folder = os.path.join(CUB_SEGMENTATIONS)

        training_dataloader, validation_dataloader, test_dataloader = create_train_val_test_loaders_distributed(
            path_images_folder,
            path_segmentations_folder,
            batch_size=BATCHSIZE,
            train_ratio=0.7,
            val_ratio=0.2,   
            gTransforms=geoTransforms,  
            pTransforms=photometricTransforms,
            num_workers=4
        )
        # ––––––––––––––––––––– Edit below hard links to point at model directory –––––––––––––––––––––

        # EDIT BELOW
        modelName = f"{arch_name}_{encoder}"
        modelDir = f"fusionLearning/models/results/{dset}/{modelName}/"

        trained = os.path.exists(modelDir + "weights/best_model.pth") 

        # Create outputs directory if it doesn't exist
        os.makedirs(modelDir + "metrics", exist_ok=True)
        os.makedirs(modelDir + "figures", exist_ok=True)
        os.makedirs(modelDir + "weights", exist_ok=True)
        

        if trained:
            if rank == 0: 
                print("Model already trained. To retrain, delete 'best_model.pth' or move it elsewhere.\n Going ahead with testing...")

            # Load best model -- EDIT BELOW

            model = MODEL(
                encoder_name=encoder,  
                encoder_weights=None,  
                in_channels=3,  
                classes=num_classes,
            ).to(device)
            
            model = DDP(model, device_ids=[], find_unused_parameters=True)
            
            print("Loading from path: ", modelDir + f"weights/best_model.pth")
            state_dict = torch.load(modelDir + f"weights/best_model.pth", map_location=device)
            
            model.load_state_dict(state_dict)
            model.to(device)

            test_dist(modelDir, model, test_dataloader, lossFunc, rank)
            print("\nTesting completed successfully.")
            plot_metrics(modelDir)
        else:
            if rank == 0: 
                print("Starting training process...") 


            train_dist(modelDir, model, optimizer, lossFunc, training_dataloader, validation_dataloader, rank)
            
            if rank == 0: 
                print("\nStarting testing process...")
            
            test_loss = test_dist(modelDir, model, test_dataloader, lossFunc, rank)
            if rank == 0: 
                print(f"\nFinal test loss: {test_loss:.4f}")
                print("\nTraining and testing completed lsuccessfully.")
            
                print("\t * Results and visualizations saved in the 'weights' directory. * ")

                plot_metrics(modelDir)

        
        if rank == 0: 
            inference_from_paths(model=model, modelDir=modelDir, test_dataloader=test_dataloader, n=20)
        
        copy_best_model_to_weights(modelDir)

        

    except Exception as e:
        print(e)
        
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

if __name__ == "__main__":


    import argparse
    
    parser = argparse.ArgumentParser(description='Distributed Training')
    parser.add_argument('dataset', type=str, choices=['CUB', 'Cityscapes'])
    parser.add_argument('arch_name', type=str, choices=arch_dict.keys())
    parser.add_argument('encoder', type=str, choices=flat_encoders)
    args = parser.parse_args()


    dset = args.dataset
    arch_name = args.arch_name 
    encoder = args.encoder
    
    mp.spawn(main, args=(WORLD_SIZE, arch_name, encoder, dset), nprocs=WORLD_SIZE, join=True)

# ––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

def launch_training(arch_name, encoder, dataset : str):
    world_size = WORLD_SIZE
    mp.spawn(main, args=(world_size, arch_name, encoder, dataset), nprocs=world_size, join=True)
