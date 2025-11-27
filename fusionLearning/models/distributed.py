

import os
import sys

parent_parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if parent_parent_dir not in sys.path:
    sys.path.insert(0, parent_parent_dir)

from fusionLearning.models.consts import MAXEPOCHS, BATCHSIZE, MOMENTUM, LEARNING_RATE, NUM_CLASSES_CUB, NUM_CLASSES_CITYSCAPES
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

from fusionLearning.models.train import train_dist
from fusionLearning.models.test import test_dist
from fusionLearning.models.vis import visualize_training_process, plot_metrics
from fusionLearning.models.inference import copy_best_model_to_weights, inference_from_paths


# debug_viz determines if logits are outputted when graphing
debug_viz = 0
# prints logits in train phase, some other verbose stuff too
DEBUG_TRAIN = False

arch_dict = {
    "UnetPlusPlus" : smp.UnetPlusPlus,
    "Unet": smp.Unet,
    "FPN": smp.FPN,
    "PSPNet": smp.PSPNet,
    "DeepLabV3": smp.DeepLabV3,
    "DeepLabV3Plus": smp.DeepLabV3Plus,
    "MAnet": smp.MAnet,
    "Linknet": smp.Linknet,
    "Segformer": smp.Segformer,
}

# Available encoders are listed [here](https://smp.readthedocs.io/en/latest/encoders.html) in SMP's documentation

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
        
        model = DDP(model, device_ids=[rank])

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

        trained = os.path.exists(modelDir + "outputs/best_model.pth") 

        # Create outputs directory if it doesn't exist
        os.makedirs(modelDir + "outputs", exist_ok=True)

        if trained:
            if rank == 0: 
                print("Model already trained. To retrain, delete 'outputs/best_model.pth' or move it elsewhere.\n Going ahead with testing...")

            # Load best model -- EDIT BELOW

            model = MODEL(
                encoder_name=encoder,  
                encoder_weights=None,  
                in_channels=3,  
                classes=num_classes,
            ).to(device)
            
            model = DDP(model, device_ids=[])
            
            print("Loading from path: ", modelDir + f"outputs/best_model.pth")
            state_dict = torch.load(modelDir + f"outputs/best_model.pth", map_location=device)
            
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
            
                print("\t * Results and visualizations saved in the 'outputs' directory. * ")

                plot_metrics(modelDir)

        
        if rank == 0: 
            inference_from_paths(model=model, modelDir=modelDir, test_dataloader=test_dataloader, n=20)
        
        copy_best_model_to_weights(modelDir)

        

    except Exception as e:
        print(e)
        raise
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":

    dset = sys.argv[1]
    arch_name = sys.argv[2] 
    encoder = sys.argv[3]
    
    # TODO: work on cli
    # parser = argparse.ArgumentParser(description='Distributed Training')
    # args = parser.parse_args()

    mp.spawn(main, args=(WORLD_SIZE, arch_name, encoder, dset), nprocs=WORLD_SIZE, join=True)

def launch_training(arch_name, encoder):
    world_size = WORLD_SIZE
    mp.spawn(main, args=(world_size, arch_name, encoder), nprocs=world_size, join=True)
