
import torch
import segmentation_models_pytorch as smp 
import dataloader


hyperparams = {
    "learningRate" : 0.01,
    "epochs" : 100,
}

# vanilla unet, untrained
unet = smp.Unet(
    encoder_name="resnet34",  
    encoder_weights=None,  
    in_channels=3,  
    classes=1  
)


 