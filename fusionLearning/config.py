import os
import torch

# Base directory of the repository (one level above the 'fusionLearning' package)
MODULE_DIR = os.path.abspath(os.path.dirname(__file__))
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Data directories
DATA_DIR = os.path.join(MODULE_DIR, "data")
CUB = os.path.join(DATA_DIR, "CUBdata")
CUB_IMAGES = os.path.join(CUB, "CUB_200_2011", "images")
CUB_SEGMENTATIONS = os.path.join(CUB, "segmentations")

CITYSCAPES_ROOT = os.path.join(DATA_DIR, "cityscapes")
ADE20K_ROOT = os.path.join(DATA_DIR, "ADE20K")
VOC_ROOT = os.path.join(DATA_DIR, "VOC")

TOMPEI_CMMD = os.path.join(DATA_DIR, "TOMPEI-CMMD")
TOMPEI_CMMD_TASK = os.path.join(TOMPEI_CMMD, "Task_classification")
TOMPEI_CMMD_TRAIN = os.path.join(TOMPEI_CMMD_TASK, "train")
TOMPEI_CMMD_TRAIN_LABEL = os.path.join(TOMPEI_CMMD_TASK, "train_label")
TOMPEI_CMMD_VAL = os.path.join(TOMPEI_CMMD_TASK, "val")
TOMPEI_CMMD_VAL_LABEL = os.path.join(TOMPEI_CMMD_TASK, "val_label")
TOMPEI_CMMD_TEST = os.path.join(TOMPEI_CMMD_TASK, "test")
TOMPEI_CMMD_TEST_LABEL = os.path.join(TOMPEI_CMMD_TASK, "test_label")

# Model, weights and output directories
LEARNED_WEIGHTS = os.path.join(MODULE_DIR, "weights")
BASE_MODELS = os.path.join(MODULE_DIR, "models")
FUSION_MODELS = os.path.join(MODULE_DIR, "ensembles")

# Where the generated segmentation masks will be stored
BASE_MODELS_SEGMENTATIONS = os.path.join(BASE_DIR, "images", "segmentations")
FUSED_MODEL_SEGMENTATIONS = os.path.join(BASE_DIR, "images", "fused_segmentations")

# Sang Machine
MASTER_ADDR = "localhost"
MASTER_PORT = "12355" # arbitrary port
WORLD_SIZE = torch.cuda.device_count() # should be 4, variable if ported to another GPU cluster

if __name__ == '__main__':
    print("Base directory : ", BASE_DIR)
    print("Module directory: ", MODULE_DIR)
    print("Data directory: ", DATA_DIR)
    print("CUB directory: ", CUB)
    print("CUB images directory: ", CUB_IMAGES)
    print("CUB segmentations directory: ", CUB_SEGMENTATIONS)
    print("Learned weights directory: ", LEARNED_WEIGHTS)
    print("Base models directory: ", BASE_MODELS)
    print("Fusion models directory: ", FUSION_MODELS)
    print("Base models segmentations directory: ", BASE_MODELS_SEGMENTATIONS)
    print("Fused models segmentations directory: ", FUSED_MODEL_SEGMENTATIONS)
    print("Master address: ", MASTER_ADDR)
    print("Master port: ", MASTER_PORT)
    print("Number of GPUs: ", WORLD_SIZE)