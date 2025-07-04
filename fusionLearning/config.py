



import os

# Base directory of the repository (one level above the 'fusionLearning' package)
MODULE_DIR = os.path.abspath(os.path.dirname(__file__))
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Data directories
DATA_DIR = os.path.join(MODULE_DIR, "data")
CUB = os.path.join(DATA_DIR, "CUBdata")
CUB_IMAGES = os.path.join(CUB, "CUB_200_2011", "images")
CUB_SEGMENTATIONS = os.path.join(CUB, "segmentations")

# Model, weights and output directories
LEARNED_WEIGHTS = os.path.join(MODULE_DIR, "weights")
BASE_MODELS = os.path.join(MODULE_DIR, "models")
FUSION_MODELS = os.path.join(MODULE_DIR, "ensembles")

# Where the generated segmentation masks will be stored
BASE_MODELS_SEGMENTATIONS = os.path.join(BASE_DIR, "images", "segmentations")

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