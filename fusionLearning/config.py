

encoders = ["resnet34", "resnet18"]
trainedModels = [FPN , "Linknet", "Unet" , "UnetPlusPlus"]

CUB = "./data/CUBdata"
CUB_IMAGES = CUB + "/CUB_200_2011/images"
CUB_SEGMENTATIONS = CUB + "/segmentations"

LEARNED_WEIGHTS = "./weights"
BASE_MODELS = "./models"
FUSION_MODELS = "./ensembles"

BASE_MODELS_SEGMENTATIONS = "./images/segmentations"

AVAILABLE_MODELS = [str(model) for model in archs]
print(AVAILABLE_MODELS)


