"""Control variables for base model arch and optimizers"""


MAXEPOCHS : int = 20
BATCHSIZE : int = 8 
MOMENTUM : float = 0.99
LEARNING_RATE : float = 0.01


NUM_CLASSES_CUB : int = 1
NUM_CLASSES_CITYSCAPES : int = 19
NUM_CLASSES_VOC : int = 21  # background + 20 object categories

# TOMPEI-CMMD binary lesion/normal classification
MAXEPOCHS_CLS : int = 30
BATCHSIZE_CLS : int = 16
LEARNING_RATE_CLS : float = 0.01
LR_MIN_CLS : float = 1e-6
NUM_CLASSES_TOMPEI_CMMD : int = 1
INPUT_SIZE_TOMPEI_CMMD : int = 512  # legacy default / fallback - roster_cls.py now varies this per-variant

# Resolution tiers for the depth x resolution roster grid (roster_cls.py). All
# divisible by 32 (conv-stride friendly) and by 16 (ViT/patch16 friendly).
# Chosen against confirmed remote hardware (4x NVIDIA L40S, 48GB each) - real
# memory profiling still needed on that hardware before trusting the largest
# depth-tier x hi-res combos not to OOM (see roster_cls.py batch_size table).
RESOLUTION_TIERS_CLS : dict[str, int] = {"lo": 384, "mid": 512, "hi": 768}