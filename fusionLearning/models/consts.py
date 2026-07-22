"""Control variables for base model arch and optimizers"""


MAXEPOCHS : int = 20
BATCHSIZE : int = 8 
MOMENTUM : float = 0.99
LEARNING_RATE : float = 0.01


NUM_CLASSES_CUB : int = 1
NUM_CLASSES_CITYSCAPES : int = 19
NUM_CLASSES_VOC : int = 21  # background + 20 object categories

# TOMPEI-CMMD binary lesion/normal classification
MAXEPOCHS_CLS : int = 20
BATCHSIZE_CLS : int = 16
LEARNING_RATE_CLS : float = 0.01
LR_MIN_CLS : float = 1e-6
NUM_CLASSES_TOMPEI_CMMD : int = 1
INPUT_SIZE_TOMPEI_CMMD : int = 512