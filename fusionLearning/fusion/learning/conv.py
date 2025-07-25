from fusion.interfaces import FusionModule
from torch import nn

config = {
    "in_channels" : 3,
    "out_channels" : 3,
    "kernel_size" : 3,
    "padding" : 1,
    "stride" : 1,
    "dilation" : 1,
    "groups" : 1,
    "bias" : True,
    "padding_mode" : 'zeros',
    "subsample" : False,
    "use_batch_norm" : False,
    "use_instance_norm" : False,
    "use_group_norm" : False,
    "use_layer_norm" : False,
    "use_weight_norm" : False,
    "use_dropout" : False,
    "dropout_rate" : 0.0,
}



class ConvFusion(FusionModule): 
    def __init__(self, name: str, config : dict):
        super().__init__(name) 

        

