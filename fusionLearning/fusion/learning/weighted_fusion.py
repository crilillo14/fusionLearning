from typing import override
from fusion.interfaces import FusionModule

import torch




""" 
weighted mean fusion
added learning step to learn how to fuse each model.

Could either be: 
1. one neuron per pixel
2. one neuron per model segmentation
3. some compromise -- patching, idk

because images vary in size, might have to start doing model-only.
zsh:1: command not found: venv
"""
class MaskwisePatchedWeightedFusion(FusionModule): 

    def __init__(self, name : str, num_models : int): 
        super().__init__(name)
        # 32 multiple, divide
        

    @override
    def forward(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        pass
        
    
    @override
    def backward(self, y : torch.Tensor) -> None:
        return super().backward(y)
