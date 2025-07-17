""" interface.py

Interface for fusion methods to follow. 

Should have the following: 

+ A forward pass
+ (OPTIONAL) A backpropagation method
+ initialization method
+ (nice to have) __repr__
"""


import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List


class FusionModule(nn.Module, ABC):
    """
    Abstract base class for fusion methods.
    
    All fusion modules must implement the forward method to combine
    multiple model predictions into a single output.

    Backward pass only if fusion framework has learnable weights (e.g. fusion by linear combination)
    """
    
    def __init__(self, name: str):
        super().__init__()
        self.name = name
    
    @abstractmethod
    def forward(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        """
        Fuse multiple predictions into a single output.
        
        Args:
            predictions: List of tensors with shape [batch_size, channels, height, width]
        
        Returns:
            fused_output: Tensor with shape [batch_size, channels, height, width]
        """
        pass

    @abstractmethod
    def backward(self, y : torch.Tensor) -> None:
        """
        Backpropagate the loss through the fusion module.
        
        Args:
            y: output tensor with shape [batch_size, channels, height, width]
        """
        pass
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"

    def __