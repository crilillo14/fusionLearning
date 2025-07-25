from typing import Optional, override
from fusion.interfaces import FusionModule
from enum import Enum
import torch



"""Basic mean based fusion techniques"""
# ========================================================================= 


class MeanTypes(Enum):
    ARITHMETIC = "arithmetic"
    GEOMETRIC = "geometric"
    HARMONIC = "harmonic"
    POWER = "power"
    MEDIAN = "median"
    ROOT_MEAN_SQUARE = "rms"


class PixelwiseMeanFusion(FusionModule): 

    def __init__(self, name: str, mean_type: MeanTypes = MeanTypes.ARITHMETIC, power: Optional[float] = None):
        """
        Args:
            name: Name of the fusion module
            mean_type: Type of mean to use for fusion
            power: Exponent for power mean (only used when mean_type is POWER)
        """
        super().__init__(name)
        self.mean_type : MeanTypes = mean_type
        self.power : Optional[float] = power
    
    @override
    def forward(self, predictions: torch.Tensor) -> torch.Tensor:
        # Input shape: [batch_size, num_models, C, H, W] = [1, 7, 3, 416, 480]

        # Remove batch dimension
        predictions = predictions.squeeze(dim=0)
        n_models = predictions.shape[0]
        
        if self.mean_type == MeanTypes.ARITHMETIC:
            return torch.mean(predictions, dim=0).unsqueeze(0)  # Mean across num_models dimension
            
        elif self.mean_type == MeanTypes.GEOMETRIC:
            return torch.prod(predictions, dim=0).unsqueeze(0) ** (1/n_models)  # Geometric mean
            
        elif self.mean_type == MeanTypes.HARMONIC:
            # Avoid division by zero by adding small epsilon
            eps = 1e-10
            return n_models / (torch.sum(1.0 / (predictions + eps), dim=0) + eps).unsqueeze(0)
            
        elif self.mean_type == MeanTypes.POWER:
            # Generalized power mean
            if self.power is None:
                raise ValueError("Power must be specified for power mean")
            return (torch.mean(torch.pow(predictions, self.power), dim=0) ** (1/self.power)).unsqueeze(0)
            
        elif self.mean_type == MeanTypes.MEDIAN:
            return torch.median(predictions, dim=0).values.unsqueeze(0)
            
        elif self.mean_type == MeanTypes.ROOT_MEAN_SQUARE:
            return torch.sqrt(torch.mean(predictions ** 2, dim=0)).unsqueeze(0)

    @override
    def backward(self, y: torch.Tensor) -> None:
        pass

# =========================================================================


class VotingType(Enum):
    MAJORITY = "majority"
    WEIGHTED = "weighted"

class VotingFusion(FusionModule): 
    def __init__(self, name : str, voting_type : VotingType = VotingType.MAJORITY):
        super().__init__(name)
        self.voting_type = voting_type

    @override
    def forward(self, predictions : torch.Tensor) -> torch.Tensor:
        if self.voting_type == VotingType.MAJORITY:
            return torch.mode(predictions, dim=0).values
        elif self.voting_type == VotingType.WEIGHTED:
            return torch.mean(predictions, dim=0)

    @override
    def backward(self, y: torch.Tensor) -> None:
        pass

