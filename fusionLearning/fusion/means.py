from typing import override
from interfaces import FusionModule
from enum import Enum



"""Basic mean based fusion techniques"""
# ========================================================================= 


class MeanTypes(Enum):
    ARITHMETIC = "arithmetic"
    GEOMETRIC = "geometric"


class PixelwiseMeanFusion(FusionModule): 

    def __init__(self, name: str, mean_type : MeanTypes = MeanTypes.ARITHMETIC):
        super().__init__(name)
        self.mean_type = mean_type
    
    @override                       # maybe stick to a torch tensor instead of tensor list
    def forward(self, predictions: torch.Tensor) -> torch.Tensor:
        if self.mean_type == MeanTypes.ARITHMETIC:
            return torch.mean(predictions, dim=0)
        elif self.mean_type == MeanTypes.GEOMETRIC:
            return torch.prod(predictions, dim=0)

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

