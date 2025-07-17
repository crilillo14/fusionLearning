from typing import override
from interfaces import FusionModule



"""Basic mean based fusion techniques"""
# ========================================================================= 
class PixelwiseMeanFusion(FusionModule): 

    def __init__(self, name: str):
        super().__init__(name)
    
    @override
    def forward(self, predictions: List[torch.Tensor]) -> torch.Tensor:
        return torch.mean(torch.stack(predictions), dim=0)

    @override
    def backward(self, y: torch.Tensor) -> None:
        return None

# =========================================================================


