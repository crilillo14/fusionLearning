""" Sinusoidal positional encoder for patches.
"""



from torch import nn
import torch
from abc import ABC, abstractmethod


# --------------- interface
class Encoder(ABC): 

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass
    
    @abstractmethod
    def batch_encode(preds : torch.Tensor) -> torch.Tensor: 
        pass

    

# --------------- coordinate based PE

class PositionalEncoder(Encoder): 
    def __init__(self, name : str, device, model_dim : int) -> None:
        super().__init__(name)
        self.device = device
        self.model_dim = model_dim


    def __call__(self, *args: Any, **kwds: Any) -> Any:
        # left to implement
        return super().__call__(*args, **kwds)


    def batch_encode(preds: torch.Tensor) -> torch.Tensor:
        # left to implement
        return super().batch_encode()



# --------------- sinusoidal encoder
class SinusoidalEncoder(nn.Module): 

    """ SPE for fixed size patches - non learnable

    Takes torch.Tensors with patches embedded into the model dimension, and 
    adds a learned positional encoding to each patch.
    
    """
    def __init__(self, embedding_dim : int): 
        super().__init__()

    def __call__(self, tokens : torch.Tensor) -> torch.Tensor:

        for i, token in enumerate(tokens):
            if  i % 2 == 0: 
                token = torch.cos(token * k)
            else: 
                token = torch.sin(token)

        return tokens

        
    def decode(self, tokens : torch.Tensor) -> torch.Tensor:
        pass