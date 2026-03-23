

""" ViT for Segmentation
"""




### scoring metrics on validation & test
from torchmetrics.segmentation import DiceScore, MeanIoU
from torch import nn



class PositionalEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x

class PatchEmbedding(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x
    


class TransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x
    
    
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        return x




class SViT(nn.Module):
    def __init__(self):
        super().__init__()
        
        
        
        
    def forward(self, x):
        return x