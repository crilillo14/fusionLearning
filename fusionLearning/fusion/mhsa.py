from fusion.interfaces import FusionModule
from torch import nn



class PatchedMHSA(FusionModule): 

    def __init__(self, name : str, embedding_dim : int, num_heads : int, patch_size : int): 
        super().__init__(name)

        # have to be a multiple of 32 (also 16)
        assert patch_size % 16 == 0

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_patches = patch_size / 16


        self.embed_patches = nn.Sequential(
            nn.Conv2d(
                in_channels=1, 
                out_channels=embedding_dim, 
                kernel_size=patch_size, 
                stride=patch_size,
            )
        )