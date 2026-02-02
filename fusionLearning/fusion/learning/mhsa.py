from fusion.interfaces import FusionModule
from torch import nn
from fusion.learning.embedding import PatchEmbedding
from fusion.learning.encoding import SinusoidalEncoder

config_hi = { 
    "embedding_dim" : 256,
    "num_heads" : 8,
    "patch_size" : 16,
}

config_lo = { 
    "embedding_dim" : 64,
    "num_heads" : 8,
    "patch_size" : 16,
}



class PatchedMHSA(FusionModule): 

    def __init__(self, name : str, config : dict): 
        super().__init__(name)

        # have to be a multiple of 32 (also 16)
        assert config["patch_size"] % 16 == 0

        self.embedding_dim = config["embedding_dim"]
        self.num_heads = config["num_heads"]
        self.patch_size = config["patch_size"]
        self.num_patches = self.patch_size / 16


        self.embed_patches = PatchEmbedding(patch_size=self.patch_size, embedding_dim=self.embedding_dim)

        # work on this.
        self.positional_encoding = SinusoidalEncoder(embedding_dim=self.embedding_dim)

        self.attention_k = nn.MultiheadAttention( 
            embed_dim=self.embedding_dim,
            num_heads=self.num_heads,
            dropout=0.1,
            batch_first=True,
        )

        self.mlp = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.GELU(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
        )

    def forward(x : torch.Tensor) -> torch.Tensor:
        pass
