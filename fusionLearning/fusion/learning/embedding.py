
class PatchEmbedding(nn.Module):
    def __init__(self, patch_size=16, embedding_dim=256):
        super().__init__()
        self.patch_size = patch_size
        self.embedding_dim = embedding_dim
        
        # Convolutional patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels=1,           # Each mask is 1 channel
            out_channels=embedding_dim,    # Embed to d_model dimensions
            kernel_size=patch_size,  # 16x16 or 32x32
            stride=patch_size        # Non-overlapping patches
        )
        
    def forward(self, mask):
        # mask: [1, 1, H, W] -> patches: [1, d_model, H//patch_size, W//patch_size]
        patches = self.patch_embed(mask)
        
        # Flatten spatial dimensions
        B, C, Ph, Pw = patches.shape
        patches_flat = patches.view(B, C, Ph * Pw).transpose(1, 2)
        
        return patches_flat  # [1, num_patches, embedding_dim]
