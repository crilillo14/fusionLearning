"""
ViT for semantic segmentation.

Encodes image as patch tokens via a standard Vision Transformer, then
decodes the patch token sequence back to a dense [B, classes, H, W] map
with a 1×1 conv head + bilinear upsample.

Positional embeddings are interpolated at runtime so the model handles
the variable spatial sizes produced by pad_collate without any cropping.

Interface matches the SMP/SOTA arch_dict convention:
    model = ViTSegmentation(classes=21)
    logits = model(x)  # [B, classes, H, W]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── building blocks ───────────────────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, dropout: float = 0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = embed_dim // num_heads
        self.scale     = self.head_dim ** -0.5

        self.qkv       = nn.Linear(embed_dim, embed_dim * 3)
        self.proj      = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.attn_drop(attn.softmax(dim=-1))

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int = 768, num_heads: int = 12,
                 mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn  = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim    = int(embed_dim * mlp_ratio)
        self.mlp   = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ── segmentation model ────────────────────────────────────────────────────────

class ViTSegmentation(nn.Module):
    """
    Vision Transformer for dense semantic segmentation.

    Args:
        classes:     Number of output segmentation classes.
        img_size:    Nominal input size used to build the base positional grid.
                     Actual inputs can differ — pos embeds are interpolated.
        patch_size:  Patch size (default 16 → 32×32 patches for a 512 input).
        embed_dim:   Transformer hidden dimension.
        depth:       Number of transformer blocks.
        num_heads:   Number of attention heads.
        mlp_ratio:   MLP hidden dim as a multiple of embed_dim.
        dropout:     Dropout rate.
        **kwargs:    Absorbs SMP-style params (encoder_name, encoder_weights,
                     in_channels) so this sits in arch_dict without changes.
    """

    def __init__(
        self,
        classes: int,
        img_size: int = 512,
        patch_size: int = 16,
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim  = embed_dim

        # Patch embedding via conv (stride == patch_size → non-overlapping patches)
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)

        base_patches = (img_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + base_patches, embed_dim))
        self.pos_drop  = nn.Dropout(dropout)

        self.blocks = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

        # 1×1 conv maps embed_dim → classes at patch resolution
        self.seg_head = nn.Conv2d(embed_dim, classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _interp_pos_embed(self, hp: int, wp: int) -> torch.Tensor:
        """Interpolate positional embeddings to (hp × wp) patch grid."""
        cls_pos   = self.pos_embed[:, :1, :]
        patch_pos = self.pos_embed[:, 1:, :]
        base_n    = patch_pos.shape[1]
        base_hw   = int(base_n ** 0.5)

        if hp == base_hw and wp == base_hw:
            return self.pos_embed

        # [1, embed_dim, base_hw, base_hw] → interpolate → [1, hp*wp, embed_dim]
        patch_pos = patch_pos.reshape(1, base_hw, base_hw, self.embed_dim).permute(0, 3, 1, 2)
        patch_pos = F.interpolate(patch_pos, size=(hp, wp), mode="bilinear", align_corners=False)
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, hp * wp, self.embed_dim)

        return torch.cat([cls_pos, patch_pos], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _, H, W = x.shape
        hp = H // self.patch_size
        wp = W // self.patch_size

        # [B, embed_dim, hp, wp]
        feats = self.patch_embed(x)
        # [B, hp*wp, embed_dim]
        tokens = feats.flatten(2).transpose(1, 2)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        tokens = self.pos_drop(tokens + self._interp_pos_embed(hp, wp))

        tokens = self.blocks(tokens)
        tokens = self.norm(tokens)

        # Drop CLS, reshape to 2-D patch grid
        patch_tokens = tokens[:, 1:, :]                            # [B, hp*wp, D]
        patch_tokens = patch_tokens.reshape(B, hp, wp, self.embed_dim).permute(0, 3, 1, 2)
        # [B, embed_dim, hp, wp]

        logits = self.seg_head(patch_tokens)                       # [B, classes, hp, wp]
        return F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)


if __name__ == "__main__":
    model = ViTSegmentation(classes=21)
    x = torch.randn(2, 3, 512, 512)
    out = model(x)
    print(f"Output shape : {out.shape}")   # (2, 21, 512, 512)
    print(f"Params       : {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    # Variable size test
    x2 = torch.randn(2, 3, 480, 640)
    out2 = model(x2)
    print(f"Variable size: {out2.shape}")  # (2, 21, 480, 640)
