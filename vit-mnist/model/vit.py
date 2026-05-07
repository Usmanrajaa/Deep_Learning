import torch
import torch.nn as nn
from .transformer_encoder import TransformerEncoderLayer

class ViT(nn.Module):
    def __init__(self, image_size=28, patch_size=4, num_classes=10, d_model=128,
                 num_heads=4, num_layers=4, d_ff=512, dropout=0.1):
        super().__init__()
        assert image_size % patch_size == 0
        self.num_patches = (image_size // patch_size) ** 2
        patch_dim = patch_size * patch_size * 1  # 1 channel (MNIST)
        self.patch_embed = nn.Linear(patch_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, d_model))

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        self.patch_size = patch_size
        self.d_model = d_model

    def forward(self, x):
        # x: (batch, 1, 28, 28)
        batch_size = x.size(0)
        # extract patches and flatten
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.contiguous().view(batch_size, 1, self.num_patches, self.patch_size * self.patch_size)
        patches = patches.permute(0, 2, 1, 3).reshape(batch_size, self.num_patches, -1)  # (batch, num_patches, patch_dim)
        patch_emb = self.patch_embed(patches)   # (batch, num_patches, d_model)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (batch, 1, d_model)
        x = torch.cat([cls_tokens, patch_emb], dim=1)           # (batch, num_patches+1, d_model)
        x = x + self.pos_embed

        for layer in self.encoder_layers:
            x = layer(x)   # no mask, fully connected attention

        x = self.norm(x)
        cls_out = x[:, 0, :]   # take CLS token output
        logits = self.head(cls_out)
        return logits
