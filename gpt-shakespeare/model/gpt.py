import torch
import torch.nn as nn
import math
from .multi_head_attention import MultiHeadAttention
from .positional_encoding import PositionalEncoding

class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_heads=4, num_layers=4, d_ff=512, max_len=256, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)
        self.d_model = d_model
        self.max_len = max_len

    def _generate_square_subsequent_mask(self, sz, device):
        # lower triangular matrix of True → attend only to past and current
        mask = torch.ones((1, sz, sz), device=device).triu(1) == 0
        return mask.unsqueeze(1)  # (1, 1, sz, sz)

    def forward(self, x):
        # x: (batch, seq_len) integers
        batch_size, seq_len = x.shape
        mask = self._generate_square_subsequent_mask(seq_len, x.device)
        emb = self.embedding(x) * math.sqrt(self.d_model)
        emb = self.pos_enc(emb)
        out = emb
        for layer in self.layers:
            out = layer(out, mask)
        out = self.norm(out)
        logits = self.head(out)          # (batch, seq_len, vocab_size)
        return logits
