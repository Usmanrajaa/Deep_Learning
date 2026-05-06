import torch
import torch.nn as nn
import math
from .attention import MultiHeadAttention

class TransformerEncoderLayer(nn.Module):
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

class BERT(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_heads=4, num_layers=4, d_ff=512, max_len=256, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.position_embedding = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)   # MLM head
        self.d_model = d_model
        self.register_buffer('pos_ids', torch.arange(max_len).unsqueeze(0))  # (1, max_len)

    def forward(self, x, mask=None):
        """
        x: (batch, seq_len) token indices
        mask: (batch, 1, 1, seq_len) boolean attention mask (True for keep)
        """
        batch_size, seq_len = x.shape
        pos_ids = self.pos_ids[:, :seq_len].expand(batch_size, -1).to(x.device)
        emb = self.token_embedding(x) + self.position_embedding(pos_ids)
        out = emb
        for layer in self.layers:
            out = layer(out, mask)
        out = self.norm(out)
        logits = self.head(out)    # (batch, seq_len, vocab_size)
        return logits
