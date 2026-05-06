import torch
import torch.nn as nn
import math
from .multi_head_attention import MultiHeadAttention
from .positional_encoding import PositionalEncoding

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

    def forward(self, src, mask=None):
        attn_out, _ = self.self_attn(src, src, src, mask)
        src = self.norm1(src + self.dropout(attn_out))
        ff_out = self.ff(src)
        src = self.norm2(src + self.dropout(ff_out))
        return src

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        self_attn_out, _ = self.self_attn(tgt, tgt, tgt, tgt_mask)
        tgt = self.norm1(tgt + self.dropout(self_attn_out))
        cross_attn_out, cross_attn_weights = self.cross_attn(tgt, memory, memory, memory_mask)
        tgt = self.norm2(tgt + self.dropout(cross_attn_out))
        ff_out = self.ff(tgt)
        tgt = self.norm3(tgt + self.dropout(ff_out))
        return tgt, cross_attn_weights

class Transformer(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_heads=4, num_layers=2, d_ff=512, max_len=20, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.out = nn.Linear(d_model, vocab_size)
        self.d_model = d_model

    def make_pad_mask(self, seq, pad_idx=0):
        # Returns boolean mask: True = keep (not pad)
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq_len)

    def make_subsequent_mask(self, seq_len, device):
        mask = torch.ones((1, seq_len, seq_len), device=device).triu(1) == 0  # lower triangular = True
        return mask.unsqueeze(1)  # (1, 1, seq_len, seq_len)

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        src = src.transpose(0,1)   # (batch, src_len)
        trg = trg.transpose(0,1)   # (batch, trg_len)
        batch_size, trg_len = trg.shape

        src_emb = self.embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_enc(src_emb)
        trg_emb = self.embedding(trg[:, :-1]) * math.sqrt(self.d_model)
        trg_emb = self.pos_enc(trg_emb)

        src_mask = self.make_pad_mask(src)
        tgt_mask = self.make_subsequent_mask(trg_len-1, src.device) & self.make_pad_mask(trg[:, :-1])

        enc_out = src_emb
        for layer in self.encoder_layers:
            enc_out = layer(enc_out, src_mask)

        dec_out = trg_emb
        cross_attns = []
        for layer in self.decoder_layers:
            dec_out, cross_attn_weights = layer(dec_out, enc_out, tgt_mask, src_mask)
            cross_attns.append(cross_attn_weights)

        logits = self.out(dec_out)                         # (batch, trg_len-1, vocab)
        return logits.transpose(0,1), cross_attns           # (trg_len-1, batch, vocab)
