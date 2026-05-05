
import torch
import torch.nn as nn

class BahdanauAttention(nn.Module):
    def __init__(self, enc_hidden_size, dec_hidden_size):
        super().__init__()
        self.W_enc = nn.Linear(enc_hidden_size, dec_hidden_size, bias=False)
        self.W_dec = nn.Linear(dec_hidden_size, dec_hidden_size, bias=False)
        self.v = nn.Linear(dec_hidden_size, 1, bias=False)

    def forward(self, decoder_hidden, encoder_outputs):
        # decoder_hidden:  (batch, dec_hidden)
        # encoder_outputs: (seq_len, batch, enc_hidden)
        seq_len = encoder_outputs.shape[0]
        dec_proj = self.W_dec(decoder_hidden).unsqueeze(0).expand(seq_len, -1, -1)
        enc_proj = self.W_enc(encoder_outputs)
        energy = torch.tanh(dec_proj + enc_proj)          # (seq_len, batch, dec_hidden)
        scores = self.v(energy).squeeze(-1)              # (seq_len, batch)
        attn_weights = torch.softmax(scores, dim=0)      # (seq_len, batch)
        context = (attn_weights.unsqueeze(-1) * encoder_outputs).sum(dim=0)   # (batch, enc_hidden)
        return context, attn_weights
