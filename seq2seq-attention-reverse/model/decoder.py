
import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, output_size, enc_hidden_size, dec_hidden_size, attn):
        super().__init__()
        self.attn = attn
        self.lstm = nn.LSTMCell(output_size + enc_hidden_size, dec_hidden_size)
        self.out = nn.Linear(dec_hidden_size, output_size)

    def forward(self, input_token, hidden, cell, encoder_outputs):
        # input_token: (batch, output_size) one-hot
        context, attn_weights = self.attn(hidden, encoder_outputs)
        lstm_input = torch.cat([input_token, context], dim=-1)
        h, c = self.lstm(lstm_input, (hidden, cell))
        out = self.out(h)
        return out, h, c, attn_weights
