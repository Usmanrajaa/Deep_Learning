
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=False)

    def forward(self, src):
        outputs, _ = self.lstm(src)   # (seq_len, batch, 2*hidden)
        return outputs
