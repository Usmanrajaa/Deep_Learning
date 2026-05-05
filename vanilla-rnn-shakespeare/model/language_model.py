import torch
import torch.nn as nn
from model.rnn_cell import VanillaRNNCell

class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.rnn_cell = VanillaRNNCell(vocab_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h0=None):
        seq_len, batch = x.shape
        if h0 is None:
            h = torch.zeros(batch, self.hidden_size, device=x.device)
        else:
            h = h0
        outputs = []
        one_hot = torch.nn.functional.one_hot(x, num_classes=self.vocab_size).float()
        for t in range(seq_len):
            h = self.rnn_cell(one_hot[t], h)
            outputs.append(h.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        logits = self.output_layer(outputs)
        return logits, h
