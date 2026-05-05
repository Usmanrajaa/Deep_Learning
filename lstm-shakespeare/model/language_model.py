import torch
import torch.nn as nn
from model.lstm_cell import LSTMCell

class CharLSTM(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm_cell = LSTMCell(vocab_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, state=None):
        seq_len, batch = x.shape
        if state is None:
            h = torch.zeros(batch, self.hidden_size, device=x.device)
            c = torch.zeros(batch, self.hidden_size, device=x.device)
        else:
            h, c = state
        outputs = []
        one_hot = torch.nn.functional.one_hot(x, num_classes=self.output_layer.out_features).float()
        for t in range(seq_len):
            h, c = self.lstm_cell(one_hot[t], (h, c))
            outputs.append(h.unsqueeze(0))
        outputs = torch.cat(outputs, dim=0)
        logits = self.output_layer(outputs)
        return logits, (h, c)
