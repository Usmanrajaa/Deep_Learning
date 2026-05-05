import torch
import torch.nn as nn

class VanillaRNNCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.W_xh = nn.Parameter(torch.randn(hidden_size, input_size) * 0.01)
        self.W_hh = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        self.b_h = nn.Parameter(torch.zeros(hidden_size))

    def forward(self, x, h_prev):
        a = torch.matmul(x, self.W_xh.t()) + torch.matmul(h_prev, self.W_hh.t()) + self.b_h
        return torch.tanh(a)
