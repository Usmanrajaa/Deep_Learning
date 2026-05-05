import torch
import torch.nn as nn

class LSTMCell(nn.Module):
    """ From‑scratch LSTM cell. """
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.W = nn.Parameter(torch.randn(4 * hidden_size, input_size) * 0.01)
        self.U = nn.Parameter(torch.randn(4 * hidden_size, hidden_size) * 0.01)
        self.b = nn.Parameter(torch.zeros(4 * hidden_size))

    def forward(self, x, state):
        h_prev, c_prev = state
        gates = torch.matmul(x, self.W.t()) + torch.matmul(h_prev, self.U.t()) + self.b
        i, f, g, o = gates.chunk(4, dim=-1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c
