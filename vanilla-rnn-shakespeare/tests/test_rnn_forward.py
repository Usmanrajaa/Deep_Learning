import sys
sys.path.insert(0, '.')
import torch
from model.rnn_cell import VanillaRNNCell

def test():
    cell = VanillaRNNCell(10, 5)
    x = torch.randn(2, 10)
    h0 = torch.zeros(2, 5)
    h1 = cell(x, h0)
    assert h1.shape == (2, 5)
    print("RNN cell forward test passed.")

if __name__ == "__main__":
    test()
