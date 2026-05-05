
import torch
import torch.nn as nn
from .encoder import Encoder
from .attention import BahdanauAttention
from .decoder import Decoder

class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, hidden_size):
        super().__init__()
        enc_hidden = hidden_size * 2   # bidirectional encoder output
        dec_hidden = hidden_size
        self.encoder = Encoder(vocab_size, hidden_size)
        attn = BahdanauAttention(enc_hidden, dec_hidden)
        self.decoder = Decoder(vocab_size, enc_hidden, dec_hidden, attn)
        self.vocab_size = vocab_size

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        trg_len, batch = trg.shape
        encoder_outputs = self.encoder(torch.nn.functional.one_hot(src, self.vocab_size).float())
        h = torch.zeros(batch, self.decoder.lstm.hidden_size, device=src.device)
        c = torch.zeros(batch, self.decoder.lstm.hidden_size, device=src.device)
        outputs = []
        input_token = torch.nn.functional.one_hot(trg[0], self.vocab_size).float()
        for t in range(1, trg_len):
            out, h, c, _ = self.decoder(input_token, h, c, encoder_outputs)
            outputs.append(out.unsqueeze(0))
            teacher_force = torch.rand(1).item() < teacher_forcing_ratio
            top1 = out.argmax(-1)
            input_token = torch.nn.functional.one_hot(trg[t] if teacher_force else top1, self.vocab_size).float()
        return torch.cat(outputs, dim=0)
