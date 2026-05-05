
class CharVocab:
    def __init__(self, chars):
        specials = ['<pad>', '<sos>', '<eos>', '<unk>']
        self.chars = specials + sorted(set(chars))
        self.char2idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx2char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        self.pad_idx = self.char2idx['<pad>']
        self.sos_idx = self.char2idx['<sos>']
        self.eos_idx = self.char2idx['<eos>']
        self.unk_idx = self.char2idx['<unk>']
