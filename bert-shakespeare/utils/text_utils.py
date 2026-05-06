class CharVocab:
    def __init__(self, chars):
        specials = ['<pad>', '<mask>', '<unk>']
        self.chars = specials + sorted(list(set(chars)))
        self.char2idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx2char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        self.pad_idx = self.char2idx['<pad>']
        self.mask_idx = self.char2idx['<mask>']
        self.unk_idx = self.char2idx['<unk>']
