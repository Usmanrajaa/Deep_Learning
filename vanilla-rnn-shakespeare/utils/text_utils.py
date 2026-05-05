class CharVocab:
    def __init__(self, chars):
        self.chars = sorted(list(set(chars)))
        self.char2idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx2char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)

    def encode(self, text):
        return [self.char2idx[ch] for ch in text]

    def decode(self, indices):
        return ''.join(self.idx2char[idx] for idx in indices)
