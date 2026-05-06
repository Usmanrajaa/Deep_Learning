class CharVocab:
    def __init__(self, chars):
        self.chars = sorted(list(set(chars)))
        self.char2idx = {ch: i for i, ch in enumerate(self.chars)}
        self.idx2char = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
