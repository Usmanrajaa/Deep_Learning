
import random

def generate_samples(num, min_len=2, max_len=8, chars='abcdefghijklmnopqrstuvwxyz'):
    src, trg = [], []
    for _ in range(num):
        length = random.randint(min_len, max_len)
        s = ''.join(random.choice(chars) for _ in range(length))
        src.append(s)
        trg.append(s[::-1])
    return src, trg

def encode_data(strs, char2idx):
    return [[char2idx[ch] for ch in s] for s in strs]

def pad_sequences(seqs, pad_idx):
    max_len = max(len(s) for s in seqs)
    return [s + [pad_idx] * (max_len - len(s)) for s in seqs]
