import torch
import numpy as np

def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def get_batches(encoded, batch_size, seq_length):
    total_batches = (len(encoded) - 1) // (batch_size * seq_length)
    encoded = encoded[:total_batches * batch_size * seq_length + 1]
    for i in range(total_batches):
        start = i * batch_size * seq_length
        x = encoded[start : start + batch_size * seq_length].reshape(batch_size, seq_length)
        y = encoded[start + 1 : start + batch_size * seq_length + 1].reshape(batch_size, seq_length)
        yield torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
