import torch
import numpy as np
import random

def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def mask_tokens(input_ids, vocab_size, mask_idx, pad_idx, masked_prob=0.15):
    """
    input_ids: (batch_size, seq_len) LongTensor
    Returns: masked_input_ids, labels (original tokens, -100 for non-masked)
    """
    labels = input_ids.clone()
    # Create a mask for tokens to be predicted (not padding)
    prob_matrix = torch.full(input_ids.shape, masked_prob)
    # do not mask padding tokens
    pad_mask = (input_ids == pad_idx)
    prob_matrix.masked_fill_(pad_mask, 0.0)
    masked_indices = torch.bernoulli(prob_matrix).bool()

    # For 80% of those masked positions, replace with mask token
    indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
    input_ids[indices_replaced] = mask_idx

    # For 10% replace with random token
    indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long)
    # avoid padding and mask tokens for random replacement (optional but helps)
    # For simplicity we just assign random; it's okay if occasionally replaced with special tokens
    input_ids[indices_random] = random_words[indices_random]

    # The rest (10%) keep original, so no change.
    # Labels for non-masked positions are set to -100 (ignored in cross-entropy)
    labels[~masked_indices] = -100

    return input_ids, labels

def get_mlm_batches(encoded, vocab_size, mask_idx, pad_idx, batch_size, seq_length, masked_prob=0.15):
    """Yield batches of masked sequences from encoded text (1D array of token indices)."""
    total_batches = (len(encoded) - 1) // (batch_size * seq_length)
    encoded = encoded[:total_batches * batch_size * seq_length]
    for i in range(total_batches):
        start = i * batch_size * seq_length
        chunk = encoded[start:start + batch_size * seq_length].reshape(batch_size, seq_length)
        chunk = torch.tensor(chunk, dtype=torch.long)
        masked_input, labels = mask_tokens(chunk, vocab_size, mask_idx, pad_idx, masked_prob)
        yield masked_input, labels
