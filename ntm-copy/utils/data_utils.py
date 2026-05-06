import torch
import random

def generate_copy_batch(batch_size, seq_len, num_tokens=10):
    """
    Generate random sequences of length seq_len of tokens 1..num_tokens.
    Input format: [start_token(0)] + sequence + [0]*seq_len (padding)
    Target:        [0]*seq_len + sequence + [end_token(num_tokens+1)?]
    We'll use a simpler format: target is just the sequence, we shift and use cross-entropy.
    We'll use start=0, end=num_tokens+1, but for simplicity we can just output the sequence after reading.
    Actually, typical NTM copy uses input: sequence, output: same sequence.
    But here we need to teach the network to copy. We'll do:
    Input:  sequence (each element one-hot) of length L
    Target: same sequence, but the model must learn to produce it. Since the NTM reads input at each step, it can output the input after some delay. We'll train it to produce the input at each step, which will force it to write and read.
    A better way: input sequence, then output the same sequence.
    We'll feed the sequence as input, and target is the same sequence.
    The model will learn to copy by using memory.
    """
    seq = torch.randint(1, num_tokens+1, (batch_size, seq_len))  # tokens 1..num_tokens
    # One-hot for input
    input_onehot = torch.zeros(batch_size, seq_len, num_tokens+2)  # 0: pad, tokens 1..num_tokens, maybe unused
    # scatter tokens
    input_onehot.scatter_(2, seq.unsqueeze(2), 1)
    # target: the same tokens, we'll use cross-entropy with token indices.
    # For output, we treat as classification.
    # We need to provide target indices (batch, seq_len)
    return input_onehot, seq   # target indices (no need for start/end for this simple task)
