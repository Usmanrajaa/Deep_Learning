import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NTM(nn.Module):
    """NTM Lite – LSTM controller, external memory, two heads (read/write) with soft attention."""
    def __init__(self, input_size, output_size, memory_size=(10, 20), controller_size=128, num_heads=1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.memory_rows, self.memory_cols = memory_size
        self.controller_size = controller_size
        self.num_heads = num_heads

        # Controller: LSTM that receives [input, previous_read] concatenated
        self.lstm = nn.LSTMCell(input_size + num_heads * self.memory_cols, controller_size)
        # Output linear
        self.out_layer = nn.Linear(controller_size, output_size)
        # Linear layers to produce read/write parameters from controller output
        # Read head parameters: read attention key (to compare with memory rows)
        self.read_key = nn.Linear(controller_size, self.memory_cols)
        self.read_strength = nn.Linear(controller_size, 1)

        # Write head parameters: erase vector, add vector, write key, write strength
        self.write_key = nn.Linear(controller_size, self.memory_cols)
        self.write_strength = nn.Linear(controller_size, 1)
        self.erase_vector = nn.Linear(controller_size, self.memory_cols)
        self.add_vector = nn.Linear(controller_size, self.memory_cols)

    def _address_memory(self, memory, key, strength, use_softmax=True):
        """Compute soft attention over memory rows using cosine similarity."""
        # memory: (batch, rows, cols)
        # key: (batch, cols)
        # strength: (batch, 1)
        key_norm = key / (torch.norm(key, dim=1, keepdim=True) + 1e-8)
        memory_norm = memory / (torch.norm(memory, dim=2, keepdim=True) + 1e-8)
        sim = torch.bmm(memory_norm, key_norm.unsqueeze(2)).squeeze(2)   # (batch, rows)
        # scale with strength (beta)
        weights = F.softmax(strength * sim, dim=1)
        return weights

    def forward(self, inputs, memory=None, state=None):
        """
        inputs: (batch, seq_len, input_size)
        Returns logits, final memory, final state
        """
        batch_size, seq_len, _ = inputs.shape
        if memory is None:
            memory = torch.zeros(batch_size, self.memory_rows, self.memory_cols, device=inputs.device)
        if state is None:
            h = torch.zeros(batch_size, self.controller_size, device=inputs.device)
            c = torch.zeros(batch_size, self.controller_size, device=inputs.device)
        else:
            h, c = state

        # For reading, initialize read vector to zeros
        read_vec = torch.zeros(batch_size, self.num_heads * self.memory_cols, device=inputs.device)

        outputs = []
        for t in range(seq_len):
            # Controller input: current input + previous read
            controller_input = torch.cat([inputs[:, t, :], read_vec], dim=1)
            h, c = self.lstm(controller_input, (h, c))

            # Compute output
            out = self.out_layer(h)
            outputs.append(out.unsqueeze(1))

            # Read head
            rk = self.read_key(h)
            rs = torch.exp(self.read_strength(h))  # ensure positive
            read_weights = self._address_memory(memory, rk, rs)               # (batch, rows)
            read_vec = torch.bmm(read_weights.unsqueeze(1), memory).squeeze(1) # (batch, cols)

            # Write head: we use one write head for simplicity
            wk = self.write_key(h)
            ws = torch.exp(self.write_strength(h))
            e = torch.sigmoid(self.erase_vector(h))          # (batch, cols)
            a = self.add_vector(h)                           # (batch, cols)

            write_weights = self._address_memory(memory, wk, ws)   # (batch, rows)

            # Memory update: M' = M * (1 - w * e) + w * a
            w_expanded = write_weights.unsqueeze(2)           # (batch, rows, 1)
            e_expanded = e.unsqueeze(1)                       # (batch, 1, cols)
            a_expanded = a.unsqueeze(1)                       # (batch, 1, cols)
            memory = memory * (1 - w_expanded * e_expanded) + w_expanded * a_expanded

        return torch.cat(outputs, dim=1), memory, (h, c)
