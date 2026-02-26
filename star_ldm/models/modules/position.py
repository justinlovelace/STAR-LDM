import torch
import torch.nn as nn

def exists(val):
    return val is not None

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=1024):
        super().__init__()
        self.scale = dim ** -0.5
        self.max_seq_len = max_seq_len
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x, pos = None):
        seq_len, device = x.shape[1], x.device
        assert seq_len <= self.max_seq_len, f'you are passing in a sequence length of {seq_len} but your absolute positional embedding has a max sequence length of {self.max_seq_len}'

        if not exists(pos):
            pos = torch.arange(seq_len, device = device)

        pos_emb = self.emb(pos)
        pos_emb = pos_emb * self.scale
        return pos_emb
