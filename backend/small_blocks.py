import torch
import torch.nn as nn


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads, dropout):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads

        assert (
            self.head_dim * heads == embed_size
        ), "embed size must be divisible by heads"

        self.values_matrix = nn.Linear(self.head_dim, embed_size, bias=False)
        self.keys_matrix = nn.Linear(self.head_dim, embed_size, bias=False)
        self.queries_matrix = nn.Linear(self.head_dim, embed_size, bias=False)
        self.dense_out = nn.Linear(embed_size, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, keys, queries, values, mask=None):
        N = queries.shape[0]
        keys = keys.reshape(N, -1, self.heads, self.head_dim)
        queries = queries.reshape(N, -1, self.heads, self.head_dim)
        values = values.reshape(N, -1, self.heads, self.head_dim)

        keys = self.keys_matrix(keys)
        queries = self.queries_matrix(queries)
        values = self.values_matrix(values)

        matmul = torch.einsum("nqhd, nkhd -> nhqk", [queries, keys])
        matmul /= self.head_dim**0.5
        if mask is not None:
            matmul.masked_fill(mask == 0, float("-inf"))
        attention = torch.softmax(matmul, dim=3)
        out = torch.einsum("nhqk, nkhd -> nqhd", [attention, values])
        out = out.reshape(N, -1, self.embed_size)
        out = self.dense_out(out)
        out = self.dropout(out)
        out = out.reshape(N, out.shape[1] // self.heads, self.heads, self.embed_size)
        return out.mean(dim=2)


class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(SelfAttentionBlock, self).__init__()
        self.attention = SelfAttention(embed_size, heads, dropout)
        self.norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, keys, queries, values, mask):
        attention = self.attention(queries, keys, values, mask)
        return self.dropout(self.norm(attention + values))


class FeedForwardBlock(nn.Module):
    def __init__(self, embed_size, dropout, forward_expansion):
        super(FeedForwardBlock, self).__init__()
        self.ff = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size),
        )
        self.norm = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, attention):
        x = self.ff(attention)
        out = self.norm(x + attention)
        return self.dropout(out)
