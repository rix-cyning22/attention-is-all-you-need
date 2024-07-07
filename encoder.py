import torch
import torch.nn as nn
from small_blocks import SelfAttentionBlock, FeedForwardBlock


class EncoderBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(EncoderBlock, self).__init__()
        self.attention = SelfAttentionBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.ff = FeedForwardBlock(embed_size, dropout, forward_expansion)

    def forward(self, x, mask):
        attention = self.attention(x, x, x, mask)
        return self.ff(attention)


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        seq_max_len,
        embed_size,
        Nx,
        heads,
        dropout,
        device,
        forward_expansion,
    ):
        super(Encoder, self).__init__()
        self.device = device
        self.word_embed = nn.Embedding(vocab_size, embed_size)
        self.pos_embed = nn.Embedding(seq_max_len, embed_size)
        self.layers = nn.ModuleList(
            [
                EncoderBlock(embed_size, heads, dropout, forward_expansion)
                for _ in range(Nx)
            ]
        )

    def forward(self, x, mask):
        N, seqlen = x.shape
        pos = torch.arange(0, seqlen).expand(N, seqlen).to(self.device)
        x = self.word_embed(x) + self.pos_embed(pos)
        for layer in self.layers:
            x = layer(x, mask)
        return x
