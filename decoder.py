import torch
import torch.nn as nn
from small_blocks import SelfAttentionBlock, FeedForwardBlock


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super(DecoderBlock, self).__init__()
        self.attention1 = SelfAttentionBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.attention2 = SelfAttentionBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.ff = FeedForwardBlock(embed_size, dropout, forward_expansion)

    def forward(self, x, encoder_out, src_mask, trg_mask):
        attention = self.attention1(x, x, x, trg_mask)
        attention = self.attention2(attention, encoder_out, encoder_out, src_mask)
        return self.ff(attention)


class Decoder(nn.Module):
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
        super(Decoder, self).__init__()
        self.device = device
        self.word_embed = nn.Embedding(vocab_size, embed_size)
        self.pos_embed = nn.Embedding(seq_max_len, embed_size)
        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_size, heads, dropout, forward_expansion)
                for _ in range(Nx)
            ]
        )

    def forward(self, x, encoder_out, src_mask, trg_mask):
        N, seqlen = x.shape
        pos = torch.arange(0, seqlen).expand(N, seqlen).to(self.device)
        x = self.word_embed(x) + self.pos_embed(pos)
        for layer in self.layers:
            x = layer(x, encoder_out, src_mask, trg_mask)
        return x
