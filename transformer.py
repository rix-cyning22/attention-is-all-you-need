import torch.nn as nn
import torch
from encoder import Encoder
from decoder import Decoder


class Transformer(nn.Module):
    def __init__(
        self,
        input_vocab_size,
        output_vocab_size,
        input_pad_idx,
        output_pad_idx,
        seq_max_len=100,
        Nx=6,
        heads=8,
        dropout=0.1,
        forward_expansion=4,
        embed_size=256,
    ):
        super(Transformer, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = Encoder(
            input_vocab_size,
            seq_max_len,
            embed_size,
            Nx,
            heads,
            dropout,
            self.device,
            forward_expansion,
        )
        self.decoder = Decoder(
            output_vocab_size,
            seq_max_len,
            embed_size,
            Nx,
            heads,
            dropout,
            self.device,
            forward_expansion,
        )
        self.linear = nn.Linear(embed_size, output_vocab_size)
        self.input_pad_idx = input_pad_idx
        self.output_pad_idx = output_pad_idx

    def make_src_mask(self, src):
        src_mask = (src != self.input_pad_idx).unsqueeze(1).unsqueeze(2)
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_pad_mask = (trg != self.output_pad_idx).unsqueeze(1).unsqueeze(2)
        trg_sub_mask = torch.tril(
            torch.ones((trg_len, trg_len), device=self.device)
        ).bool()
        trg_mask = trg_pad_mask & trg_sub_mask
        return trg_mask

    def forward(self, src, trg):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg)
        enc_out = self.encoder(src, src_mask)
        out = self.decoder(trg, enc_out, src_mask, trg_mask)
        return self.linear(out)
