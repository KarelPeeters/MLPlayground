from typing import Optional

import torch
from torch import nn
from torch.nn import functional as nnf


class MultiHeadAttention(nn.Module):
    def __init__(self, size_stream: int, heads: int, size_kq: int, size_v: int):
        super().__init__()

        self.d_stream = size_stream
        self.heads = heads
        self.d_kq = size_kq
        self.d_v = size_v

        # TODO proper initialization
        self.project_qkv = nn.Linear(self.d_stream, heads * (2 * self.d_kq + self.d_v))
        self.project_out = nn.Linear(heads * self.d_v, self.d_stream)

    def forward(self, stream, mask):
        # stream: (seq, batch, size)
        n, b, _ = stream.shape
        heads = self.heads
        d_stream = self.d_stream
        d_kq = self.d_kq
        d_v = self.d_v

        # proj inputs
        qkv = self.project_qkv(stream.view(n * b, d_stream)).view(n, b * heads, 2 * d_kq + d_v)
        q = qkv[:, :, :d_kq]
        k = qkv[:, :, d_kq:2 * d_kq]
        v = qkv[:, :, 2 * d_kq:]

        # attention
        logits = torch.bmm(q.transpose(0, 1), k.transpose(0, 1).transpose(1, 2))
        if mask is not None:
            logits = logits + mask
        weights = nnf.softmax(logits, -1)
        values = torch.bmm(weights, v.transpose(0, 1)).transpose(0, 1).contiguous()

        # proj output
        result = self.project_out(values.view(n * b, heads * d_v)).view(n, b, d_stream)
        return result


class TransformerBlock(nn.Module):
    def __init__(self, size_stream: int, size_ff: int, heads: int):
        super().__init__()

        self.ln0 = nn.LayerNorm(size_stream)
        self.att = MultiHeadAttention(size_stream, heads, size_stream // heads, size_stream // heads)
        self.ln1 = nn.LayerNorm(size_stream)
        self.ff = nn.Sequential(
            nn.Linear(size_stream, size_ff),
            nn.ReLU(),
            nn.Linear(size_ff, size_stream),
        )

    def forward(self, stream, mask):
        stream = stream + self.att(self.ln0(stream), mask)
        stream = stream + self.ff(self.ln1(stream))
        return stream


class TransformerModel(nn.Module):
    def __init__(self, depth: int, size_stream: int, size_ff: int, heads: int):
        super().__init__()

        self.size_stream = size_stream

        self.blocks = nn.ModuleList([
            TransformerBlock(size_stream, size_ff, heads) for _ in range(depth)
        ])

    def forward(self, stream, mask):
        for block in self.blocks:
            stream = block(stream, mask)
        return stream


class TokenTransformerModel(nn.Module):
    def __init__(
            self,
            transformer: TransformerModel,
            tokens: int, max_seq_len: int, padding_idx: Optional[int],
    ):
        super().__init__()

        self.tokens = tokens
        self.max_seq_len = max_seq_len
        size_stream = transformer.size_stream

        self.embed = nn.Embedding(tokens, size_stream, padding_idx=padding_idx)
        self.pos_encoding = nn.Parameter(torch.randn(max_seq_len, size_stream) / size_stream ** .5)
        self.un_embed = nn.Linear(size_stream, tokens)

        self.transformer = transformer

        with torch.no_grad():
            self.un_embed.weight *= 0.001
            self.un_embed.bias *= 0.0

    def forward(self, x, mask):
        # (seq, batch)
        seq_len, batch_size = x.shape
        assert seq_len <= self.max_seq_len

        embedded = self.embed(x)
        stream_in = embedded + self.pos_encoding[:seq_len, None, :]
        stream_out = self.transformer(stream_in, mask)
        logits = self.un_embed(stream_out)

        # (seq, batch, token)
        return logits


def build_causal_mask(seq_len: int):
    full = torch.tensor(-torch.inf).expand(seq_len, seq_len)
    return torch.triu(full, diagonal=1)


def main():
    seq_len = 4

    model = TransformerModel(1, 8, 8, 1)
    mask = build_causal_mask(seq_len)

    input = torch.randn(seq_len, 2, 8, requires_grad=True)
    output = model(input, mask)

    output[2, 0, :].sum().backward()
    print(input.grad)


if __name__ == '__main__':
    main()
