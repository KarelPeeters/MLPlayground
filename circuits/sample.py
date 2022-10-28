from dataclasses import dataclass

import torch


@dataclass
class Sample:
    batch_size: int
    seq_len: int
    output_token_count: int

    # B x S (int)
    input_tokens: torch.tensor
    # B x S x C (int)
    output_tokens: torch.tensor
    # B x S x C (bool)
    knowable: torch.tensor

    def __post_init__(self):
        b = self.batch_size
        s = self.seq_len
        c = self.output_token_count

        assert self.input_tokens.shape == (b, s)
        assert self.output_tokens.shape == (b, s, c)
        assert self.knowable.shape == (b, s, c)

    def to(self, device):
        return Sample(
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            output_token_count=self.output_token_count,
            input_tokens=self.input_tokens.to(device),
            output_tokens=self.output_tokens.to(device),
            knowable=self.knowable.to(device),
        )


def ceil_div(x, y):
    return -(x // -y)


def generate_sample_counting(batch_size: int, seq_len: int, tokens: int):
    assert tokens > seq_len
    starts = torch.randint(tokens - seq_len, (batch_size,))

    all_tokens = starts[:, None] + torch.arange(seq_len + 1)[None, :]

    return Sample(
        batch_size, seq_len, 1,
        all_tokens[:, :-1],
        all_tokens[:, 1:, None],
        torch.tensor(True).expand(batch_size, seq_len, 1),
    )

# def generate_sample_repeating(batch_size: int, seq_len: int, tokens: int, period: int):
#     starts = torch.randint(tokens, (batch_size, period))
#     repeated = starts.repeat(1, ceil_div(seq_len, period))
#     data_int = repeated[:, :seq_len]
#     return data_int

# def generate_sample_lookup(batch_size: int, seq_len: int, tokens: int):
#     noise = torch.randint(1, tokens, (batch_size, seq_len))
#
#     second = torch.randint(3, seq_len, (batch_size,))
#     first = (torch.rand(batch_size) * (second - 2) + 1).long()
#
#     assert torch.all(second - first > 1)
#
#     bi = torch.arange(batch_size)
#
#     noise[bi, first] = noise[bi, second]
#     noise[bi, second - 1] = 0
#     noise[bi, first - 1] = 0
#
#     return noise, second
