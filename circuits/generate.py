from dataclasses import dataclass

import einops
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
    predictable: torch.tensor

    def __post_init__(self):
        b = self.batch_size
        s = self.seq_len
        c = self.output_token_count

        assert self.input_tokens.shape == (b, s)
        assert self.output_tokens.shape == (b, s, c)
        assert self.predictable.shape == (b, s, c)

    def to(self, device):
        return Sample(
            batch_size=self.batch_size,
            seq_len=self.seq_len,
            output_token_count=self.output_token_count,
            input_tokens=self.input_tokens.to(device),
            output_tokens=self.output_tokens.to(device),
            predictable=self.predictable.to(device),
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
        # everything is predictable
        torch.tensor(True).expand(batch_size, seq_len, 1),
    )


def generate_sample_repeating(batch_size: int, seq_len: int, tokens: int, period: int):
    starts = torch.randint(tokens, (batch_size, period))
    repeated = starts.repeat(1, ceil_div(seq_len + 1, period))
    all_tokens = repeated[:, :seq_len + 1]

    return Sample(
        batch_size, seq_len, 1,
        all_tokens[:, :-1],
        all_tokens[:, 1:, None],
        # everything except the first period-1 is predictable
        einops.repeat(torch.arange(seq_len) >= period - 1, "s -> b s 1", b=batch_size)
    )


def generate_sample_lookup(batch_size: int, seq_len: int, tokens: int):
    all_tokens = torch.randint(1, tokens, (batch_size, seq_len + 1))

    second = torch.randint(3, seq_len + 1, (batch_size,))
    first = (torch.rand(batch_size) * (second - 2) + 1).long()

    assert torch.all(second - first > 1)

    bi = torch.arange(batch_size)

    all_tokens[bi, first] = all_tokens[bi, second]
    all_tokens[bi, second - 1] = 0
    all_tokens[bi, first - 1] = 0

    predictable = torch.zeros(batch_size, seq_len, 1, dtype=torch.bool)
    predictable[bi, second - 1] = True

    return Sample(
        batch_size, seq_len, 1,
        all_tokens[:, :-1],
        all_tokens[:, 1:, None],
        # nothing except the token right after zero is predictable
        # (we're still missing a small bonus for the possibly higher zero probability)
        predictable
    )
