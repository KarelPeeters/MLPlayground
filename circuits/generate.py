from dataclasses import dataclass
from typing import List

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


def rand_delta_ints(min: int, max: int, delta: int, batch_size: int, count: int):
    # this approximates rejection sampling pretty well when (max-min) is large enough
    #   the only difference is that equal pre-delta values are overrepresented

    total_delta = delta * (count - 1)

    assert max > min
    assert delta >= 0
    assert total_delta < max - min

    x_raw = torch.randint(max - min - total_delta, (batch_size, count))
    x_sorted = torch.sort(x_raw, -1).values
    x_delta = x_sorted + torch.arange(count) * delta + min

    return x_delta


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

    zero_indices = rand_delta_ints(1, seq_len + 1, 2, batch_size, 2)

    second = zero_indices[:, 1]
    first = zero_indices[:, 0]

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


def generate_multi_lookback(batch_size: int, seq_len: int, tokens: int, deltas: List[int]):
    assert all(d >= 0 for d in deltas)
    max_delta = max(deltas)

    all_tokens = torch.randint(tokens, (batch_size, seq_len + max_delta))
    all_pred = torch.full((seq_len + max_delta,), True)
    all_pred[:max_delta] = False

    input_tokens = all_tokens[:, max_delta:]

    output_tokens = []
    pred = []

    for d in deltas:
        offset = max_delta - d
        output_tokens.append(all_tokens[:, offset:offset + seq_len])
        pred.append(all_pred[offset:offset + seq_len])

    sample = Sample(
        batch_size, seq_len, len(deltas),
        input_tokens,
        torch.stack(output_tokens, dim=2),
        einops.repeat(torch.stack(pred, dim=1), "s c -> b s c", b=batch_size),
    )

    return sample
