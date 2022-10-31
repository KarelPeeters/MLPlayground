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


def generate_sample_lookup(batch_size: int, seq_len: int, tokens: int, flip: bool = False):
    assert tokens >= 2
    all_tokens = torch.randint(1, tokens, (batch_size, seq_len + 1))

    zero_indices = rand_delta_ints(1, seq_len + 1, 2, batch_size, 2)

    second = zero_indices[:, 1]
    first = zero_indices[:, 0]

    assert torch.all(second - first > 1)

    brange = torch.arange(batch_size)

    all_tokens[brange, first] = all_tokens[brange, second]
    all_tokens[brange, second - 1] = 0
    all_tokens[brange, first - 1] = 0

    predictable = torch.zeros(batch_size, seq_len, 1, dtype=torch.bool)
    predictable[brange, second - 1] = True

    if flip:
        tmp = all_tokens[brange, first].clone()
        all_tokens[brange, first] = all_tokens[brange, first - 1]
        all_tokens[brange, first - 1] = tmp

    return Sample(
        batch_size, seq_len, 1,
        all_tokens[:, :-1],
        all_tokens[:, 1:, None],
        # nothing except the token right after zero is predictable
        # (we're still missing a small bonus for the possibly higher zero probability)
        predictable
    )


def generate_sample_lookup_all(batch_size: int, seq_len: int, tokens: int):
    """
    For every token, predict the token that followed the last occurrence of it.
    If this is the first time we've seen this token, predict 0.
    """

    brange = torch.arange(batch_size)

    input_tokens = torch.randint(1, tokens, (batch_size, seq_len))

    output_tokens = torch.zeros(batch_size, seq_len, dtype=torch.int64)
    memory = torch.zeros(batch_size, tokens, dtype=torch.int64)

    prev_input = torch.zeros(batch_size, dtype=torch.int64)

    for i in range(seq_len):
        curr_input = input_tokens[:, i]
        # store to memory first, so we can immediately output the current input if applicable
        memory[brange, prev_input] = curr_input

        curr_output = memory[brange, curr_input]
        output_tokens[brange, i] = curr_output

        prev_input = curr_input

    return Sample(
        batch_size, seq_len, 1,
        input_tokens,
        output_tokens[:, :, None],
        # everything is predictable
        torch.tensor(True).expand(batch_size, seq_len, 1),
    )


def generate_sample_multi_lookback(batch_size: int, seq_len: int, tokens: int, deltas: List[int]):
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


def main():
    torch.random.manual_seed(10)
    sample = generate_sample_lookup_all(2, 16, 10)

    print(sample.input_tokens)
    print(sample.output_tokens.mT)
    print(sample.predictable.mT)


if __name__ == '__main__':
    main()
