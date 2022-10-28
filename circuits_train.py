import itertools
import os
import shutil
from typing import Optional, List

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as nnf
import torchvision.utils
from torch import nn

from lib.logger import Logger
from lib.plotter import LogPlotter, run_with_plotter


def causal_mask(seq_len: int):
    full = torch.tensor(-torch.inf).expand(seq_len, seq_len)
    return torch.triu(full, diagonal=1)


class Head(nn.Module):
    def __init__(self, stream_size: int, proj_size: int):
        super().__init__()
        self.proj_size = proj_size

        self.wq = nn.Linear(stream_size, proj_size, bias=False)
        self.wk = nn.Linear(stream_size, proj_size, bias=False)
        self.wv = nn.Linear(stream_size, proj_size, bias=False)
        self.wo = nn.Linear(proj_size, stream_size, bias=False)

    def forward(self, stream, att_mask):
        # stream: BxSxN
        # * we use ... instead of B to allow arbitrarily many batch dims, including none
        # * in einsum we use q/k for the respective dims corresponding to S

        q = self.wq(stream)
        k = self.wk(stream)
        v = self.wv(stream)

        scale = self.proj_size ** 0.5
        att_logit = torch.einsum("...qn,...kn->...qk", q, k) / scale

        if att_mask is not None:
            att_logit = att_logit + att_mask

        att = nnf.softmax(att_logit, dim=-1)
        att_combined = torch.einsum("...qk,...kn->...qn", att, v)

        result = self.wo(att_combined)
        return result, att


class TokenTransformer(nn.Module):
    def __init__(
            self,
            depth: int, heads: int,
            stream_size: int, proj_size: int,
            tokens: int, pos_encoding_length: Optional[int],
    ):
        super().__init__()

        if pos_encoding_length is not None:
            self.pos_encoding = nn.Parameter(torch.randn(pos_encoding_length, stream_size))
        else:
            self.pos_encoding = None

        self.embed = nn.Linear(tokens, stream_size, bias=False)
        self.un_embed = nn.Linear(stream_size, tokens, bias=False)

        self.transformer = Transformer(depth, heads, stream_size, proj_size)

    def forward(self, tokens, att_mask):
        # tokens: ...xSxT one-hot encoded
        embedded = self.embed(tokens)
        result, attn, streams = self.transformer(embedded, att_mask)
        logits = self.un_embed(result)
        return logits, attn, streams


class Transformer(nn.Module):
    def __init__(
            self,
            depth: int, heads: int,
            stream_size: int, proj_size: int,
    ):
        super().__init__()
        self.stream_size = stream_size

        self.layers = nn.ModuleList(
            nn.ModuleList(
                Head(stream_size, proj_size)
                for _ in range(heads)
            )
            for _ in range(depth)
        )

    def forward(self, stream, att_mask):
        atts = []
        streams = [stream]

        for layer in self.layers:
            layer: nn.ModuleList

            layer_delta = 0
            layer_atts = []

            for head in layer:
                head_delta, head_att = head(stream, att_mask)
                layer_delta = layer_delta + head_delta
                layer_atts.append(head_att)

            stream = stream + layer_delta
            streams.append(stream)
            atts.append(layer_atts)

        return stream, atts, streams


def generate_counting_seq(batch_size: int, seq_len: int, tokens: int):
    assert tokens > seq_len
    starts = torch.randint(tokens - seq_len + 1, (batch_size,))
    data_int = starts[:, None] + torch.arange(seq_len)[None, :]
    return data_int


def ceil_div(x, y):
    return -(x // -y)


def generate_repeating_sequence(batch_size: int, seq_len: int, tokens: int, period: int):
    starts = torch.randint(tokens, (batch_size, period))
    repeated = starts.repeat(1, ceil_div(seq_len, period))
    data_int = repeated[:, :seq_len]
    return data_int


def generate_lookup_sequence(batch_size: int, seq_len: int, tokens: int):
    noise = torch.randint(1, tokens, (batch_size, seq_len))

    second = torch.randint(3, seq_len, (batch_size,))
    first = (torch.rand(batch_size) * (second - 2) + 1).long()

    assert torch.all(second - first > 1)

    bi = torch.arange(batch_size)

    noise[bi, first] = noise[bi, second]
    noise[bi, second - 1] = 0
    noise[bi, first - 1] = 0

    return noise, second


@torch.no_grad()
def plots(model: TokenTransformer, atts: List[List[torch.tensor]], plot_weights: bool, run_path: str, bi: int):
    plot_path = os.path.join(run_path, "plots")
    os.makedirs(plot_path, exist_ok=True)

    if plot_weights:
        embed_matrix = model.un_embed.weight @ model.embed.weight

        plt.matshow(embed_matrix.cpu())
        plt.ylabel("output token")
        plt.xlabel("input token")
        plt.title("Wu @ We")
        plt.colorbar()
        plt.savefig(os.path.join(plot_path, f"embedding_{bi}.png"))
        plt.close()

        if model.pos_encoding is not None:
            plt.matshow((model.un_embed.weight @ model.pos_encoding.T).cpu())
            plt.ylabel("output token")
            plt.xlabel("stream_size")
            plt.title("Wu @ Epos")
            plt.colorbar()
            plt.savefig(os.path.join(plot_path, f"pos_encoding_{bi}.png"))
            plt.close()

        if len(model.transformer.layers) > 0:
            if len(model.transformer.layers[0]) > 0:
                head: nn.Module = model.transformer.layers[0][0]
                head: Head

                head_matrix = model.un_embed.weight @ head.wo.weight @ head.wv.weight @ model.embed.weight
                plt.matshow(head_matrix.cpu())
                plt.title("Wu @ (Wo_11 @ Wv_11) @ We")
                plt.colorbar()
                plt.savefig(os.path.join(plot_path, f"head_matrix_{bi}.png"))
                plt.close()

                plt.matshow((embed_matrix + head_matrix).cpu())
                plt.title("Wu @ We + Wu @ (Wo_11 @ Wv_11) @ We")
                plt.colorbar()
                plt.savefig(os.path.join(plot_path, f"combined_matrix_{bi}.png"))
                plt.close()

    if len(atts) > 0:
        atts_tensor = torch.stack([torch.stack(layer_atts) for layer_atts in atts])

        # shape (layers, heads, batch, query, key)
        layers, heads, _, q_count, k_count = atts_tensor.shape
        total_heads = layers * heads

        att_image = atts_tensor[:, :, 0, :, :].view(total_heads, 1, q_count, k_count)
        att_path = os.path.join(plot_path, f"att_{bi}.png")
        torchvision.utils.save_image(att_image, att_path, nrow=heads, normalize=True)


def main(plotter: LogPlotter):
    run_name = "token_transformer"
    run_path = f"ignored/circuits/{run_name}/"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_freq = 100
    plot_freq = 100
    plot_weights = True

    if os.path.exists(run_path):
        shutil.rmtree(run_path)

    tokens = 10
    batch_size = 1024
    seq_length = 16

    stream_size = 128
    proj_size = 32
    heads = 1
    depth = 2

    mask = causal_mask(seq_length).to(device)

    model = TokenTransformer(depth, heads, stream_size, proj_size, tokens, None)
    model.to(device)

    weight_decay = 0.1
    stream_decay = 0.0
    weight_decay_abs = 0.0
    knowable_focus = 0.0

    optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=weight_decay)

    logger = Logger()
    os.makedirs(run_path, exist_ok=False)

    for bi in itertools.count():
        print(f"Starting batch {bi}")
        logger.start_batch()

        if save_freq != 0 and bi % save_freq == 0:
            os.makedirs(os.path.join(run_path, "models"), exist_ok=True)
            torch.save(model, os.path.join(run_path, "models", f"model_{bi}.pt"))
            logger.save(os.path.join(run_path, "log.npz"))

        # generate data
        # data_int = generate_counting_seq(batch_size, seq_length + 1, tokens)
        # predictable_tokens = seq_length

        # period = 2
        # data_int = generate_repeating_sequence(batch_size, seq_length + 1, tokens, period)
        # predictable_tokens = seq_length - period + 1

        data_int, index_second = generate_lookup_sequence(batch_size, seq_length + 1, tokens)
        predictable_tokens = 0

        data_int = data_int.to(device)
        data_one_hot = nnf.one_hot(data_int, tokens).float()
        model_input = data_one_hot[:, :-1, :]
        model_target_int = data_int[:, 1:]

        model.train()
        model_output, atts, streams = model(model_input, mask)

        if plot_freq != 0 and bi % plot_freq == 0:
            plots(model, atts, plot_weights, run_path, bi)

            print("Sequence predictions:")
            for si in range(seq_length):
                topk = torch.topk(model_output[0, si, :], k=4)
                print(f"  {data_int[0, si]} -> {topk.indices.tolist()}")

        loss = nnf.cross_entropy(model_output.reshape(-1, tokens).double(), model_target_int.reshape(-1))
        acc = (torch.argmax(model_output, -1) == model_target_int).float().mean()

        stream_weight = sum((s * s).mean() for s in streams)

        brange = torch.arange(batch_size)
        model_output_second = model_output[brange, index_second - 1, :]
        model_target_int_second = model_target_int[brange, index_second - 1]

        loss_second = nnf.cross_entropy(model_output_second, model_target_int_second)
        acc_second = (torch.argmax(model_output_second, -1) == model_target_int_second).float().mean()

        loss_uniform = nnf.cross_entropy(torch.full((tokens,), 1 / tokens), torch.tensor(0))
        acc_uniform = 1 / tokens
        acc_max = ((seq_length - predictable_tokens) * 1 / tokens + predictable_tokens) / seq_length

        logger.log("loss", "train", loss)
        logger.log("loss", "uniform", loss_uniform)
        logger.log("loss", "second", loss_second)
        logger.log("log(loss)", "train", torch.log10(loss))
        logger.log("log(loss)", "uniform", torch.log10(loss_uniform))
        logger.log("log(loss)", "second", torch.log10(loss_second))
        logger.log("acc", "train", acc)
        logger.log("acc", "uniform", acc_uniform)
        logger.log("acc", "max", acc_max)
        logger.log("acc", "second", acc_second)
        logger.log("acc", "1", 1)

        logger.log("norm", "stream_weight", stream_weight)

        optim.zero_grad()

        total_loss = loss
        total_loss += knowable_focus * loss_second
        total_loss += stream_decay * stream_weight
        total_loss += weight_decay_abs * sum(param.abs().mean() for param in model.parameters())

        total_loss.backward()
        optim.step()

        for name, param in model.named_parameters():
            logger.log("param", name, param.abs().mean())
            grad_ratio = param.grad.abs().mean() / param.abs().mean()
            logger.log("grad/param", name, grad_ratio)
            logger.log("log(grad/param)", name, torch.log10(grad_ratio))

        plotter.update(logger)


if __name__ == '__main__':
    run_with_plotter(main)
