import itertools
import os
import shutil
from dataclasses import dataclass
from typing import Optional, List

import einops
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as nnf
import torchvision.utils
from torch import nn

from circuits import generate
from lib.logger import Logger
from lib.plotter import LogPlotter, run_with_plotter


def causal_mask(seq_len: int):
    full = torch.tensor(-torch.inf).expand(seq_len, seq_len)
    return torch.triu(full, diagonal=1)


@dataclass
class Composition:
    q: bool
    k: bool
    v: bool


class Head(nn.Module):
    def __init__(self, stream_size: int, proj_size: int, comp: Composition):
        super().__init__()
        self.proj_size = proj_size

        self.wq = nn.Linear(stream_size, proj_size, bias=False)
        self.wk = nn.Linear(stream_size, proj_size, bias=False)
        self.wv = nn.Linear(stream_size, proj_size, bias=False)
        self.wo = nn.Linear(proj_size, stream_size, bias=False)

        self.comp = comp

    def forward(self, stream, att_mask, prev_stream):
        # stream: BxSxN
        # * we use ... instead of B to allow arbitrarily many batch dims, including none
        # * in einsum we use q/k for the respective dims corresponding to S

        q = self.wq(stream if self.comp.q else prev_stream)
        k = self.wk(stream if self.comp.k else prev_stream)
        v = self.wv(stream if self.comp.v else prev_stream)

        scale = self.proj_size ** 0.5
        att_logit = torch.einsum("...qn,...kn->...qk", q, k) / scale

        if att_mask is not None:
            att_logit = att_logit + att_mask

        att = nnf.softmax(att_logit, dim=-1)
        att_combined = torch.einsum("...qk,...kn->...qn", att, v)

        result = self.wo(att_combined)
        return result, att


class Transformer(nn.Module):
    def __init__(
            self,
            depth: int, heads: int,
            stream_size: int, proj_size: int,
            comp: Composition,
    ):
        super().__init__()
        self.stream_size = stream_size

        self.layers = nn.ModuleList(
            nn.ModuleList(
                Head(stream_size, proj_size, comp)
                for _ in range(heads)
            )
            for _ in range(depth)
        )

    def forward(self, stream, att_mask):
        atts = []
        streams = [stream]
        prev_stream = stream

        for layer in self.layers:
            layer: nn.ModuleList

            layer_delta = 0
            layer_atts = []

            for head in layer:
                head_delta, head_att = head(stream, att_mask, prev_stream)
                layer_delta = layer_delta + head_delta
                layer_atts.append(head_att)

            prev_stream = stream
            stream = stream + layer_delta
            streams.append(stream)
            atts.append(layer_atts)

        return stream, atts, streams


class TokenTransformer(nn.Module):
    def __init__(
            self,
            transformer: Transformer,
            tokens: int,
            output_token_count: int,
            pos_encoding_length: Optional[int],
    ):
        super().__init__()

        stream_size = transformer.stream_size
        self.output_token_count = output_token_count

        if pos_encoding_length is not None:
            self.pos_encoding = nn.Parameter(torch.randn(pos_encoding_length, stream_size) / stream_size ** .5)
        else:
            self.pos_encoding = None

        self.embed = nn.Linear(tokens, stream_size, bias=False)
        self.transformer = transformer

        assert stream_size % output_token_count == 0
        self.un_embed = nn.Linear(stream_size // output_token_count, tokens, bias=False)

    def forward(self, tokens, att_mask):
        # tokens: ...xSxT one-hot encoded
        embedded = self.embed(tokens)

        if self.pos_encoding is not None:
            seq_len = tokens.shape[-2]
            embedded += self.pos_encoding[:seq_len, :]

        result, attn, streams = self.transformer(embedded, att_mask)

        result_split = einops.rearrange(result, "... s (c n) -> ... s c n", c=self.output_token_count)

        logits_split = self.un_embed(result_split)

        return logits_split, attn, streams


@torch.no_grad()
def plots(model: TokenTransformer, atts: List[List[torch.tensor]], plot_weights: bool, run_path: str, bi: int):
    plot_path = os.path.join(run_path, "plots")
    os.makedirs(plot_path, exist_ok=True)

    if plot_weights and model.output_token_count == 1:
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
    run_name = "lookup_all"
    run_path = f"../ignored/circuits/{run_name}/"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_freq = 100
    plot_freq = 100
    plot_weights = True
    print_wrong_freq = 0

    if os.path.exists(run_path):
        shutil.rmtree(run_path)

    tokens = 10
    batch_size = 1024
    seq_len = 256

    stream_size = 128
    proj_size = 32
    heads = 1
    depth = 2
    pos_encoding = True

    def generator():
        # return generate.generate_sample_counting(batch_size, seq_len, tokens)
        # return generate.generate_sample_repeating(batch_size, seq_len, tokens, 3)
        # return generate.generate_sample_lookup(batch_size, seq_len, tokens, flip=True)
        return generate.generate_sample_lookup_all(batch_size, seq_len, tokens)
        # return generate.generate_sample_multi_lookback(batch_size, seq_len, tokens, [1])

    output_token_count = 1

    mask = causal_mask(seq_len).to(device)

    comp = Composition(q=False, k=True, v=False)
    model = TokenTransformer(
        Transformer(depth, heads, stream_size, proj_size, comp),
        tokens, output_token_count, seq_len if pos_encoding else None,
    )
    model.to(device)

    l2_weight = 0.0
    l2_stream = 0.0
    l1_weight = 0.0
    predictable_focus = 0

    optim = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=l2_weight)
    # optim = torch.optim.SGD(model.parameters(), lr=1e-3, weight_decay=l2_weight, momentum=0.0)

    logger = Logger()
    os.makedirs(run_path, exist_ok=False)

    for bi in itertools.count():
        plotter.block_while_paused()

        print(f"Starting batch {bi}")
        logger.start_batch()

        if save_freq != 0 and bi % save_freq == 0:
            os.makedirs(os.path.join(run_path, "models"), exist_ok=True)
            torch.save(model, os.path.join(run_path, "models", f"model_{bi}.pt"))
            logger.save(os.path.join(run_path, "log.npz"))

        # generate data
        sample = generator()

        assert sample.output_token_count == model.output_token_count
        sample = sample.to(device)
        predictable_count = sample.predictable.sum()
        token_count = batch_size * seq_len * output_token_count

        # run the model
        model_input = nnf.one_hot(sample.input_tokens, tokens).float()
        model.train()
        model_output, atts, streams = model(model_input, mask)

        #  plot things
        if plot_freq != 0 and bi % plot_freq == 0:
            plots(model, atts, plot_weights, run_path, bi)

            print("Sequence predictions:")
            topk = torch.topk(model_output[0, :, :, :], k=2, dim=-1)
            for si in range(seq_len):
                topk_list = topk.indices[si, :, ].tolist()
                pred_list = sample.predictable[0, si, :].tolist()
                target_list = sample.output_tokens[0, si, :].tolist()
                print(f"  {sample.input_tokens[0, si]} -> {topk_list} {target_list} pred {pred_list}")

        # compute metrics
        assert model_output.shape == sample.output_tokens.shape + (tokens,), \
            f"Output shape mismatch, {model_output.shape} vs {sample.output_tokens.shape}"

        loss_individual = nnf.cross_entropy(
            model_output.reshape(-1, tokens).double(),
            sample.output_tokens.reshape(-1),
            reduction="none"
        ).view(sample.output_tokens.shape)
        acc_individual = (torch.argmax(model_output, -1) == sample.output_tokens)

        loss_all = loss_individual.mean()
        loss_predictable = (loss_individual * sample.predictable).sum() / predictable_count

        acc_all = acc_individual.float().mean()
        acc_predictable = (acc_individual * sample.predictable).sum() / predictable_count

        loss_uniform = nnf.cross_entropy(torch.full((tokens,), 1 / tokens), torch.tensor(0))
        acc_uniform = 1 / tokens

        acc_all_max = (predictable_count + (token_count - predictable_count) / tokens) / token_count

        if print_wrong_freq != 0 and bi % print_wrong_freq == 0:
            torch.set_printoptions(linewidth=1000)
            wrong_indices = torch.argwhere(acc_individual * sample.predictable)
            if len(wrong_indices) > 0:
                wrong_bi = wrong_indices[0, 0]
                print("Wrong sequence:", sample.input_tokens[wrong_bi])
                print(f"Input {sample.input_tokens[wrong_bi]}")
                print(f"Expected {sample.output_tokens[wrong_bi].T}")
                print(f"Got {torch.argmax(model_output, -1)[bi].T}")

        logger.log("loss", "train", loss_all)
        logger.log("loss", "uniform", loss_uniform)
        logger.log("loss", "predictable", loss_predictable)
        logger.log("log(loss)", "train", torch.log10(loss_all))
        logger.log("log(loss)", "uniform", torch.log10(loss_uniform))
        logger.log("log(loss)", "predictable", torch.log10(loss_predictable))
        logger.log("acc", "train", acc_all)
        logger.log("acc", "uniform", acc_uniform)
        logger.log("acc", "max", acc_all_max)
        logger.log("acc", "predictable", acc_predictable)
        logger.log("acc", "1", 1)

        token_output_freq = (sample.output_tokens.unsqueeze(-1) == torch.arange(tokens, device=device)) \
            .view(-1, tokens).float().mean(dim=0)
        for t in range(tokens):
            logger.log("freq", f"token {t}", token_output_freq[t])

        stream_weight = sum((s * s).mean() for s in streams)
        logger.log("norm", "stream_weight", stream_weight)

        # training loss and backward step
        total_loss = loss_all
        total_loss += predictable_focus * loss_predictable
        total_loss += l2_stream * stream_weight
        total_loss += l1_weight * sum(param.abs().mean() for param in model.parameters())

        # TODO remove this
        # total_loss += 0.01 * (q_norm + v_norm)

        optim.zero_grad()
        total_loss.backward()
        optim.step()

        # log more things
        for name, param in model.named_parameters():
            logger.log("param", name, param.abs().mean())
            grad_ratio = param.grad.abs().mean() / param.abs().mean()
            logger.log("grad/param", name, grad_ratio)
            logger.log("log(grad/param)", name, torch.log10(grad_ratio))

        plotter.update(logger)


if __name__ == '__main__':
    run_with_plotter(main)
