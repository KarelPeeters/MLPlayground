import json
import os
from typing import List

import ktoken
import torch
import torch.nn.functional as nnf

from lib.logger import Logger
from lib.plotter import run_with_plotter, LogPlotter
from nlp.model import TransformerModel, build_causal_mask, TokenTransformerModel


def main(plotter: LogPlotter):
    run_name = "first_nlp_run"
    run_path = os.path.join("ignored/nlp", run_name)
    os.makedirs(run_path, exist_ok=False)

    token_path = r"C:\Documents\Programming\Rust\kToken\ignored\tokens.json"
    data_paths = [
        r"\\192.168.0.10\Documents\Download\the-pile\00.jsonl.zst",
        r"\\192.168.0.10\Documents\Download\the-pile\01.jsonl.zst",
        r"\\192.168.0.10\Documents\Download\the-pile\29.jsonl.zst",
    ]

    save_freq = 512

    batch_size = 32
    seq_len = 128

    with open(token_path, "r") as f:
        tokens: List[List[int]] = json.load(f)["tokens"]
    reader = ktoken.BatchTokenReader(tokens, data_paths, batch_size, seq_len + 1, 2 * batch_size, 4)

    print(f"Tokens: {len(tokens)}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    depth = 4
    size_stream = 1024
    size_ff = 1024
    heads = 8

    causal_mask = build_causal_mask(seq_len).to(device)

    model = TokenTransformerModel(
        TransformerModel(depth, size_stream, size_ff, heads),
        len(tokens) + 1, seq_len, len(tokens)
    )
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    logger = Logger()

    for bi, batch_tokens_raw in enumerate(reader):
        if save_freq != 0 and bi % save_freq == 0:
            os.makedirs(os.path.join(run_path, "models"), exist_ok=True)
            torch.save(model, os.path.join(run_path, "models", f"model_{bi}.pt"))
            logger.save(os.path.join(run_path, "log.npz"))

        plotter.block_while_paused()
        logger.start_batch()

        batch_tokens_raw = torch.tensor(batch_tokens_raw, device=device).T
        assert batch_tokens_raw.shape == (seq_len + 1, batch_size)

        # replace -1 padding with final token (which is marked as padding in the transformer)
        is_present = batch_tokens_raw != -1
        batch_tokens = torch.where(is_present, batch_tokens_raw, len(tokens))
        logger.log("fill", "fill rate", is_present.sum() / batch_tokens_raw.numel())

        batch_input = batch_tokens[:-1, :]
        batch_target = batch_tokens[1:, :]
        is_present_target = is_present[1:, :]

        model.train()
        batch_logits = model(batch_input, causal_mask)

        # TODO lower the loss for tokens that also start with the same characters?
        loss_full = nnf.cross_entropy(
            batch_logits.reshape(-1, len(tokens) + 1),
            batch_target.reshape(-1).long(),
            reduction="none"
        ).view(seq_len, batch_size)

        loss = (loss_full * is_present_target).sum() / is_present_target.sum()

        logger.log("loss", "loss", loss)
        logger.log("log(loss)", "log(loss)", torch.log10(loss))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        plotter.update(logger)


if __name__ == '__main__':
    run_with_plotter(main)
