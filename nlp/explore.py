import itertools
import json

import torch
import torchvision.utils
from ktoken import Tokenizer

from nlp.model import TokenTransformerModel, build_causal_mask


def token_to_str(raw_tokens, token):
    return ''.join(chr(c) for c in raw_tokens[token])


@torch.no_grad()
def main():
    model_path = "../ignored/nlp/first_nlp_run/models/model_24576.pt"
    token_path = r"C:\Documents\Programming\Rust\kToken\ignored\tokens.json"

    with open(token_path, "r") as f:
        raw_tokens = json.load(f)["tokens"]
        tokenizer = Tokenizer(raw_tokens)

    tokens = tokenizer.tokenize("The United States of ")
    print(tokens)

    print([token_to_str(raw_tokens, t) for t in list(tokens)])

    mask = build_causal_mask(len(tokens))
    model: TokenTransformerModel = torch.load(model_path, map_location="cpu")

    print(f"Parameters: {sum(p.numel() for p in model.parameters())}")

    logits = model(torch.tensor(tokens).view(-1, 1), mask)
    print(logits.shape)
    logits = logits.squeeze(1)

    topk = torch.topk(logits, 4, dim=1).indices
    # print(topk)

    print("Topk:")
    for row in topk:
        print(" ", [token_to_str(raw_tokens, t) for t in row.tolist()])

    torchvision.utils.save_image(model.pos_encoding, "pos_encoding.png", normalize=True)
    torchvision.utils.save_image(torch.fft.rfft(model.pos_encoding, dim=0).abs(), "pos_encoding_fft.png", normalize=True)

    bigram = model.embed.weight @ model.un_embed.weight.T
    torchvision.utils.save_image(bigram, "bigram.png", normalize=True)
    torchvision.utils.save_image(torch.log10(bigram), "bigram_log.png", normalize=True)

    top_k = torch.topk(bigram.flatten(0), 32)
    bottom_k = torch.topk(bigram.flatten(0), 32, largest=False)
    model_tokens = len(bigram)

    print("Topk:")
    for i, value in itertools.chain(zip(top_k.indices, top_k.values), zip(bottom_k.indices, bottom_k.values)):
        a_i = i // model_tokens
        b_i = i % model_tokens

        a = token_to_str(raw_tokens, a_i) if a_i < len(raw_tokens) else None
        b = token_to_str(raw_tokens, b_i) if b_i < len(raw_tokens) else None
        print(f"({repr(a)})->({repr(b)}): value {value}")


if __name__ == '__main__':
    main()
