import itertools
import os

import torch
import torch.nn.functional as nnf
import torchvision.utils
from torch import nn

device = "cuda"
tokens = 1024
token_size = 256
model = nn.Embedding(tokens, token_size)
model.to(device)

batch_size = 1024

optimizer = torch.optim.AdamW(model.parameters(), 1e-3)

# token_target = torch.eye(tokens).to(device)
token_target = (torch.arange(tokens)[None, :] - torch.arange(tokens)[:, None]).abs() < 5
token_target = token_target.float().to(device)

os.makedirs("ignored/trivial_emb/", exist_ok=True)
torchvision.utils.save_image(token_target, f"ignored/trivial_emb/token_target.png", normalize=True)

for bi in itertools.count():
    # i = torch.randint(tokens, (batch_size, 2))
    # y_target = torch.eq(i[:, 0], i[:, 1]).float()

    # hidden = model(i)
    # y_pred_logit = (hidden[:, 0] * hidden[:, 1]).sum(dim=1) / token_size ** .5
    # loss_split = nnf.binary_cross_entropy_with_logits(y_pred_logit, y_target, reduction="none")
    # loss = loss_split.mean()

    i = torch.arange(tokens, device=device)
    model.train()
    w = model(i)
    sim_logit = w @ w.T

    loss_sim = nnf.binary_cross_entropy_with_logits(sim_logit, token_target)
    loss_weight = w.abs().mean()

    loss_total = loss_sim + .01 * loss_weight

    if bi % 500 == 0:
        os.makedirs("ignored/trivial_emb/", exist_ok=True)
        torchvision.utils.save_image(model.weight, f"ignored/trivial_emb/emb_{bi}.png", normalize=True)
        torchvision.utils.save_image(
            sim_logit.sigmoid(), f"ignored/trivial_emb/sim_{bi}.png",
            normalize=True, value_range=(0, 1)
        )

    optimizer.zero_grad()
    loss_total.backward()
    optimizer.step()

    print(f"Batch {bi}: loss_sim={loss_sim.item()}, loss_weight={loss_weight.item()}")
