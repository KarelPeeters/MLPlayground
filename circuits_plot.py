import os
from typing import Optional

import matplotlib.pyplot as plt
import torch.jit
import torch.nn.functional as nnf

from circuits_train import Transformer, Head, causal_mask


def plot_matrix(value, name: str, y: Optional[str], x: Optional[str], min_max=None):
    if min_max is not None:
        plt.matshow(value, vmin=min_max[0], vmax=min_max[1])
    else:
        plt.matshow(value)

    if x is not None:
        plt.xlabel(x)
    if y is not None:
        plt.ylabel(y)

    plt.colorbar()
    plt.title(name)
    plt.tight_layout()

    os.makedirs("ignored/circuits/plots", exist_ok=True)
    plt.savefig(f"ignored/circuits/plots/{name}.png", bbox_inches='tight')
    plt.close()


@torch.no_grad()
def main():
    # import classes and load model
    _ = Transformer(8, 0, 8, 8, 0, None)
    _ = Head(8, 8)
    model: Transformer = torch.load("ignored/circuits/repeating_new/models/model_6000.pt")
    model.to("cpu")
    model.eval()

    tokens = 10

    # seq_len = 16
    # torch.random.manual_seed(0)
    # data_int = generate_lookup_sequence(1, seq_len + 1, tokens)[0].squeeze(0)

    # TODO try forcing sharp attention patterns and see if they still work
    #   (easily achievable by adding things to the masks)

    data_int = torch.tensor([5, 2, 0, 7, 8, 6, 4, 9, 0, 7, 3, 4])
    seq_len = len(data_int) - 1

    assert len(data_int) == seq_len + 1
    assert torch.all(data_int < tokens)

    mask = causal_mask(seq_len)
    input = nnf.one_hot(data_int[:-1], tokens).float()

    mask0 = mask.clone()
    mask1 = mask.clone()

    # mask0[2:, 0:2] = -torch.inf
    # mask0[2:6, 3:6] = -torch.inf
    # mask0[:, 7:10] = -torch.inf
    # mask0[:, 11:] = -torch.inf

    head0 = model.layers[0][0]
    head1 = model.layers[1][0]

    stream0 = model.embed(input)
    delta0, att0 = head0(stream0, mask0)
    stream1 = stream0 + delta0
    delta1, att1 = head1(stream1, mask1)
    stream2 = stream1 + delta1
    logits = model.un_embed(stream2)

    print(f"sequence: {data_int}")
    for i, token in enumerate(data_int.tolist()):
        if i > 0:
            topk = logits[i - 1].softmax(0).topk(3)
            token_pred = {t.item(): p.item() for t, p in zip(topk.indices, topk.values)}
        else:
            token_pred = {}

        token_pred_str = ", ".join(f"{t}: {v:.2f}" for t, v in token_pred.items())
        print(f"    {i:>2}: {token} (pred {token_pred_str})")

    # plot weights
    # for name, param in model.named_parameters():
    #     if len(param.shape) == 2:
    #         plot_matrix(param, f"weight {name}", None, None)

    # extract weights
    Wu = model.un_embed.weight
    We = model.embed.weight

    Wq0 = head0.wq.weight
    Wk0 = head0.wk.weight
    Wv0 = head0.wv.weight
    Wo0 = head0.wo.weight

    Wq1 = head1.wq.weight
    Wk1 = head1.wk.weight
    Wv1 = head1.wv.weight
    Wo1 = head1.wo.weight

    # activations
    plot_matrix(nnf.softmax(logits, dim=1), "act softmax", "seq", "digit")
    plot_matrix(att0, "act att head0", "q", "k")
    plot_matrix(att1, "act att head1", "q", "k")
    plot_matrix(att1 @ att0, "act att virtual", "q", "k")

    plot_matrix(Wu @ stream0.T, "act token stream 0", "seq", "token")
    plot_matrix(Wu @ stream1.T, "act token stream 1", "seq", "token")
    plot_matrix(Wu @ stream2.T, "act token stream 2", "seq", "token")
    plot_matrix(Wu @ delta0.T, "act delta 0", "seq", "token")
    plot_matrix(Wu @ delta1.T, "act delta 1", "seq", "token")

    s_mats = [stream0, stream1, stream2, delta0, delta1]
    s_min_max = min(x.min() for x in s_mats), max(x.max() for x in s_mats)
    plot_matrix(stream0, "act raw stream 0", "seq", "n", s_min_max)
    plot_matrix(stream1, "act raw stream 1", "seq", "n", s_min_max)
    plot_matrix(stream2, "act raw stream 2", "seq", "n", s_min_max)
    plot_matrix(delta0, "act raw delta 0", "seq", "n", s_min_max)
    plot_matrix(delta1, "act raw delta 1", "seq", "n", s_min_max)

    plot_matrix(head0.wq(stream0), "act head 0 Q", None, None)
    plot_matrix(head0.wk(stream0), "act head 0 K", None, None)
    plot_matrix(head0.wv(stream0), "act head 0 V", None, None)

    plot_matrix(head1.wq(stream1), "act head 1 Q", None, None)
    plot_matrix(head1.wk(stream1), "act head 1 K", None, None)
    plot_matrix(head1.wv(stream1), "act head 1 V", None, None)

    plot_matrix(Wu @ Wo0 @ head0.wv(stream0).T, "act token head 0 V", None, None)
    plot_matrix(Wu @ Wo1 @ head1.wv(stream1).T, "act token head 1 V", None, None)

    # plot circuits
    Wov0 = Wo0 @ Wv0
    Wov1 = Wo1 @ Wv1
    Wqk0 = Wq0.T @ Wk0
    Wqk1 = Wq1.T @ Wk1

    plot_matrix(Wu @ We, "circuit direct", "dest token", "src token")
    plot_matrix(We.T @ Wqk0 @ We, "circuit qk 0", "dest token", "src token")
    plot_matrix(We.T @ Wqk1 @ We, "circuit qk 1", "dest token", "src token")
    plot_matrix(Wu @ Wov0 @ We, "circuit ov 0", "dest token", "src token")
    plot_matrix(Wu @ Wov1 @ We, "circuit ov 1", "dest token", "src token")
    plot_matrix(Wu @ Wov1 @ Wov0 @ We, "circuit ov virtual", "dest token", "src token")

    plot_matrix(We.T @ Wov0 @ Wqk1 @ We, "circuit comp Q", None, None)
    plot_matrix(We.T @ Wqk1 @ Wov0 @ We, "circuit comp K", None, None)
    plot_matrix(We.T @ Wov1 @ Wqk1 @ Wov0 @ We, "circuit comp QK", None, None)

    q_norm = torch.linalg.matrix_norm(Wqk1.T @ Wov0) / \
             (torch.linalg.matrix_norm(Wqk1.T) * torch.linalg.matrix_norm(Wov0))
    k_norm = torch.linalg.matrix_norm(Wqk1 @ Wov0) / \
             (torch.linalg.matrix_norm(Wqk1) * torch.linalg.matrix_norm(Wov0))
    v_norm = torch.linalg.matrix_norm(Wov1 @ Wov0) / \
             (torch.linalg.matrix_norm(Wov1) * torch.linalg.matrix_norm(Wov0))

    print("Q comp", q_norm)
    print("K comp", k_norm)
    print("V comp", v_norm)


if __name__ == '__main__':
    main()
