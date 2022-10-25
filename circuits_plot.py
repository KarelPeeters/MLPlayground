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
    os.makedirs("ignored/circuits/plots", exist_ok=True)
    plt.rcParams.update({'font.size': 16})
    plt.savefig(f"ignored/circuits/plots/{name}.png")
    plt.close()


@torch.no_grad()
def main():
    # import classes and load model
    _ = Transformer(8, 0, 8, 8, 0, None)
    _ = Head(8, 8)
    model: Transformer = torch.load("ignored/circuits/repeating/model_5000.pt")
    model.to("cpu")
    model.eval()

    tokens = 10

    # seq_len = 16
    # torch.random.manual_seed(0)
    # data_int = generate_lookup_sequence(1, seq_len + 1, tokens)[0].squeeze(0)

    data_int = torch.tensor([5, 2, 0, 7, 8, 6, 0, 7, 3, 4, 0, 7, 3])
    seq_len = len(data_int) - 1

    print(f"sequence: {data_int}")
    for i, token in enumerate(data_int.tolist()):
        print(f"    {i:>2}: {token}")

    assert len(data_int) == seq_len + 1
    assert torch.all(data_int < tokens)

    mask = causal_mask(seq_len)
    input = nnf.one_hot(data_int[:-1], tokens).float()

    stream0 = model.embed(input)
    delta0, att0 = model.layers[0][0](stream0, mask)
    stream1 = stream0 + delta0
    delta1, att1 = model.layers[1][0](stream1, mask)
    stream2 = stream1 + delta1
    logits = model.un_embed(stream2)

    plot_matrix(nnf.softmax(logits, dim=1), "softmax", "seq", "digit")
    plot_matrix(att0, "att0", "q", "k")
    plot_matrix(att1, "att1", "q", "k")
    plot_matrix(att1 @ att0, "att virtual", "q", "k")

    Wu = model.un_embed.weight
    We = model.embed.weight

    Wq0 = model.layers[0][0].wq.weight
    Wk0 = model.layers[0][0].wk.weight
    Wv0 = model.layers[0][0].wv.weight
    Wo0 = model.layers[0][0].wo.weight

    Wq1 = model.layers[1][0].wq.weight
    Wk1 = model.layers[1][0].wk.weight
    Wv1 = model.layers[1][0].wv.weight
    Wo1 = model.layers[1][0].wo.weight

    Wov0 = Wo0 @ Wv0
    Wov1 = Wo1 @ Wv1
    Wqk0 = Wq0.T @ Wk0
    Wqk1 = Wq1.T @ Wk1

    plot_matrix(Wu @ We, "direct circuit", "dest token", "src token")

    plot_matrix(We.T @ Wqk0 @ We, "qk circuit 0", "dest token", "src token")
    plot_matrix(We.T @ Wqk1 @ We, "qk circuit 1", "dest token", "src token")

    plot_matrix(Wu @ Wov0 @ We, "ov circuit 0", "dest token", "src token")
    plot_matrix(Wu @ Wov1 @ We, "ov circuit 1", "dest token", "src token")

    plot_matrix(Wu @ Wov1 @ Wov0 @ We, "ov circuit virtual", "dest token", "src token")

    plot_matrix(Wu @ stream0.T, "stream 0", "seq", "token")
    plot_matrix(Wu @ stream1.T, "stream 1", "seq", "token")
    plot_matrix(Wu @ stream2.T, "stream 2", "seq", "token")
    plot_matrix(Wu @ delta0.T, "delta 0", "seq", "token")
    plot_matrix(Wu @ delta1.T, "delta 1", "seq", "token")

    s_mats = [stream0, stream1, stream2, delta0, delta1]
    s_min_max = min(x.min() for x in s_mats), max(x.max() for x in s_mats)

    plot_matrix(stream0, "raw stream 0", "seq", "n", s_min_max)
    plot_matrix(stream1, "raw stream 1", "seq", "n", s_min_max)
    plot_matrix(stream2, "raw stream 2", "seq", "n", s_min_max)
    plot_matrix(delta0, "raw delta 0", "seq", "n", s_min_max)
    plot_matrix(delta1, "raw delta 1", "seq", "n", s_min_max)

    plot_matrix(We.T @ Wov0 @ Wqk1 @ We, "comp Q", None, None)
    plot_matrix(We.T @ Wqk1 @ Wov0 @ We, "comp K", None, None)
    plot_matrix(We.T @ Wov1 @ Wqk1 @ Wov0 @ We, "comp QK", None, None)

    q_norm = torch.linalg.matrix_norm(Wqk1.T @ Wov0) / \
             (torch.linalg.matrix_norm(Wqk1.T) * torch.linalg.matrix_norm(Wov0))
    k_norm = torch.linalg.matrix_norm(Wqk1 @ Wov0) / \
             (torch.linalg.matrix_norm(Wqk1) * torch.linalg.matrix_norm(Wov0))
    v_norm = torch.linalg.matrix_norm(Wov1 @ Wov0) / \
             (torch.linalg.matrix_norm(Wov1) * torch.linalg.matrix_norm(Wov0))

    print("Q comp", q_norm)
    print("K comp", k_norm)
    print("V comp", v_norm)

    # print(torch.linalg.eig(Wu @ Wov0 @ We))
    # print(torch.linalg.eig(Wu @ Wov1 @ We))


if __name__ == '__main__':
    main()
