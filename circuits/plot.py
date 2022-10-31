import os
from typing import Optional

import matplotlib.pyplot as plt
import torch.jit
import torch.nn.functional as nnf

from circuits import generate
from circuits.train import Transformer, Head, TokenTransformer, causal_mask, Composition


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

    os.makedirs("../ignored/circuits/plots", exist_ok=True)
    plt.savefig(f"../ignored/circuits/plots/{name}.png", bbox_inches='tight')
    plt.close()


@torch.no_grad()
def main():
    # import classes and load model
    comp = Composition(True, True, True)
    _ = TokenTransformer(Transformer(0, 0, 8, 8, comp), 8, 1, None)
    _ = Head(8, 8, comp)
    model: TokenTransformer = torch.load("../ignored/circuits/lookup_all/models/model_24900.pt")
    model.to("cpu")
    model.eval()

    tokens = 10

    # seq_len = 16
    # torch.random.manual_seed(0)
    # data_int = generate_lookup_sequence(1, seq_len + 1, tokens)[0].squeeze(0)

    # TODO try forcing sharp attention patterns and see if they still work
    #   (easily achievable by adding things to the masks)

    torch.random.manual_seed(1234)
    sample = generate.generate_sample_lookup_all(1, 32, 10)

    seq_len = sample.seq_len
    assert sample.batch_size == 1
    input_tokens = sample.input_tokens.squeeze(0)
    output_tokens = sample.output_tokens.squeeze(0)
    predictable = sample.predictable.squeeze(0)

    mask = causal_mask(seq_len)
    input = nnf.one_hot(input_tokens, tokens).float()

    mask0 = mask.clone()
    mask1 = mask.clone()

    # TODO be careful, modifying both heads at once can hide issues!
    force_mask_0 = False
    force_mask_1 = False

    if force_mask_0:
        pass
        # GOOD: queries for zero tokens don't matter, can be set to first token
        mask0[3, 1:] = -torch.inf
        mask0[13, 1:] = -torch.inf

        # MEH: removing most of the triangle, except second line (probably any extra key would be fine)
        # mask0[4:13, 4:] = -torch.inf # BROKEN by itself, we need something
        # mask0[5:13, 5] = 0  # MEH: mostly fixes the triangle, but second head loses a bit of focus

        # GOOD: keys for tokens before first zero don't matter
        mask0[4:, :3] = -torch.inf
        mask0[13, 0] = 0

    if force_mask_1:
        mask1[4:13, 4:] = -torch.inf
        mask1[4:, :3] = -torch.inf
        mask1[13, 5:] = -torch.inf

    head0: Head = model.transformer.layers[0][0]
    head1: Head = model.transformer.layers[1][0]

    stream_n1 = model.embed(input)
    if model.pos_encoding is not None:
        stream0 = stream_n1 + model.pos_encoding[:seq_len, :]
    else:
        stream0 = stream_n1
    delta0, att0 = head0(stream0, mask0, stream0)
    stream1 = stream0 + delta0
    delta1, att1 = head1(stream1, mask1, stream0)
    stream2 = stream1 + delta1
    logits = model.un_embed(stream2)

    print(f"Sequence: {sample.input_tokens}")
    for i, (token_in, token_out, token_predictable) in enumerate(
            zip(input_tokens.tolist(), output_tokens.tolist(), predictable.tolist())):
        topk = logits[i].softmax(0).topk(3)
        token_pred = {t.item(): p.item() for t, p in zip(topk.indices, topk.values)}
        token_pred_str = ", ".join(f"{t}: {v:.2f}" for t, v in token_pred.items())
        print(f"    {i:>2}: {token_in} -> {token_out} {token_predictable} (pred {token_pred_str})")

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

    plot_matrix(att0, "act att unit head0", "q", "k")
    plot_matrix(att1, "act att unit head1", "q", "k")
    plot_matrix(att1 @ att0, "act att unit virtual", "q", "k")

    def clip_log10(x):
        return torch.clip(torch.log10(x), -10, None)

    plot_matrix(clip_log10(att0), "act att log head0", "q", "k")
    plot_matrix(clip_log10(att1), "act att log head1", "q", "k")
    plot_matrix(clip_log10(att1 @ att0), "act att log virtual", "q", "k")

    plot_matrix(Wu @ stream0.T, "act token stream 0", "token", "seq")
    plot_matrix(Wu @ stream1.T, "act token stream 1", "token", "seq")
    plot_matrix(Wu @ stream2.T, "act token stream 2", "token", "seq")
    plot_matrix(Wu @ delta0.T, "act token delta 0", "token", "seq")
    plot_matrix(Wu @ delta1.T, "act token delta 1", "token", "seq")

    s_mats = [stream0, stream1, stream2, delta0, delta1]
    s_min_max = min(x.min() for x in s_mats), max(x.max() for x in s_mats)
    plot_matrix(stream0, "act raw stream 0", "seq", "n", s_min_max)
    plot_matrix(stream1, "act raw stream 1", "seq", "n", s_min_max)
    plot_matrix(stream2, "act raw stream 2", "seq", "n", s_min_max)
    plot_matrix(delta0, "act raw delta 0", "seq", "n", s_min_max)
    plot_matrix(delta1, "act raw delta 1", "seq", "n", s_min_max)

    head0_q = head0.wq(stream0)
    head0_k = head0.wk(stream0)
    head0_v = head0.wv(stream0)
    head1_q = head1.wq(stream0)  # no Q-comp
    head1_k = head1.wk(stream1)
    head1_v = head1.wv(stream0)  # no V-comp

    plot_matrix(head0_q, "act head 0 0.Q", "seq", None)
    plot_matrix(head0_k, "act head 0 1.K", "seq", None)
    plot_matrix(head0_v, "act head 0 2.V", "seq", None)

    plot_matrix(head1_q, "act head 1 0.Q", "seq", None)
    plot_matrix(head1_k, "act head 1 1.K", "seq", None)
    plot_matrix(head1_v, "act head 1 2.V", "seq", None)

    # head0_q_n = head0_q[0, :]
    # head0_q_z = head0_q[3, :]
    # head0_k_n = head0_k[0, :]
    # head0_k_z = head0_k[3, :]
    # scale = len(head0_q_n) ** .5
    # print(f"head0 qn*kn = {(head0_q_n * head0_k_n).sum() / scale}")
    # print(f"head0 qn*kz = {(head0_q_n * head0_k_z).sum() / scale}")
    # print(f"head0 qz*kn = {(head0_q_z * head0_k_n).sum() / scale}")
    # print(f"head0 qz*kz = {(head0_q_z * head0_k_z).sum() / scale}")

    # print("Head 1 Q13 matches:")
    # logits_part = head1_q[13, :] * head1_k[:, :]
    # plt.plot(logits_part[4:13, :])
    # plt.show()
    # logits = logits_part.sum(1)
    # plt.plot(logits)
    # plt.show()

    # logit = (head1_q[13, :] * head1_k[i, :]).sum() / scale
    # print(f"Logit q13 x k{i}: {logit.item()}")

    # scale = head1.proj_size ** 0.5
    # att_logit = torch.einsum("...qn,...kn->...qk", head1.wq(stream0), head1.wk(stream1)) / scale
    # att_logit = att_logit + mask1
    # plt.plot(att_logit[13, :])
    # plt.show()

    plot_matrix(Wu @ Wo0 @ head0_v.T, "act token head 0 V", "token", "seq")
    plot_matrix(Wu @ Wo1 @ head1_v.T, "act token head 1 V", "token", "seq")

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
    # plot_matrix(Wu @ Wov1 @ Wov0 @ We, "circuit ov virtual", "dest token", "src token")

    # plot_matrix(We.T @ Wov0 @ Wqk1 @ We, "circuit comp Q", None, None)
    plot_matrix(We.T @ Wqk1 @ Wov0 @ We, "circuit comp K", None, None)
    # plot_matrix(We.T @ Wov1 @ Wqk1 @ Wov0 @ We, "circuit comp QK", None, None)

    if head1.comp.q:
        q_norm = torch.linalg.matrix_norm(Wqk1.T @ Wov0) / \
                 (torch.linalg.matrix_norm(Wqk1.T) * torch.linalg.matrix_norm(Wov0))
        print("Q comp", q_norm)
    if head1.comp.k:
        k_norm = torch.linalg.matrix_norm(Wqk1 @ Wov0) / \
                 (torch.linalg.matrix_norm(Wqk1) * torch.linalg.matrix_norm(Wov0))
        print("K comp", k_norm)
    if head1.comp.v:
        v_norm = torch.linalg.matrix_norm(Wov1 @ Wov0) / \
                 (torch.linalg.matrix_norm(Wov1) * torch.linalg.matrix_norm(Wov0))
        print("V comp", v_norm)


if __name__ == '__main__':
    main()
