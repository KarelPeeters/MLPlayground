import matplotlib.pyplot as plt
import torch

from circuits_train import Head, causal_mask


def print_tensor(name: str, t):
    print(f"{name}:")
    print(str(t).replace("0.", "  "))
    print()


def ex1():
    I = torch.eye(16)

    Wv1 = torch.zeros(4, 16)
    Wv1[:, 0:4] = torch.eye(4)

    Wo1 = torch.zeros(16, 4)
    Wo1[8:12, :] = torch.eye(4)

    Wv2 = torch.zeros(4, 16)
    Wv2[0:3, 4:7] = torch.eye(3)
    Wv2[3, 11] = 1

    Wo2 = torch.zeros(16, 4)
    Wo2[4:8, :] = torch.eye(4)

    print_tensor("Wv1", Wv1)
    print_tensor("Wo1", Wo1)
    print_tensor("Wv2", Wv2)
    print_tensor("Wo2", Wo2)

    Wov1 = Wo1 @ Wv1
    Wov2 = Wo2 @ Wv2
    print_tensor("Wov1", Wov1)
    print_tensor("Wov2", Wov2)

    W_two_token = Wov2 @ Wov1
    W_one_token = Wov2 @ I + I @ Wov1

    print_tensor("W_two_token", W_two_token)
    print_tensor("W_one_token", W_one_token)


def init_heads_ov(head1: Head, head2: Head):
    Wv1 = torch.zeros(4, 16)
    Wv1[:, 0:4] = torch.eye(4)

    Wo1 = torch.zeros(16, 4)
    Wo1[8:12, :] = torch.eye(4)

    Wv2 = torch.zeros(4, 16)
    Wv2[0:3, 4:7] = torch.eye(3)
    Wv2[3, 11] = 1

    Wo2 = torch.zeros(16, 4)
    Wo2[4:8, :] = torch.eye(4)

    Wov1 = Wo1 @ Wv1
    Wov2 = Wo2 @ Wv2
    print_tensor("Wov1", Wov1)
    print_tensor("Wov2", Wov2)

    head1.wo.weight.copy_(Wo1)
    head1.wv.weight.copy_(Wv1)
    head2.wo.weight.copy_(Wo2)
    head2.wv.weight.copy_(Wv2)


def ex1_test():
    stream_size = 16
    proj_size = 4
    seq_len = 8

    mask = causal_mask(seq_len)
    head1 = Head(stream_size, proj_size)
    head2 = Head(stream_size, proj_size)

    with torch.no_grad():
        # ensure uniform attention pattern
        head1.wq.weight *= 0
        head2.wq.weight *= 0

        # set custom ov weights
        init_heads_ov(head1, head2)

    stream0 = torch.zeros(seq_len, stream_size)
    stream0[1, 3] = 1

    delta1, att1 = head1(stream0, mask)
    stream1 = stream0 + delta1

    delta2, att2 = head2(stream1, mask)
    stream2 = stream1 + delta2

    with torch.no_grad():
        atts = [("att1", att1), ("att2", att2)]

        f, axes = plt.subplots(1, len(atts))
        for ax, (name, att) in zip(axes, atts):
            ax.matshow(att, vmin=0, vmax=1)
            ax.set_title(name)
        f.show()

        # noinspection PyTypeChecker
        mats = [("stream0", stream0), ("delta1", delta1), ("stream1", stream1), ("delta2", delta2),
                ("stream2", stream2)]

        f, axes = plt.subplots(len(mats), 1)
        vmin = min([m.min() for _, m in mats])
        vmax = max([m.max() for _, m in mats])

        m = None
        for ax, (name, mat) in zip(axes, mats):
            m = ax.matshow(mat, vmin=vmin, vmax=vmax)
            ax.set_ylabel(name)

        f.colorbar(m, ax=axes)
        f.show()

    print(delta1)


def main():
    # ex1()
    ex1_test()


if __name__ == '__main__':
    main()
