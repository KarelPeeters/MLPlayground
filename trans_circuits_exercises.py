import torch


def print_tensor(name: str, t):
    print(f"{name}:")
    print(str(t).replace("0.", "  "))
    print()


def main():
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


if __name__ == '__main__':
    main()
