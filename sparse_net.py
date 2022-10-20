import torch.nn.functional as nnf
import torch.optim
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision.transforms import ToTensor


class Residual(nn.Module):
    def __init__(self, *inner: nn.Module):
        super().__init__()
        self.inner = nn.Sequential(*inner)

    def forward(self, x):
        return x + self.inner(x)


def loader(data, batch_size: int):
    return DataLoader(data, batch_size, shuffle=True, drop_last=True)


def inf_loader(data, batch_size: int):
    while True:
        for batch in loader(data, batch_size):
            yield batch


def main():
    batch_size = 1024
    device = "cuda"
    test_freq = 10

    data_train = CIFAR100(root="ignored/data", download=True, train=True, transform=ToTensor())
    data_test = CIFAR100(root="ignored/data", download=True, train=False, transform=ToTensor())

    h = 512
    network = nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 32 * 32, h),

        *(
            Residual(
                # nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Linear(h, h),
                # nn.Dropout(p=0.5),
                nn.Dropout1d(p=0.9)  # dropout entire residual branch for some elements of the batch?
            )

            for _ in range(8)
        ),

        # nn.BatchNorm1d(h),
        nn.Linear(h, 100),
    )

    optimizer = torch.optim.Adam(network.parameters())
    network.to(device)

    losses_train = []
    accs_train = []

    indices_test = []
    losses_test = []
    accs_test = []

    test_loader = inf_loader(data_test, batch_size)

    for ei in range(128):
        for bi, (x_train, y_train) in enumerate(loader(data_train, batch_size)):
            should_test = bi % test_freq == 0

            if should_test:
                x_test, y_test = next(test_loader)
                network.eval()
                y_test = y_test.to(device)

                with torch.no_grad():
                    y_test_pred = network(x_test.to(device))
                    loss_test = nnf.cross_entropy(y_test_pred, y_test)
                    acc_test = (torch.argmax(y_test_pred, 1) == y_test).float().mean()

                index = len(losses_train)
                indices_test.append(index)
                losses_test.append(loss_test.item())
                accs_test.append(acc_test.item())

                print(f"Epoch {ei} batch {bi}:")
                print(f"  test: loss {loss_test.item()} acc {acc_test.item()}")

            network.train()
            y_train = y_train.to(device)
            y_train_pred = network(x_train.to(device))
            loss_train = nnf.cross_entropy(y_train_pred, y_train)
            acc_train = (torch.argmax(y_train_pred, 1) == y_train).float().mean()

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            losses_train.append(loss_train.item())
            accs_train.append(acc_train.item())

            if should_test:
                print(f"  train: loss {loss_train.item()} acc {acc_train.item()}")

    plt.plot(losses_train, label="train")
    plt.plot(indices_test, losses_test, label="test")
    plt.title("Loss")
    plt.legend()
    plt.show()

    plt.plot(accs_train, label="train")
    plt.plot(indices_test, accs_test, label="test")
    plt.title("Accuracy")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
