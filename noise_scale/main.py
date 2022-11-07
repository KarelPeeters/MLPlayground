import time

import torch.nn.functional as nnf
import torch.optim
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from lib.logger import Logger
from lib.plotter import run_with_plotter, LogPlotter
from noise_scale.model import ResNet9

DATA_ROOT = "../ignored/data"


def data_loader(batch_size: int, train: bool):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = torchvision.datasets.CIFAR10(
        root='../ignored/data/', train=train,
        download=True, transform=transform
    )

    while True:
        loader = DataLoader(
            dataset, batch_size=batch_size,
            shuffle=True  # , num_workers=2
        )

        for batch in loader:
            yield batch


def main(plotter: LogPlotter):
    sample_budget = 128 * 1024

    batch_sizes = [4, 8, 128, 256, 512]
    device = "cuda"

    final_losses = []
    final_accs = []
    final_times = []

    logger = Logger()

    for batch_size in batch_sizes:
        print(f"Running with batch size {batch_size}")

        # model = ResNet(64, 8, 3, 4, 10, 32)
        model = ResNet9(3, 10, 32)
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        start = time.perf_counter()

        for bi, (x, y) in enumerate(data_loader(batch_size, True)):
            if bi * batch_size >= sample_budget:
                break

            print(f"  batch {bi} / {sample_budget // batch_size}")

            plotter.block_while_paused()
            logger.start_batch()

            logger.log("batch", "batch size", batch_size)

            x = x.to(device)
            y = y.to(device)

            y_pred = model(x)

            loss = nnf.cross_entropy(y_pred, y)
            acc = (torch.argmax(y_pred, dim=-1) == y).float().mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            logger.log("loss", "train", loss)
            logger.log("acc", "train", acc)

            logger.log("acc", "uniform", 1 / 10)
            logger.log("acc", "perfect", 1)

            plotter.update(logger)

        final_loss = logger.finished_data().values[("loss", "train")][-10:].mean()
        final_acc = logger.finished_data().values[("acc", "train")][-10:].mean()

        final_losses.append(final_loss)
        final_accs.append(final_acc)
        final_times.append(time.perf_counter() - start)

    for name, data in [("loss", final_losses), ("acc", final_accs), ("time", final_times)]:
        plt.plot(batch_sizes, data)
        plt.xscale("log")
        plt.title(name)
        plt.xlabel("Batch size")
        plt.ylabel(name)
        plt.show()

    logger.save("log.npz")


if __name__ == '__main__':
    run_with_plotter(main)
