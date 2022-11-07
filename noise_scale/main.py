import torch.optim
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from lib.logger import Logger
from lib.plotter import run_with_plotter, LogPlotter
from noise_scale.model import ResNet, ResNet9

import torch.nn.functional as nnf

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
    batch_size = 128
    device = "cuda"

    # model = ResNet(64, 8, 3, 4, 10, 32)
    model = ResNet9(3, 10, 32)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    logger = Logger()

    for x, y in data_loader(batch_size, True):
        plotter.block_while_paused()
        logger.start_batch()

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

        logger.log("acc", "uniform", 1/10)
        logger.log("acc", "perfect", 1)

        plotter.update(logger)


if __name__ == '__main__':
    run_with_plotter(main)
