from typing import Optional

from torch import nn


class Residual(nn.Module):
    def __init__(self, *inner: nn.Module):
        super().__init__()
        self.inner = nn.Sequential(*inner)

    def forward(self, x):
        return x + self.inner(x)


class ResNet(nn.Module):
    def __init__(self, channels: int, depth: int, channels_in: int, channels_out: int, categories: int, img_size: int):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(channels_in, channels, 1),

            *(
                Residual(
                    nn.BatchNorm2d(channels),
                    nn.ReLU(),
                    nn.Conv2d(channels, channels, 3, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(),
                    nn.Conv2d(channels, channels, 3, padding=1),
                )
                for _ in range(depth)
            ),

            nn.BatchNorm2d(channels),
            nn.Conv2d(channels, channels_out, 1),
            nn.Flatten(),
            nn.Linear(channels_out * img_size * img_size, categories),
        )

    def forward(self, x):
        return self.seq(x)


class ResNet9(nn.Module):
    def __init__(self, in_channels: int, categories: int, img_size: int):
        super().__init__()

        self.seq = nn.Sequential(
            conv_block(in_channels, 64, None),
            conv_block(64, 128, 2),
            Residual(
                conv_block(128, 128, None),
                conv_block(128, 128, None),
            ),
            conv_block(128, 256, 2),
            conv_block(256, 256, 2),
            Residual(
                conv_block(256, 256, None),
                conv_block(256, 256, None),
            ),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(256 * (img_size // (2 * 2 * 2 * 2)) ** 2, categories)
        )

    def forward(self, x):
        return self.seq(x)


def conv_block(in_channels, out_channels, pool: Optional[int]):
    return nn.Sequential(
        *(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.Mish(),
            *((nn.MaxPool2d(pool),) if pool is not None else ())
        )
    )
