import itertools
import random

import numpy as np

from lib.logger import Logger
from lib.plotter import run_with_plotter, LogPlotter


def main(plotter: LogPlotter):
    logger = Logger()

    x = 0.0
    y = 0.0

    for i in itertools.count():
        plotter.block_while_paused()
        logger.start_batch()

        target = np.sin(i * 0.0002)

        logger.log("test", "x", x)
        logger.log("test", "y", y)
        logger.log("test", "target", target)

        # random walk
        x += (random.random() * 2 - 1) * 0.01

        # decay towards target
        x = target + (x - target) * 0.999
        y = target + (y - target) * 0.999

        plotter.update(logger)


if __name__ == '__main__':
    run_with_plotter(main)
