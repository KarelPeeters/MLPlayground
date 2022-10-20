import numpy as np


class ContinuousBuffer:
    def __init__(self, window_size: int):
        self.window_size = window_size

        self.offset = 0
        self.buffer0 = np.zeros(2 * window_size)
        self.buffer1 = np.zeros(2 * window_size)

    def push(self, data):
        offset = self.offset
        window = self.window_size

        assert len(data.shape) == 1
        assert len(data) < window

        assert False, "TODO implement this"

        self.offset = (offset + len(data)) % (window * 2)

    def window(self):
        offset = self.offset
        window = self.window_size

        if offset > window:
            return self.buffer0[offset - window, offset]
        else:
            return self.buffer1[offset:offset + window]


def main():
    buffer = ContinuousBuffer(8)

    for _ in range(8):
        buffer.push(np.array([1, 2]))
        print(buffer.window())


if __name__ == '__main__':
    main()
