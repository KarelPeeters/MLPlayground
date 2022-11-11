import math

import numpy as np
from numba import cuda, typeof


@cuda.jit
def scalar_sqrt_kernel(x, y):
    i = cuda.grid(1)
    y[i] = math.sqrt(x[i])


def main():
    N = 1024

    x_cpu = np.random.rand(N)
    y_expected = np.sqrt(x_cpu)

    x_gpu = cuda.to_device(x_cpu)
    y_gpu = cuda.device_array_like(y_expected)

    scalar_sqrt_kernel[32, 128](x_gpu, y_gpu)

    y_cpu = y_gpu.copy_to_host()

    print(y_expected)
    print(y_cpu)

    max_err = np.abs(y_cpu - y_expected).max()
    print(f"Max error: {max_err}")

    args = [typeof(a) for a in [x_gpu, y_gpu]]

    print("====================\nTypes\n====================")
    scalar_sqrt_kernel.inspect_types()

    print("====================\nLLVM\n====================")
    print(scalar_sqrt_kernel.inspect_llvm(tuple(args)))
    print("====================\nASM\n====================")
    print(scalar_sqrt_kernel.inspect_asm(tuple(args)))
    print("====================\nSASS\n====================")
    print(scalar_sqrt_kernel.inspect_sass(tuple(args)))


if __name__ == '__main__':
    main()
