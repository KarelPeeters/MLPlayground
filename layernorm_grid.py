import gc
import time

import torch

sizes_inert = [2 ** i for i in range(4, 20)]
sizes_op = [2 ** i for i in range(4, 20)]

max_elements = 1024 * 1024 * 1024 // 4
iterations = 100

max_input = torch.randn(max_elements, device="cuda")
throughputs = []

for size_inert in sizes_inert:
    for size_op in sizes_op:
        print(f"Testing {size_inert}x{size_op}")
        elements = size_inert * size_op
        if elements > max_elements:
            throughputs.append(torch.nan)
            continue

        input = max_input[:elements].view(size_inert, size_op)
        # results = []

        # warmup
        torch.layer_norm(input, (size_op,))
        torch.cuda.synchronize()

        # profile
        start = time.perf_counter()
        for _ in range(iterations):
            torch.layer_norm(input, (size_op,))
        torch.cuda.synchronize()
        end = time.perf_counter()

        gc.collect()
        torch.cuda.empty_cache()

        throughput = elements * iterations / (end - start) / 1024 ** 3
        throughputs.append(throughput)

print(f"size_inert = {sizes_inert}")
print(f"size_op = {sizes_op}")
print(f"caches = {[False]}")
print(f"throughput = {throughputs}")
