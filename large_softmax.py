import time

import torch
from cupy.cuda import Stream
from torch.cuda import Event

x = torch.randn(32, 2097152, device="cuda")
# x = torch.randn(32, 512, device="cuda")
dim = 1
# x = torch.randn(int(1e8), device="cuda")
# dim = 0

y = torch.softmax(x, dim)

start = time.perf_counter()

iterations = 100
for _ in range(iterations):
    y = torch.softmax(x, dim)

torch.cuda.synchronize()
end = time.perf_counter()

throughput = x.numel() * iterations / (end - start)
print("Throughput: {} Gel/s", throughput / 1024**3)
