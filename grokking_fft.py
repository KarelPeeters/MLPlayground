import os
import shutil

import torch
from matplotlib import pyplot as plt

with torch.no_grad():
    path = f"ignored/grokking/dense/models/model_{97000}.pt"

    model = torch.jit.load(path, "cpu")

    params = dict(model.named_parameters())
    embedding = params["inner.0.weight"].view(256, 2, 97)

    rfft = torch.fft.rfft(embedding, dim=2)
    rfft[:, :, 0] = 0
    rfft_size = rfft.shape[2]

    max_freq = torch.argmax(rfft[:, :, :].abs(), dim=2)

    # get a mask for plots with clear max frequencies
    top_k = torch.topk(rfft.abs(), k=3, dim=2)
    high_max = torch.all(top_k.values[:, :, 0] > 2 * top_k.values[:, :, -1], dim=1)
    same_max = max_freq[:, 0] == max_freq[:, 1]
    clear_max = same_max & high_max

    # order plots, low to high freq with unclear last
    max_freq_clear = torch.clone(max_freq[:, 0])
    max_freq_clear[~clear_max] = 1000

    argsort = torch.argsort(max_freq_clear)

    # plt.hist(max_freq[clear_max, 0], bins=rfft_size)
    # plt.title("Embeddings per frequency")
    # plt.show()

    for i in range(len(rfft)):
        if max_freq[i, 0] != max_freq[i, 1]:
            print(f"Max frequency differs for {i}: {max_freq[i, 0]} vs {max_freq[i, 1]}")

        f, (ax0, ax1, ax2) = plt.subplots(3, 1)
        f.tight_layout()

        ax0.set_title(f"emb_{i}")
        ax0.plot(embedding[i].T)

        ax1.set_title(f"abs(rfft(emb_{i}))")
        ax1.plot(rfft[i].abs().T)

        ax2.set_title(f"angle(rfft(emb_{i}))")
        ax2.plot(rfft[i].angle().T)

        ax1.axvline(max_freq[i][0], color="C0")
        ax1.axvline(max_freq[i][1], color="C1")
        # ax1.axvline(mean_freq[i][0], color="C0", linestyle="--")
        # ax1.axvline(mean_freq[i][1], color="C1", linestyle="--")

        j = (argsort == i).nonzero().item()

        fig_path_i = os.path.join(f"ignored/grokking/ftt/plots/", f"emb_{i}.png")
        fig_path_j = os.path.join(f"ignored/grokking/ftt/plots_ordered/", f"emb_{j}.png")

        os.makedirs(os.path.dirname(fig_path_i), exist_ok=True)
        os.makedirs(os.path.dirname(fig_path_j), exist_ok=True)

        f.savefig(fig_path_i)
        shutil.copyfile(fig_path_i, fig_path_j)

        plt.close(f)
