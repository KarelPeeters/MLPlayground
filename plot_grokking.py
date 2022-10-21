import itertools
import os

import torch.jit
import torchvision
from matplotlib import pyplot as plt


@torch.no_grad()
def main():
    # stacked = {}

    for i in itertools.count(0, 1000):
        path = f"ignored/grokking/dense/models/model_{i}.pt"
        next_path = f"ignored/grokking/dense/models/model_{i + 1000}.pt"
        if not os.path.exists(path):
            break
        is_last = not os.path.exists(next_path)

        print(path)
        model = torch.jit.load(path, "cpu")

        for name, param in model.named_parameters():
            print(f"  {name}: {param.shape}")

            if len(param.shape) == 2:
                os.makedirs(f"ignored/grokking/images/{name}/", exist_ok=True)
                torchvision.utils.save_image(
                    param.unsqueeze(0), f"ignored/grokking/images/{name}/{i}.png",
                    normalize=True,
                )

                if name == "inner.0.weight":
                    assert param.shape == (256, 2 * 97)
                    param_split = param.view(256, 2, 97)

                    # save split image
                    img_split = param_split.permute(1, 0, 2).unsqueeze(1)
                    img_doubled = param_split.reshape(256 * 2, 97)
                    img_doubled_split = param_split.unsqueeze(1)

                    os.makedirs(f"ignored/grokking/images/{name}_split/", exist_ok=True)
                    torchvision.utils.save_image(
                        img_split, f"ignored/grokking/images/{name}_split/{i}.png",
                        normalize=True,
                    )

                    os.makedirs(f"ignored/grokking/images/{name}_doubled/", exist_ok=True)
                    torchvision.utils.save_image(
                        img_doubled, f"ignored/grokking/images/{name}_doubled/{i}.png",
                        normalize=True,
                    )

                    os.makedirs(f"ignored/grokking/images/{name}_doubled_split/", exist_ok=True)
                    torchvision.utils.save_image(
                        img_doubled_split, f"ignored/grokking/images/{name}_doubled_split/{i}.png",
                        normalize=True, nrow=4, padding=1,
                    )

                    # draw pca
                    # if is_last:
                    #     param_first = param_split[:, 0, :]
                    #     u, s, v = torch.pca_lowrank(param_first)
                    #
                    #     coords = torch.einsum("ij,is->js", param_first, u)
                    #
                    #     plt.plot(s)
                    #     plt.show()
                    #
                    #     plt.scatter(coords[:, 0], coords[:, 1])
                    #     plt.show()

            # shaped = param.unsqueeze(0)
            # if name in stacked:
            #     stacked[name] = torch.cat([stacked[name], shaped], dim=0)
            # else:
            #     stacked[name] = shaped

    # print("Total:")
    # for name, total in stacked.items():
    #     print(f"  {name}: {total.shape}")
    #
    #     if len(total.shape) != 3:
    #         continue
    #
    #     shaped = total.unsqueeze(1)
    #
    #     os.makedirs(f"ignored/grokking/images_total/{name}", exist_ok=True)
    #     torchvision.utils.save_image(
    #         shaped, f"ignored/grokking/images_total/{name}.png",
    #         nrow=len(total),
    #     )


if __name__ == '__main__':
    main()
