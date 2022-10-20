import itertools
import os

import torch.jit
import torchvision.utils


def main():
    # stacked = {}

    for i in itertools.count(0, 1000):
        path = f"ignored/grokking/model_{i}.pt"
        if not os.path.exists(path):
            break

        print(path)
        model = torch.jit.load(path)

        for name, param in model.named_parameters():
            print(f"  {name}: {param.shape}")

            if len(param.shape) == 2:
                os.makedirs(f"ignored/grokking/images/{name}/", exist_ok=True)
                torchvision.utils.save_image(
                    param.unsqueeze(0), f"ignored/grokking/images/{name}/{i}.png",
                )

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
