import os

import torch.nn.functional as nnf
import torch.optim
from torch import nn

from lib.logger import Logger
from lib.plotter import LogPlotter, run_with_plotter


# TODO does grokking also happen with non-transformer networks?

def print_params(model):
    print("Parameters:")
    total_param_count = 0
    for name, param in model.named_parameters():
        print(f"  {name}: {param.shape} ({param.numel()})")
        total_param_count += param.numel()
    print(f"Total param count: {total_param_count}")


def operation(x, y, p):
    # return x
    # return (x + y) % p
    return (x - y) % p


def generate_data(p: int):
    x = torch.stack([
        torch.arange(p).view(p, 1).expand(p, p),
        torch.arange(p).view(1, p).expand(p, p)
    ], dim=2).view(p * p, 2)
    y = operation(x[:, 0], x[:, 1], p)

    both = torch.concat([x, y.unsqueeze(1)], dim=1)
    return both


def split_data(data, train_fraction: float):
    shuffle = torch.randperm(len(data))
    data_shuffled = data[shuffle]

    split = int(train_fraction * len(data))
    data_train = data_shuffled[:split]
    data_test = data_shuffled[split:]

    return data_train, data_test


class Model(nn.Module):
    def __init__(self, p: int, hidden_size: int, dropout: float):
        super().__init__()
        self.p = p

        self.embedding = nn.Embedding(2 + p, hidden_size)
        self.position = nn.Embedding(4, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size, nhead=4, dim_feedforward=hidden_size,
            dropout=dropout, batch_first=False, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, 2)
        self.mask = nn.Parameter(nn.Transformer.generate_square_subsequent_mask(4), requires_grad=False)

        self.final_linear = nn.Linear(hidden_size, p)
        # with torch.no_grad():
        #     self.final_linear.bias *= 0
        #     self.final_linear.weight *= 0.02

        self.zero = nn.Parameter(torch.tensor(0), requires_grad=False)
        self.one = nn.Parameter(torch.tensor(1), requires_grad=False)

    def forward(self, x, y):
        batch_size = len(x)

        in_embedded = torch.stack([
            self.embedding(2 + x),
            self.embedding(self.zero).view(1, -1).expand(batch_size, -1),
            self.embedding(2 + y),
            self.embedding(self.one).view(1, -1).expand(batch_size, -1),
        ])
        in_encoded = in_embedded + self.position.weight.unsqueeze(1)

        out_hidden = self.transformer(in_encoded, self.mask)
        out_final = self.final_linear(out_hidden)

        pred_final = out_final[-1]

        return pred_final


def eval_model(logger: Logger, prefix: str, model, batch):
    x = batch[:, 0]
    y = batch[:, 1]
    z = batch[:, 2]

    pred = model(x, y)

    loss = nnf.cross_entropy(pred, z)
    acc = (torch.argmax(pred, dim=1) == z).float().mean()
    mse = nnf.mse_loss(nnf.softmax(pred, dim=1), nnf.one_hot(z, model.p).float())

    logger.log("loss", prefix, loss)
    logger.log("mse", prefix, mse)
    logger.log("acc", prefix, acc)

    return loss
    # return mse


def set_optim_lr(optim, lr):
    for g in optim.param_groups:
        assert "lr" in g
        g["lr"] = lr


def main(plotter: LogPlotter):
    run_name = "derp"

    os.makedirs(f"ignored/grokking/{run_name}", exist_ok=False)
    plotter.set_title(f"Grokking - {run_name}")

    p = 97

    dropout = 0.1
    hidden_size = 128
    batch_size = 512
    batches = int(1e5)
    lr = 1e-3
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_fraction = 0.4
    weight_decay = 0.01

    data_all = generate_data(p).to(device)
    data_train, data_test = split_data(data_all, train_fraction)

    model = Model(p=p, hidden_size=hidden_size, dropout=dropout)
    model.to(device)
    # print_params(model)

    # TODO linear learning rate warmup over first 10 steps
    optim = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=weight_decay)

    logger = Logger()

    for bi in range(batches):
        if bi % 1000 == 0:
            os.makedirs(f"ignored/grokking/{run_name}/models/", exist_ok=True)
            torch.jit.save(torch.jit.script(model), f"ignored/grokking/{run_name}/models/model_{bi}.pt")
            logger.save(f"ignored/grokking/{run_name}/log.npz")

        print(f"bi {bi}")
        logger.start_batch()

        model.eval()
        with torch.no_grad():
            test_i = torch.randint(len(data_test), (batch_size,))
            test_batch = data_test[test_i]
            eval_model(logger, "test", model, test_batch)

        model.train()
        train_i = torch.randint(len(data_train), (batch_size,))
        train_batch = data_train[train_i]
        loss = eval_model(logger, "train", model, train_batch)

        logger.log("acc", "uniform", 1 / p)
        logger.log("loss", "uniform", nnf.cross_entropy(torch.full((p,), 1 / p), torch.tensor(0)))

        optim.zero_grad()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.log("param", name, param.abs().mean())
                logger.log("grad", name, param.grad.abs().mean())

                if bi >= 10:
                    logger.log("grad/param", name, param.grad.abs().mean() / param.abs().mean())
                    logger.log("log(grad/param)", name, torch.log10(param.grad.abs().mean() / param.abs().mean()))

        if bi < 10:
            set_optim_lr(optim, bi / 10 * lr)
        else:
            set_optim_lr(optim, lr)
        optim.step()

        plotter.update(logger)
        plotter.block_while_paused()


if __name__ == '__main__':
    run_with_plotter(main)
