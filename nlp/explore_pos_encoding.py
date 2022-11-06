import torch
import torchvision.utils
from matplotlib import pyplot as plt


def bert():
    model = torch.load("C:/Users/Karel/Downloads/BERT/pytorch_model.bin")

    for k, v in model.items():
        print(k, v.shape)

    pos_encoding = model["bert.embeddings.position_embeddings.weight"]

    q_min = torch.quantile(pos_encoding.flatten(0), 0.01)
    q_max = torch.quantile(pos_encoding.flatten(0), 0.99)

    torchvision.utils.save_image(
        pos_encoding, "../ignored/embedding/bert_pos_encoding.png",
        normalize=True, value_range=(q_min, q_max)
    )
    torchvision.utils.save_image(
        torch.fft.rfft(pos_encoding, dim=0).abs(), "../ignored/embedding/bert_pos_encoding_fft.png",
        normalize=True
    )


def fft_based():
    seq_len = 128
    stream_size = 128

    # peak_freqs = torch.randint(seq_len, (stream_size,))
    peak_freqs = torch.arange(stream_size)

    freqs = torch.randn(seq_len, stream_size) * .2 / seq_len ** .5
    freqs[peak_freqs, torch.arange(stream_size)] = 1

    phase = torch.rand(stream_size)

    fft = torch.polar(freqs, phase)

    print(freqs.shape)
    print(phase.shape)
    print(fft.shape)

    print(freqs)

    pos = torch.fft.irfft(fft, dim=0)

    print(pos)

    torchvision.utils.save_image(pos, "../ignored/embedding/manual_pos_encoding.png", normalize=True)
    torchvision.utils.save_image(fft.abs(), "../ignored/embedding/manual_pos_encoding_fft.png", normalize=True)


def manual():
    seq_len = 256
    stream_size = 1024

    freq = torch.linspace(1, seq_len / 2, stream_size // 2)
    i = torch.arange(seq_len)
    # x = torch.arange(seq_len)

    t = freq[None, :] * i[:, None]

    pos = torch.stack([
        torch.sin(t * 2 * torch.pi / seq_len),
        torch.cos(t * 2 * torch.pi / seq_len),
    ], dim=2).view(seq_len, stream_size)

    pos += torch.randn(seq_len, stream_size) / stream_size ** .5

    pos_fft = torch.fft.rfft(pos, dim=0)

    plt.plot(pos[:, 0])
    plt.plot(pos[:, 2])
    plt.show()

    torchvision.utils.save_image(pos, "../ignored/embedding/manual.png", normalize=True)
    torchvision.utils.save_image(pos_fft.abs(), "../ignored/embedding/manual_fft.png", normalize=True)
    plt.matshow(pos_fft.abs())
    plt.show()


def main():
    manual()


if __name__ == '__main__':
    main()
