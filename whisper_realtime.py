from dataclasses import dataclass
from threading import Lock, Thread, Event

import numpy as np
import pyaudio as pa
import torch
import whisper
from matplotlib import pyplot as plt


@dataclass
class Config:
    rate: int
    window_len: int
    sample_len: int
    loopback: bool


@dataclass
class Shared:
    buffer: np.array
    buffer_mutex: Lock = Lock()
    stop_event: Event = Event()


def main_audio(config: Config, shared: Shared):
    rate = config.rate
    sample_len = config.sample_len

    p = pa.PyAudio()

    input_index = p.get_default_input_device_info()["index"]
    output_index = p.get_default_output_device_info()["index"]

    stream = p.open(rate=rate, channels=1, format=pa.paInt16, input=True, input_device_index=input_index)
    stream_output = p.open(rate=rate, channels=1, format=pa.paInt16, output=True, output_device_index=output_index)

    while True:
        if shared.stop_event.is_set():
            break

        # record and loopback
        sample_bytes = stream.read(sample_len)
        if config.loopback:
            stream_output.write(sample_bytes)

        # TODO try direct float32 again
        sample = np.frombuffer(sample_bytes, dtype=np.int16).astype(np.float32) / 32768.0

        # append to buffer and roll it around
        with shared.buffer_mutex:
            shared.buffer[-sample_len:] = sample
            shared.buffer[:-sample_len] = shared.buffer[sample_len:]


def main_model(model, config: Config, shared: Shared):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    with shared.buffer_mutex:
        line, = ax.plot(shared.buffer)
    fig.show()

    while True:
        with shared.buffer_mutex:
            buffer_copy = np.copy(shared.buffer)

        mel = whisper.log_mel_spectrogram(torch.tensor(buffer_copy).to(model.device))
        options = whisper.DecodingOptions(language="en")
        result = whisper.decode(model, mel, options)

        print(result.text)
        print(np.max(buffer_copy), np.mean(buffer_copy))

        line.set_ydata(buffer_copy)
        fig.canvas.draw()
        fig.canvas.flush_events()


def main():
    print("Loading model")
    model = whisper.load_model("base")

    rate = whisper.audio.SAMPLE_RATE
    window_len = whisper.audio.N_SAMPLES
    sample_len = 1024
    loopback = False

    config = Config(rate=rate, window_len=window_len, sample_len=sample_len, loopback=loopback)
    buffer = np.zeros(window_len, dtype=np.float32)
    shared = Shared(buffer=buffer)

    print("Starting audio thread")
    audio_thread = Thread(target=lambda: main_audio(config, shared))
    audio_thread.start()

    try:
        print("Running model")
        main_model(model, config, shared)
    except Exception as e:
        shared.stop_event.set()
        raise e


if __name__ == '__main__':
    main()
