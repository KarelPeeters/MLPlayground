from sentencepiece import SentencePieceTrainer

# path = r"\\192.168.0.10\Documents\Download\the-pile\test.txt"
path = r"C:\Users\Karel\Desktop\the-pile\00_10k.txt"

result = SentencePieceTrainer.Train(
    input=path,
    model_prefix='m',
    vocab_size=1024,
    byte_fallback=True,
)

print(result)
