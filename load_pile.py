import time

from lm_dataformat import Reader

reader = Reader(r"\\192.168.0.10\Documents\Download\the-pile\00.jsonl.zst")
# reader = Reader(r"\\192.168.0.10\Documents\Download\the-pile\test.jsonl.zst")

chars_seen = 0
docs_seen = 0

prev_time = time.perf_counter()
prev_chars_seen = 0

for doc in reader.stream_data():
    if docs_seen % 10000 == 0:
        now = time.perf_counter()
        char_tp = (chars_seen - prev_chars_seen) / (now - prev_time)

        print(f"Seen {docs_seen} docs => {chars_seen / 1024 ** 3:.3} GChar {char_tp / 1024 ** 3:.3} GChar/s")

        prev_time = now
        prev_chars_seen = chars_seen

    chars_seen += len(doc)
    docs_seen += 1

print(f"Seen {docs_seen} docs => {chars_seen / 1024 ** 3:.3} GChar")
