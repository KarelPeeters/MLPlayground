from lm_dataformat import Reader

# reader = Reader(r"\\192.168.0.10\Documents\Download\the-pile\00.jsonl.zst")
reader = Reader(r"\\192.168.0.10\Documents\Download\the-pile\test.jsonl.zst")

chars_seen = 0
docs_seen = 0

for doc in reader.stream_data():
    if docs_seen % 10000 == 0:
        print(f"Seen {docs_seen} docs => {chars_seen / 1024 ** 3:.3} GChar")

    chars_seen += len(doc)
    docs_seen += 1

print(f"Seen {docs_seen} docs => {chars_seen / 1024 ** 3:.3} GChar")