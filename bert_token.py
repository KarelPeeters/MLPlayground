from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

text = "One unusual feature of our task is the length of the games; each rollout can take up to two hours to complete"
ids = tokenizer.encode(text)
tokens = tokenizer.convert_ids_to_tokens(ids)

print(tokens)
print(" ".join(t.replace("#", "") for t in tokens))
