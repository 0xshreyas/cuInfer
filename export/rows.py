import numpy as np
from transformers import GPT2Tokenizer
from gpt2_numpy import forward_pass

tok = GPT2Tokenizer.from_pretrained("gpt2")
text = "Hello, my name is"
ids = tok.encode(text)
print("token ids:", ids)
print("pieces:   ", [tok.decode([i]) for i in ids])

logits = forward_pass(ids)
print("logits shape:", logits.shape)
print()

for t in range(len(ids)):
    best = int(np.argmax(logits[t]))
    seen = tok.decode(ids[:t + 1])
    print(f"after {seen!r:25} -> predicts {tok.decode([best])!r}")
