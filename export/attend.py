import numpy as np
from transformers import GPT2Tokenizer
from gpt2_numpy import wte, wpe, layers, layerNorm, block, softMax

tok = GPT2Tokenizer.from_pretrained("gpt2")
ids = tok.encode("The cat sat on the mat because it was tired")
pieces = [tok.decode([i]) for i in ids]
T = len(ids)

LAYER = 5
x = wte[ids] + wpe[:T]
for L in layers[:LAYER]:            # run layers 0-4 normally
    x = block(x, L)

L = layers[LAYER]                   # now open up layer 5's attention by hand
h = layerNorm(x, L["ln_1_w"], L["ln_1_b"])
qkv = h @ L["attn_w"] + L["attn_b"]
q = qkv[:, 0:768].reshape(T, 12, 64).transpose(1, 0, 2)      # (12, T, 64)
k = qkv[:, 768:1536].reshape(T, 12, 64).transpose(1, 0, 2)
scores = q @ k.transpose(0, 2, 1) / 8.0                      # (12, T, T)
scores[:, np.triu(np.ones((T, T), dtype=bool), k=1)] = -np.inf
weights = softMax(scores)                                    # each row sums to 1

it = pieces.index(" it")
avg = weights[:, it, :].mean(axis=0)                         # average of 12 heads
print(f"Layer {LAYER}: where does ' it' look? (average of the 12 heads)")
for j in range(T):
    print(f"  {pieces[j]!r:10} {avg[j]:.2f}  " + "#" * int(avg[j] * 60))
print(f"  row sums to {avg.sum():.2f}")

print("\nEach head's favourite word for ' it':")
for head in range(12):
    j = int(np.argmax(weights[head, it]))
    print(f"  head {head:2d} -> {pieces[j]!r:10} ({weights[head, it, j]:.2f})")
