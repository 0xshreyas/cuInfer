import numpy as np
from transformers import GPT2Tokenizer
from gpt2_numpy import wte, wpe, layers, layerNorm, block, softMax

tok = GPT2Tokenizer.from_pretrained("gpt2")
ids = tok.encode("The cat sat on the mat because it was tired")
pieces = [tok.decode([i]) for i in ids]
T = len(ids)

x = wte[ids] + wpe[:T]
for L in layers[:5]:
    x = block(x, L)
L = layers[5]
h = layerNorm(x, L["ln_1_w"], L["ln_1_b"])
qkv = h @ L["attn_w"] + L["attn_b"]
q = qkv[:, 0:768].reshape(T, 12, 64).transpose(1, 0, 2)
k = qkv[:, 768:1536].reshape(T, 12, 64).transpose(1, 0, 2)
raw = q @ k.transpose(0, 2, 1)                  # scores with NO division
mask = np.triu(np.ones((T, T), dtype=bool), k=1)
it = pieces.index(" it")

for name, s in [("with / 8   ", raw / 8.0), ("without / 8", raw.copy())]:
    s[:, mask] = -np.inf
    w = softMax(s)
    top = w.max(axis=-1)                        # biggest weight in every row
    saturated = (top > 0.99).mean() * 100
    print(f"{name}: rows where ONE word gets >99% of the weight: {saturated:5.1f}%")
    row = w[10, it]
    print(f"   head 10, ' it':  " +
          "  ".join(f"{pieces[j].strip()}={row[j]:.2f}" for j in range(8)))
