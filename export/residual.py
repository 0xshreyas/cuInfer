import numpy as np
from transformers import GPT2Tokenizer
from gpt2_numpy import wte, wpe, layers, layerNorm, attention, MLP

tok = GPT2Tokenizer.from_pretrained("gpt2")
ids = tok.encode("Hello, my name is")
x = wte[ids] + wpe[:len(ids)]

print("layer | biggest in x before | biggest in LN copy | attn adds | MLP adds | biggest in x after")
for i, L in enumerate(layers):
    before = np.abs(x).max()
    ln1 = layerNorm(x, L["ln_1_w"], L["ln_1_b"])
    a = attention(ln1, L)
    x = x + a
    ln2 = layerNorm(x, L["ln_2_w"], L["ln_2_b"])
    m = MLP(ln2, L)
    x = x + m
    print(f"{i:5d} | {before:19.1f} | {np.abs(ln1).max():18.1f} | "
          f"{np.abs(a).max():9.1f} | {np.abs(m).max():8.1f} | {np.abs(x).max():18.1f}")
