import numpy as np
from gpt2_numpy import forward_pass, layerNorm, layers, wte, wpe, ln_f_w, ln_f_b

tokens = [15496, 11, 616, 1438, 318]
np.array(tokens, dtype=np.int32).tofile("oracle_tokens.bin")

logits = forward_pass(tokens)
logits.astype(np.float32).tofile("oracle_logits.bin")
print("Logits: ", logits.shape)

x = wte[tokens] + wpe[:len(tokens)]
x.astype(np.float32).tofile("oracle_embed.bin")

from gpt2_numpy import block
for i, L in enumerate(layers):
    x = block(x, L)
    x.astype(np.float32).tofile(f"Oracle_layer{i:02d}.bin")
print("Saved 12 Layer Outputs")