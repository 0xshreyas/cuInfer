import numpy as np
from gpt2_numpy import wte, wpe

tokens = [15496, 11, 616, 1438, 318]
expected = (wte[tokens] + wpe[:5]).astype(np.float32)
saved = np.fromfile("oracle_embed.bin", dtype=np.float32).reshape(5, 768)

print("expected first 3:", expected[0][:3])
print("saved    first 3:", saved[0][:3])
print("max diff:", np.abs(expected - saved).max())
