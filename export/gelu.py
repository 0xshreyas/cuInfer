import numpy as np
from gpt2_numpy import GELU, layers

print("GELU on a few numbers:")
for v in [-4, -2, -1, -0.5, 0, 0.5, 1, 2, 4]:
    print(f"  GELU({v:5}) = {GELU(np.float32(v)):7.3f}")

L = layers[0]
A, B = L["fc_w"], L["fcproj_w"]            # the MLP's two weight matrices
x = np.random.randn(5, 768).astype(np.float32)

merged = x @ (A @ B)                       # one single multiplication
no_gelu = (x @ A) @ B                      # two multiplications, nothing between
with_gelu = GELU(x @ A) @ B                # two multiplications, GELU between

print("\nhidden shape (the widened middle):", (x @ A).shape)
print("without GELU vs merged, biggest difference:", np.abs(no_gelu - merged).max())
print("with GELU    vs merged, biggest difference:", np.abs(with_gelu - merged).max())
