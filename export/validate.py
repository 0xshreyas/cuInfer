import numpy as np
import torch
from transformers import GPT2LMHeadModel
from gpt2_numpy import forward_pass

tokens = [15496, 11, 616, 1438, 318] # Hello, my name is (From gpt2_numpy.py)

# Reference
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()
with torch.no_grad():
    ref = model(torch.tensor([tokens])).logits[0].numpy()

# Values produced from the forward pass function
here = forward_pass(tokens)

print("Shapes: ", ref.shape, here.shape)

diff = np.abs(ref - here)
print(f"Max Abs Difference: {diff.max():.6e}")
print(f"Mean Abs Difference: {diff.mean():.6e}")

# We need the difference to be less than 1e-3, error here means there is an issue with the implementation of the formaula somewhere in GELU, layerNorm etc
if diff.max() < 1e-3:
    print("PASS")
else:
    print("FAIL")
    worst = np.unravel_index(diff.argmax(), diff.shape)
    print(f"Worst at position {worst}: ref={ref[worst]:.6f} here={here[worst]:.6f}")

# Bugs detected where calculation error in v was detected, GELU had a + instead of * and MLP where * is there instead of +, 