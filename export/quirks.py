import numpy as np
import torch
from transformers import GPT2LMHeadModel
from gpt2_numpy import wte, layers

model = GPT2LMHeadModel.from_pretrained("gpt2")

# 1. Weight tying: is the output matrix the same object as the embedding table?
out_w = model.lm_head.weight
emb_w = model.transformer.wte.weight
print("output matrix shape:", tuple(out_w.shape), "  wte shape:", tuple(emb_w.shape))
print("same memory?       ", out_w.data_ptr() == emb_w.data_ptr())

# 2. Conv1D vs a normal PyTorch Linear layer, both 768 -> 3072
print("\nGPT-2 MLP weight (Conv1D):     ", layers[0]["fc_w"].shape)
print("PyTorch Linear 768 -> 3072:    ", tuple(torch.nn.Linear(768, 3072).weight.shape))

# 3. wte_T: flip rows and columns
wte_T = np.ascontiguousarray(wte.T)
print("\nwte shape:", wte.shape, "   wte_T shape:", wte_T.shape)
print("wte[1][0]   =", wte[1, 0])
print("wte_T[0][1] =", wte_T[0, 1])
