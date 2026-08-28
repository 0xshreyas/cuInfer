from transformers import GPT2LMHeadModel, GPT2Config
import torch

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()

cfg = model.config
print(f"Layers:     {cfg.n_layer}")
print(f"Heads:      {cfg.n_head}")
print(f"d_model:    {cfg.n_embd}")
print(f"vocab:      {cfg.vocab_size}")
print(f"max_pos:   {cfg.n_positions}")
print()

total = 0
for name, param in model.named_parameters():
    total += param.numel()
    print(f"{name:45s} {str(tuple(param.shape)):20s}")
print(f"\nTotal Parameters: {total:,}")