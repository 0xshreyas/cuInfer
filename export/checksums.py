from transformers import GPT2LMHeadModel
import numpy as np

model = GPT2LMHeadModel.from_pretrained("gpt2")
sd = model.state_dict()

for name in ["transformer.wte.weight", "transformer.h.0.attn.c_attn.weight", "transformer.h.11.mlp.c_proj.weight", "transformer.ln_f.bias"]:
    t = sd[name].detach().numpy().astype(np.float32)
    print(f"{name}")
    print(f"   first 3:   {t.flatten()[:3]}")
    print(f"   sum:   {t.sum():.6f}")