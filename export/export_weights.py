from transformers import GPT2LMHeadModel
import numpy as np
import struct

model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()
sd = model.state_dict()

MAGIC = 0x63754C4D #cuLM in binary

with open("gpt2_124m.bin", "wb") as f:
    f.write(struct.pack("7i", MAGIC, 1, 12, 12, 768, 50257, 1024))

    def dump(name):
        t = sd[name].detach().numpy().astype(np.float32)
        t.tofile(f)
        return t.size
    total = 0
    total += dump("transformer.wte.weight")
    total += dump("transformer.wpe.weight")

    for i in range(12):
        p = f"transformer.h.{i}."
        for suffix in ["ln_1.weight", "ln_1.bias", "attn.c_attn.weight", "attn.c_attn.bias", "attn.c_proj.weight", "attn.c_proj.bias", "ln_2.weight", "ln_2.bias", "mlp.c_fc.weight", "mlp.c_fc.bias", "mlp.c_proj.weight", "mlp.c_proj.bias"]:
            total += dump(p + suffix)
    total += dump("transformer.ln_f.weight")
    total += dump("transformer.ln_f.bias")
    print(f"Wrote {total:,} floats = {total * 4 / 1e6:.1f} MB")