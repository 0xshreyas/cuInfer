import numpy as np
from transformers import GPT2Tokenizer
from gpt2_numpy import wte

tok = GPT2Tokenizer.from_pretrained("gpt2")

def vec(word):
    ids = tok.encode(word)
    return wte[ids[0]]          # the word's 768 numbers

def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

print("' cat' first 8 of 768 numbers:", np.round(vec(" cat")[:8], 3))
print()

pairs = [(" cat", " dog"), (" cat", " kitten"), (" cat", " car"),
         (" king", " queen"), (" Monday", " Tuesday"), (" cat", " whereas")]
for a, b in pairs:
    print(f"{a!r:10} vs {b!r:10}  similarity = {cos(vec(a), vec(b)):.3f}")
