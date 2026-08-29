import numpy as np
import struct

# Load the binary that was exported using the same layout the loader.c uses

with open("gpt2_124m.bin", "rb") as f:
    magic, version, n_layer, n_head, d_model, vocab_size, max_pos = struct.unpack("7i", f.read(28))
    assert magic == 0x63754C4D, f"Bad Magic: {magic:#x}"
    blob = np.fromfile(f, dtype=np.float32)

print(f"Loaded {blob.size:,} floats")

# Each cell returns the view of the next float, basically like pointers
offset = 0
def take(*shape):
    global offset
    n = int(np.prod(shape))
    arr = blob[offset:offset + n].reshape(shape)
    offset += n
    return arr

wte = take(vocab_size, d_model) # token embeddings
wpe = take(max_pos, d_model) # position embeddings

layers = []
for i in range(n_layer):
    layers.append({
        "ln_1_w":  take(d_model),
        "ln_1_b":  take(d_model),
        "attn_w":  take(d_model, 3 * d_model),
        "attn_b":  take(3 * d_model),
        "proj_w":  take(d_model, d_model),
        "proj_b":  take(d_model),
        "ln_2_w":  take(d_model),
        "ln_2_b":  take(d_model),
        "fc_w":    take(d_model, 4 * d_model),
        "fc_b":    take(4 * d_model),
        "fcproj_w": take(4 * d_model, d_model),
        "fcproj_b": take(d_model),
    })

ln_f_w = take(d_model)
ln_f_b = take(d_model)

assert offset == blob.size, f"Used {offset} / {blob.size} floats"
print("All weights have been consumed fully!")

# Operations with their formula

# x = (T, d_model)
def layerNorm(x, g, b, eps=1e-5):
    mean = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    return g * (x - mean) / np.sqrt(var + eps) + b

# TanH approximation which is used by GPT2
def GELU(x):
    return 0.5 * x * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)))

# Subtracting max first as it stops exp() from overflowing
def softMax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)

def attention(x, L):
    T = x.shape[0]
    head_dim = d_model // n_head

    # One matmul returns Q, K and V concatenated in (T, 2304)
    qkv = x @ L["attn_w"] + L["attn_b"]

    #Split the QKV into each Q, K and V respective parts

    q = qkv[:, 0:d_model]
    k = qkv[:, d_model:2 * d_model]
    v = qkv[:, 2 * d_model:3 * d_model]

    # (T, 768) is being converted to (T, 12, 64), splitting the floats each layer

    q = q.reshape(T, n_head, head_dim).transpose(1, 0, 2)
    k = k.reshape(T, n_head, head_dim).transpose(1, 0, 2)
    v = v.reshape(T, n_head, head_dim).transpose(1, 0, 2)

    # (12, T, 64) @ (12, 64, T) = (12, T, T)
    scores = q @ k.transpose(0, 2, 1) / np.sqrt(head_dim)

    # Mask, token i should not be able to peek into any j > i
    mask = np.triu(np.ones((T, T), dtype=bool), k = 1)
    scores[:, mask] = -np.inf

    weights = softMax(scores)
    out = weights @ v

    out = out.transpose(1, 0, 2).reshape(T, d_model)
    return out @ L["proj_w"] + L["proj_b"]

def MLP(x, L):
    h = GELU(x @ L["fc_w"] + L["fc_b"])
    return h @ L["fcproj_w"] + L["fcproj_b"]

def block(x, L):
    x = x + attention(layerNorm(x, L["ln_1_w"], L["ln_1_b"]), L)
    x = x + MLP(layerNorm(x, L["ln_2_w"], L["ln_2_b"]), L)
    return x

def forward_pass(tokens):
    T = len(tokens)
    x = wte[tokens] + wpe[:T] # Gives (T, 768)
    for L in layers:
        x = block(x, L)
    x = layerNorm(x, ln_f_w, ln_f_b)
    return x @ wte.T

# Testing

if __name__ == '__main__':
    tokens = [15496, 11, 616, 1438, 318] # Hello, my name is
    logits = forward_pass(tokens)
    print("Logits Shape: ", logits.shape)

    last = logits[-1]
    top5 = np.argsort(last)[-5:][::-1]
    print("Top 5 token IDs: ",top5)
    print("Their respective logits:   ", last[top5])
