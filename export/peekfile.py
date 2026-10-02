import os, struct
import numpy as np
from gpt2_numpy import wpe

path = "gpt2_124m.bin"
print("file size in bytes:", os.path.getsize(path))
print("expected:          ", 28 + 124439808 * 4)

with open(path, "rb") as f:
    header = f.read(28)
    print("\nfirst 28 bytes, raw:", header.hex(" "))
    values = struct.unpack("7i", header)
    print("read as 7 integers: ", values)
    print("first one in hex:   ", hex(values[0]))

    wpe_start = 28 + 50257 * 768 * 4        # skip the header, then all of wte
    f.seek(wpe_start)                       # jump straight to that byte
    first3 = np.frombuffer(f.read(12), dtype=np.float32)

print("\nwpe starts at byte:          ", wpe_start)
print("3 floats read from that byte:", first3)
print("wpe's first 3 in gpt2_numpy: ", wpe[0, :3])
