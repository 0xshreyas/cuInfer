# cuInfer results (RTX 5060 Laptop GPU, fp32)

## Matmul kernels, 128 tokens: GFLOPS (% of cuBLAS)

| Shape | cuBLAS | naive | tiled | 1d | 2d | 2d vs naive |
|---|---|---|---|---|---|---|
| qkv | 6596 | 1042 (16%) | 1445 (22%) | 2177 (33%) | 2008 (30%) | 1.9x |
| proj | 4435 | 983 (22%) | 1355 (31%) | 1714 (39%) | 670 (15%) | 0.7x |
| fc | 7480 | 1050 (14%) | 1456 (19%) | 2285 (31%) | 2678 (36%) | 2.6x |
| fcproj | 7359 | 1038 (14%) | 1395 (19%) | 1776 (24%) | 705 (10%) | 0.7x |
| logits | 9144 | 1061 (12%) | 1441 (16%) | 2303 (25%) | 3447 (38%) | 3.2x |

## Matmul kernels, 512 tokens: GFLOPS (% of cuBLAS)

| Shape | cuBLAS | naive | tiled | 1d | 2d | 2d vs naive |
|---|---|---|---|---|---|---|
| qkv | 8534 | 1091 (13%) | 1497 (18%) | 2522 (30%) | 3727 (44%) | 3.4x |
| proj | 7532 | 1069 (14%) | 1453 (19%) | 2273 (30%) | 2676 (36%) | 2.5x |
| fc | 8487 | 1090 (13%) | 1496 (18%) | 2619 (31%) | 4314 (51%) | 4.0x |
| fcproj | 8488 | 1086 (13%) | 1463 (17%) | 2345 (28%) | 2800 (33%) | 2.6x |
| logits | 10953 | 1068 (10%) | 1445 (13%) | 2342 (21%) | 3502 (32%) | 3.3x |

## End-to-end GPT-2 (whole forward pass)

| | Before (naive matmul) | After | Speedup |
|---|---|---|---|
| Prefill, 5 tokens | 4.79 ms | 4.66 ms | 1.03x |
| Prefill, 128 tokens | 33.91 ms | 15.84 ms | 2.14x |
| Prefill, 512 tokens | 160.36 ms | 85.72 ms | 1.87x |
| Decode | 213 tokens/s | 216 tokens/s | 1.01x |
