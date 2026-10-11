#!/usr/bin/env python3
# plot_results.py: turns the raw benchmark outputs in results/ into
# results/summary.md (tables) and two charts. Run from ~/cuinfer.
import re
import matplotlib
matplotlib.use("Agg")                      # draw straight to image files, no window
import matplotlib.pyplot as plt

R = "results/"
SHAPES = ["qkv", "proj", "fc", "fcproj", "logits"]
KERNELS = ["naive", "tiled", "1d", "2d"]
COLORS = {"naive": "#2a78d6", "tiled": "#eb6834", "1d": "#1baf7a", "2d": "#eda100"}
AFTER = "#6250d6"
GRAY, INK, INK2, SURFACE, GRID = "#a3a29c", "#0b0b0b", "#52514e", "#fcfcfb", "#e6e5e0"

def read_kernels(path):
    """benchmark output -> {shape: {kernel: (ms, GFLOPS, % of cuBLAS)}}"""
    rows = {}
    pat = re.compile(r"^(\w+)\s+(cublas|naive|tiled|1d|2d)\s+([\d.]+)\s+(\d+)\s+(\d+)%")
    for line in open(path):
        m = pat.match(line)
        if m:
            shape, k, ms, gf, pct = m.groups()
            rows.setdefault(shape, {})[k] = (float(ms), int(gf), int(pct))
    return rows

def read_e2e(path):
    """first 'End-to-end' block -> {5: ms, 128: ms, 512: ms, 'tps': tokens/s}"""
    block = open(path).read().split("=== End-to-end (clean timing) ===")[1]
    out = {int(t): float(ms) for t, ms in
           re.findall(r"prefill T=\s*(\d+) :\s+([\d.]+) ms", block)}
    out["tps"] = float(re.search(r"=\s+([\d.]+) tokens/s", block).group(1))
    return out

def style(ax, title):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK2, length=0)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=12)

k128 = read_kernels(R + "kernels_128.txt")
k512 = read_kernels(R + "kernels_512.txt")
old = read_e2e(R + "e2e_old.txt")
new = read_e2e(R + "e2e_new.txt")

# ---------- summary.md: the tables for the README ----------
lines = ["# cuInfer results (RTX 5060 Laptop GPU, fp32)", ""]
for M, rows in [(128, k128), (512, k512)]:
    lines += [f"## Matmul kernels, {M} tokens: GFLOPS (% of cuBLAS)", "",
              "| Shape | cuBLAS | naive | tiled | 1d | 2d | 2d vs naive |",
              "|---|---|---|---|---|---|---|"]
    for s in SHAPES:
        r = rows[s]
        cells = " | ".join(f"{r[k][1]} ({r[k][2]}%)" for k in KERNELS)
        lines.append(f"| {s} | {r['cublas'][1]} | {cells} | "
                     f"{r['2d'][1] / r['naive'][1]:.1f}x |")
    lines.append("")
lines += ["## End-to-end GPT-2 (whole forward pass)", "",
          "| | Before (naive matmul) | After | Speedup |", "|---|---|---|---|"]
for T in (5, 128, 512):
    lines.append(f"| Prefill, {T} tokens | {old[T]:.2f} ms | {new[T]:.2f} ms | "
                 f"{old[T] / new[T]:.2f}x |")
lines.append(f"| Decode | {old['tps']:.0f} tokens/s | {new['tps']:.0f} tokens/s | "
             f"{new['tps'] / old['tps']:.2f}x |")
open(R + "summary.md", "w").write("\n".join(lines) + "\n")

# ---------- chart 1: every kernel on every shape, 512 tokens ----------
fig, ax = plt.subplots(figsize=(10, 4.8), facecolor=SURFACE)
w = 0.16
for i, k in enumerate(KERNELS + ["cublas"]):
    xs = [j + (i - 2) * w for j in range(len(SHAPES))]
    ys = [k512[s][k][1] for s in SHAPES]
    ax.bar(xs, ys, width=w, color=GRAY if k == "cublas" else COLORS[k],
           edgecolor=SURFACE, linewidth=1.5,
           label="cuBLAS (reference)" if k == "cublas" else k)
    if k == "2d":                                   # label only the final kernel
        for x, y, s in zip(xs, ys, SHAPES):
            ax.text(x, y + 120, f"{k512[s]['2d'][2]}%", ha="center",
                    va="bottom", fontsize=8, color=INK2)
ax.set_xticks(range(len(SHAPES)))
ax.set_xticklabels(SHAPES, color=INK)
ax.set_ylabel("GFLOPS", color=INK2)
ax.set_ylim(0, max(k512[s]["cublas"][1] for s in SHAPES) * 1.3)
style(ax, "Matmul throughput on GPT-2 shapes, 512 tokens (higher is better)")
ax.legend(ncol=5, frameon=False, loc="upper left", labelcolor=INK2)
fig.text(0.01, 0.01, "Labels above the 2d bars: % of cuBLAS.",
         fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.03, 1, 1))
fig.savefig(R + "kernels_512.png", dpi=160, facecolor=SURFACE)

# ---------- chart 2: the whole model, before vs after ----------
fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
Ts, w = [128, 512], 0.36
for i, (name, d, color) in enumerate([("Before (naive matmul)", old, GRAY),
                                      ("After (dispatched kernels)", new, AFTER)]):
    xs = [j + (i - 0.5) * w for j in range(len(Ts))]
    ys = [d[T] for T in Ts]
    ax.bar(xs, ys, width=w, color=color, edgecolor=SURFACE, linewidth=1.5, label=name)
    for x, y in zip(xs, ys):
        ax.text(x, y + 2, f"{y:.1f} ms", ha="center", va="bottom", fontsize=9, color=INK)
ax.set_xticks(range(len(Ts)))
ax.set_xticklabels([f"{T}-token prompt" for T in Ts], color=INK)
ax.set_ylabel("milliseconds", color=INK2)
ax.set_ylim(0, max(old[T] for T in Ts) * 1.25)
style(ax, "GPT-2 prefill time, whole model (lower is better)")
ax.legend(frameon=False, loc="upper left", labelcolor=INK2)
fig.text(0.01, 0.01,
         f"Speedup: {old[128] / new[128]:.1f}x at 128 tokens, {old[512] / new[512]:.1f}x "
         f"at 512. Decode: {old['tps']:.0f} -> {new['tps']:.0f} tokens/s "
         f"(memory-bound, unchanged).", fontsize=8, color=INK2)
fig.tight_layout(rect=(0, 0.04, 1, 1))
fig.savefig(R + "e2e_prefill.png", dpi=160, facecolor=SURFACE)

print(open(R + "summary.md").read())
print("Charts written: results/kernels_512.png, results/e2e_prefill.png")
