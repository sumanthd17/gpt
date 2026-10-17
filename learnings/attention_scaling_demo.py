"""
Why attention divides scores by sqrt(hs).

If q and k entries are ~N(0, 1), then q·k is a sum of hs products, each with
variance 1, so Var(q·k) = hs and std = sqrt(hs). Large scores make softmax
nearly one-hot and its gradients vanish. Dividing by sqrt(hs) restores std ~1.

Run: uv run python attention_scaling_demo.py
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(0)
T = 32  # sequence length (number of keys each query attends over)


def scores(hs, scaled, n_batch=256):
    q = torch.randn(n_batch, T, hs)
    k = torch.randn(n_batch, T, hs)
    att = q @ k.transpose(-2, -1)  # (B, T, T)
    if scaled:
        att = att / hs**0.5  # same as * (1.0 / math.sqrt(k.size(-1)))
    return att


def softmax_grad_norm(att):
    # Push a random upstream gradient back through softmax and measure how
    # much reaches the scores. This is what the q/k weights get to learn from.
    att = att.detach().requires_grad_(True)
    w = F.softmax(att, dim=-1)
    upstream = torch.randn_like(w)
    (w * upstream).sum().backward()
    return att.grad.norm(dim=-1).mean().item()


fig, axes = plt.subplots(2, 2, figsize=(13, 9))
fig.suptitle("Scaled dot-product attention: why divide by sqrt(hs)?", fontsize=14)

# 1. Score distribution at hs=64
hs = 64
raw, scl = scores(hs, False).flatten(), scores(hs, True).flatten()
ax = axes[0, 0]
ax.hist(raw, bins=120, alpha=0.6, density=True, label=f"q·k  (std={raw.std():.2f})")
ax.hist(scl, bins=120, alpha=0.6, density=True, label=f"q·k / sqrt(64)  (std={scl.std():.2f})")
ax.set_title(f"1. Distribution of attention scores, hs={hs}")
ax.set_xlabel("score")
ax.set_xlim(-30, 30)
ax.legend()

# 2. One softmax row, unscaled vs scaled
q = torch.randn(1, hs)
k = torch.randn(T, hs)
row = (q @ k.T).squeeze(0)
w_raw = F.softmax(row, dim=-1)
w_scl = F.softmax(row / hs**0.5, dim=-1)
ax = axes[0, 1]
x = torch.arange(T)
ax.bar(x - 0.2, w_raw, width=0.4, label=f"softmax(q·k), max={w_raw.max():.2f}")
ax.bar(x + 0.2, w_scl, width=0.4, label=f"softmax(q·k/sqrt(hs)), max={w_scl.max():.2f}")
ax.axhline(1 / T, color="gray", ls="--", lw=1, label=f"uniform = 1/T = {1/T:.3f}")
ax.set_title(f"2. Attention weights for one query over {T} keys (hs={hs})")
ax.set_xlabel("key position")
ax.set_ylabel("attention weight")
ax.legend()

# 3 & 4. Sweep head size
head_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
stats = {False: {"std": [], "max": [], "grad": []}, True: {"std": [], "max": [], "grad": []}}
for h in head_sizes:
    for scaled in (False, True):
        a = scores(h, scaled)
        stats[scaled]["std"].append(a.std().item())
        stats[scaled]["max"].append(F.softmax(a, dim=-1).max(dim=-1).values.mean().item())
        stats[scaled]["grad"].append(softmax_grad_norm(a))

ax = axes[1, 0]
ax.plot(head_sizes, stats[False]["std"], "o-", label="std of q·k")
ax.plot(head_sizes, stats[True]["std"], "o-", label="std of q·k / sqrt(hs)")
ax.plot(head_sizes, [h**0.5 for h in head_sizes], "k:", label="sqrt(hs) (theory)")
ax.set_xscale("log", base=2)
ax.set_yscale("log", base=2)
ax.set_title("3. Score std grows like sqrt(hs) unless scaled")
ax.set_xlabel("head size hs")
ax.set_ylabel("std of scores")
ax.legend()

ax = axes[1, 1]
ax.plot(head_sizes, stats[False]["max"], "o-", color="C0", label="max weight, unscaled")
ax.plot(head_sizes, stats[True]["max"], "o-", color="C1", label="max weight, scaled")
ax.set_xscale("log", base=2)
ax.set_ylim(0, 1.05)
ax.set_xlabel("head size hs")
ax.set_ylabel("avg max softmax weight (1.0 = one-hot)")
ax2 = ax.twinx()
ax2.plot(head_sizes, stats[False]["grad"], "s--", color="C0", alpha=0.7, label="grad norm, unscaled")
ax2.plot(head_sizes, stats[True]["grad"], "s--", color="C1", alpha=0.7, label="grad norm, scaled")
ax2.set_ylabel("gradient norm reaching scores (dashed)")
ax.set_title("4. Unscaled: softmax goes one-hot, gradients vanish")
h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="center right", fontsize=8)

fig.tight_layout()
fig.savefig("attention_scaling_demo.png", dpi=130)
print("saved attention_scaling_demo.png")
for h, gu, gs, mu, ms in zip(head_sizes, stats[False]["grad"], stats[True]["grad"],
                             stats[False]["max"], stats[True]["max"]):
    print(f"hs={h:5d}  max_w unscaled={mu:.3f} scaled={ms:.3f}   grad unscaled={gu:.4f} scaled={gs:.4f}")
