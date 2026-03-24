import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEQ = 256
BS = 8
D = 384
NL = 4
NH = 6
STEPS = 10000
LR = 3e-4
CLIP = 1.0

TOKENS = Path("tokens.npy")
META = Path("tokens_meta.json")
CKPT = Path("checkpoint.pt")


def _vocab():
    if META.is_file():
        return int(json.loads(META.read_text())["vocab_size"])
    return 50257


class Attn(nn.Module):
    def __init__(self, d, nh, T):
        super().__init__()
        self.nh, self.dh = nh, d // nh
        self.scale = 1.0 / math.sqrt(self.dh)

        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d)

        self.register_buffer("mask", torch.tril(torch.ones(T, T)).view(1, 1, T, T))

    def forward(self, x):
        b, t, c = x.shape
        q, k, v = self.qkv(x).chunk(3, -1)

        q = q.view(b, t, self.nh, self.dh).transpose(1, 2)
        k = k.view(b, t, self.nh, self.dh).transpose(1, 2)
        v = v.view(b, t, self.nh, self.dh).transpose(1, 2)

        w = (q @ k.transpose(-2, -1)) * self.scale

        w = w.masked_fill(self.mask[:, :, :t, :t] == 0, float("-inf"))
        w = F.softmax(w, -1) @ v
        w = w.transpose(1, 2).reshape(b, t, c)

        return self.o(w)


class Block(nn.Module):
    def __init__(self, d, nh, T):
        super().__init__()
        self.l1 = nn.LayerNorm(d)
        self.a = Attn(d, nh, T)
        self.l2 = nn.LayerNorm(d)

        self.up = nn.Linear(d, 4 * d)
        self.down = nn.Linear(4 * d, d)

    def forward(self, x):
        x = x + self.a(self.l1(x))
        x = x + self.down(F.gelu(self.up(self.l2(x))))
        return x


class GPT(nn.Module):
    def __init__(self, V, T, d, nl, nh):
        super().__init__()
        self.T = T
        self.wte = nn.Embedding(V, d)
        self.wpe = nn.Embedding(T, d)
        self.blocks = nn.ModuleList(Block(d, nh, T) for _ in range(nl))
        self.ln = nn.LayerNorm(d)
        self.lm = nn.Linear(d, V, bias=False)

    def forward(self, idx):
        t = idx.size(1)
        pos = torch.arange(t, device=idx.device)
        x = self.wte(idx) + self.wpe(pos)

        for b in self.blocks:
            x = b(x)

        return self.lm(self.ln(x))


def _batch(data, dev):
    hi = len(data) - SEQ - 1
    ix = torch.randint(0, hi, (BS,))
    xs, ys = [], []

    for i in ix:
        ii = int(i.item())
        chunk = data[ii : ii + SEQ + 1].astype(np.int64, copy=False)
        xs.append(torch.from_numpy(chunk[:-1]))
        ys.append(torch.from_numpy(chunk[1:]))
    return torch.stack(xs).to(dev), torch.stack(ys).to(dev)


def train():
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    V = _vocab()

    data = np.load(TOKENS, mmap_mode="r")

    if len(data) <= SEQ:
        raise ValueError(f"tokens.npy length {len(data)} must be > SEQ ({SEQ})")

    m = GPT(V, SEQ, D, NL, NH).to(dev)

    opt = torch.optim.AdamW(m.parameters(), lr=LR)

    m.train()

    for s in range(1, STEPS + 1):
        x, y = _batch(data, dev)
        logits = m(x)
        loss = F.cross_entropy(logits.view(-1, V), y.view(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), CLIP)
        opt.step()
        if s % 50 == 0 or s == 1:
            print(f"step {s:5d} loss {loss.item():.4f}")

    torch.save(
        {"state": m.state_dict(), "cfg": {"V": V, "SEQ": SEQ, "D": D, "NL": NL, "NH": NH}},
        CKPT,
    )

    print("saved", CKPT.resolve())


if __name__ == "__main__":
    train()
