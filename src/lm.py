"""Small character-level GPT on TinyStories: do the toy-task findings carry over to a real LM?

Every sequence is [BOS] + ctx-1 characters (so position 0 is a BOS token, as in the synthetic task):
  --sample start   the first ctx-1 characters of one story (padded)
  --sample random  ctx-1 characters from a random offset in the concatenated story stream (packed, as in
                   real LM pre-training)
Attention variants:
  softmax   standard causal softmax
  softmax1  softmax with an extra zero logit (implemented as a zero key/value slot)
  gated     softmax + elementwise, head-specific, query-dependent sigmoid output gate (Qiu et al. 2025, G1)
            --gate_lr_mult lambda  : learning-rate multiplier for the gate parameters only
            --gate_kind scaled     : race-matched gate, kappa = gate_scale*sqrt(d_head), muP-style
                                     (init/kappa, multiplier kappa)  ==  gate_lr_mult = kappa under Adam

Logged every --eval_every steps (and at the end) on held-out stories:
  val_loss (nats/char), per-head BOS attention mass (mean over query positions >= 1),
  fraction of sink heads (BOS mass > 0.5), massive activations (max |h| of the residual stream and its
  ratio to the median |h|), and per-head mean gate value on non-BOS queries.

  python src/lm.py --attn gated --gate_lr_mult 10 --out results/lm/gated_lam10_s0.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
import urllib.request

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BASE = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
TRAIN_URL, VALID_URL = BASE + "TinyStoriesV2-GPT4-train.txt", BASE + "TinyStoriesV2-GPT4-valid.txt"
SEP = "<|endoftext|>"
BOS, PAD, UNK = 0, 1, 2


# ----------------------------------------------------------------------------- data
def fetch(url, nbytes, cache):
    if os.path.exists(cache):
        return open(cache, encoding="utf-8", errors="ignore").read()
    req = urllib.request.Request(url, headers={"Range": f"bytes=0-{nbytes - 1}", "User-Agent": "python-urllib"})
    with urllib.request.urlopen(req, timeout=300) as r:
        data = r.read()
    os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
    open(cache, "wb").write(data)
    return data.decode("utf-8", errors="ignore")


def load_stories(args):
    tr = fetch(TRAIN_URL, args.train_mb << 20, os.path.join(args.data_dir, f"ts_train_{args.train_mb}mb.txt"))
    va = fetch(VALID_URL, 8 << 20, os.path.join(args.data_dir, "ts_valid_8mb.txt"))
    tr = [s.strip() for s in tr.split(SEP)[:-1] if len(s.strip()) > 20]   # drop the cut-off last story
    va = [s.strip() for s in va.split(SEP)[:-1] if len(s.strip()) > 20]
    return tr, va


def build_vocab(stories, max_chars=96):
    cnt = {}
    for s in stories[:20000]:
        for ch in s:
            cnt[ch] = cnt.get(ch, 0) + 1
    chars = sorted(cnt, key=lambda c: -cnt[c])[:max_chars]
    return {c: i + 3 for i, c in enumerate(sorted(chars))}


def encode(stories, vocab, ctx):
    X = torch.full((len(stories), ctx), PAD, dtype=torch.long)
    X[:, 0] = BOS
    for n, s in enumerate(stories):
        ids = [vocab.get(c, UNK) for c in s[:ctx - 1]]
        X[n, 1:1 + len(ids)] = torch.tensor(ids)
    return X


def encode_stream(stories, vocab):
    """All stories joined by newlines, as a uint8 id stream (fast: unicode code point -> id lookup table)."""
    lut = np.full(0x110000, UNK, dtype=np.uint8)
    for c, i in vocab.items():
        lut[ord(c)] = i
    text = "\n".join(stories)
    cp = np.frombuffer(text.encode("utf-32-le"), dtype=np.uint32)
    return torch.from_numpy(lut[cp])


def windows(stream, idx, ctx):
    """[BOS] + stream[idx : idx+ctx-1] for every offset in idx."""
    X = stream[idx[:, None] + torch.arange(ctx - 1, device=stream.device)[None, :]].long()
    return torch.cat([torch.full((len(idx), 1), BOS, dtype=torch.long, device=stream.device), X], 1)


# ----------------------------------------------------------------------------- model
class Attn(nn.Module):
    def __init__(self, d, H, attn, gate_kind="linear", gate_scale=2.5):
        super().__init__()
        self.H, self.dh, self.attn = H, d // H, attn
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.kappa = 1.0
        if attn == "gated":
            self.gate = nn.Linear(d, d, bias=True)
            if gate_kind == "scaled":
                self.kappa = gate_scale * math.sqrt(self.dh)
            nn.init.normal_(self.gate.weight, std=0.02 / self.kappa)
            nn.init.zeros_(self.gate.bias)

    def forward(self, x, want_stats=False):
        B, T, d = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.H, self.dh).permute(2, 0, 3, 1, 4)
        stats = {}
        if self.attn == "softmax1":
            zk = torch.zeros(B, self.H, 1, self.dh, device=x.device, dtype=k.dtype)
            k, v = torch.cat([zk, k], 2), torch.cat([zk, v], 2)
            mask = torch.ones(T, T + 1, dtype=torch.bool, device=x.device).tril(1)   # null slot + causal
        else:
            mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        if want_stats:   # explicit attention to read the weights
            a = ((q @ k.transpose(-1, -2)) / math.sqrt(self.dh)).float().masked_fill(~mask, float("-inf"))
            a = a.softmax(-1)
            off = 1 if self.attn == "softmax1" else 0
            stats["bos"] = a[:, :, 1:, off].mean((0, 2))                         # H
            stats["maxkey"] = a[:, :, 1:, off:].max(-1).values.mean((0, 2))      # H: anchor on ANY key
            stats["null"] = a[:, :, 1:, 0].mean((0, 2)) if off else None
            y = (a.to(v.dtype) @ v)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        if self.attn == "gated":
            g = torch.sigmoid(self.kappa * self.gate(x)).view(B, T, self.H, self.dh).transpose(1, 2)
            y = y * g
            if want_stats:
                stats["gate"] = g[:, :, 1:].float().mean((0, 2, 3))              # H
                stats["gate_bos"] = g[:, :, 0].float().mean((0, 2))
        return self.proj(y.transpose(1, 2).reshape(B, T, d)), stats


class Block(nn.Module):
    def __init__(self, d, H, attn, gate_kind, gate_scale):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attn(d, H, attn, gate_kind, gate_scale)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, want_stats=False):
        y, st = self.attn(self.ln1(x), want_stats)
        x = x + y
        return x + self.mlp(self.ln2(x)), st


class GPT(nn.Module):
    def __init__(self, V, ctx, d, H, D, attn, gate_kind="linear", gate_scale=2.5):
        super().__init__()
        self.tok, self.pos = nn.Embedding(V, d), nn.Embedding(ctx, d)
        self.blocks = nn.ModuleList(Block(d, H, attn, gate_kind, gate_scale) for _ in range(D))
        self.ln = nn.LayerNorm(d)
        for n, p in self.named_parameters():
            if p.dim() == 2 and "gate" not in n:
                nn.init.normal_(p, std=0.02)
        for b in self.blocks:   # GPT-2 style scaled init of residual projections
            nn.init.normal_(b.attn.proj.weight, std=0.02 / math.sqrt(2 * D))
            nn.init.normal_(b.mlp[2].weight, std=0.02 / math.sqrt(2 * D))

    def forward(self, idx, want_stats=False):
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        stats, acts = [], []
        for b in self.blocks:
            x, st = b(x, want_stats)
            stats.append(st)
            if want_stats:
                acts.append(x.detach().float())
        return self.ln(x) @ self.tok.weight.T, stats, acts


# ----------------------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(model, Xva, amp, n_stats=64):
    model.eval()
    tot, cnt = 0.0, 0
    for i in range(0, len(Xva), 128):
        X = Xva[i:i + 128]
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            logits, _, _ = model(X)
        tgt = X[:, 1:]
        l = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]), tgt.reshape(-1),
                            ignore_index=PAD, reduction="sum")
        tot += l.item()
        cnt += (tgt != PAD).sum().item()
    full = Xva[(Xva != PAD).all(1)][:n_stats]          # stories that fill the whole context
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        _, stats, acts = model(full, want_stats=True)
    heads = []
    for li, st in enumerate(stats):
        for h in range(len(st["bos"])):
            r = dict(layer=li, head=h, bos=st["bos"][h].item(), maxkey=st["maxkey"][h].item())
            if st.get("null") is not None:
                r["null"] = st["null"][h].item()
            if "gate" in st:
                r["gate"], r["gate_bos"] = st["gate"][h].item(), st["gate_bos"][h].item()
            heads.append(r)
    act = [dict(layer=li, max=a.abs().max().item(), median=a.abs().median().item(),
                max_pos0=a[:, 0].abs().max().item(), max_rest=a[:, 1:].abs().max().item())
           for li, a in enumerate(acts)]
    model.train()
    return dict(val_loss=tot / cnt, heads=heads, acts=act,
                sink_frac=sum(h["bos"] > 0.5 for h in heads) / len(heads),
                anchor_frac=sum(h["maxkey"] > 0.5 for h in heads) / len(heads),
                mean_bos=sum(h["bos"] for h in heads) / len(heads),
                massive_ratio=max(a["max"] / max(a["median"], 1e-6) for a in act))


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--attn", default="softmax", choices=["softmax", "softmax1", "gated"])
    ap.add_argument("--gate_kind", default="linear", choices=["linear", "scaled"])
    ap.add_argument("--gate_scale", type=float, default=2.5)
    ap.add_argument("--gate_lr_mult", type=float, default=1.0)
    ap.add_argument("--d", type=int, default=384)
    ap.add_argument("--H", type=int, default=6)
    ap.add_argument("--D", type=int, default=6)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=300)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--train_mb", type=int, default=200)
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--prefetch_only", action="store_true", help="download the data and exit")
    ap.add_argument("--sample", default="start", choices=["start", "random"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    amp = a.device == "cuda"

    tr, va = load_stories(a)
    if a.prefetch_only:
        print(f"prefetched {len(tr)} train / {len(va)} valid stories", flush=True)
        return
    vocab = build_vocab(tr)
    V = len(vocab) + 3
    if a.sample == "start":
        Xtr, Xva = encode(tr, vocab, a.ctx).to(a.device), encode(va[:4096], vocab, a.ctx).to(a.device)
        ntok = (Xtr != PAD).sum().item()
    else:
        Str, Sva = encode_stream(tr, vocab).to(a.device), encode_stream(va, vocab).to(a.device)
        gv = torch.Generator(device=a.device).manual_seed(1234)
        Xva = windows(Sva, torch.randint(0, len(Sva) - a.ctx, (4096,), device=a.device, generator=gv), a.ctx)
        ntok = len(Str)
    print(f"{len(tr)} train / {len(va)} valid stories, vocab {V}, train tokens {ntok:,}, sample={a.sample}",
          flush=True)

    model = GPT(V, a.ctx, a.d, a.H, a.D, a.attn, a.gate_kind, a.gate_scale).to(a.device)
    nparam = sum(p.numel() for p in model.parameters())
    gate_p = [p for n, p in model.named_parameters() if ".gate." in n]
    decay = [p for n, p in model.named_parameters() if ".gate." not in n and p.dim() == 2]
    no_decay = [p for n, p in model.named_parameters() if ".gate." not in n and p.dim() < 2]
    groups = [dict(params=decay, weight_decay=a.wd, lr_mult=1.0),
              dict(params=no_decay, weight_decay=0.0, lr_mult=1.0)]
    if gate_p:
        groups.append(dict(params=gate_p, weight_decay=0.0, lr_mult=a.gate_lr_mult))
    opt = torch.optim.AdamW(groups, lr=a.lr, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    g = torch.Generator(device=a.device).manual_seed(a.seed)

    def lr_at(t):
        if t < a.warmup:
            return a.lr * (t + 1) / a.warmup
        p = (t - a.warmup) / max(1, a.steps - a.warmup)
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    log, t0 = [], time.time()
    for step in range(a.steps + 1):
        if step % a.eval_every == 0 or step == a.steps:
            ev = evaluate(model, Xva, amp)
            ev.update(step=step, secs=time.time() - t0)
            log.append(ev)
            print(f"step {step:5d} val {ev['val_loss']:.3f} sinkfrac {ev['sink_frac']:.2f} "
                  f"anchorfrac {ev['anchor_frac']:.2f} "
                  f"meanBOS {ev['mean_bos']:.2f} massive {ev['massive_ratio']:.0f}"
                  + (f" gate {sum(h['gate'] for h in ev['heads']) / len(ev['heads']):.2f}" if a.attn == "gated" else "")
                  + f" {time.time() - t0:.0f}s", flush=True)
            os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
            json.dump(dict(args=vars(a), nparam=nparam, vocab=V, log=log), open(a.out, "w"))
        if step == a.steps:
            break
        for grp in opt.param_groups:
            grp["lr"] = lr_at(step) * grp["lr_mult"]
        if a.sample == "start":
            X = Xtr[torch.randint(0, len(Xtr), (a.batch,), device=a.device, generator=g)]
        else:
            X = windows(Str, torch.randint(0, len(Str) - a.ctx, (a.batch,), device=a.device, generator=g), a.ctx)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            logits, _, _ = model(X)
        loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, V), X[:, 1:].reshape(-1), ignore_index=PAD)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()


if __name__ == "__main__":
    main()
