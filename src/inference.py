# inference.py — compat loader for trained checkpoint
import argparse
import json
import math
import os
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

import re

punct_end = re.compile(r'[.!?…]+$')

def clean_caption(text: str, vibe_tag: str | None = None) -> str:
    # first line only
    text = text.splitlines()[0]

    # normalize spaces & stray '#'
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'#+', '#', text)          # "##moody" -> "#moody"
    text = re.sub(r'\s#\s', ' ', text)       # " # " -> " "
    text = re.sub(r'(^| )#( |$)', ' ', text) # lone '#' -> drop
    text = text.strip()

    # extract hashtags (letters+digits+underscores, 2..20 chars)
    raw_tags = re.findall(r'(?<!\w)#([A-Za-z][A-Za-z0-9_]{1,19})', text)

    # filter: prefer tags with mostly letters (avoid gibberish)
    def good_tag(t: str) -> bool:
        letters = len(re.findall(r'[A-Za-z]', t))
        return letters >= max(2, int(0.6 * len(t)))

    tags = []
    seen = set()
    # vibe tag first if provided
    if vibe_tag:
        vt = vibe_tag.lower()
        if good_tag(vt) and vt not in seen:
            tags.append(vt); seen.add(vt)
    # then keep up to 2 more distinct tags
    for t in raw_tags:
        t = t.lower()
        if t not in seen and good_tag(t):
            tags.append(t); seen.add(t)
        if len(tags) >= 3:
            break

    # strip all hashtags from sentence part
    sentence = re.sub(r'(?<!\w)#([A-Za-z][A-Za-z0-9_]{1,19})', '', text).strip()

    # fix repeated-char junk words (e.g., "beartbeainsby" or "loooove")
    sentence = re.sub(r'\b(\w{2,})\1{1,}\b', r'\1', sentence)

    # titlecase first letter
    if sentence:
        sentence = sentence[0].upper() + sentence[1:]

    # cap length ~140 chars for the sentence (not counting tags)
    sentence = sentence[:140].rstrip()

    # ensure sentence ends with punctuation
    if sentence and not punct_end.search(sentence):
        sentence += '.'

    # stitch final with up to 3 tags, prefixed by single spaces
    if tags:
        sentence += ' ' + ' '.join('#' + t for t in tags)
    return sentence.strip()


# ---------- Utils ----------
def top_k_filter(logits, k=None):
    if k is None:
        return logits
    k = min(k, logits.size(-1))
    v, _ = torch.topk(logits, k)
    logits[logits < v[..., [-1]]] = -float("inf")
    return logits

# ---------- Compat Model ----------
class CompatSelfAttention(nn.Module):
    def __init__(self, n_embd, block_size, dropout=0.1):
        super().__init__()
        # fused qkv like in training code: weight shape [3*n_embd, n_embd]
        self.qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.proj = nn.Linear(n_embd, n_embd, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        # causal mask buffer (T, T) — 2D so it broadcasts cleanly to (B,T,T)
        mask = torch.tril(torch.ones(block_size, block_size, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=True)

    def forward(self, x):
        # Robustness: accept (B,1,T,C) by squeezing the singleton
        if x.dim() == 4 and x.size(1) == 1:
            x = x[:, 0]  # (B, T, C)
        if x.dim() != 3:
            raise ValueError(f"CompatSelfAttention expected (B,T,C); got {tuple(x.shape)}")

        B, T, C = x.shape
        qkv = self.qkv(x)                         # (B, T, 3C)
        q, k, v = qkv.chunk(3, dim=-1)            # each (B, T, C)

        # scaled dot-prod attention (single-head to match your checkpoint)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(C)      # (B, T, T)
        att = att.masked_fill(~self.mask[:T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v                                         # (B, T, C)
        y = self.proj(y)
        y = self.proj_drop(y)
        return y

class CompatMLP(nn.Module):
    def __init__(self, n_embd, expansion=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, expansion * n_embd),
            nn.GELU(),
            nn.Linear(expansion * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x): return self.net(x)

class CompatBlock(nn.Module):
    def __init__(self, n_embd, block_size, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = CompatSelfAttention(n_embd, block_size, dropout=dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        self.mlp = CompatMLP(n_embd, expansion=4, dropout=dropout)
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x

class CompatGPT(nn.Module):
    def __init__(self, vocab_size, n_embd, n_layer, block_size, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        # positional as (1, T, C) broadcast — implemented via nn.Embedding on index 0..T-1
        self.pos_emb = nn.Embedding(block_size, n_embd)
        self.blocks = nn.ModuleList([
            CompatBlock(n_embd, block_size, dropout=dropout) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(n_embd)
        # checkpoint has lm_head.weight only (no bias)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        if T > self.block_size:
            idx = idx[:, -self.block_size:]
            T = idx.size(1)
        pos = torch.arange(T, device=idx.device)               # (T,)
        x = self.tok_emb(idx) + self.pos_emb(pos)[None, :, :]  # (B, T, C)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# ---------- Loading checkpoint & tokenizer ----------
# -- open and edit /content/MoodMuse/src/inference.py --
# Replace your current load_checkpoint() with this version:

def load_checkpoint(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)

    # stoi/itos & block_size saved by your training loop
    stoi = ckpt.get("stoi"); itos = ckpt.get("itos")
    block_size = ckpt.get("block_size", 128)

    # infer sizes from state_dict
    sd = ckpt["model"]
    vocab_size, n_embd = sd["tok_emb.weight"].shape
    n_layer = max(int(k.split(".")[1]) for k in sd.keys() if k.startswith("blocks.")) + 1

    # prune attn.mask buffers saved in ckpt (shape mismatch vs our runtime buffer)
    pruned = 0
    for k in list(sd.keys()):
        if k.endswith(".attn.mask"):
            sd.pop(k)
            pruned += 1

    model = CompatGPT(vocab_size=vocab_size, n_embd=n_embd, n_layer=n_layer, block_size=block_size)

    # load with strict=False because we intentionally dropped buffers
    missing, unexpected = model.load_state_dict(sd, strict=False)

    # optional: small sanity print
    print(f"Loaded checkpoint. Pruned {pruned} mask tensors. Missing={len(missing)}, Unexpected={len(unexpected)}.")

    model.to(device).eval()
    return model, stoi, itos, block_size

def encode(text, stoi):
    return torch.tensor([stoi[c] for c in text if c in stoi], dtype=torch.long)

def decode(ids, itos):
    return "".join(itos[int(i)] for i in ids)

# ---------- Generation ----------
@torch.no_grad()
def generate_caption(model, stoi, itos, block_size, desc="", vibe="aesthetic",
                     max_new_tokens=120, temperature=0.8, top_k=40, device="cpu"):
    style_hint = f" #{vibe.replace(' ', '')}" if vibe else ""
    prompt = (desc.strip() + style_hint).strip() or "\n"

    idx = encode(prompt, stoi)[None, :].to(device)
    hashtag_seen = 0
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-8)
        if top_k:  # top-k filter
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)

        ch = itos[int(next_id)]
        if ch == "\n":  # stop on newline
            break
        if ch == "#":
            hashtag_seen += 1
            if hashtag_seen >= 4:  # hard cap while sampling
                break

    raw = "".join(itos[int(i)] for i in idx[0].tolist())
    vtag = vibe.replace(' ', '') if vibe else None
    return clean_caption(raw, vibe_tag=vtag)


@torch.no_grad()
def generate_best_of_n(model, stoi, itos, block_size, desc, vibe="moody",
                       n=5, device="cpu"):
    def score(s: str) -> float:
        # simple heuristic: fewer hashtags + more letters
        tags = len(re.findall(r'#\w+', s))
        letters = len(re.findall(r'[A-Za-z]', s))
        return letters - 10*max(0, tags-2)

    cands = [generate_caption(model, stoi, itos, block_size, desc, vibe, device=device)
             for _ in range(n)]
    best = max(cands, key=score)
    return best, cands

# ---------- CLI ----------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="ckpts/moodmuse_best.pt")
    p.add_argument("--desc", type=str, default="golden hour on the beach")
    p.add_argument("--vibe", type=str, default="moody")
    p.add_argument("--max_new_tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=60)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(args.ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt}")

    model, stoi, itos, block_size = load_checkpoint(args.ckpt, device)
    caption = generate_caption(
        model, stoi, itos, block_size,
        desc=args.desc, vibe=args.vibe,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=device
    )
    print("\n=== Caption ===")
    print(caption)

if __name__ == "__main__":
    main()
