"""A tiny character-level GPT trained from scratch on Tiny Shakespeare."""
import sys
import time
import urllib.request
from pathlib import Path

import torch
import torch.nn as nn
from torch.nn import functional as F

sys.stdout.reconfigure(encoding="utf-8")

DATA_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
DATA_PATH = Path(__file__).parent / "input.txt"
CHECKPOINT_PATH = Path(__file__).parent / "tiny_gpt.pt"

BLOCK = 256        # context length in characters
BATCH = 64
N_EMBD = 384
N_HEAD = 6
N_LAYER = 6
DROPOUT = 0.2
LR = 3e-4
MAX_ITERS = 3000
EVAL_EVERY = 500
EVAL_ITERS = 50

device = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(1337)


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(N_EMBD, 3 * N_EMBD, bias=False)
        self.proj = nn.Linear(N_EMBD, N_EMBD)
        self.drop = nn.Dropout(DROPOUT)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, N_HEAD, C // N_HEAD).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=DROPOUT if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.drop(self.proj(y))


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.attn = Attention()
        self.ln2 = nn.LayerNorm(N_EMBD)
        self.mlp = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD),
            nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD),
            nn.Dropout(DROPOUT),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.mlp(self.ln2(x))


class TinyGPT(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, N_EMBD)
        self.pos = nn.Embedding(BLOCK, N_EMBD)
        self.blocks = nn.Sequential(*[Block() for _ in range(N_LAYER)])
        self.ln = nn.LayerNorm(N_EMBD)
        self.head = nn.Linear(N_EMBD, vocab_size)

    def forward(self, idx, targets=None):
        _, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        logits = self.head(self.ln(self.blocks(x)))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx[:, -BLOCK:])
            probs = F.softmax(logits[:, -1, :], dim=-1)
            idx = torch.cat([idx, torch.multinomial(probs, 1)], dim=1)
        return idx


def load_text() -> str:
    if not DATA_PATH.exists():
        print(f"Downloading training text to {DATA_PATH} ...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    return DATA_PATH.read_text(encoding="utf-8")


def get_batch(data):
    ix = torch.randint(len(data) - BLOCK - 1, (BATCH,))
    x = torch.stack([data[i:i + BLOCK] for i in ix])
    y = torch.stack([data[i + 1:i + BLOCK + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, splits):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.tensor([model(*get_batch(data))[1].item() for _ in range(EVAL_ITERS)])
        out[name] = losses.mean().item()
    model.train()
    return out


def main():
    text = load_text()
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}

    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    split = int(0.9 * len(data))
    splits = {"train": data[:split], "val": data[split:]}

    model = TinyGPT(len(chars)).to(device)
    print(f"device={device}  vocab={len(chars)}  params={sum(p.numel() for p in model.parameters()) / 1e6:.1f}M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    start = time.time()

    for step in range(MAX_ITERS + 1):
        if step % EVAL_EVERY == 0:
            losses = estimate_loss(model, splits)
            print(f"step {step:>5}  train loss {losses['train']:.3f}  val loss {losses['val']:.3f}  ({time.time() - start:.0f}s)")
        if step == MAX_ITERS:
            break
        _, loss = model(*get_batch(splits["train"]))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.save({"model": model.state_dict(), "chars": chars}, CHECKPOINT_PATH)
    print(f"Saved model to {CHECKPOINT_PATH}\n")

    model.eval()
    seed = torch.zeros((1, 1), dtype=torch.long, device=device)
    sample = model.generate(seed, 500)[0].tolist()
    print("--- Generated text ---")
    print("".join(itos[i] for i in sample))


if __name__ == "__main__":
    main()
