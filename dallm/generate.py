from __future__ import annotations

from collections.abc import Iterator

import torch
from transformers import AutoTokenizer

from dallm.pretrain import CKPT, GPT


@torch.no_grad()
def generate_text_stream(
    prompt: str = "Wikipedia is",
    max_new: int = 120,
    temperature: float = 0.9,
    checkpoint_path: str | None = None,
    tokenizer_id: str = "gpt2",
    device: torch.device | None = None,
) -> Iterator[str]:
    """Yield newly decoded text after each sampled token (GPT-style streaming)."""
    path = checkpoint_path or str(CKPT)
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["cfg"]
    block_size = cfg["SEQ"]

    model = GPT(cfg["V"], cfg["SEQ"], cfg["D"], cfg["NL"], cfg["NH"]).to(device)
    model.load_state_dict(ckpt["state"])
    model.eval()

    tok = AutoTokenizer.from_pretrained(tokenizer_id)
    ids = tok.encode(prompt, add_special_tokens=False)
    if not ids:
        ids = tok.encode(" ", add_special_tokens=False)

    idx = torch.tensor([ids], dtype=torch.long, device=device)
    prev_text = tok.decode(ids, skip_special_tokens=True)

    for _ in range(max_new):
        idx_in = idx[:, -block_size:]
        logits = model(idx_in)
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_id], dim=1)
        full_text = tok.decode(idx[0].tolist(), skip_special_tokens=True)
        delta = full_text[len(prev_text) :]
        prev_text = full_text
        if delta:
            yield delta


@torch.no_grad()
def generate_text(
    prompt: str = "Wikipedia is",
    max_new: int = 120,
    temperature: float = 0.9,
    checkpoint_path: str | None = None,
    tokenizer_id: str = "gpt2",
    device: torch.device | None = None,
) -> str:
    return "".join(
        generate_text_stream(
            prompt=prompt,
            max_new=max_new,
            temperature=temperature,
            checkpoint_path=checkpoint_path,
            tokenizer_id=tokenizer_id,
            device=device,
        )
    )
