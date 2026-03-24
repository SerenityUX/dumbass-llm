from __future__ import annotations

import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F

from dallm.data_remote import ensure_alpaca_json
from dallm.tokenize import load_tokenizer

ALPACA_PATH = Path("./alpaca_data.json")
SFT_CKPT = Path("checkpoint_sft.pt")

IGNORE = -100
SFT_STEPS = 2000
SFT_LR = 3e-5
SFT_BS = 4
SFT_CLIP = 1.0


def grab_data(path: Path) -> list[dict]:
    ensure_alpaca_json(path)
    raw = path.read_text(encoding="utf-8")
    return json.loads(raw)


def alpaca_prompt(instruction: str, input_text: str = "") -> str:
    instruction = str(instruction).strip()
    input_text = str(input_text or "").strip()
    if input_text:
        return (
            f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        )
    return f"### Instruction:\n{instruction}\n\n### Response:\n"


def _format_prompt(row: dict) -> str:
    return alpaca_prompt(str(row["instruction"]), str(row.get("input") or ""))


def _format_full(row: dict) -> str:
    return _format_prompt(row) + str(row["output"]).strip()


def format_data(rows: list[dict]) -> list[tuple[str, str]]:
    return [(_format_full(r), _format_prompt(r)) for r in rows]


def tokenize_data(
    pairs: list[tuple[str, str]],
    tokenizer,
    seq: int,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("tokenizer needs eos_token_id")

    out: list[tuple[torch.Tensor, torch.Tensor]] = []
    for full, prompt in pairs:
        output = full[len(prompt) :]
        p_ids = tokenizer.encode(prompt, add_special_tokens=False)
        o_ids = tokenizer.encode(output, add_special_tokens=False)
        if not o_ids:
            continue
        prompt_len = len(p_ids)
        actual_len = prompt_len + len(o_ids)
        if prompt_len >= actual_len:
            continue
        if actual_len > seq + 1:
            continue

        ids_full = p_ids + o_ids
        while len(ids_full) < seq + 1:
            ids_full.append(eos_id)

        x = torch.tensor(ids_full[:seq], dtype=torch.long)
        y = torch.full((seq,), IGNORE, dtype=torch.long)
        for i in range(seq):
            nxt = ids_full[i + 1]
            if i + 1 < prompt_len:
                continue
            if i + 1 >= actual_len:
                continue
            y[i] = nxt

        out.append((x, y))

    return out


def train_loop(
    model: torch.nn.Module,
    device: torch.device,
    examples: list[tuple[torch.Tensor, torch.Tensor]],
    cfg: dict,
    *,
    steps: int = SFT_STEPS,
    lr: float = SFT_LR,
    batch_size: int = SFT_BS,
    grad_clip: float = SFT_CLIP,
) -> None:
    if not examples:
        raise ValueError("No training examples after tokenize_data (check alpaca_data.json and seq length).")

    V = cfg["V"]
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    for s in range(1, steps + 1):
        idx = random.sample(range(len(examples)), min(batch_size, len(examples)))
        xs = torch.stack([examples[i][0] for i in idx]).to(device)
        ys = torch.stack([examples[i][1] for i in idx]).to(device)

        logits = model(xs)
        loss = F.cross_entropy(logits.view(-1, V), ys.view(-1), ignore_index=IGNORE)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

        if s % 50 == 0 or s == 1:
            print(f"sft step {s:5d} loss {loss.item():.4f}")


def save_model(model: torch.nn.Module, path: Path, cfg: dict) -> None:
    torch.save({"state": model.state_dict(), "cfg": cfg}, path)
    print("saved", path.resolve())
