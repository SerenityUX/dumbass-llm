from __future__ import annotations

import csv
from pathlib import Path

import torch

from dallm.finetune_alpaca import (
    SFT_BS,
    SFT_CLIP,
    SFT_LR,
    SFT_STEPS,
    alpaca_prompt,
    format_data,
    save_model,
    tokenize_data,
    train_loop,
)
from dallm.pretrain import CKPT, GPT
from dallm.tokenize import load_tokenizer

CSV_PATH = Path("./SpongeBobTranscript.csv")
SPONGE_CKPT = Path("./checkpoint_sponge.pt")

PATRICK_INSTRUCTION = (
    "You are Patrick Star from SpongeBob SquarePants. "
    "Reply with exactly one short line of spoken dialogue in his voice: "
    "simple words, earnest and enthusiastic, sometimes confused or a non sequitur, "
    "like on the show. No narration or stage directions—only what Patrick would say out loud."
)


def patrick_prompt(user_message: str = "") -> str:
    u = str(user_message).strip()
    if not u:
        return alpaca_prompt(PATRICK_INSTRUCTION)
    instr = f'{PATRICK_INSTRUCTION}\n\nSomeone asks you: "{u}"'
    return alpaca_prompt(instr)


def default_base_for_patrick() -> Path:
    from dallm.finetune_alpaca import SFT_CKPT

    if SFT_CKPT.is_file():
        return SFT_CKPT
    return CKPT


def load_patrick_utterances(csv_path: Path) -> list[str]:
    if not csv_path.is_file():
        raise FileNotFoundError(csv_path.resolve())

    out: list[str] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "speaker" not in (reader.fieldnames or []) or "utterance" not in (reader.fieldnames or []):
            raise ValueError("CSV needs 'speaker' and 'utterance' columns")

        for row in reader:
            if row.get("speaker", "").strip() != "Patrick":
                continue
            u = (row.get("utterance") or "").strip()
            if u:
                out.append(u)

    return out


def utterances_to_sft_rows(utterances: list[str]) -> list[dict]:
    return [
        {
            "instruction": PATRICK_INSTRUCTION,
            "input": "",
            "output": text,
        }
        for text in utterances
    ]


def load_base_model(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["cfg"]
    m = GPT(cfg["V"], cfg["SEQ"], cfg["D"], cfg["NL"], cfg["NH"])
    m.load_state_dict(ckpt["state"])
    return m, cfg


def run_sponge_fine_tune(
    csv_path: Path = CSV_PATH,
    out_ckpt: Path = SPONGE_CKPT,
    *,
    base_ckpt: Path | None = None,
    steps: int = SFT_STEPS,
    lr: float = SFT_LR,
    batch_size: int = SFT_BS,
) -> None:
    utterances = load_patrick_utterances(csv_path)
    if not utterances:
        raise ValueError("No Patrick lines found (check speaker column == Patrick).")

    print(f"Patrick lines: {len(utterances)}")

    rows = utterances_to_sft_rows(utterances)
    pairs = format_data(rows)
    tok = load_tokenizer("gpt2")
    src = base_ckpt if base_ckpt is not None else default_base_for_patrick()
    print(f"Patrick SFT loading weights from {src.resolve()}")
    model, cfg = load_base_model(src)
    examples = tokenize_data(pairs, tok, cfg["SEQ"])
    print(f"SFT examples after tokenize (len <= seq+1): {len(examples)}")

    if not examples:
        raise ValueError(
            "No examples fit in context length; try increasing SEQ in dallm.pretrain or shortening outputs."
        )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model.to(device)
    train_loop(
        model,
        device,
        examples,
        cfg,
        steps=steps,
        lr=lr,
        batch_size=batch_size,
        grad_clip=SFT_CLIP,
    )
    save_model(model, out_ckpt, cfg)


if __name__ == "__main__":
    run_sponge_fine_tune()
