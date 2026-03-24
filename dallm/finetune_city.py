from __future__ import annotations

import json
import random
import sys
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

CITY_DATA_PATH = Path("./city_data.json")
CITY_CKPT = Path("./checkpoint_city.pt")

US_STATE_CAPITALS: list[tuple[str, str]] = [
    ("Alabama", "Montgomery"),
    ("Alaska", "Juneau"),
    ("Arizona", "Phoenix"),
    ("Arkansas", "Little Rock"),
    ("California", "Sacramento"),
    ("Colorado", "Denver"),
    ("Connecticut", "Hartford"),
    ("Delaware", "Dover"),
    ("Florida", "Tallahassee"),
    ("Georgia", "Atlanta"),
    ("Hawaii", "Honolulu"),
    ("Idaho", "Boise"),
    ("Illinois", "Springfield"),
    ("Indiana", "Indianapolis"),
    ("Iowa", "Des Moines"),
    ("Kansas", "Topeka"),
    ("Kentucky", "Frankfort"),
    ("Louisiana", "Baton Rouge"),
    ("Maine", "Augusta"),
    ("Maryland", "Annapolis"),
    ("Massachusetts", "Boston"),
    ("Michigan", "Lansing"),
    ("Minnesota", "Saint Paul"),
    ("Mississippi", "Jackson"),
    ("Missouri", "Jefferson City"),
    ("Montana", "Helena"),
    ("Nebraska", "Lincoln"),
    ("Nevada", "Carson City"),
    ("New Hampshire", "Concord"),
    ("New Jersey", "Trenton"),
    ("New Mexico", "Santa Fe"),
    ("New York", "Albany"),
    ("North Carolina", "Raleigh"),
    ("North Dakota", "Bismarck"),
    ("Ohio", "Columbus"),
    ("Oklahoma", "Oklahoma City"),
    ("Oregon", "Salem"),
    ("Pennsylvania", "Harrisburg"),
    ("Rhode Island", "Providence"),
    ("South Carolina", "Columbia"),
    ("South Dakota", "Pierre"),
    ("Tennessee", "Nashville"),
    ("Texas", "Austin"),
    ("Utah", "Salt Lake City"),
    ("Vermont", "Montpelier"),
    ("Virginia", "Richmond"),
    ("Washington", "Olympia"),
    ("West Virginia", "Charleston"),
    ("Wisconsin", "Madison"),
    ("Wyoming", "Cheyenne"),
]


def _instruction_variants(state: str) -> list[str]:
    s = state
    return [
        f"What is the capital of {s}?",
        f"What's the capital of {s}?",
        f"Name the capital of {s}.",
        f"Which city is the capital of {s}?",
        f"Capital of {s}?",
        f"The capital city of {s} is?",
        f"State capital of {s}?",
        f"What is the capital city of the state of {s}?",
        f"Tell me the capital of {s}.",
        f"{s}: what is its capital?",
        f"Geography question: capital of {s}?",
        f"Answer with one city: capital of {s}.",
        f"US state trivia: capital of {s}?",
        f"Name {s}'s capital.",
        f"Which city serves as capital of {s}?",
        f"{s} — capital?",
        f"What city is the seat of government of {s}?",
        f"Identify the capital of {s}.",
        f"Quick: capital of {s}?",
        f"For the state of {s}, name the capital.",
        f"{s}'s state capital is called what?",
        f"Quiz: capital of {s}?",
        f"One-word style answer expected. Capital of {s}?",
        f"What is {s}'s capital city?",
        f"Give the capital of {s}.",
        f"{s}: capital city?",
        f"States and capitals: {s}?",
        f"Educational: what is the capital of {s}?",
        f"Fill in the capital: {s} -> ?",
        f"Memorization: capital of {s}?",
        f"AP Geography: capital of {s}?",
        f"Name the seat of {s}.",
        f"Capital city name for {s}?",
        f"{s} (USA): capital?",
        f"Which US state capital belongs to {s}?",
        f"State of {s} — capital?",
        f"Correct capital of {s}?",
        f"Not the largest city necessarily: capital of {s}?",
        f"Official capital of {s}?",
        f"Government capital of {s}?",
        f"List the capital: {s}.",
        f"Short answer: capital of {s}?",
        f"Jeopardy style: This state, {s}, has its capital where?",
    ]


def _output_variants(capital: str) -> list[str]:
    c = capital
    return [
        f"{c}.",
        f"{c}",
        f"The capital is {c}.",
        f"The capital of this state is {c}.",
        f"{c} is the capital.",
        f"It's {c}.",
        f"The answer is {c}.",
    ]


def build_city_rows() -> list[dict]:
    rows: list[dict] = []
    for state, capital in US_STATE_CAPITALS:
        outs = _output_variants(capital)
        for instr in _instruction_variants(state):
            rows.append(
                {
                    "instruction": instr,
                    "input": "",
                    "output": random.choice(outs),
                }
            )
    random.shuffle(rows)
    return rows


def write_city_data_json(path: Path = CITY_DATA_PATH) -> int:
    random.seed(42)
    rows = build_city_rows()
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} examples -> {path.resolve()}")
    return len(rows)


def grab_city_data(path: Path = CITY_DATA_PATH) -> list[dict]:
    if not path.is_file():
        write_city_data_json(path)
    return json.loads(path.read_text(encoding="utf-8"))


def default_base_for_city() -> Path:
    from dallm.finetune_alpaca import SFT_CKPT

    if SFT_CKPT.is_file():
        return SFT_CKPT
    return CKPT


def load_base_model(ckpt_path: Path):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt["cfg"]
    m = GPT(cfg["V"], cfg["SEQ"], cfg["D"], cfg["NL"], cfg["NH"])
    m.load_state_dict(ckpt["state"])
    return m, cfg


def run_city_fine_tune(
    data_path: Path = CITY_DATA_PATH,
    out_ckpt: Path = CITY_CKPT,
    *,
    base_ckpt: Path | None = None,
    steps: int = SFT_STEPS,
    lr: float = SFT_LR,
    batch_size: int = SFT_BS,
) -> None:
    rows = grab_city_data(data_path)
    pairs = format_data(rows)
    tok = load_tokenizer("gpt2")
    src = base_ckpt if base_ckpt is not None else default_base_for_city()
    print(f"City SFT loading weights from {src.resolve()}")
    model, cfg = load_base_model(src)
    examples = tokenize_data(pairs, tok, cfg["SEQ"])
    print(f"City SFT examples after tokenize: {len(examples)}")
    if not examples:
        raise ValueError("No city examples fit in SEQ; increase SEQ in dallm.pretrain.")

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
    if len(sys.argv) > 1 and sys.argv[1] == "--build-data":
        write_city_data_json()
    else:
        run_city_fine_tune()
