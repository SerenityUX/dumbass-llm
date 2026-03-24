from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import torch

from dallm.finetune_alpaca import (
    ALPACA_PATH,
    SFT_CKPT,
    alpaca_prompt,
    format_data,
    grab_data,
    save_model,
    tokenize_data,
    train_loop,
)
from dallm.generate import generate_text_stream
from dallm.pretrain import CKPT, GPT, train
from dallm.tokenize import load_tokenizer, tokenize_extracted_shards
from dallm.wiki_extract import extract_data, ensure_extracted_shards


def _color_enabled() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


_PRIMARY = 34


def _st(text: str, *codes: int) -> str:
    if not _color_enabled():
        return text
    return f"\033[{';'.join(str(c) for c in codes)}m{text}\033[0m"


def _banner_line() -> str:
    if not _color_enabled():
        return " DUMBASS LLM "
    return _st(" DUMBASS LLM ", 1, _PRIMARY)


def _prompt_model() -> str:
    return _st("Which model (1-4, q to quit): ", 1, _PRIMARY)


def _prompt_question() -> str:
    return _st("Your question (q to quit): ", 1, _PRIMARY)


def _sep() -> str:
    return _st("· ─── · ─── · ─── ·", 2, 37)


def _bye() -> str:
    return _st("Bye.", 2, 37)


def _word_sound_enabled() -> bool:
    """One short click per word; set DUMBASSLLM_WORD_SOUND=0 to disable."""
    return os.environ.get("DUMBASSLLM_WORD_SOUND", "1") != "0"


def _play_word_sound() -> None:
    if not _word_sound_enabled():
        return
    if sys.platform == "darwin":
        try:
            subprocess.Popen(
                ["afplay", "/System/Library/Sounds/Tink.aiff"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (FileNotFoundError, OSError):
            print("\a", end="", flush=True)
    else:
        # Linux/Windows: terminal bell (no extra deps; non-blocking)
        print("\a", end="", flush=True)


def _print_stream_with_word_sounds(chunks: Iterator[str]) -> None:
    """Print streamed text; after each whitespace-delimited word, play a short sound."""
    buf = ""
    for chunk in chunks:
        buf += chunk
        while True:
            split_at = None
            for i, c in enumerate(buf):
                if c in " \n\t\r":
                    split_at = i
                    break
            if split_at is None:
                break
            word = buf[: split_at + 1]
            print(word, end="", flush=True)
            _play_word_sound()
            buf = buf[split_at + 1 :]
    if buf:
        print(buf, end="", flush=True)
        _play_word_sound()


def _parse_model_input(s: str) -> str | None:
    m = s.strip().lower()
    if m in ("q", "quit"):
        return "quit"
    if m in ("1", "base"):
        return "1"
    if m in ("2", "sft"):
        return "2"
    if m in ("3", "patrick"):
        return "3"
    if m in ("4", "city"):
        return "4"
    return None


def _print_title_screen() -> None:
    print(_banner_line())
    rows = (
        ("1", "Base", "base model with limited training"),
        ("2", "SFT", "base model fine-tuned on question–answer data"),
        ("3", "Patrick", "fine-tuned on SpongeBob SquarePants' Patrick Star"),
        ("4", "City", "fine-tuned on the capital of every US state"),
    )
    for num, name, desc in rows:
        print(
            _st(f"  {num} ", 1, _PRIMARY)
            + _st(f"{name} — ", 2, 37)
            + _st(desc, 2, 37)
        )
    print()


data_need_extraction = False
data_need_tokenization = False
data_need_training = False
data_need_fine_tuning = False
data_need_train_patrick = False
data_need_train_city = False


def model_architecture():
    ckpt = torch.load(CKPT, map_location="cpu")
    cfg = ckpt["cfg"]
    m = GPT(cfg["V"], cfg["SEQ"], cfg["D"], cfg["NL"], cfg["NH"])
    m.load_state_dict(ckpt["state"])
    return m, cfg


def run_training_pipeline() -> None:
    if data_need_train_patrick and data_need_train_city:
        raise ValueError("Pick one: data_need_train_patrick or data_need_train_city per run.")

    if data_need_extraction:
        extract_data()

    if data_need_tokenization:
        ensure_extracted_shards()
        tokenize_extracted_shards(
            Path("./extracted").resolve(),
            Path("./tokens.npy"),
            Path("./tokens_meta.json"),
            "gpt2",
            None,
            None,
            False,
        )

    if data_need_training:
        train()

    if data_need_fine_tuning:
        rows = grab_data(ALPACA_PATH)
        pairs = format_data(rows)
        tok = load_tokenizer("gpt2")
        model, cfg = model_architecture()
        examples = tokenize_data(pairs, tok, cfg["SEQ"])
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        model.to(device)
        train_loop(model, device, examples, cfg)
        save_model(model, SFT_CKPT, cfg)

    if data_need_train_patrick:
        from dallm.finetune_sponge import run_sponge_fine_tune

        run_sponge_fine_tune()

    if data_need_train_city:
        from dallm.finetune_city import run_city_fine_tune

        run_city_fine_tune()


def run_interactive_cli() -> None:
    from dallm.finetune_city import CITY_CKPT
    from dallm.finetune_sponge import SPONGE_CKPT, patrick_prompt

    _print_title_screen()

    while True:
        choice = ""
        while not choice:
            raw = input(_prompt_model()).strip()
            parsed = _parse_model_input(raw)
            if parsed == "quit":
                print(_bye())
                return
            if parsed is not None:
                choice = parsed
            else:
                print(_st("  Type 1, 2, 3, or 4 (or q to quit).", 2, 37))

        print()
        question = input(_prompt_question()).strip()
        if question.lower() in ("q", "quit"):
            print(_bye())
            return
        if not question:
            question = "Hello."

        if choice == "1":
            prompt = question
            path = str(CKPT)
            temp = 0.9
            max_new = 32
        elif choice == "2":
            prompt = alpaca_prompt(question)
            path = str(SFT_CKPT)
            temp = 0.7
            max_new = 32
        elif choice == "3":
            prompt = patrick_prompt(question)
            path = str(SPONGE_CKPT)
            temp = 0.82
            max_new = 48
        else:
            prompt = alpaca_prompt(question)
            path = str(CITY_CKPT)
            temp = 0.5
            max_new = 12

        print()
        print(_sep())
        _print_stream_with_word_sounds(
            generate_text_stream(
                prompt=prompt,
                max_new=max_new,
                temperature=temp,
                checkpoint_path=path,
            )
        )
        print()
        print(_sep())
        print()
        _print_title_screen()


def main() -> None:
    training = any(
        (
            data_need_extraction,
            data_need_tokenization,
            data_need_training,
            data_need_fine_tuning,
            data_need_train_patrick,
            data_need_train_city,
        )
    )
    if training:
        run_training_pipeline()
    run_interactive_cli()


if __name__ == "__main__":
    main()
