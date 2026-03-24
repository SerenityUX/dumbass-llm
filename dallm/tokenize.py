from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer


def load_tokenizer(model: str = "gpt2"):
    tok = AutoTokenizer.from_pretrained(model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def iter_shard_paths(data_dir: Path) -> list[Path]:
    paths = [p for p in data_dir.rglob("*") if p.is_file()]
    paths.sort()
    return paths


def tokenize_extracted_shards(
    data_dir: Path,
    out: Path,
    meta_out: Path,
    model: str,
    max_tokens: int | None,
    max_files: int | None,
    eos_between_shards: bool,
) -> None:
    data_dir = data_dir.resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"Data directory not found: {data_dir}")

    print(f"Load tokenizer: {model}")
    tok = AutoTokenizer.from_pretrained(model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    paths = iter_shard_paths(data_dir)
    if max_files is not None:
        paths = paths[:max_files]
    if not paths:
        raise SystemExit(f"No files under {data_dir}")

    eos_id = tok.eos_token_id
    ids: list[int] = []

    for i, path in enumerate(paths):
        text = path.read_text(encoding="utf-8", errors="replace")
        chunk = tok.encode(text, add_special_tokens=False)
        ids.extend(chunk)
        if eos_between_shards and eos_id is not None:
            ids.append(eos_id)

        if max_tokens is not None and len(ids) >= max_tokens:
            ids = ids[:max_tokens]
            print(f"Stopped at --max-tokens ({max_tokens}) after file {i + 1}/{len(paths)}: {path}")
            break

        if (i + 1) % 50 == 0 or i == 0:
            print(f"files {i + 1}/{len(paths)} token_ids so far: {len(ids):,}")

    arr = np.asarray(ids, dtype=np.uint32)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, arr)

    meta = {
        "tokenizer_model": model,
        "vocab_size": tok.vocab_size,
        "num_token_ids": int(arr.shape[0]),
        "data_dir": str(data_dir),
        "eos_between_shards": eos_between_shards,
    }

    meta_out.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote {arr.shape[0]:,} token IDs -> {out}")
    print(f"Meta -> {meta_out} (vocab_size={tok.vocab_size})")


def _cli() -> None:
    p = argparse.ArgumentParser(description="Tokenize extracted Wikipedia shards to a .npy file.")
    p.add_argument("--data-dir", type=Path, default=Path("./extracted"), help="WikiExtractor output root")
    p.add_argument("--out", type=Path, default=Path("./tokens.npy"), help="Output: NumPy uint32 token IDs")
    p.add_argument("--meta-out", type=Path, default=Path("./tokens_meta.json"), help="Output: vocab info JSON")
    p.add_argument(
        "--model",
        default="gpt2",
        help="Tokenizer id; LM embedding size must match tok.vocab_size",
    )
    p.add_argument("--max-tokens", type=int, default=None, help="Stop after N token IDs (RAM-friendly tests)")
    p.add_argument("--max-files", type=int, default=None, help="Only first N shard files after sort")
    p.add_argument(
        "--eos-between-shards",
        action="store_true",
        help="Append EOS after each file so the model can learn document boundaries",
    )
    ns = p.parse_args()
    tokenize_extracted_shards(
        ns.data_dir,
        ns.out,
        ns.meta_out,
        ns.model,
        ns.max_tokens,
        ns.max_files,
        ns.eos_between_shards,
    )


if __name__ == "__main__":
    _cli()
