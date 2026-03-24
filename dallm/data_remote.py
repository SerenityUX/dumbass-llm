from __future__ import annotations

import bz2
import json
import os
import shutil
import urllib.request
from pathlib import Path

ALPACA_URL = (
    "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
)
SIMPLEWIKI_BZ2_URL = (
    "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles.xml.bz2"
)

USER_AGENT = "DumbAssLLM-data-fetch/1.0"
CHUNK = 256 * 1024


def _request(url: str) -> urllib.request.Request:
    return urllib.request.Request(url, headers={"User-Agent": USER_AGENT})


def stream_download(url: str, dest: Path, *, label: str) -> None:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    try:
        with urllib.request.urlopen(_request(url), timeout=600) as resp:
            total = resp.length or int(resp.headers.get("Content-Length") or 0)
            n = 0
            with open(part, "wb") as f:
                while True:
                    chunk = resp.read(CHUNK)
                    if not chunk:
                        break
                    f.write(chunk)
                    n += len(chunk)
                    if total and n % (CHUNK * 40) < CHUNK:
                        print(f"  [{label}] {n // (1024 * 1024)} / {total // (1024 * 1024)} MiB")
        part.replace(dest)
    except BaseException:
        part.unlink(missing_ok=True)
        raise


def ensure_alpaca_json(path: Path | str = Path("./alpaca_data.json")) -> None:
    path = Path(path)
    if path.is_file() and path.stat().st_size > 1000:
        return
    print(f"Downloading Alpaca JSON -> {path.resolve()}")
    stream_download(ALPACA_URL, path, label="alpaca")
    raw = path.read_text(encoding="utf-8")
    json.loads(raw)


def ensure_simplewiki_xml(xml_path: Path | str = Path("./simplewiki.xml")) -> None:
    xml_path = Path(xml_path)
    if xml_path.is_file() and xml_path.stat().st_size > 10_000:
        return

    bz2_path = Path("./simplewiki-latest-pages-articles.xml.bz2")
    if not bz2_path.is_file() or bz2_path.stat().st_size < 1_000_000:
        print(f"Downloading Simple English Wikipedia dump (bz2) -> {bz2_path.resolve()}")
        stream_download(SIMPLEWIKI_BZ2_URL, bz2_path, label="simplewiki.bz2")

    print(f"Decompressing {bz2_path.name} -> {xml_path.name} (this may take a few minutes)")
    with bz2.open(bz2_path, "rb") as src, open(xml_path, "wb") as dst:
        shutil.copyfileobj(src, dst, length=1024 * 1024)

    if not os.environ.get("KEEP_SIMPLEWIKI_BZ2"):
        bz2_path.unlink(missing_ok=True)
        print(f"Removed {bz2_path.name} (set KEEP_SIMPLEWIKI_BZ2=1 to keep the bz2)")
