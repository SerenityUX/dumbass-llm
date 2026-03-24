from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

from dallm.data_remote import ensure_simplewiki_xml

SIMPLEWIKI_XML = Path("./simplewiki.xml")
EXTRACTED_ROOT = Path("./extracted")

_DOC_OPEN = re.compile(r"<doc[^>]*>\s*", re.MULTILINE)
_DOC_CLOSE = re.compile(r"</doc>\s*", re.MULTILINE)


def strip_wiki_doc_markup(text: str) -> str:
    text = _DOC_OPEN.sub("", text)
    text = _DOC_CLOSE.sub("", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_extracted_dir(root: Path | str = "./extracted") -> int:
    root = Path(root).resolve()
    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")

    written = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        cleaned = strip_wiki_doc_markup(raw)
        if cleaned == raw:
            continue
        if cleaned and not cleaned.endswith("\n"):
            cleaned += "\n"
        path.write_text(cleaned, encoding="utf-8")
        written += 1
    return written


def resolve_wikiextractor() -> str:
    env = os.environ.get("WIKIEXTRACTOR_BIN")
    if env and Path(env).is_file():
        return env
    which = shutil.which("wikiextractor")
    if which:
        return which
    for minor in range(9, 14):
        legacy = Path.home() / f"Library/Python/3.{minor}/bin/wikiextractor"
        if legacy.is_file():
            return str(legacy)
    raise FileNotFoundError(
        "wikiextractor not found. Install with: python3 -m pip install wikiextractor "
        "(ensure the wikiextractor CLI is on PATH), or set WIKIEXTRACTOR_BIN to its path."
    )


def _extracted_has_files(root: Path) -> bool:
    root = root.resolve()
    if not root.is_dir():
        return False
    return any(p.is_file() for p in root.rglob("*"))


def ensure_extracted_shards(
    extracted: Path | str = EXTRACTED_ROOT,
    xml: Path | str = SIMPLEWIKI_XML,
) -> None:
    extracted, xml = Path(extracted), Path(xml)
    if _extracted_has_files(extracted):
        return
    ensure_simplewiki_xml(xml)
    extract_data()


def extract_data() -> None:
    ensure_simplewiki_xml(SIMPLEWIKI_XML)
    wx = resolve_wikiextractor()
    subprocess.run(
        [wx, "-o", str(EXTRACTED_ROOT.resolve()), str(SIMPLEWIKI_XML.resolve())],
        check=True,
    )
    n = clean_extracted_dir(EXTRACTED_ROOT)
    print(f"stripped <doc> markup from {n} file(s) under {EXTRACTED_ROOT}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--clean":
        n = clean_extracted_dir("./extracted")
        print(f"stripped <doc> markup from {n} file(s) under ./extracted")
    else:
        extract_data()
