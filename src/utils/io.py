from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator, List, Dict, Any
from datetime import datetime


def ensure_dir(path: str | Path) -> Path:
    """Ensure a directory exists and return it as Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def timestamp() -> str:
    """Return a compact timestamp string: YYYYMMDD_HHMMSS."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def read_jsonl(path: str | Path, as_list: bool = False) -> Iterator[Dict[str, Any]] | List[Dict[str, Any]]:
    """Read a JSONL file. Return iterator by default, or list if as_list=True."""
    fpath = Path(path)
    if not fpath.exists():
        raise FileNotFoundError(f"JSONL not found: {fpath}")
    def _iter() -> Iterator[Dict[str, Any]]:
        with fpath.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)
    return list(_iter()) if as_list else _iter()


def write_jsonl(path: str | Path, data: Iterable[Dict[str, Any]]) -> None:
    """Write iterable of dicts to JSONL file. Creates parent dirs."""
    fpath = Path(path)
    ensure_dir(fpath.parent)
    with fpath.open("w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

