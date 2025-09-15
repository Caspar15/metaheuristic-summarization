import argparse
import os
import re
from typing import Dict, List, Optional

import pandas as pd

from src.utils.io import ensure_dir, write_jsonl


# 在中英文標點（. ! ? 。 ！ ？）後切分，允許零或多個空白。
# 例如中文常見無空白的句界，也能正確切分。
_SPLIT_REGEX = re.compile(r"(?<=[.!?。！？])\s*")


def simple_sentence_split(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)
    # 依標點切分（允許無空白），並去除空片段
    parts = [p.strip() for p in _SPLIT_REGEX.split(text) if p.strip()]
    return parts


def preprocess_row(row: Dict, min_tokens: int = 3, max_sentences: Optional[int] = None) -> Dict:
    doc_id = row.get("id")
    article = str(row.get("article", ""))
    highlights = str(row.get("highlights", ""))
    sents = simple_sentence_split(article)
    # ADM #1: filter short sentences
    kept: List[str] = []
    for s in sents:
        if len(s.split()) >= min_tokens:
            kept.append(s)
    if max_sentences is not None and max_sentences > 0:
        kept = kept[: max_sentences]
    return {"id": doc_id, "sentences": kept, "highlights": highlights}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="input CSV path")
    ap.add_argument("--split", required=True, help="dataset split name")
    ap.add_argument("--out", default=None, help="output JSONL path")
    ap.add_argument("--sample_n", type=int, default=None, help="randomly sample N rows")
    ap.add_argument("--sample_frac", type=float, default=None, help="randomly sample a fraction of rows (0,1]")
    ap.add_argument("--limit", type=int, default=None, help="take first N rows (after sampling if any)")
    ap.add_argument("--seed", type=int, default=2024)
    ap.add_argument("--max_sentences", type=int, default=None, help="cap sentences per document after filtering")
    args = ap.parse_args()

    out = args.out or os.path.join("data", "processed", f"{args.split}.jsonl")
    ensure_dir(os.path.dirname(out))

    df = pd.read_csv(args.input)
    # sampling
    if args.sample_n is not None or args.sample_frac is not None:
        df = df.sample(
            n=args.sample_n if args.sample_n is not None else None,
            frac=args.sample_frac if args.sample_frac is not None else None,
            random_state=args.seed,
        )
    if args.limit is not None:
        df = df.head(args.limit)
    rows = (
        preprocess_row(rec, max_sentences=args.max_sentences)
        for rec in df.to_dict(orient="records")
    )
    write_jsonl(out, rows)
    print(f"Wrote processed JSONL to {out}")


if __name__ == "__main__":
    main()
