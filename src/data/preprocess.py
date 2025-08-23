# src/data/preprocess.py
from __future__ import annotations
import argparse, json, os
from typing import Iterator, Dict, Any, List
import pandas as pd
import yaml

from ..utils.io import ensure_dir
from ..utils.text import split_into_sentences, tokenize_sentence, clean_text

def iter_rows_from_csv(path: str, id_col: str, art_col: str, ref_col: str, chunksize: int = 0) -> Iterator[Dict[str, str]]:
    if chunksize and chunksize > 0:
        for chunk in pd.read_csv(path, chunksize=chunksize):
            for _, row in chunk.iterrows():
                yield {
                    "id": str(row.get(id_col, "")),
                    "article": "" if pd.isna(row.get(art_col, "")) else str(row[art_col]),
                    "reference": "" if pd.isna(row.get(ref_col, "")) else str(row[ref_col]),
                }
    else:
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            yield {
                "id": str(row.get(id_col, "")),
                "article": "" if pd.isna(row.get(art_col, "")) else str(row[art_col]),
                "reference": "" if pd.isna(row.get(ref_col, "")) else str(row[ref_col]),
            }

def iter_rows_from_jsonl(path: str) -> Iterator[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def process_example(ex: Dict[str, str], lang_cfg: Dict[str, Any]) -> Dict[str, Any]:
    article = clean_text(ex.get("article", ""))
    sents: List[str] = split_into_sentences(
        article,
        lang=lang_cfg.get("lang", "en"),
        sent_tokenizer=lang_cfg.get("sent_tokenizer", "nltk"),
    )
    sent_objs = []
    for s in sents:
        toks = tokenize_sentence(
            s,
            lang=lang_cfg.get("lang", "en"),
            word_tokenizer=lang_cfg.get("word_tokenizer", "nltk"),
            lowercase=lang_cfg.get("lowercase", True),
            remove_stopwords=lang_cfg.get("remove_stopwords", True),
        )
        sent_objs.append({"text": s, "tokens": toks})
    return {"id": ex.get("id", ""), "sentences": sent_objs, "reference": ex.get("reference", "")}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    ds = cfg["dataset"]
    # Preprocess settings may be at root level or nested under dataset
    pp = cfg.get("preprocess", {}) or d.get("preprocess", {})
    src_path = ds["source_path"]
    out_path = ds["output_path"]
    ensure_dir(os.path.dirname(out_path))

    fmt = ds.get("format", "jsonl").lower()
    if fmt == "csv":
        rows_iter = iter_rows_from_csv(
            src_path,
            id_col=ds.get("id_column", "id"),
            art_col=ds.get("article_column", "article"),
            ref_col=ds.get("reference_column", "highlights"),
            chunksize=int(pp.get("chunksize", 0) or 0),
        )
    else:
        rows_iter = iter_rows_from_jsonl(src_path)

    limit = int(pp.get("max_docs", 0) or pp.get("limit", 0) or 0)
    count = 0
    with open(out_path, "w", encoding="utf-8") as w:
        for ex in rows_iter:
            obj = process_example(ex, pp)
            w.write(json.dumps(obj, ensure_ascii=False) + "\n")
            count += 1
            if limit and count >= limit:
                break

if __name__ == "__main__":
    main()
