import argparse
import json
import time
from typing import Dict, List, Set


def load_jsonl(path: str) -> List[Dict]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
    return out


def write_jsonl(path: str, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def index_by_id(rows: List[Dict]) -> Dict[str, Dict]:
    out = {}
    for r in rows:
        out[str(r.get("id"))] = r
    return out


def main():
    ap = argparse.ArgumentParser(description="Build stage2 union input from two Top-K runs")
    ap.add_argument("--input", required=True, help="original processed jsonl (with full sentences)")
    ap.add_argument("--base_pred", required=True, help="stage1 base predictions.jsonl (e.g., greedy)")
    ap.add_argument("--bert_pred", required=True, help="stage1 bert predictions.jsonl")
    ap.add_argument("--out", required=True, help="output jsonl path for stage2 union")
    ap.add_argument("--cap", type=int, default=None, help="optional cap for union size (e.g., 25)")
    ap.add_argument("--dedup_threshold", type=float, default=None, help="optional TF-IDF cosine threshold for pre-dedup (e.g., 0.95)")
    args = ap.parse_args()

    t0 = time.perf_counter()
    docs = load_jsonl(args.input)
    base = index_by_id(load_jsonl(args.base_pred))
    bert = index_by_id(load_jsonl(args.bert_pred))

    # lazy import scoring to rank when capping
    try:
        from src.features.tf_isf import sentence_tf_isf_scores  # type: ignore
        from src.features.length import length_scores  # type: ignore
        from src.features.position import position_scores  # type: ignore
        from src.features.compose import combine_scores  # type: ignore
    except Exception:
        sentence_tf_isf_scores = length_scores = position_scores = combine_scores = None  # type: ignore

    out_rows = []
    for d in docs:
        _id = str(d.get("id"))
        sentences: List[str] = d.get("sentences", [])
        u: Set[int] = set()
        if _id in base:
            u.update(base[_id].get("selected_indices", []))
        if _id in bert:
            u.update(bert[_id].get("selected_indices", []))
        idx = [i for i in sorted(u) if 0 <= int(i) < len(sentences)]
        # optional pre-dedup within union using TF-IDF cosine
        if args.dedup_threshold is not None and len(idx) > 1:
            try:
                from sklearn.feature_extraction.text import TfidfVectorizer
                from sklearn.metrics.pairwise import cosine_similarity
                import numpy as _np

                sub = [sentences[i] for i in idx]
                X = TfidfVectorizer(lowercase=True, sublinear_tf=True, ngram_range=(1, 2)).fit_transform(sub)
                S = cosine_similarity(X)
                order = list(range(len(sub)))
                kept = []
                for j in order:
                    ok = True
                    for k in kept:
                        if float(S[j, k]) >= float(args.dedup_threshold):
                            ok = False
                            break
                    if ok:
                        kept.append(j)
                idx = [idx[j] for j in kept]
            except Exception:
                pass

        if args.cap is not None and args.cap > 0 and len(idx) > int(args.cap):
            # Rank union by base feature score if available; fallback to sentence length
            if combine_scores is not None:
                f_importance = sentence_tf_isf_scores(sentences)
                f_len = length_scores(sentences)
                f_pos = position_scores(sentences)
                feats = {"importance": f_importance, "length": f_len, "position": f_pos}
                weights = {"importance": 1.0, "length": 0.3, "position": 0.3}
                scores = combine_scores(feats, weights)
            else:
                scores = [len((s or "").split()) for s in sentences]
            # sort candidate idx by score desc and take top cap
            idx = sorted(idx, key=lambda i: scores[i], reverse=True)[: int(args.cap)]
            # keep stable ordering by original index for readability
            idx = sorted(idx)
        sub_sentences = [sentences[i] for i in idx]
        out_rows.append({
            "id": d.get("id"),
            "sentences": sub_sentences,
            "highlights": d.get("highlights", ""),
        })

    write_jsonl(args.out, out_rows)
    t1 = time.perf_counter()
    elapsed = t1 - t0
    # also emit a sidecar time file next to output for downstream aggregation if needed
    try:
        with open(str(args.out) + ".time_union_seconds.txt", "w", encoding="utf-8") as f:
            f.write(f"{elapsed:.6f}")
    except Exception:
        pass
    print(f"Wrote union stage2 input to {args.out} (time: {elapsed:.6f}s)")


if __name__ == "__main__":
    main()
