# Pipeline（兩階段流程）

目標：以模組化方式完成抽取式三句摘要，並支援二階段（聯集）流程。

## 名詞與模組
- Base 特徵：TF‑ISF / 長度 / 句位（`src/features/*` → `combine_scores`）。
- 相似度表示：TF‑IDF 或 SBERT（`src/representations/*`）。
- 選句器（Stage1 Base）：`greedy` / `grasp` / `nsga2`（`src/models/extractive/*`）。
- BERT 排序（Stage1 另一條路、或 Stage2 最終）：`src/models/extractive/bert_rank.py`。
- Fused（Stage2 最終）：base 分數 + BERT 分數融合後，以 BERT 相似度做 MMR（`src/models/extractive/fused.py`）。

## 兩階段（做法 B）
1) Stage1 並行兩條路，各自直接選句
   - Base 路徑：特徵 + 相似度 + 演算法 → 取 K1 句
   - BERT 路徑：BERT 句向量對文件重心排序 → 取 K2 句
2) 聯集 U + cap 截斷
   - U = K1 ∪ K2；若超 cap，先用 Base 分數對 U 排序取前 cap（僅為保留候選）
   - 產物：`data/processed/validation.stage2.union.k1_<K1>.k2_<K2>.cap_<cap>.<stamp>.jsonl`
3) Stage2 最終三句（在 U 上）
   - method=bert：BERT 排序直接取 3 句
   - method=fused：S = w_base*base + w_bert*bert；用 BERT 向量算 pairwise 相似度，跑 MMR 取 3 句

備註：若要「快速 Stage2」且不使用 BERT，可直接在 U 上用 `optimizer=greedy`（TF‑IDF 表示）選 3 句，但品質通常低於 bert/fused。

## 典型命令
- 前處理：`python -m src.data.preprocess --input data/raw/.../validation.csv --split validation --out data/processed/validation.jsonl --max_sentences 25`
- 單階段三句：`python -m src.pipeline.select_sentences --config configs/features_3sent.yaml --split validation --input data/processed/validation.jsonl --run_dir runs --optimizer greedy`
- 二階段聯集與調參：`python scripts/tune_union_fusion.py --input data/processed/validation.jsonl --run_dir runs --k1 20 --k2 20 --cap 25 --methods bert fused --optimizer1 greedy`

