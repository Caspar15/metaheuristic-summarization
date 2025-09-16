# Pipeline（單段與二階段）

本文描述從資料前處理到摘要輸出的完整流程，並給出單段與二階段的指令範例。

## 單段（Single-Stage）
1) 前處理（CSV→JSONL）
```
python -m src.data.preprocess \
  --input data/raw/validation.csv \
  --split validation \
  --out data/processed/validation.jsonl \
  --max_sentences 25
```

2) 選句輸出（以 Greedy 為例）
```
python -m src.pipeline.select_sentences \
  --config configs/stage1/base/k20.yaml \
  --split validation \
  --input data/processed/validation.jsonl \
  --run_dir runs \
  --optimizer greedy
```

3) 評估
```
python -m src.pipeline.evaluate \
  --pred runs/<stamp>/predictions.jsonl \
  --out  runs/<stamp>/metrics.csv
```

## 二階段（Stage1/Stage2）
Stage1 並行兩路（Base 與 LLM 編碼器）各自產生 Top‑K，組合為聯集 U 後，Stage2 在 U 上做最終決策。

1) Stage1 Base（K1）
```
python -m src.pipeline.select_sentences \
  --config configs/stage1/base/k20.yaml \
  --split validation \
  --input data/processed/validation.jsonl \
  --run_dir runs \
  --optimizer greedy \
  --stamp stage1-base-top20
```

2) Stage1 LLM（K2；三擇一）
```
# BERT
python -m src.pipeline.select_sentences --config configs/stage1/llm/bert/k20.yaml \
  --split validation --input data/processed/validation.jsonl --run_dir runs \
  --optimizer bert --stamp stage1-bert-top20

# RoBERTa
python -m src.pipeline.select_sentences --config configs/stage1/llm/roberta/k20.yaml \
  --split validation --input data/processed/validation.jsonl --run_dir runs \
  --optimizer roberta --stamp stage1-roberta-top20

# XLNet
python -m src.pipeline.select_sentences --config configs/stage1/llm/xlnet/k20.yaml \
  --split validation --input data/processed/validation.jsonl --run_dir runs \
  --optimizer xlnet --stamp stage1-xlnet-top20
```

3) 聯集 U（可選去重與 cap）
```
python scripts/build_union_stage2.py \
  --input data/processed/validation.jsonl \
  --base_pred runs/stage1-base-top20/predictions.jsonl \
  --bert_pred runs/stage1-bert-top20/predictions.jsonl \
  --out data/processed/validation.stage2.union.jsonl \
  --cap 25
```

4) Stage2 最終決策（常見三句）
```
python -m src.pipeline.select_sentences \
  --config configs/stage2/fast/3sent.yaml \
  --split validation \
  --input data/processed/validation.stage2.union.jsonl \
  --run_dir runs \
  --optimizer fast    # 或 fast_grasp / fast_nsga2
```

## 一鍵計時（二階段 + 時間彙總）
- 腳本：`scripts/run_two_stage_timed.py`
- 範例（RoBERTa）：
```
python scripts/run_two_stage_timed.py \
  --input data/processed/validation.jsonl \
  --run_dir runs \
  --base_cfg configs/stage1/base/k20.yaml --opt1 nsga2 \
  --llm_cfg  configs/stage1/llm/roberta/k20.yaml --opt2 roberta \
  --cap 15 \
  --stage2_cfg configs/stage2/fast/3sent.yaml --opt3 fast
```

備註
- Encoder（`bert|roberta|xlnet`）為統一路徑：AutoModel 最後層均值池化 → 重心 cosine 排序。
- Stage2 僅 fast 系列（TF‑IDF 語義 + MMR/GRASP/NSGA2）。



