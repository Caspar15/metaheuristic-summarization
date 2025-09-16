# configs 使用說明（結構與範例）

本資料夾存放所有 Pipeline 設定檔。自 2025-09 起，建議採用「按用途與 K 值分層」的結構，避免未來新增 K 變體時越來越混亂。

## 結構總覽
- Stage1/Base：`configs/stage1/base/k{K}.yaml`
- Stage1/LLM：`configs/stage1/llm/{bert|roberta|xlnet}/k{K}.yaml`
- Stage2/Fast：`configs/stage2/fast/3sent.yaml`
- 歷史/歸檔：`configs/_archive/**`
- 產生器產物（舊）：`configs/_generated/**`（僅保留參考；建議改用上面結構）

## 產生 Stage1 K 變體（建議）
- 使用腳本：`scripts/gen_stage1_cfg.py`
- 範例：
```
# 產生 Stage1 Base（K=7）
python scripts/gen_stage1_cfg.py --type base --k 7
# 產生 Stage1 LLM（K=7，RoBERTa）
python scripts/gen_stage1_cfg.py --type llm --k 7 --model roberta
```
- 生成位置：
  - `configs/stage1/base/k7.yaml`
  - `configs/stage1/llm/roberta/k7.yaml`

## 快速使用指令（示例）
- Stage1 Base（K=20）：
```
python -m src.pipeline.select_sentences \
  --config configs/stage1/base/k20.yaml \
  --split validation \
  --input data/processed/validation.jsonl \
  --run_dir runs \
  --optimizer nsga2 \
  --stamp stage1-base-k20
```
- Stage1 LLM（K=20，BERT/RoBERTa/XLNet）：
```
python -m src.pipeline.select_sentences \
  --config configs/stage1/llm/roberta/k20.yaml \
  --split validation \
  --input data/processed/validation.jsonl \
  --run_dir runs \
  --optimizer roberta \
  --stamp stage1-roberta-k20
```
- 聯集 U + Stage2（fast, 3 句）：
```
python scripts/build_union_stage2.py \
  --input data/processed/validation.jsonl \
  --base_pred runs/<stage1-base>/predictions.jsonl \
  --bert_pred runs/<stage1-llm>/predictions.jsonl \
  --out data/processed/validation.stage2.union.k20.jsonl \
  --cap 25

python -m src.pipeline.select_sentences \
  --config configs/stage2/fast/3sent.yaml \
  --split validation \
  --input data/processed/validation.stage2.union.k20.jsonl \
  --run_dir runs \
  --optimizer fast \
  --stamp stage2-fast-k20
```

## 備註
- Stage1 Base/LLM 可透過 CLI 的 `--optimizer` 切換方法（base: greedy|grasp|nsga2；llm: bert|roberta|xlnet）。
- Stage2 僅使用 fast 系列（`fast|fast_grasp|fast_nsga2`）。
- 舊的 `features_*.yaml` 已移至 `_archive/`（legacy）；示例與文件均指向新結構。
- 若需一次計時與產出總表，請參考 `scripts/run_two_stage_timed.py`。
