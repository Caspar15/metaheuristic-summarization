# Scripts 與 Configs（已整理）

本文快速總結 `scripts/` 與 `configs/` 的用途、常見用法與注意事項。

## Scripts 一覽
- `scripts/benchmark_small.py`：在小樣本上對多個優化器跑選句+評估，輸出簡表（<stamp>-summary.csv）。
- `scripts/run_7_2_1.py`：一次性對 train/validation/test 三個 split 跑前處理→選句→評估。
- `scripts/build_union_stage2.py`：將 Stage1 Base（K1）與 Stage1 LLM（K2）的 `predictions.jsonl` 合併為聯集 U，可選 `--cap` 與 TF‑IDF 去重。
- `scripts/tune_union_fusion.py`：二階段（Stage1/2）調參與彙總：自動產生 `_generated/` YAML；Stage2 僅支援 `fast|fast_grasp|fast_nsga2`；輸出 `runs/tune_summary_*.csv`。
- `scripts/grid_stage2_fast.py`：Stage2 fast 系列（不依賴 BERT）的網格搜尋，產生 `_generated/` 變體，執行與彙總。
- `scripts/run_two_stage_timed.py`：二階段整流程執行＋各階段時間彙總（Stage1 Base/LLM 與 Stage2），輸出 `timed_summary_*.csv`。
- 資料維護：
  - `scripts/split_dataset.py`：將單一 CSV 依比例切分為 train/validation/test（欄位 `id, article, highlights`）。
  - `scripts/jsonl_to_csv.py`：將 JSONL（含 `id, article, reference/highlights`）轉為 CSV。
  - `scripts/summarize_runs.py`：匯總多個 `tune_summary_*.csv` 為 `runs/summary_all.csv`。
  - `scripts/organize_runs.py`：重組 `runs/` 結構（支援 `--apply` 實際移動）。
  - `scripts/cleanup_runs.ps1`（已歸檔於 `scripts/_archive/`）：舊的 PowerShell 輔助腳本。

## Configs 一覽
- 常用模板：
  - `configs/stage1/base/k20.yaml`：單段基線（TF‑IDF + greedy，含候選池設定）
  - `configs/stage2/fast/3sent.yaml`：三句摘要（建議搭配 greedy|grasp|nsga2 或在 U 上使用 fast 系列）
  - `configs/stage1/base/k20.yaml`：Stage1 Base Top‑K 模板（K1）
  - `configs/stage1/llm/bert/k20.yaml`：Stage1 LLM（BERT）Top‑K 模板（K2）；RoBERTa/XLNet 有對應模板
  - `configs/_generated/**`：由腳本產出（歷史上包含 bert/fused 的 Stage2 變體；現行 Stage2 僅 fast 系列）

## 相依與注意事項
- 編碼器/Stage1：`bert|roberta|xlnet` 需 `torch`, `transformers`（SBERT 另需 `sentence-transformers`）。
- NSGA‑II：`nsga2` 與 `fast_nsga2` 需 `pymoo==0.6.1.1`；若 `representations.use=false` 無相似度，NSGA‑II 將退回 greedy。
- `recall_target` 需 `rouge-score` 以計算 oracle recall。
- `_generated/` 常含大量調參輸出，建議定期整理/歸檔。

## 典型流程（示例）
- Stage1：
  - Base：`python -m src.pipeline.select_sentences --config configs/stage1/base/k20.yaml --split validation --input data/processed/validation.jsonl --run_dir runs --optimizer greedy --stamp stage1-greedy-k20`
  - LLM：`python -m src.pipeline.select_sentences --config configs/stage1/llm/bert/k20.yaml --split validation --input data/processed/validation.jsonl --run_dir runs --optimizer bert --stamp stage1-bert-k20`
- 聯集 U + Stage2：
  - `python scripts/build_union_stage2.py --input data/processed/validation.jsonl --base_pred runs/stage1-greedy-k20/predictions.jsonl --bert_pred runs/stage1-bert-k20/predictions.jsonl --out data/processed/validation.stage2.union.jsonl --cap 25`
  - `python -m src.pipeline.select_sentences --config configs/stage2/fast/3sent.yaml --split validation --input data/processed/validation.stage2.union.jsonl --run_dir runs --optimizer fast --stamp stage2-fast-top3`

## 清理建議
- `configs/`：常用模板放置於此；大量 `_generated/` 產物可批次移至 `configs/_archive/`。
- `runs/`：使用 `scripts/organize_runs.py` 規整；用 `scripts/summarize_runs.py` 彙總 `tune_summary_*.csv`。


### 產生 Stage1 K 變體（建議做法）
- 腳本：scripts/gen_stage1_cfg.py
- 範例：
``
python scripts/gen_stage1_cfg.py --type base --k 7
python scripts/gen_stage1_cfg.py --type llm --k 7 --model roberta
```n- 生成位置：configs/stage1/base/k7.yaml、configs/stage1/llm/roberta/k7.yaml（可用 --out_root 自訂）

### 建議結構
- Stage1 Base：configs/stage1/base/k{K}.yaml
- Stage1 LLM：configs/stage1/llm/{bert|roberta|xlnet}/k{K}.yaml
- Stage2（fast）：configs/stage2/fast/3sent.yaml



