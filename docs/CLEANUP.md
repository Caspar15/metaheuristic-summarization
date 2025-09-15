# 清理與整理建議

## runs/ 整理
- 建議規則：
  - 每次大網格調參完，把 `runs/tune_summary_*.csv` 蒐集成一份總表（參見 `scripts/summarize_runs.py`）。
  - 將不再需要的 `runs/tune1-*` 與舊 `runs/tune2-*` 移動到 `runs/archive/`。
- 提供工具：
  - `scripts/summarize_runs.py`：彙總所有 `tune_summary_*.csv` 至 `runs/summary_all.csv`。
  - `scripts/cleanup_runs.ps1`：互動式選擇要搬移到 `runs/archive/` 的 run 目錄。

## configs/ 整理
- `_generated/`：由調參腳本產出，建議保留最新一批的資料夾，舊批次移至 `configs/_archive/`（手動建立）。
- 固定模板：`features_*` 系列維持只讀；需要客製時複製一份到 `configs/local/`。

## data/ 整理
- `data/processed/validation.stage2.union.*.jsonl`：只保留近期最佳組合的 U 檔即可，其餘可移到 `data/processed/archive/`。

