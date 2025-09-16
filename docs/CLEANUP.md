# 清理與整理建議（CLEANUP）

## runs/
- 建議規劃：
  - 每次大規模調參輸出 `runs/tune_summary_*.csv`，可用 `scripts/summarize_runs.py` 匯總。
  - 將過舊的 `runs/<stamp>` 移動至 `runs/archive/`。
- 輔助工具：
  - `scripts/summarize_runs.py`：彙整多份 `tune_summary_*.csv` 為 `runs/summary_all.csv`。
  - `scripts/cleanup_runs.ps1`：將舊 run 批次搬移到 `runs/archive/`。

## configs/
- `_generated/`：調參腳本輸出；可視情況批次移至 `configs/_archive/`。
- 常用模板：`features_*` 系列維持乾淨；客製化建議放在 `configs/local/`。

## data/
- `data/processed/<split>.stage2.union.*.jsonl`：只保留近期最佳的 U 檔，其餘移至 `data/processed/archive/`。
