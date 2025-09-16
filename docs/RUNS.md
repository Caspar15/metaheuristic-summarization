# 讀取輸出（runs 與 tune_summary）

## 單次 Run（runs/<stamp>/）
- `predictions.jsonl`：逐筆輸出
  - `id`, `selected_indices`, `summary`, `reference`
  - 若為 Stage2，索引依據聯集 U 的子集重新對應
- `metrics.csv`：`metric,value`（rouge1/rouge2/rougeL 等）
- `config_used.json`：本次使用的完整設定（便於溯源）
- `time_select_seconds.txt`：選句耗時統計

## 二階段中間產物
- `data/processed/<split>.stage2.union*.jsonl`
  - `sentences`：聯集候選句；Stage2 在其上選最終摘要

## 調參總表（tune_union_fusion.py）
- `runs/tune_summary_<stamp>.csv`
  - 欄位：`base_optimizer,k1,k2,cap,method,w_bert,alpha,rouge1,rouge2,rougeL,pred`
  - `pred`：對應 `predictions.jsonl` 的路徑

## 快速比較範例（Python）
```python
import pandas as pd, glob
files = glob.glob('runs/tune_summary_*.csv')
df = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
print(df.sort_values(['rouge1','rouge2','rougeL'], ascending=False).head(10))
```
